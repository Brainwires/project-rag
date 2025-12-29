//! Code relationships module for definition/reference tracking and call graphs.
//!
//! This module provides capabilities for understanding code relationships:
//! - Find where symbols are defined
//! - Find all references to a symbol
//! - Build call graphs for functions/methods
//!
//! ## Architecture
//!
//! The module uses a hybrid approach:
//! - **Stack-graphs** (optional feature): High-precision name resolution for Python,
//!   TypeScript, Java, and Ruby (~95% accuracy)
//! - **RepoMap**: AST-based extraction with heuristic matching for all languages
//!   (~70% accuracy)
//!
//! ## Usage
//!
//! ```ignore
//! use project_rag::relations::{HybridRelationsProvider, RelationsProvider};
//!
//! let provider = HybridRelationsProvider::new()?;
//! let definitions = provider.extract_definitions(&file_info)?;
//! let references = provider.extract_references(&file_info, &symbol_index)?;
//! ```

pub mod repomap;
pub mod storage;
pub mod types;

#[cfg(feature = "stack-graphs")]
pub mod stack_graphs;

use anyhow::Result;

pub use types::{
    CallEdge, CallGraphNode, Definition, DefinitionResult, PrecisionLevel, Reference,
    ReferenceKind, ReferenceResult, SymbolId, SymbolInfo, SymbolKind, Visibility,
};

use crate::indexer::FileInfo;
use std::collections::HashMap;

/// Trait for extracting code relationships from source files.
///
/// Implementors of this trait can extract symbol definitions and references
/// from source code files.
pub trait RelationsProvider: Send + Sync {
    /// Extract definitions from a file.
    ///
    /// Returns a list of all symbol definitions (functions, classes, etc.)
    /// found in the given file.
    fn extract_definitions(&self, file_info: &FileInfo) -> Result<Vec<Definition>>;

    /// Extract references from a file.
    ///
    /// `symbol_index` maps symbol names to their definitions, used for
    /// resolving which symbol a reference points to.
    fn extract_references(
        &self,
        file_info: &FileInfo,
        symbol_index: &HashMap<String, Vec<Definition>>,
    ) -> Result<Vec<Reference>>;

    /// Check if this provider supports the given language.
    fn supports_language(&self, language: &str) -> bool;

    /// Get the precision level of this provider for the given language.
    fn precision_level(&self, language: &str) -> PrecisionLevel;
}

/// Hybrid provider that selects the best available provider per language.
///
/// Uses stack-graphs for supported languages (Python, TypeScript, Java, Ruby)
/// when the feature is enabled, and falls back to RepoMap for all other languages.
pub struct HybridRelationsProvider {
    /// Stack-graphs provider (if feature enabled)
    #[cfg(feature = "stack-graphs")]
    stack_graphs: Option<stack_graphs::StackGraphsProvider>,

    /// RepoMap provider (always available)
    repomap: repomap::RepoMapProvider,
}

impl HybridRelationsProvider {
    /// Create a new hybrid relations provider.
    ///
    /// If `enable_stack_graphs` is true and the feature is enabled,
    /// stack-graphs will be used for supported languages.
    pub fn new(_enable_stack_graphs: bool) -> Result<Self> {
        #[cfg(feature = "stack-graphs")]
        let stack_graphs = if _enable_stack_graphs {
            match stack_graphs::StackGraphsProvider::new() {
                Ok(sg) => Some(sg),
                Err(e) => {
                    tracing::warn!("Failed to initialize stack-graphs: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            #[cfg(feature = "stack-graphs")]
            stack_graphs,
            repomap: repomap::RepoMapProvider::new(),
        })
    }

    /// Get the best provider for a given language.
    fn provider_for_language(&self, _language: &str) -> &dyn RelationsProvider {
        #[cfg(feature = "stack-graphs")]
        if let Some(ref sg) = self.stack_graphs {
            if sg.supports_language(_language) {
                return sg;
            }
        }

        &self.repomap
    }

    /// Check if stack-graphs is available for a language.
    #[cfg(feature = "stack-graphs")]
    pub fn has_stack_graphs_for(&self, language: &str) -> bool {
        self.stack_graphs
            .as_ref()
            .is_some_and(|sg| sg.supports_language(language))
    }

    /// Check if stack-graphs is available for a language.
    #[cfg(not(feature = "stack-graphs"))]
    pub fn has_stack_graphs_for(&self, _language: &str) -> bool {
        false
    }
}

impl RelationsProvider for HybridRelationsProvider {
    fn extract_definitions(&self, file_info: &FileInfo) -> Result<Vec<Definition>> {
        let language = file_info.language.as_deref().unwrap_or("Unknown");
        self.provider_for_language(language)
            .extract_definitions(file_info)
    }

    fn extract_references(
        &self,
        file_info: &FileInfo,
        symbol_index: &HashMap<String, Vec<Definition>>,
    ) -> Result<Vec<Reference>> {
        let language = file_info.language.as_deref().unwrap_or("Unknown");
        self.provider_for_language(language)
            .extract_references(file_info, symbol_index)
    }

    fn supports_language(&self, language: &str) -> bool {
        // We support all languages through RepoMap fallback
        self.repomap.supports_language(language)
    }

    fn precision_level(&self, language: &str) -> PrecisionLevel {
        #[cfg(feature = "stack-graphs")]
        if self.has_stack_graphs_for(language) {
            return PrecisionLevel::High;
        }

        self.repomap.precision_level(language)
    }
}

/// Configuration for relations extraction
#[derive(Debug, Clone)]
pub struct RelationsConfig {
    /// Whether relations extraction is enabled
    pub enabled: bool,
    /// Whether to use stack-graphs when available
    pub use_stack_graphs: bool,
    /// Maximum call graph traversal depth
    pub max_call_depth: usize,
}

impl Default for RelationsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            use_stack_graphs: cfg!(feature = "stack-graphs"),
            max_call_depth: 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_provider_creation() {
        let provider = HybridRelationsProvider::new(false).unwrap();
        assert!(provider.supports_language("Rust"));
        assert!(provider.supports_language("Python"));
        assert!(provider.supports_language("Unknown"));
    }

    #[test]
    fn test_precision_level_without_stack_graphs() {
        let provider = HybridRelationsProvider::new(false).unwrap();
        // Without stack-graphs, everything uses RepoMap (medium precision)
        assert_eq!(provider.precision_level("Rust"), PrecisionLevel::Medium);
        assert_eq!(provider.precision_level("Python"), PrecisionLevel::Medium);
    }

    #[test]
    fn test_relations_config_default() {
        let config = RelationsConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_call_depth, 3);
    }

    #[test]
    fn test_has_stack_graphs() {
        let provider = HybridRelationsProvider::new(false).unwrap();
        // Without the feature, should always return false
        #[cfg(not(feature = "stack-graphs"))]
        assert!(!provider.has_stack_graphs_for("Python"));
    }
}
