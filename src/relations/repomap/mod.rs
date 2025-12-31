//! RepoMap-style code relationship extraction.
//!
//! This module provides AST-based extraction of symbol definitions and references
//! for all languages supported by tree-sitter. It's the fallback provider when
//! stack-graphs isn't available for a language.
//!
//! ## How it works
//!
//! 1. **Definition extraction**: Parse AST and extract function/class/method definitions
//! 2. **Reference finding**: Find identifier occurrences that match known symbol names
//!
//! ## Accuracy
//!
//! This approach achieves ~70% accuracy because it uses heuristic name matching
//! rather than full semantic analysis. It may:
//! - Miss references to renamed imports
//! - Include false positives for common names
//! - Not resolve which overload is being called

pub mod reference_finder;
pub mod symbol_extractor;

use std::collections::HashMap;

use anyhow::Result;

use crate::indexer::FileInfo;
use crate::relations::{Definition, PrecisionLevel, Reference, RelationsProvider};

pub use reference_finder::ReferenceFinder;
pub use symbol_extractor::SymbolExtractor;

/// RepoMap-style relations provider using AST-based extraction.
pub struct RepoMapProvider {
    /// Symbol extractor for parsing definitions
    symbol_extractor: SymbolExtractor,
    /// Reference finder for locating symbol usages
    reference_finder: ReferenceFinder,
}

impl RepoMapProvider {
    /// Create a new RepoMap provider
    pub fn new() -> Self {
        Self {
            symbol_extractor: SymbolExtractor::new(),
            reference_finder: ReferenceFinder::new(),
        }
    }
}

impl Default for RepoMapProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl RelationsProvider for RepoMapProvider {
    fn extract_definitions(&self, file_info: &FileInfo) -> Result<Vec<Definition>> {
        self.symbol_extractor.extract_definitions(file_info)
    }

    fn extract_references(
        &self,
        file_info: &FileInfo,
        symbol_index: &HashMap<String, Vec<Definition>>,
    ) -> Result<Vec<Reference>> {
        self.reference_finder
            .find_references(file_info, symbol_index)
    }

    fn supports_language(&self, _language: &str) -> bool {
        // RepoMap supports all languages (with varying accuracy)
        true
    }

    fn precision_level(&self, _language: &str) -> PrecisionLevel {
        PrecisionLevel::Medium
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repomap_provider_creation() {
        let provider = RepoMapProvider::new();
        assert!(provider.supports_language("Rust"));
        assert!(provider.supports_language("Python"));
        assert!(provider.supports_language("Unknown"));
    }

    #[test]
    fn test_precision_level() {
        let provider = RepoMapProvider::new();
        assert_eq!(provider.precision_level("Rust"), PrecisionLevel::Medium);
        assert_eq!(provider.precision_level("Python"), PrecisionLevel::Medium);
    }
}
