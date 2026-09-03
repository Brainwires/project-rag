//! High-precision name resolution via stack-graphs (Python, TypeScript, Java, Ruby).
//!
//! **Status: not yet implemented.** The `stack-graphs` feature flag and this module
//! exist so [`crate::relations::HybridRelationsProvider`] has a real type to hold and
//! a real fallback path to exercise, but no stack-graphs crate is wired in yet.
//! [`StackGraphsProvider::new`] always returns an error, which `HybridRelationsProvider`
//! already handles by logging a warning and falling back to [`crate::relations::repomap::RepoMapProvider`]
//! for every language -- so enabling this feature today changes nothing observable.

use anyhow::{Result, bail};
use std::collections::HashMap;

use crate::indexer::FileInfo;
use crate::relations::{Definition, PrecisionLevel, Reference, RelationsProvider};

/// Placeholder for the future stack-graphs-backed provider.
///
/// Not constructible via the normal path: [`StackGraphsProvider::new`] always errors,
/// so [`crate::relations::HybridRelationsProvider`] never actually holds one of these
/// today. The type exists to keep the feature-gated field and call sites in
/// `relations/mod.rs` compiling against a real API shape.
pub struct StackGraphsProvider {
    _private: (),
}

impl StackGraphsProvider {
    /// Always fails: stack-graphs support has not been implemented yet.
    pub fn new() -> Result<Self> {
        bail!("stack-graphs support is not yet implemented")
    }

    /// No languages are supported yet.
    pub fn supports_language(&self, _language: &str) -> bool {
        false
    }
}

impl RelationsProvider for StackGraphsProvider {
    fn extract_definitions(&self, _file_info: &FileInfo) -> Result<Vec<Definition>> {
        bail!("stack-graphs support is not yet implemented")
    }

    fn extract_references(
        &self,
        _file_info: &FileInfo,
        _symbol_index: &HashMap<String, Vec<Definition>>,
    ) -> Result<Vec<Reference>> {
        bail!("stack-graphs support is not yet implemented")
    }

    fn supports_language(&self, _language: &str) -> bool {
        false
    }

    fn precision_level(&self, _language: &str) -> PrecisionLevel {
        PrecisionLevel::High
    }
}
