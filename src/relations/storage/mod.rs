//! Storage layer for code relationships.
//!
//! This module provides persistent storage for definitions and references
//! using LanceDB tables.

mod lance_store;

pub use lance_store::LanceRelationsStore;

use anyhow::Result;
use async_trait::async_trait;

use crate::relations::types::{CallEdge, Definition, Reference};

/// Trait for storing and querying code relationships.
#[async_trait]
pub trait RelationsStore: Send + Sync {
    /// Store definitions for a file
    async fn store_definitions(
        &self,
        definitions: Vec<Definition>,
        root_path: &str,
    ) -> Result<usize>;

    /// Store references for a file
    async fn store_references(&self, references: Vec<Reference>, root_path: &str) -> Result<usize>;

    /// Find definition at a specific location
    async fn find_definition_at(
        &self,
        file_path: &str,
        line: usize,
        column: usize,
    ) -> Result<Option<Definition>>;

    /// Find all definitions with a given name
    async fn find_definitions_by_name(&self, name: &str) -> Result<Vec<Definition>>;

    /// Find all references to a symbol
    async fn find_references(&self, target_symbol_id: &str) -> Result<Vec<Reference>>;

    /// Get callers of a function (incoming call edges)
    async fn get_callers(&self, symbol_id: &str) -> Result<Vec<CallEdge>>;

    /// Get callees of a function (outgoing call edges)
    async fn get_callees(&self, symbol_id: &str) -> Result<Vec<CallEdge>>;

    /// Delete all relationships for a file (for incremental updates)
    async fn delete_by_file(&self, file_path: &str) -> Result<usize>;

    /// Clear all relationships
    async fn clear(&self) -> Result<()>;

    /// Get statistics
    async fn get_stats(&self) -> Result<RelationsStats>;
}

/// Statistics about stored relationships
#[derive(Debug, Clone, Default)]
pub struct RelationsStats {
    /// Total number of definitions
    pub definition_count: usize,
    /// Total number of references
    pub reference_count: usize,
    /// Number of unique files with definitions
    pub files_with_definitions: usize,
}
