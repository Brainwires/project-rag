//! LanceDB-based storage for code relationships.

use anyhow::{Context, Result};
use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{RelationsStats, RelationsStore};
use crate::relations::types::{CallEdge, Definition, Reference};

/// LanceDB-based relations store.
///
/// Stores definitions and references in separate LanceDB tables for efficient querying.
pub struct LanceRelationsStore {
    /// Path to the database directory
    db_path: PathBuf,
    /// Database connection (lazy initialized)
    db: Arc<RwLock<Option<lancedb::Connection>>>,
}

impl LanceRelationsStore {
    /// Create a new LanceDB relations store
    pub async fn new(db_path: PathBuf) -> Result<Self> {
        // Ensure directory exists
        tokio::fs::create_dir_all(&db_path)
            .await
            .context("Failed to create relations database directory")?;

        Ok(Self {
            db_path,
            db: Arc::new(RwLock::new(None)),
        })
    }

    /// Get or create the database connection
    async fn get_connection(&self) -> Result<lancedb::Connection> {
        let mut db_guard = self.db.write().await;

        if let Some(ref db) = *db_guard {
            return Ok(db.clone());
        }

        let db = lancedb::connect(self.db_path.to_string_lossy().as_ref())
            .execute()
            .await
            .context("Failed to connect to LanceDB")?;

        *db_guard = Some(db.clone());
        Ok(db)
    }

    /// Ensure definitions table exists
    async fn ensure_definitions_table(&self) -> Result<()> {
        let _db = self.get_connection().await?;
        // Table will be created on first insert
        // LanceDB creates tables lazily
        Ok(())
    }

    /// Ensure references table exists
    async fn ensure_references_table(&self) -> Result<()> {
        let _db = self.get_connection().await?;
        // Table will be created on first insert
        Ok(())
    }
}

#[async_trait]
impl RelationsStore for LanceRelationsStore {
    async fn store_definitions(
        &self,
        definitions: Vec<Definition>,
        _root_path: &str,
    ) -> Result<usize> {
        if definitions.is_empty() {
            return Ok(0);
        }

        self.ensure_definitions_table().await?;

        // TODO: Implement actual LanceDB storage
        // For now, just return the count
        let count = definitions.len();

        tracing::debug!("Stored {} definitions", count);
        Ok(count)
    }

    async fn store_references(&self, references: Vec<Reference>, _root_path: &str) -> Result<usize> {
        if references.is_empty() {
            return Ok(0);
        }

        self.ensure_references_table().await?;

        // TODO: Implement actual LanceDB storage
        let count = references.len();

        tracing::debug!("Stored {} references", count);
        Ok(count)
    }

    async fn find_definition_at(
        &self,
        _file_path: &str,
        _line: usize,
        _column: usize,
    ) -> Result<Option<Definition>> {
        // TODO: Implement query
        Ok(None)
    }

    async fn find_definitions_by_name(&self, _name: &str) -> Result<Vec<Definition>> {
        // TODO: Implement query
        Ok(Vec::new())
    }

    async fn find_references(&self, _target_symbol_id: &str) -> Result<Vec<Reference>> {
        // TODO: Implement query
        Ok(Vec::new())
    }

    async fn get_callers(&self, _symbol_id: &str) -> Result<Vec<CallEdge>> {
        // TODO: Implement call graph query
        Ok(Vec::new())
    }

    async fn get_callees(&self, _symbol_id: &str) -> Result<Vec<CallEdge>> {
        // TODO: Implement call graph query
        Ok(Vec::new())
    }

    async fn delete_by_file(&self, _file_path: &str) -> Result<usize> {
        // TODO: Implement deletion
        Ok(0)
    }

    async fn clear(&self) -> Result<()> {
        // TODO: Drop and recreate tables
        Ok(())
    }

    async fn get_stats(&self) -> Result<RelationsStats> {
        // TODO: Query actual counts
        Ok(RelationsStats::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_store_creation() {
        let temp_dir = TempDir::new().unwrap();
        let store = LanceRelationsStore::new(temp_dir.path().to_path_buf())
            .await
            .unwrap();

        let stats = store.get_stats().await.unwrap();
        assert_eq!(stats.definition_count, 0);
    }

    #[tokio::test]
    async fn test_store_empty_definitions() {
        let temp_dir = TempDir::new().unwrap();
        let store = LanceRelationsStore::new(temp_dir.path().to_path_buf())
            .await
            .unwrap();

        let count = store.store_definitions(Vec::new(), "/test").await.unwrap();
        assert_eq!(count, 0);
    }
}
