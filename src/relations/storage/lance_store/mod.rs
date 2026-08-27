//! LanceDB-based storage for code relationships.
//!
//! Definitions and references live in schema-versioned relation tables inside
//! the same LanceDB directory as the embeddings
//! table, so one database directory holds everything the index knows.
//!
//! Writes are idempotent per file: storing rows for a file first deletes
//! whatever that file had, so re-indexing never accumulates duplicates.

mod codec;

use anyhow::{Context, Result};
use arrow_array::{Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::Schema;
use async_trait::async_trait;
use futures::stream::TryStreamExt;
use lancedb::index::{Index, scalar::BTreeIndexBuilder};
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, Table};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{RelationsStats, RelationsStore};
use crate::relations::types::{CallEdge, Definition, Reference, ReferenceKind, ResolutionStatus};

const DEFINITIONS_TABLE: &str = "relations_definitions_v4";
const REFERENCES_TABLE: &str = "relations_references_v4";

/// Delete filters are built as `file_path IN (...)`; chunked so a large batch
/// of files cannot produce an absurdly long filter string.
const DELETE_CHUNK: usize = 400;

/// LanceDB-based relations store.
pub struct LanceRelationsStore {
    /// Path to the database directory
    db_path: PathBuf,
    /// Database connection (lazy initialized)
    db: Arc<RwLock<Option<Connection>>>,
    /// Avoid listing/rebuilding scalar adjacency indexes on warm graph queries.
    adjacency_ready: Arc<RwLock<bool>>,
}

impl LanceRelationsStore {
    /// Create a new LanceDB relations store
    pub async fn new(db_path: PathBuf) -> Result<Self> {
        tokio::fs::create_dir_all(&db_path)
            .await
            .context("Failed to create relations database directory")?;

        Ok(Self {
            db_path,
            db: Arc::new(RwLock::new(None)),
            adjacency_ready: Arc::new(RwLock::new(false)),
        })
    }

    /// Get or create the database connection
    async fn get_connection(&self) -> Result<Connection> {
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

    /// Open a table, creating it empty with the given schema if it does not exist.
    async fn open_or_create(&self, name: &str, schema: Arc<Schema>) -> Result<Table> {
        let db = self.get_connection().await?;

        if let Ok(table) = db.open_table(name).execute().await {
            return Ok(table);
        }

        let empty = RecordBatch::new_empty(schema.clone());
        let batches = RecordBatchIterator::new(vec![empty].into_iter().map(Ok), schema);
        match db.create_table(name, Box::new(batches)).execute().await {
            Ok(table) => Ok(table),
            // Lost a creation race; the table exists now, so open it.
            Err(_) => db
                .open_table(name)
                .execute()
                .await
                .with_context(|| format!("Failed to open or create table {}", name)),
        }
    }

    async fn definitions_table(&self) -> Result<Table> {
        self.open_or_create(DEFINITIONS_TABLE, codec::definitions_schema())
            .await
    }

    async fn references_table(&self) -> Result<Table> {
        self.open_or_create(REFERENCES_TABLE, codec::references_schema())
            .await
    }

    async fn collect_batches(table: &Table, filter: &str) -> Result<Vec<RecordBatch>> {
        let stream = table
            .query()
            .only_if(filter)
            .execute()
            .await
            .with_context(|| format!("Failed to query with filter: {}", filter))?;
        stream
            .try_collect()
            .await
            .context("Failed to collect query results")
    }

    async fn query_definitions(&self, filter: &str) -> Result<Vec<Definition>> {
        let table = self.definitions_table().await?;
        let batches = Self::collect_batches(&table, filter).await?;
        let mut out = Vec::new();
        for batch in &batches {
            out.extend(codec::batch_to_definitions(batch)?);
        }
        Ok(out)
    }

    async fn query_references(&self, filter: &str) -> Result<Vec<Reference>> {
        let table = self.references_table().await?;
        let batches = Self::collect_batches(&table, filter).await?;
        let mut out = Vec::new();
        for batch in &batches {
            out.extend(codec::batch_to_references(batch)?);
        }
        Ok(out)
    }

    /// Rebuild the two scalar indexes that are the persisted incoming/outgoing
    /// adjacency maps. M2 currently publishes references in one project-wide
    /// batch, so rebuilding here also makes newly published rows warm immediately.
    async fn refresh_adjacency_indexes(&self, table: &Table) -> Result<()> {
        for (column, name) in [
            ("source_symbol_id", "relations_outgoing_v1"),
            ("target_symbol_id", "relations_incoming_v1"),
        ] {
            table
                .create_index(&[column], Index::BTree(BTreeIndexBuilder::default()))
                .name(name.to_string())
                .replace(true)
                .execute()
                .await
                .with_context(|| format!("Failed to refresh {} adjacency index", column))?;
        }
        *self.adjacency_ready.write().await = true;
        Ok(())
    }

    /// Materialize adjacency indexes for an existing M2 table on first graph use.
    /// This is the schema-compatible M3 migration path: correctness does not
    /// require reindexing, while the first query pays the one-time index build.
    async fn ensure_adjacency_indexes(&self, table: &Table) -> Result<()> {
        if *self.adjacency_ready.read().await {
            return Ok(());
        }
        let mut ready = self.adjacency_ready.write().await;
        if *ready {
            return Ok(());
        }
        let existing = table
            .list_indices()
            .await
            .context("Failed to inspect relation adjacency indexes")?;
        for (column, name) in [
            ("source_symbol_id", "relations_outgoing_v1"),
            ("target_symbol_id", "relations_incoming_v1"),
        ] {
            if existing.iter().any(|index| index.name == name) {
                continue;
            }
            table
                .create_index(&[column], Index::BTree(BTreeIndexBuilder::default()))
                .name(name.to_string())
                .execute()
                .await
                .with_context(|| format!("Failed to create {} adjacency index", column))?;
        }
        *ready = true;
        Ok(())
    }

    /// Delete every row belonging to the given files within one project root.
    async fn delete_files_in_root(table: &Table, files: &[String], root_path: &str) -> Result<()> {
        for chunk in files.chunks(DELETE_CHUNK) {
            let filter = format!(
                "root_path = '{}' AND file_path IN ({})",
                codec::escape_sql(root_path),
                codec::sql_in_list(chunk)
            );
            table
                .delete(&filter)
                .await
                .context("Failed to delete rows by file")?;
        }
        Ok(())
    }
}

#[async_trait]
impl RelationsStore for LanceRelationsStore {
    async fn store_definitions(
        &self,
        mut definitions: Vec<Definition>,
        root_path: &str,
    ) -> Result<usize> {
        if definitions.is_empty() {
            return Ok(0);
        }

        for definition in &mut definitions {
            match definition.root_path.as_deref() {
                Some(stored_root) if stored_root != root_path => anyhow::bail!(
                    "Definition root '{}' does not match storage root '{}'",
                    stored_root,
                    root_path
                ),
                None => definition.root_path = Some(root_path.to_string()),
                _ => {}
            }
        }

        let table = self.definitions_table().await?;

        // Idempotent per file: replace whatever rows those files had.
        let files: Vec<String> = definitions
            .iter()
            .map(|d| d.file_path().to_string())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        Self::delete_files_in_root(&table, &files, root_path).await?;

        let batch = codec::definitions_to_batch(&definitions)?;
        let count = batch.num_rows();
        let batches =
            RecordBatchIterator::new(vec![batch].into_iter().map(Ok), codec::definitions_schema());
        table
            .add(Box::new(batches))
            .execute()
            .await
            .context("Failed to store definitions")?;

        tracing::debug!("Stored {} definitions for {} files", count, files.len());
        Ok(count)
    }

    async fn store_references(
        &self,
        mut references: Vec<Reference>,
        root_path: &str,
    ) -> Result<usize> {
        if references.is_empty() {
            return Ok(0);
        }

        for reference in &mut references {
            match reference.root_path.as_deref() {
                Some(stored_root) if stored_root != root_path => anyhow::bail!(
                    "Reference root '{}' does not match storage root '{}'",
                    stored_root,
                    root_path
                ),
                None => reference.root_path = Some(root_path.to_string()),
                _ => {}
            }
        }

        let table = self.references_table().await?;

        let files: Vec<String> = references
            .iter()
            .map(|r| r.file_path.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        Self::delete_files_in_root(&table, &files, root_path).await?;

        let batch = codec::references_to_batch(&references)?;
        let count = batch.num_rows();
        let batches =
            RecordBatchIterator::new(vec![batch].into_iter().map(Ok), codec::references_schema());
        table
            .add(Box::new(batches))
            .execute()
            .await
            .context("Failed to store references")?;
        self.refresh_adjacency_indexes(&table).await?;

        tracing::debug!("Stored {} references for {} files", count, files.len());
        Ok(count)
    }

    async fn find_definition_at(
        &self,
        file_path: &str,
        line: usize,
        _column: usize,
    ) -> Result<Option<Definition>> {
        let filter = format!(
            "file_path = '{}' AND start_line <= {} AND end_line >= {}",
            codec::escape_sql(file_path),
            line,
            line
        );
        let matches = self.query_definitions(&filter).await?;
        // Innermost definition wins: the nested symbol, not its container.
        Ok(matches
            .into_iter()
            .min_by_key(|d| d.end_line.saturating_sub(d.symbol_id.start_line)))
    }

    async fn find_definitions_by_name(&self, name: &str) -> Result<Vec<Definition>> {
        let filter = format!("name = '{}'", codec::escape_sql(name));
        self.query_definitions(&filter).await
    }

    async fn find_definitions_by_symbol_id_in_root(
        &self,
        symbol_id: &str,
        root_path: &str,
    ) -> Result<Vec<Definition>> {
        let filter = format!(
            "id = '{}' AND root_path = '{}'",
            codec::escape_sql(symbol_id),
            codec::escape_sql(root_path)
        );
        self.query_definitions(&filter).await
    }

    async fn find_definitions_by_symbol_ids_in_root(
        &self,
        symbol_ids: &[String],
        root_path: &str,
    ) -> Result<Vec<Definition>> {
        if symbol_ids.is_empty() {
            return Ok(Vec::new());
        }
        let filter = format!(
            "id IN ({}) AND root_path = '{}'",
            codec::sql_in_list(symbol_ids),
            codec::escape_sql(root_path)
        );
        self.query_definitions(&filter).await
    }

    async fn find_definitions_in_root(&self, root_path: &str) -> Result<Vec<Definition>> {
        self.query_definitions(&format!("root_path = '{}'", codec::escape_sql(root_path)))
            .await
    }

    async fn find_definitions_by_files_in_root(
        &self,
        file_paths: &[String],
        root_path: &str,
    ) -> Result<Vec<Definition>> {
        if file_paths.is_empty() {
            return Ok(Vec::new());
        }
        self.query_definitions(&format!(
            "root_path = '{}' AND file_path IN ({})",
            codec::escape_sql(root_path),
            codec::sql_in_list(file_paths)
        ))
        .await
    }

    async fn find_reference_at_in_root(
        &self,
        file_path: &str,
        root_path: &str,
        line: usize,
        column: usize,
    ) -> Result<Option<Reference>> {
        let filter = format!(
            "file_path = '{}' AND root_path = '{}' AND start_line <= {} AND end_line >= {}",
            codec::escape_sql(file_path),
            codec::escape_sql(root_path),
            line,
            line
        );
        let matches = self.query_references(&filter).await?;
        Ok(matches
            .into_iter()
            .filter(|reference| {
                reference.start_line < line
                    || reference.end_line > line
                    || (column >= reference.start_col && column <= reference.end_col)
            })
            .min_by_key(|reference| {
                (
                    reference.end_line.saturating_sub(reference.start_line),
                    reference.end_col.saturating_sub(reference.start_col),
                )
            }))
    }

    async fn find_references(&self, target_symbol_id: &str) -> Result<Vec<Reference>> {
        let filter = format!(
            "target_symbol_id = '{}'",
            codec::escape_sql(target_symbol_id)
        );
        self.query_references(&filter).await
    }

    async fn find_references_by_name_in_root(
        &self,
        symbol_name: &str,
        root_path: &str,
    ) -> Result<Vec<Reference>> {
        let filter = format!(
            "target_name = '{}' AND root_path = '{}'",
            codec::escape_sql(symbol_name),
            codec::escape_sql(root_path)
        );
        self.query_references(&filter).await
    }

    async fn find_references_by_names_in_root(
        &self,
        symbol_names: &[String],
        root_path: &str,
    ) -> Result<Vec<Reference>> {
        if symbol_names.is_empty() {
            return Ok(Vec::new());
        }
        self.query_references(&format!(
            "target_name IN ({}) AND root_path = '{}'",
            codec::sql_in_list(symbol_names),
            codec::escape_sql(root_path)
        ))
        .await
    }

    async fn get_outgoing_references_in_root(
        &self,
        symbol_ids: &[String],
        root_path: &str,
    ) -> Result<Vec<Reference>> {
        if symbol_ids.is_empty() {
            return Ok(Vec::new());
        }
        let table = self.references_table().await?;
        self.ensure_adjacency_indexes(&table).await?;
        let filter = format!(
            "source_symbol_id IN ({}) AND root_path = '{}'",
            codec::sql_in_list(symbol_ids),
            codec::escape_sql(root_path)
        );
        let batches = Self::collect_batches(&table, &filter).await?;
        let mut out = Vec::new();
        for batch in &batches {
            out.extend(codec::batch_to_references(batch)?);
        }
        Ok(out)
    }

    async fn get_incoming_references_in_root(
        &self,
        symbol_ids: &[String],
        root_path: &str,
    ) -> Result<Vec<Reference>> {
        if symbol_ids.is_empty() {
            return Ok(Vec::new());
        }
        let table = self.references_table().await?;
        self.ensure_adjacency_indexes(&table).await?;
        let filter = format!(
            "target_symbol_id IN ({}) AND root_path = '{}'",
            codec::sql_in_list(symbol_ids),
            codec::escape_sql(root_path)
        );
        let batches = Self::collect_batches(&table, &filter).await?;
        let mut out = Vec::new();
        for batch in &batches {
            out.extend(codec::batch_to_references(batch)?);
        }
        Ok(out)
    }

    async fn get_callers(&self, symbol_id: &str) -> Result<Vec<CallEdge>> {
        let filter = format!(
            "target_symbol_id = '{}' AND reference_kind = '{}' AND resolution_status = '{}'",
            codec::escape_sql(symbol_id),
            codec::enum_to_str(&ReferenceKind::Call),
            codec::enum_to_str(&ResolutionStatus::Resolved)
        );
        let call_refs = self.query_references(&filter).await?;
        if call_refs.is_empty() {
            return Ok(Vec::new());
        }

        let mut seen = HashSet::new();
        let mut edges = Vec::new();
        for r in &call_refs {
            if let Some(caller_id) = r.source_symbol_id.clone()
                && seen.insert((caller_id.clone(), r.start_line))
            {
                edges.push(CallEdge {
                    caller_id,
                    callee_id: symbol_id.to_string(),
                    call_site_file: r.file_path.clone(),
                    call_site_line: r.start_line,
                    call_site_col: r.start_col,
                    reference_kind: r.reference_kind,
                    resolution_status: r.resolution_status,
                    evidence_kind: r.evidence_kind,
                    parser: r.parser.clone(),
                });
            }
        }
        Ok(edges)
    }

    async fn get_callees(&self, symbol_id: &str) -> Result<Vec<CallEdge>> {
        let refs_filter = format!(
            "source_symbol_id = '{}' AND reference_kind = '{}' AND resolution_status = '{}'",
            codec::escape_sql(symbol_id),
            codec::enum_to_str(&ReferenceKind::Call),
            codec::enum_to_str(&ResolutionStatus::Resolved)
        );
        let call_refs = self.query_references(&refs_filter).await?;

        let mut seen = HashSet::new();
        Ok(call_refs
            .into_iter()
            .filter(|r| seen.insert((r.target_symbol_id.clone(), r.start_line)))
            .map(|r| CallEdge {
                caller_id: symbol_id.to_string(),
                callee_id: r.target_symbol_id,
                call_site_file: r.file_path,
                call_site_line: r.start_line,
                call_site_col: r.start_col,
                reference_kind: r.reference_kind,
                resolution_status: r.resolution_status,
                evidence_kind: r.evidence_kind,
                parser: r.parser,
            })
            .collect())
    }

    async fn delete_by_file(&self, file_path: &str) -> Result<usize> {
        let filter = format!("file_path = '{}'", codec::escape_sql(file_path));

        let defs_table = self.definitions_table().await?;
        let refs_table = self.references_table().await?;

        // LanceDB's delete does not report a count, so count first.
        let removed = defs_table
            .count_rows(Some(filter.clone()))
            .await
            .unwrap_or(0)
            + refs_table
                .count_rows(Some(filter.clone()))
                .await
                .unwrap_or(0);

        defs_table
            .delete(&filter)
            .await
            .context("Failed to delete definitions for file")?;
        refs_table
            .delete(&filter)
            .await
            .context("Failed to delete references for file")?;

        Ok(removed)
    }

    async fn delete_by_file_in_root(&self, file_path: &str, root_path: &str) -> Result<usize> {
        let filter = format!(
            "file_path = '{}' AND root_path = '{}'",
            codec::escape_sql(file_path),
            codec::escape_sql(root_path)
        );

        let defs_table = self.definitions_table().await?;
        let refs_table = self.references_table().await?;

        // LanceDB's delete does not report a count, so count first.
        let removed = defs_table
            .count_rows(Some(filter.clone()))
            .await
            .unwrap_or(0)
            + refs_table
                .count_rows(Some(filter.clone()))
                .await
                .unwrap_or(0);

        defs_table
            .delete(&filter)
            .await
            .context("Failed to delete definitions for scoped file")?;
        refs_table
            .delete(&filter)
            .await
            .context("Failed to delete references for scoped file")?;

        Ok(removed)
    }

    async fn delete_definitions_by_files_in_root(
        &self,
        file_paths: &[String],
        root_path: &str,
    ) -> Result<usize> {
        if file_paths.is_empty() {
            return Ok(0);
        }
        let table = self.definitions_table().await?;
        Self::delete_files_in_root(&table, file_paths, root_path).await?;
        Ok(file_paths.len())
    }

    async fn delete_references_by_files_in_root(
        &self,
        file_paths: &[String],
        root_path: &str,
    ) -> Result<usize> {
        if file_paths.is_empty() {
            return Ok(0);
        }
        let table = self.references_table().await?;
        Self::delete_files_in_root(&table, file_paths, root_path).await?;
        Ok(file_paths.len())
    }

    async fn delete_by_root(&self, root_path: &str) -> Result<usize> {
        let filter = format!("root_path = '{}'", codec::escape_sql(root_path));
        let defs_table = self.definitions_table().await?;
        let refs_table = self.references_table().await?;
        let removed = defs_table
            .count_rows(Some(filter.clone()))
            .await
            .unwrap_or(0)
            + refs_table
                .count_rows(Some(filter.clone()))
                .await
                .unwrap_or(0);
        defs_table
            .delete(&filter)
            .await
            .context("Failed to delete definitions by root")?;
        refs_table
            .delete(&filter)
            .await
            .context("Failed to delete references by root")?;
        Ok(removed)
    }

    async fn clear(&self) -> Result<()> {
        let db = self.get_connection().await?;
        for name in [DEFINITIONS_TABLE, REFERENCES_TABLE] {
            if let Err(e) = db.drop_table(name, &[]).await {
                // Dropping a table that was never created is not an error worth failing on.
                tracing::debug!("Dropping relations table {} failed: {}", name, e);
            }
        }
        *self.adjacency_ready.write().await = false;
        Ok(())
    }

    async fn get_stats(&self) -> Result<RelationsStats> {
        let defs_table = self.definitions_table().await?;
        let refs_table = self.references_table().await?;

        let definition_count = defs_table
            .count_rows(None)
            .await
            .context("Failed to count definitions")?;
        let reference_count = refs_table
            .count_rows(None)
            .await
            .context("Failed to count references")?;
        let all_references = self.query_references("id IS NOT NULL").await?;
        let code_reference_count = all_references
            .iter()
            .filter(|reference| reference.reference_kind.is_code())
            .count();

        // Distinct files with definitions.
        let stream = defs_table
            .query()
            .select(lancedb::query::Select::Columns(vec![
                "file_path".to_string(),
                "root_path".to_string(),
            ]))
            .execute()
            .await
            .context("Failed to query definition files")?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .context("Failed to collect definition files")?;

        let mut files = HashSet::new();
        for batch in &batches {
            if let Some(paths) = batch
                .column_by_name("file_path")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            {
                let roots = batch
                    .column_by_name("root_path")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                for i in 0..batch.num_rows() {
                    let root = roots
                        .filter(|array| !array.is_null(i))
                        .map(|array| array.value(i))
                        .unwrap_or("");
                    files.insert((root.to_string(), paths.value(i).to_string()));
                }
            }
        }

        Ok(RelationsStats {
            definition_count,
            reference_count,
            code_reference_count,
            files_with_definitions: files.len(),
        })
    }
}

#[cfg(test)]
mod tests;
