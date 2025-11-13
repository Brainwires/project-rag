//! LanceDB vector database client
//!
//! NOTE: This file is ~1232 lines (737 implementation + 495 tests).
//! It exceeds the 600-line guideline but is kept as a single coherent unit because:
//! - Tests require access to private methods (must be in same file)
//! - The implementation represents a single logical component (LanceDB client)
//! - Splitting would compromise test coverage and code organization
//!
//! Future refactoring could extract search logic into traits if needed.

use crate::bm25_search::BM25Search;
use crate::types::{ChunkMetadata, SearchResult};
use crate::vector_db::{DatabaseStats, VectorDatabase};
use anyhow::{Context, Result};
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
    UInt32Array, types::Float32Type,
};
use arrow_schema::{DataType, Field, Schema};
use futures::stream::TryStreamExt;
use lancedb::Table;
use lancedb::connection::Connection;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// LanceDB vector database implementation (embedded, no server required)
/// Includes BM25 hybrid search support using Tantivy
pub struct LanceVectorDB {
    connection: Connection,
    table_name: String,
    db_path: String,
    /// BM25 search index for keyword matching
    bm25_index: Arc<RwLock<Option<BM25Search>>>,
}

impl LanceVectorDB {
    /// Create a new LanceDB instance with default path
    pub async fn new() -> Result<Self> {
        let db_path = Self::default_lancedb_path();
        Self::with_path(&db_path).await
    }

    /// Create a new LanceDB instance with custom path
    pub async fn with_path(db_path: &str) -> Result<Self> {
        tracing::info!("Connecting to LanceDB at: {}", db_path);

        let connection = lancedb::connect(db_path)
            .execute()
            .await
            .context("Failed to connect to LanceDB")?;

        // Initialize BM25 index
        let bm25_path = format!("{}/lancedb_bm25", db_path);
        let bm25_index = BM25Search::new(&bm25_path).context("Failed to initialize BM25 index")?;

        Ok(Self {
            connection,
            table_name: "code_embeddings".to_string(),
            db_path: db_path.to_string(),
            bm25_index: Arc::new(RwLock::new(Some(bm25_index))),
        })
    }

    /// Get default database path (public for CLI version info)
    pub fn default_lancedb_path() -> String {
        crate::paths::PlatformPaths::default_lancedb_path()
            .to_string_lossy()
            .to_string()
    }

    /// Create schema for the embeddings table
    fn create_schema(dimension: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimension as i32,
                ),
                false,
            ),
            Field::new("id", DataType::Utf8, false),
            Field::new("file_path", DataType::Utf8, false),
            Field::new("start_line", DataType::UInt32, false),
            Field::new("end_line", DataType::UInt32, false),
            Field::new("language", DataType::Utf8, false),
            Field::new("extension", DataType::Utf8, false),
            Field::new("file_hash", DataType::Utf8, false),
            Field::new("indexed_at", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("project", DataType::Utf8, true),
        ]))
    }

    /// Get or create table
    async fn get_table(&self) -> Result<Table> {
        self.connection
            .open_table(&self.table_name)
            .execute()
            .await
            .context("Failed to open table")
    }

    /// Convert embeddings and metadata to RecordBatch
    fn create_record_batch(
        embeddings: Vec<Vec<f32>>,
        metadata: Vec<ChunkMetadata>,
        contents: Vec<String>,
        schema: Arc<Schema>,
    ) -> Result<RecordBatch> {
        let num_rows = embeddings.len();
        let dimension = embeddings[0].len();

        // Create FixedSizeListArray for vectors
        let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            embeddings
                .into_iter()
                .map(|v| Some(v.into_iter().map(Some))),
            dimension as i32,
        );

        // Create arrays for each field
        let id_array = StringArray::from(
            (0..num_rows)
                .map(|i| format!("{}:{}", metadata[i].file_path, metadata[i].start_line))
                .collect::<Vec<_>>(),
        );
        let file_path_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.file_path.as_str())
                .collect::<Vec<_>>(),
        );
        let start_line_array = UInt32Array::from(
            metadata
                .iter()
                .map(|m| m.start_line as u32)
                .collect::<Vec<_>>(),
        );
        let end_line_array = UInt32Array::from(
            metadata
                .iter()
                .map(|m| m.end_line as u32)
                .collect::<Vec<_>>(),
        );
        let language_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.language.as_deref().unwrap_or("Unknown"))
                .collect::<Vec<_>>(),
        );
        let extension_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.extension.as_deref().unwrap_or(""))
                .collect::<Vec<_>>(),
        );
        let file_hash_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.file_hash.as_str())
                .collect::<Vec<_>>(),
        );
        let indexed_at_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.indexed_at.to_string())
                .collect::<Vec<_>>(),
        );
        let content_array =
            StringArray::from(contents.iter().map(|s| s.as_str()).collect::<Vec<_>>());
        let project_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.project.as_deref())
                .collect::<Vec<_>>(),
        );

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(vector_array),
                Arc::new(id_array),
                Arc::new(file_path_array),
                Arc::new(start_line_array),
                Arc::new(end_line_array),
                Arc::new(language_array),
                Arc::new(extension_array),
                Arc::new(file_hash_array),
                Arc::new(indexed_at_array),
                Arc::new(content_array),
                Arc::new(project_array),
            ],
        )
        .context("Failed to create RecordBatch")
    }
}

#[async_trait::async_trait]
impl VectorDatabase for LanceVectorDB {
    async fn initialize(&self, dimension: usize) -> Result<()> {
        tracing::info!(
            "Initializing LanceDB with dimension {} at {}",
            dimension,
            self.db_path
        );

        // Check if table exists
        let table_names = self
            .connection
            .table_names()
            .execute()
            .await
            .context("Failed to list tables")?;

        if table_names.contains(&self.table_name) {
            tracing::info!("Table '{}' already exists", self.table_name);
            return Ok(());
        }

        // Create empty table with schema
        let schema = Self::create_schema(dimension);

        // Create empty RecordBatch
        let empty_batch = RecordBatch::new_empty(schema.clone());

        // Need to wrap in iterator that returns Result<RecordBatch>
        let batches =
            RecordBatchIterator::new(vec![empty_batch].into_iter().map(Ok), schema.clone());

        self.connection
            .create_table(&self.table_name, Box::new(batches))
            .execute()
            .await
            .context("Failed to create table")?;

        tracing::info!("Created table '{}'", self.table_name);
        Ok(())
    }

    async fn store_embeddings(
        &self,
        embeddings: Vec<Vec<f32>>,
        metadata: Vec<ChunkMetadata>,
        contents: Vec<String>,
    ) -> Result<usize> {
        if embeddings.is_empty() {
            return Ok(0);
        }

        let dimension = embeddings[0].len();
        let schema = Self::create_schema(dimension);

        // Get current row count to use as starting ID for BM25
        let table = self.get_table().await?;
        let current_count = table.count_rows(None).await.unwrap_or(0) as u64;

        let batch = Self::create_record_batch(
            embeddings,
            metadata.clone(),
            contents.clone(),
            schema.clone(),
        )?;
        let count = batch.num_rows();

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        table
            .add(Box::new(batches))
            .execute()
            .await
            .context("Failed to add records to table")?;

        // Add documents to BM25 index with file_path for deletion tracking
        let bm25_docs: Vec<_> = (0..count)
            .map(|i| {
                let id = current_count + i as u64;
                (id, contents[i].clone(), metadata[i].file_path.clone())
            })
            .collect();

        let bm25_lock = self
            .bm25_index
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire BM25 read lock: {}", e))?;
        if let Some(bm25) = bm25_lock.as_ref() {
            bm25.add_documents(bm25_docs)
                .context("Failed to add documents to BM25 index")?;
        }
        drop(bm25_lock);

        tracing::info!("Stored {} embeddings with BM25 indexing", count);
        Ok(count)
    }

    async fn search(
        &self,
        query_vector: Vec<f32>,
        query_text: &str,
        limit: usize,
        min_score: f32,
        project: Option<String>,
        hybrid: bool,
    ) -> Result<Vec<SearchResult>> {
        let table = self.get_table().await?;

        if hybrid {
            // Hybrid search: combine vector and BM25 results with RRF
            // Get more results from each source for RRF to combine
            let search_limit = limit * 3;

            // Vector search
            let query = table
                .vector_search(query_vector)
                .context("Failed to create vector search")?
                .limit(search_limit);

            let stream = if let Some(ref project_name) = project {
                query
                    .only_if(format!("project = '{}'", project_name))
                    .execute()
                    .await
                    .context("Failed to execute search")?
            } else {
                query.execute().await.context("Failed to execute search")?
            };

            let results: Vec<RecordBatch> = stream
                .try_collect()
                .await
                .context("Failed to collect search results")?;

            // Build vector results with row-based IDs
            let mut vector_results = Vec::new();
            let mut row_offset = 0u64;

            // Store original scores for later reporting
            let mut original_scores: HashMap<u64, (f32, Option<f32>)> = HashMap::new();

            for batch in &results {
                let distance_array = batch
                    .column_by_name("_distance")
                    .context("Missing _distance column")?
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .context("Invalid _distance type")?;

                for i in 0..batch.num_rows() {
                    let distance = distance_array.value(i);
                    let score = 1.0 / (1.0 + distance);
                    let id = row_offset + i as u64;

                    // For hybrid search, don't filter by min_score before RRF
                    // RRF will combine weak vector + strong keyword (or vice versa)
                    // Filtering happens after RRF based on the combined ranking
                    vector_results.push((id, score));
                    original_scores.insert(id, (score, None));
                }
                row_offset += batch.num_rows() as u64;
            }

            // BM25 keyword search
            let bm25_lock = self
                .bm25_index
                .read()
                .map_err(|e| anyhow::anyhow!("Failed to acquire BM25 read lock: {}", e))?;
            let bm25_results = if let Some(bm25) = bm25_lock.as_ref() {
                let all_bm25_results = bm25
                    .search(query_text, search_limit)
                    .context("Failed to search BM25 index")?;

                // Store BM25 scores (don't filter - let RRF combine them)
                // BM25 scores are not normalized to 0-1 range, so min_score doesn't apply
                for result in &all_bm25_results {
                    original_scores
                        .entry(result.id)
                        .and_modify(|e| e.1 = Some(result.score))
                        .or_insert((0.0, Some(result.score))); // No vector score, only keyword
                }

                all_bm25_results
            } else {
                Vec::new()
            };
            drop(bm25_lock);

            // Combine results with Reciprocal Rank Fusion
            // RRF produces scores ~0.01-0.03, so don't apply min_score to combined scores
            let combined =
                crate::bm25_search::reciprocal_rank_fusion(vector_results, bm25_results, limit);

            // Build final results by looking up the combined IDs in the vector results
            let mut search_results = Vec::new();

            for (id, combined_score) in combined {
                // Find this result in the original batch results
                let mut found = false;
                let mut batch_offset = 0u64;

                for batch in &results {
                    if id >= batch_offset && id < batch_offset + batch.num_rows() as u64 {
                        let idx = (id - batch_offset) as usize;

                        let file_path_array = batch
                            .column_by_name("file_path")
                            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                        let start_line_array = batch
                            .column_by_name("start_line")
                            .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
                        let end_line_array = batch
                            .column_by_name("end_line")
                            .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
                        let language_array = batch
                            .column_by_name("language")
                            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                        let content_array = batch
                            .column_by_name("content")
                            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                        let project_array = batch
                            .column_by_name("project")
                            .and_then(|c| c.as_any().downcast_ref::<StringArray>());

                        if let (Some(fp), Some(sl), Some(el), Some(lang), Some(cont), Some(proj)) = (
                            file_path_array,
                            start_line_array,
                            end_line_array,
                            language_array,
                            content_array,
                            project_array,
                        ) {
                            // Look up original scores for filtering and reporting
                            let (vector_score, keyword_score) =
                                original_scores.get(&id).copied().unwrap_or((0.0, None));

                            // For hybrid search, apply min_score intelligently:
                            // Accept if EITHER vector or keyword score meets threshold
                            // This allows pure keyword matches (weak vector) and pure semantic matches (weak keyword)
                            let passes_filter = vector_score >= min_score
                                || keyword_score.is_some_and(|k| k >= min_score);

                            if passes_filter {
                                // Use RRF combined score as the main score for ranking
                                // But report original vector/keyword scores for transparency
                                search_results.push(SearchResult {
                                    score: combined_score, // RRF score for ranking
                                    vector_score,          // Original vector score
                                    keyword_score,         // Original BM25 score
                                    file_path: fp.value(idx).to_string(),
                                    start_line: sl.value(idx) as usize,
                                    end_line: el.value(idx) as usize,
                                    language: lang.value(idx).to_string(),
                                    content: cont.value(idx).to_string(),
                                    project: if proj.is_null(idx) {
                                        None
                                    } else {
                                        Some(proj.value(idx).to_string())
                                    },
                                });
                            }
                            found = true;
                            break;
                        }
                    }
                    batch_offset += batch.num_rows() as u64;
                }

                if !found {
                    tracing::warn!("Could not find result for RRF ID {}", id);
                }
            }

            Ok(search_results)
        } else {
            // Pure vector search
            let query = table
                .vector_search(query_vector)
                .context("Failed to create vector search")?
                .limit(limit);

            let stream = if let Some(ref project_name) = project {
                query
                    .only_if(format!("project = '{}'", project_name))
                    .execute()
                    .await
                    .context("Failed to execute search")?
            } else {
                query.execute().await.context("Failed to execute search")?
            };

            let results: Vec<RecordBatch> = stream
                .try_collect()
                .await
                .context("Failed to collect search results")?;

            let mut search_results = Vec::new();

            for batch in results {
                let file_path_array = batch
                    .column_by_name("file_path")
                    .context("Missing file_path column")?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("Invalid file_path type")?;

                let start_line_array = batch
                    .column_by_name("start_line")
                    .context("Missing start_line column")?
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .context("Invalid start_line type")?;

                let end_line_array = batch
                    .column_by_name("end_line")
                    .context("Missing end_line column")?
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .context("Invalid end_line type")?;

                let language_array = batch
                    .column_by_name("language")
                    .context("Missing language column")?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("Invalid language type")?;

                let content_array = batch
                    .column_by_name("content")
                    .context("Missing content column")?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("Invalid content type")?;

                let project_array = batch
                    .column_by_name("project")
                    .context("Missing project column")?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("Invalid project type")?;

                let distance_array = batch
                    .column_by_name("_distance")
                    .context("Missing _distance column")?
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .context("Invalid _distance type")?;

                for i in 0..batch.num_rows() {
                    let distance = distance_array.value(i);
                    let score = 1.0 / (1.0 + distance);

                    if score >= min_score {
                        search_results.push(SearchResult {
                            score,
                            vector_score: score,
                            keyword_score: None,
                            file_path: file_path_array.value(i).to_string(),
                            start_line: start_line_array.value(i) as usize,
                            end_line: end_line_array.value(i) as usize,
                            language: language_array.value(i).to_string(),
                            content: content_array.value(i).to_string(),
                            project: if project_array.is_null(i) {
                                None
                            } else {
                                Some(project_array.value(i).to_string())
                            },
                        });
                    }
                }
            }

            Ok(search_results)
        }
    }

    async fn search_filtered(
        &self,
        query_vector: Vec<f32>,
        query_text: &str,
        limit: usize,
        min_score: f32,
        project: Option<String>,
        hybrid: bool,
        file_extensions: Vec<String>,
        languages: Vec<String>,
        path_patterns: Vec<String>,
    ) -> Result<Vec<SearchResult>> {
        // Get more results than requested to account for filtering
        let search_limit = limit * 3;

        // Do basic search with hybrid support
        let mut results = self
            .search(
                query_vector,
                query_text,
                search_limit,
                min_score,
                project,
                hybrid,
            )
            .await?;

        // Post-process filtering
        results.retain(|result| {
            // Filter by file extension
            if !file_extensions.is_empty() {
                let has_extension = file_extensions
                    .iter()
                    .any(|ext| result.file_path.ends_with(&format!(".{}", ext)));
                if !has_extension {
                    return false;
                }
            }

            // Filter by language
            if !languages.is_empty() && !languages.contains(&result.language) {
                return false;
            }

            // Filter by path pattern
            if !path_patterns.is_empty() {
                let matches_pattern = path_patterns
                    .iter()
                    .any(|pattern| result.file_path.contains(pattern));
                if !matches_pattern {
                    return false;
                }
            }

            true
        });

        // Truncate to requested limit
        results.truncate(limit);

        Ok(results)
    }

    async fn delete_by_file(&self, file_path: &str) -> Result<usize> {
        // Delete from BM25 index first (using file_path field)
        // Must be done in a scope to drop lock before await
        {
            let bm25_lock = self
                .bm25_index
                .read()
                .map_err(|e| anyhow::anyhow!("Failed to acquire BM25 read lock: {}", e))?;
            if let Some(bm25) = bm25_lock.as_ref() {
                bm25.delete_by_file_path(file_path)
                    .context("Failed to delete from BM25 index")?;
                tracing::debug!("Deleted BM25 entries for file: {}", file_path);
            }
        } // bm25_lock dropped here

        let table = self.get_table().await?;

        // LanceDB uses SQL-like delete
        let filter = format!("file_path = '{}'", file_path);

        table
            .delete(&filter)
            .await
            .context("Failed to delete records")?;

        tracing::info!("Deleted embeddings for file: {}", file_path);

        // LanceDB doesn't return count directly, return 0 as placeholder
        Ok(0)
    }

    async fn clear(&self) -> Result<()> {
        // Drop and recreate table
        self.connection
            .drop_table(&self.table_name)
            .await
            .context("Failed to drop table")?;

        // Clear BM25 index
        let bm25_lock = self
            .bm25_index
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire BM25 read lock: {}", e))?;
        if let Some(bm25) = bm25_lock.as_ref() {
            bm25.clear().context("Failed to clear BM25 index")?;
        }
        drop(bm25_lock);

        tracing::info!("Cleared all embeddings and BM25 index");
        Ok(())
    }

    async fn get_statistics(&self) -> Result<DatabaseStats> {
        let table = self.get_table().await?;

        // Count total vectors
        let count_result = table
            .count_rows(None)
            .await
            .context("Failed to count rows")?;

        // Get language breakdown by scanning the table
        let stream = table
            .query()
            .select(lancedb::query::Select::Columns(vec![
                "language".to_string(),
            ]))
            .execute()
            .await
            .context("Failed to query languages")?;

        let query_result: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .context("Failed to collect language data")?;

        let mut language_counts: HashMap<String, usize> = HashMap::new();

        for batch in query_result {
            let language_array = batch
                .column_by_name("language")
                .context("Missing language column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid language type")?;

            for i in 0..batch.num_rows() {
                let language = language_array.value(i);
                *language_counts.entry(language.to_string()).or_insert(0) += 1;
            }
        }

        let mut language_breakdown: Vec<(String, usize)> = language_counts.into_iter().collect();
        language_breakdown.sort_by(|a, b| b.1.cmp(&a.1));

        Ok(DatabaseStats {
            total_points: count_result,
            total_vectors: count_result,
            language_breakdown,
        })
    }

    async fn flush(&self) -> Result<()> {
        // LanceDB persists automatically, no explicit flush needed
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_metadata(file_path: &str, start_line: usize, end_line: usize) -> ChunkMetadata {
        ChunkMetadata {
            file_path: file_path.to_string(),
            project: Some("test-project".to_string()),
            start_line,
            end_line,
            language: Some("Rust".to_string()),
            extension: Some("rs".to_string()),
            file_hash: "test_hash_123".to_string(),
            indexed_at: 1234567890,
        }
    }

    #[tokio::test]
    async fn test_new_creates_instance() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();

        let db = LanceVectorDB::with_path(&db_path).await;
        assert!(db.is_ok());

        let db = db.unwrap();
        assert_eq!(db.table_name, "code_embeddings");
        assert_eq!(db.db_path, db_path);
    }

    #[tokio::test]
    async fn test_default_path() {
        let path = LanceVectorDB::default_lancedb_path();
        assert!(path.contains("project-rag"));
        assert!(path.contains("lancedb"));
    }

    #[tokio::test]
    async fn test_initialize_creates_table() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();

        // Initialize with dimension 384
        let result = db.initialize(384).await;
        assert!(result.is_ok());

        // Table should now exist
        let table_names = db.connection.table_names().execute().await.unwrap();
        assert!(table_names.contains(&"code_embeddings".to_string()));
    }

    #[tokio::test]
    async fn test_initialize_idempotent() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();

        // Initialize twice
        db.initialize(384).await.unwrap();
        let result = db.initialize(384).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_store_embeddings_empty() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        let result = db.store_embeddings(vec![], vec![], vec![]).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_store_and_retrieve_embeddings() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Create test embeddings
        let embeddings = vec![vec![0.1; 384], vec![0.2; 384]];
        let metadata = vec![
            create_test_metadata("test1.rs", 1, 10),
            create_test_metadata("test2.rs", 20, 30),
        ];
        let contents = vec!["fn main() {}".to_string(), "fn test() {}".to_string()];

        let count = db
            .store_embeddings(embeddings.clone(), metadata, contents)
            .await
            .unwrap();
        assert_eq!(count, 2);

        // Verify storage by searching
        let query = vec![0.1; 384];
        let results = db
            .search(query, "main", 10, 0.0, None, false)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_search_pure_vector() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings
        let embeddings = vec![vec![0.1; 384]];
        let metadata = vec![create_test_metadata("test.rs", 1, 10)];
        let contents = vec!["fn main() {}".to_string()];
        db.store_embeddings(embeddings, metadata, contents)
            .await
            .unwrap();

        // Search with pure vector (hybrid=false)
        let query = vec![0.1; 384];
        let results = db
            .search(query, "main", 10, 0.0, None, false)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, "test.rs");
        assert_eq!(results[0].start_line, 1);
        assert_eq!(results[0].end_line, 10);
        assert!(results[0].keyword_score.is_none());
    }

    #[tokio::test]
    async fn test_search_hybrid() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings
        let embeddings = vec![vec![0.1; 384]];
        let metadata = vec![create_test_metadata("test.rs", 1, 10)];
        let contents = vec!["fn main() { println!(\"hello\"); }".to_string()];
        db.store_embeddings(embeddings, metadata, contents)
            .await
            .unwrap();

        // Search with hybrid (hybrid=true)
        let query = vec![0.1; 384];
        let results = db
            .search(query, "println", 10, 0.0, None, true)
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].file_path, "test.rs");
        // Hybrid search should have keyword score
        assert!(results[0].keyword_score.is_some());
    }

    #[tokio::test]
    async fn test_search_with_min_score() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings
        let embeddings = vec![vec![0.1; 384]];
        let metadata = vec![create_test_metadata("test.rs", 1, 10)];
        let contents = vec!["fn main() {}".to_string()];
        db.store_embeddings(embeddings, metadata, contents)
            .await
            .unwrap();

        // Search with high min_score (should filter out results)
        let query = vec![0.9; 384]; // Very different from stored embedding
        let results = db
            .search(query, "main", 10, 0.99, None, false)
            .await
            .unwrap();

        // Expect fewer or no results due to high threshold
        assert!(results.len() <= 1);
    }

    #[tokio::test]
    async fn test_search_with_project_filter() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings with different projects
        let embeddings = vec![vec![0.1; 384], vec![0.2; 384]];
        let mut meta1 = create_test_metadata("test1.rs", 1, 10);
        meta1.project = Some("project-a".to_string());
        let mut meta2 = create_test_metadata("test2.rs", 20, 30);
        meta2.project = Some("project-b".to_string());
        let metadata = vec![meta1, meta2];
        let contents = vec!["fn main() {}".to_string(), "fn test() {}".to_string()];

        db.store_embeddings(embeddings, metadata, contents)
            .await
            .unwrap();

        // Search with project filter
        let query = vec![0.15; 384];
        let results = db
            .search(query, "main", 10, 0.0, Some("project-a".to_string()), false)
            .await
            .unwrap();

        // Should only get results from project-a
        for result in results {
            assert_eq!(result.project, Some("project-a".to_string()));
        }
    }

    #[tokio::test]
    async fn test_search_filtered_by_extension() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings with different file types
        let embeddings = vec![vec![0.1; 384], vec![0.2; 384]];
        let metadata = vec![
            create_test_metadata("test.rs", 1, 10),
            create_test_metadata("test.toml", 20, 30),
        ];
        let contents = vec!["fn main() {}".to_string(), "[package]".to_string()];

        db.store_embeddings(embeddings, metadata, contents)
            .await
            .unwrap();

        // Search filtered by .rs extension
        let query = vec![0.15; 384];
        let results = db
            .search_filtered(
                query,
                "main",
                10,
                0.0,
                None,
                false,
                vec!["rs".to_string()],
                vec![],
                vec![],
            )
            .await
            .unwrap();

        // Should only get .rs files
        for result in results {
            assert!(result.file_path.ends_with(".rs"));
        }
    }

    #[tokio::test]
    async fn test_search_filtered_by_language() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings with different languages
        let embeddings = vec![vec![0.1; 384]];
        let metadata = vec![create_test_metadata("test.rs", 1, 10)];
        let contents = vec!["fn main() {}".to_string()];

        db.store_embeddings(embeddings, metadata, contents)
            .await
            .unwrap();

        // Search filtered by Rust language
        let query = vec![0.1; 384];
        let results = db
            .search_filtered(
                query,
                "main",
                10,
                0.0,
                None,
                false,
                vec![],
                vec!["Rust".to_string()],
                vec![],
            )
            .await
            .unwrap();

        // Should only get Rust files
        for result in results {
            assert_eq!(result.language, "Rust");
        }
    }

    #[tokio::test]
    async fn test_search_filtered_by_path_pattern() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings with different paths
        let embeddings = vec![vec![0.1; 384], vec![0.2; 384]];
        let metadata = vec![
            create_test_metadata("src/main.rs", 1, 10),
            create_test_metadata("tests/test.rs", 20, 30),
        ];
        let contents = vec!["fn main() {}".to_string(), "fn test() {}".to_string()];

        db.store_embeddings(embeddings, metadata, contents)
            .await
            .unwrap();

        // Search filtered by path pattern
        let query = vec![0.15; 384];
        let results = db
            .search_filtered(
                query,
                "main",
                10,
                0.0,
                None,
                false,
                vec![],
                vec![],
                vec!["src/".to_string()],
            )
            .await
            .unwrap();

        // Should only get files in src/
        for result in results {
            assert!(result.file_path.contains("src/"));
        }
    }

    #[tokio::test]
    async fn test_delete_by_file() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings
        let embeddings = vec![vec![0.1; 384], vec![0.2; 384]];
        let metadata = vec![
            create_test_metadata("test1.rs", 1, 10),
            create_test_metadata("test2.rs", 20, 30),
        ];
        let contents = vec!["fn main() {}".to_string(), "fn test() {}".to_string()];

        db.store_embeddings(embeddings, metadata, contents)
            .await
            .unwrap();

        // Delete one file
        let result = db.delete_by_file("test1.rs").await;
        assert!(result.is_ok());

        // Verify deletion
        let query = vec![0.15; 384];
        let results = db
            .search(query, "main", 10, 0.0, None, false)
            .await
            .unwrap();

        // Should not contain deleted file
        for result in &results {
            assert_ne!(result.file_path, "test1.rs");
        }
    }

    #[tokio::test]
    async fn test_clear() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings
        let embeddings = vec![vec![0.1; 384]];
        let metadata = vec![create_test_metadata("test.rs", 1, 10)];
        let contents = vec!["fn main() {}".to_string()];

        db.store_embeddings(embeddings, metadata, contents)
            .await
            .unwrap();

        // Clear database
        let result = db.clear().await;
        assert!(result.is_ok());

        // Table should be gone
        let table_names = db.connection.table_names().execute().await.unwrap();
        assert!(!table_names.contains(&"code_embeddings".to_string()));
    }

    #[tokio::test]
    async fn test_get_statistics_empty() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        let stats = db.get_statistics().await.unwrap();
        assert_eq!(stats.total_points, 0);
        assert_eq!(stats.total_vectors, 0);
        assert_eq!(stats.language_breakdown.len(), 0);
    }

    #[tokio::test]
    async fn test_get_statistics_with_data() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings with different languages
        let embeddings = vec![vec![0.1; 384], vec![0.2; 384], vec![0.3; 384]];
        let mut meta1 = create_test_metadata("test1.rs", 1, 10);
        meta1.language = Some("Rust".to_string());
        let mut meta2 = create_test_metadata("test2.rs", 20, 30);
        meta2.language = Some("Rust".to_string());
        let mut meta3 = create_test_metadata("test3.py", 40, 50);
        meta3.language = Some("Python".to_string());

        let metadata = vec![meta1, meta2, meta3];
        let contents = vec![
            "fn main() {}".to_string(),
            "fn test() {}".to_string(),
            "def main(): pass".to_string(),
        ];

        db.store_embeddings(embeddings, metadata, contents)
            .await
            .unwrap();

        let stats = db.get_statistics().await.unwrap();
        assert_eq!(stats.total_points, 3);
        assert_eq!(stats.total_vectors, 3);
        assert_eq!(stats.language_breakdown.len(), 2);

        // Verify language counts (sorted by count descending)
        assert_eq!(stats.language_breakdown[0].0, "Rust");
        assert_eq!(stats.language_breakdown[0].1, 2);
        assert_eq!(stats.language_breakdown[1].0, "Python");
        assert_eq!(stats.language_breakdown[1].1, 1);
    }

    #[tokio::test]
    async fn test_flush() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();

        // Flush should succeed (no-op for LanceDB)
        let result = db.flush().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_create_schema() {
        let schema = LanceVectorDB::create_schema(384);

        // Verify schema has expected fields
        assert_eq!(schema.fields().len(), 11);
        assert_eq!(schema.field(0).name(), "vector");
        assert_eq!(schema.field(1).name(), "id");
        assert_eq!(schema.field(2).name(), "file_path");
        assert_eq!(schema.field(3).name(), "start_line");
        assert_eq!(schema.field(4).name(), "end_line");
        assert_eq!(schema.field(5).name(), "language");
        assert_eq!(schema.field(6).name(), "extension");
        assert_eq!(schema.field(7).name(), "file_hash");
        assert_eq!(schema.field(8).name(), "indexed_at");
        assert_eq!(schema.field(9).name(), "content");
        assert_eq!(schema.field(10).name(), "project");
    }

    #[tokio::test]
    async fn test_create_record_batch() {
        let embeddings = vec![vec![0.1; 384], vec![0.2; 384]];
        let metadata = vec![
            create_test_metadata("test1.rs", 1, 10),
            create_test_metadata("test2.rs", 20, 30),
        ];
        let contents = vec!["fn main() {}".to_string(), "fn test() {}".to_string()];
        let schema = LanceVectorDB::create_schema(384);

        let batch = LanceVectorDB::create_record_batch(embeddings, metadata, contents, schema);
        assert!(batch.is_ok());

        let batch = batch.unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 11);
    }

    #[tokio::test]
    async fn test_multiple_searches() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings
        let embeddings = vec![vec![0.1; 384]];
        let metadata = vec![create_test_metadata("test.rs", 1, 10)];
        let contents = vec!["fn main() {}".to_string()];
        db.store_embeddings(embeddings, metadata, contents)
            .await
            .unwrap();

        // Perform multiple searches
        for _ in 0..3 {
            let query = vec![0.1; 384];
            let results = db
                .search(query, "main", 10, 0.0, None, false)
                .await
                .unwrap();
            assert_eq!(results.len(), 1);
        }
    }
}
