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
/// Now includes BM25 hybrid search support using Tantivy
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
        let db_path = Self::default_path();
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

    /// Get default database path
    fn default_path() -> String {
        let data_dir = if cfg!(target_os = "windows") {
            std::env::var("LOCALAPPDATA").unwrap_or_else(|_| ".".to_string())
        } else if cfg!(target_os = "macos") {
            format!(
                "{}/Library/Application Support",
                std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
            )
        } else {
            // Linux/Unix
            std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| {
                format!(
                    "{}/.local/share",
                    std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
                )
            })
        };

        format!("{}/project-rag/lancedb", data_dir)
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

        // Add documents to BM25 index
        let bm25_docs: Vec<_> = (0..count)
            .map(|i| {
                let id = current_count + i as u64;
                (id, contents[i].clone())
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
                                || keyword_score.map_or(false, |k| k >= min_score);

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
            if !languages.is_empty() {
                if !languages.contains(&result.language) {
                    return false;
                }
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
        let table = self.get_table().await?;

        // LanceDB uses SQL-like delete
        let filter = format!("file_path = '{}'", file_path);

        table
            .delete(&filter)
            .await
            .context("Failed to delete records")?;

        // Note: BM25 index entries are not deleted here to avoid maintaining
        // a file_path -> BM25 ID mapping. Stale BM25 entries will be filtered
        // out by the vector search results and cleaned up on next full index.
        // In production, consider maintaining an ID mapping for precise deletion.

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
