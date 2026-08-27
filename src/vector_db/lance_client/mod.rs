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
use crate::glob_utils;
use crate::types::{ChunkMetadata, SearchResult};
use crate::vector_db::{DatabaseStats, LanguageBreakdown, VectorDatabase};
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
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, RwLock};

/// LanceDB vector database implementation (embedded, no server required)
/// Includes BM25 hybrid search support using Tantivy with per-project indexes
pub struct LanceVectorDB {
    connection: Connection,
    table_name: String,
    db_path: String,
    /// Per-project BM25 search indexes for keyword matching
    /// Key: hashed root path, Value: BM25Search instance
    bm25_indexes: Arc<RwLock<HashMap<String, BM25Search>>>,
}

/// Total on-disk size of every file under the given directory.
///
/// LanceDB stores a directory tree rather than a single file. Entries that
/// cannot be read are skipped rather than failing the whole statistics call,
/// so this is a best-effort figure.
fn directory_size_bytes(path: &Path) -> u64 {
    walkdir::WalkDir::new(path)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.metadata().ok())
        .filter(|metadata| metadata.is_file())
        .map(|metadata| metadata.len())
        .sum()
}

impl LanceVectorDB {
    /// Create a new LanceDB instance with an explicit path
    pub async fn with_path(db_path: &str) -> Result<Self> {
        tracing::info!("Connecting to LanceDB at: {}", db_path);

        let connection = lancedb::connect(db_path)
            .execute()
            .await
            .context("Failed to connect to LanceDB")?;

        // Initialize empty per-project BM25 index map
        // BM25 indexes are created on-demand per root path
        let bm25_indexes = Arc::new(RwLock::new(HashMap::new()));

        Ok(Self {
            connection,
            table_name: "code_embeddings".to_string(),
            db_path: db_path.to_string(),
            bm25_indexes,
        })
    }

    /// Hash a root path to create a unique identifier for per-project BM25 indexes
    fn hash_root_path(root_path: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(root_path.as_bytes());
        let result = hasher.finalize();
        // Use first 16 characters of hex hash for brevity
        format!("{:x}", result)[..16].to_string()
    }

    /// Get the BM25 index path for a specific root path
    fn bm25_path_for_root(&self, root_path: &str) -> String {
        let hash = Self::hash_root_path(root_path);
        format!("{}/bm25_{}", self.db_path, hash)
    }

    /// Get or create a BM25 index for a specific root path
    fn get_or_create_bm25(&self, root_path: &str) -> Result<()> {
        let hash = Self::hash_root_path(root_path);

        // Check if already exists (read lock)
        {
            let indexes = self.bm25_indexes.read().map_err(|e| {
                anyhow::anyhow!("Failed to acquire read lock on BM25 indexes: {}", e)
            })?;
            if indexes.contains_key(&hash) {
                return Ok(()); // Already exists
            }
        }

        // Need to create new index (write lock)
        let mut indexes = self
            .bm25_indexes
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire write lock on BM25 indexes: {}", e))?;

        // Double-check after acquiring write lock (another thread might have created it)
        if indexes.contains_key(&hash) {
            return Ok(());
        }

        let bm25_path = self.bm25_path_for_root(root_path);
        tracing::info!(
            "Creating BM25 index for root path '{}' at: {}",
            root_path,
            bm25_path
        );

        let bm25_index = BM25Search::new(&bm25_path)
            .with_context(|| format!("Failed to initialize BM25 index for root: {}", root_path))?;

        indexes.insert(hash, bm25_index);

        Ok(())
    }

    /// The chunk's stable identity, and the fusion key shared by the vector table's `id`
    /// column and the BM25 index. Both arms MUST derive it here; deriving it in two places
    /// is how they silently drifted apart and killed hybrid search.
    ///
    /// `file_hash` is part of the key because git commits are stored as chunks whose
    /// file_path is `git://<repo>` and whose start_line is 0 -- identical for EVERY commit
    /// in a repository. Keyed on path and line alone they would all collapse onto one id
    /// and the BM25 index would hold a single document for the entire history. For code
    /// chunks the hash is constant within a file, so start_line still provides uniqueness.
    fn chunk_id(meta: &ChunkMetadata) -> String {
        format!("{}:{}:{}", meta.file_path, meta.start_line, meta.file_hash)
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
            Field::new("root_path", DataType::Utf8, true),
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
                .map(|i| Self::chunk_id(&metadata[i]))
                .collect::<Vec<_>>(),
        );
        let file_path_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.file_path.as_str())
                .collect::<Vec<_>>(),
        );
        let root_path_array = StringArray::from(
            metadata
                .iter()
                .map(|m| m.root_path.as_deref())
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
                Arc::new(root_path_array),
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
        root_path: &str,
    ) -> Result<usize> {
        if embeddings.is_empty() {
            return Ok(0);
        }

        let dimension = embeddings[0].len();
        let schema = Self::create_schema(dimension);

        let table = self.get_table().await?;

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

        // Ensure BM25 index exists for this root path
        self.get_or_create_bm25(root_path)?;

        // Add documents to per-project BM25 index, keyed by the SAME stable chunk id the
        // vector table stores in its `id` column -- see Self::chunk_id. The previous
        // `count_rows() + i` scheme was not merely a different key space, it was unstable:
        // incremental re-indexing deletes and re-adds rows, so the row numbers drifted.
        let bm25_docs: Vec<_> = (0..count)
            .map(|i| {
                (
                    Self::chunk_id(&metadata[i]),
                    contents[i].clone(),
                    metadata[i].file_path.clone(),
                )
            })
            .collect();

        let hash = Self::hash_root_path(root_path);
        let bm25_indexes = self
            .bm25_indexes
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire BM25 read lock: {}", e))?;

        if let Some(bm25) = bm25_indexes.get(&hash) {
            bm25.add_documents(bm25_docs)
                .context("Failed to add documents to BM25 index")?;
        }
        drop(bm25_indexes);

        tracing::info!(
            "Stored {} embeddings with BM25 indexing for root: {}",
            count,
            root_path
        );
        Ok(count)
    }

    async fn search(
        &self,
        query_vector: Vec<f32>,
        query_text: &str,
        limit: usize,
        min_score: f32,
        project: Option<String>,
        root_path: Option<String>,
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

            // Build vector results keyed by the chunk's stable `id` column.
            //
            // This used to key on `row_offset + i`, a position within THIS query's result
            // batches, while the BM25 arm keyed on a table row number assigned at index
            // time. The two spaces coincide only for the first insert into an empty table,
            // so in any real index RRF fused two disjoint key sets: every vector hit scored
            // exactly 1/(60+rank), keyword_score never populated, and BM25-only hits were
            // dropped. Both arms now use `file_path:start_line`.
            let mut vector_results: Vec<(String, f32)> = Vec::new();
            let mut original_scores: HashMap<String, (f32, Option<f32>)> = HashMap::new();
            // chunk id -> (index into `results`, row within that batch)
            let mut chunk_pos: HashMap<String, (usize, usize)> = HashMap::new();

            for (batch_idx, batch) in results.iter().enumerate() {
                let distance_array = batch
                    .column_by_name("_distance")
                    .context("Missing _distance column")?
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .context("Invalid _distance type")?;
                let id_array = batch
                    .column_by_name("id")
                    .context("Missing id column")?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("Invalid id type")?;

                for i in 0..batch.num_rows() {
                    let distance = distance_array.value(i);
                    let score = 1.0 / (1.0 + distance);
                    let chunk_id = id_array.value(i).to_string();

                    // For hybrid search, don't filter by min_score before RRF
                    // RRF will combine weak vector + strong keyword (or vice versa)
                    // Filtering happens after RRF based on the combined ranking
                    vector_results.push((chunk_id.clone(), score));
                    original_scores.insert(chunk_id.clone(), (score, None));
                    chunk_pos.insert(chunk_id, (batch_idx, i));
                }
            }

            // BM25 keyword search across all per-project indexes.
            //
            // Scoped so the RwLockReadGuard is released at the end of the block: this
            // function now awaits further down, and a guard held across an await makes the
            // whole future non-Send, which the VectorDatabase trait requires. An explicit
            // drop() is not enough -- the generator transform still captures it.
            let bm25_results = {
                let bm25_indexes = self
                    .bm25_indexes
                    .read()
                    .map_err(|e| anyhow::anyhow!("Failed to acquire BM25 read lock: {}", e))?;

                let mut all_bm25_results = Vec::new();
                for (root_hash, bm25) in bm25_indexes.iter() {
                    tracing::debug!("Searching BM25 index for root hash: {}", root_hash);
                    let results = bm25
                        .search(query_text, search_limit)
                        .context("Failed to search BM25 index")?;

                    // Store BM25 scores (don't filter - let RRF combine them)
                    // BM25 scores are not normalized to 0-1 range, so min_score doesn't apply
                    for result in &results {
                        original_scores
                            .entry(result.chunk_id.clone())
                            .and_modify(|e| e.1 = Some(result.score))
                            .or_insert((0.0, Some(result.score))); // No vector score, only keyword
                    }

                    all_bm25_results.extend(results);
                }
                all_bm25_results
            };

            // Combine results with Reciprocal Rank Fusion
            // RRF produces scores ~0.01-0.03, so don't apply min_score to combined scores
            let combined =
                crate::bm25_search::reciprocal_rank_fusion(vector_results, bm25_results, limit);

            // Build final results from the fused ranking.
            //
            // Vector-arm rows are materialised from the batches already in hand. Ids that
            // ONLY BM25 matched are fetched from the table below -- without that step a
            // pure keyword hit, which is the exact-symbol case hybrid search exists for,
            // would be ranked and then silently dropped.
            let missing: Vec<String> = combined
                .iter()
                .map(|(id, _)| id)
                .filter(|id| !chunk_pos.contains_key(*id))
                .cloned()
                .collect();

            let extra_batches: Vec<RecordBatch> = if missing.is_empty() {
                Vec::new()
            } else {
                let quoted: Vec<String> = missing
                    .iter()
                    .map(|id| format!("'{}'", id.replace('\'', "''")))
                    .collect();
                match table
                    .query()
                    .only_if(format!("id IN ({})", quoted.join(", ")))
                    .execute()
                    .await
                {
                    Ok(stream) => stream.try_collect().await.unwrap_or_else(|e| {
                        tracing::warn!("Failed to collect keyword-only rows: {}", e);
                        Vec::new()
                    }),
                    Err(e) => {
                        tracing::warn!("Failed to fetch keyword-only rows: {}", e);
                        Vec::new()
                    }
                }
            };

            let mut extra_pos: HashMap<String, (usize, usize)> = HashMap::new();
            for (batch_idx, batch) in extra_batches.iter().enumerate() {
                if let Some(id_array) = batch
                    .column_by_name("id")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                {
                    for i in 0..batch.num_rows() {
                        extra_pos.insert(id_array.value(i).to_string(), (batch_idx, i));
                    }
                }
            }

            let mut search_results = Vec::new();

            for (chunk_id, combined_score) in combined {
                let (batch, idx) = match chunk_pos.get(&chunk_id) {
                    Some(&(b, i)) => (&results[b], i),
                    None => match extra_pos.get(&chunk_id) {
                        Some(&(b, i)) => (&extra_batches[b], i),
                        None => {
                            tracing::warn!("Could not materialise fused result {}", chunk_id);
                            continue;
                        }
                    },
                };

                let file_path_array = batch
                    .column_by_name("file_path")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                let root_path_array = batch
                    .column_by_name("root_path")
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

                if let (
                    Some(fp),
                    Some(rp),
                    Some(sl),
                    Some(el),
                    Some(lang),
                    Some(cont),
                    Some(proj),
                ) = (
                    file_path_array,
                    root_path_array,
                    start_line_array,
                    end_line_array,
                    language_array,
                    content_array,
                    project_array,
                ) {
                    // Look up original scores for filtering and reporting
                    let (vector_score, keyword_score) = original_scores
                        .get(&chunk_id)
                        .copied()
                        .unwrap_or((0.0, None));

                    // For hybrid search, apply min_score intelligently:
                    // Accept if EITHER vector or keyword score meets threshold
                    // This allows pure keyword matches (weak vector) and pure semantic matches (weak keyword)
                    let passes_filter =
                        vector_score >= min_score || keyword_score.is_some_and(|k| k >= min_score);

                    if !passes_filter {
                        continue;
                    }

                    let result_root_path = if rp.is_null(idx) {
                        None
                    } else {
                        Some(rp.value(idx).to_string())
                    };

                    // Filter by root_path if specified
                    if let Some(ref filter_path) = root_path
                        && result_root_path.as_ref() != Some(filter_path)
                    {
                        continue;
                    }

                    // Use RRF combined score as the main score for ranking
                    // But report original vector/keyword scores for transparency
                    search_results.push(SearchResult {
                        score: combined_score, // RRF score for ranking
                        vector_score,          // Original vector score
                        keyword_score,         // Original BM25 score
                        file_path: fp.value(idx).to_string(),
                        root_path: result_root_path,
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

                let root_path_array = batch
                    .column_by_name("root_path")
                    .context("Missing root_path column")?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("Invalid root_path type")?;

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
                        let result_root_path = if root_path_array.is_null(i) {
                            None
                        } else {
                            Some(root_path_array.value(i).to_string())
                        };

                        // Filter by root_path if specified
                        if let Some(ref filter_path) = root_path {
                            if result_root_path.as_ref() != Some(filter_path) {
                                continue;
                            }
                        }

                        search_results.push(SearchResult {
                            score,
                            vector_score: score,
                            keyword_score: None,
                            file_path: file_path_array.value(i).to_string(),
                            root_path: result_root_path,
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
        root_path: Option<String>,
        hybrid: bool,
        file_extensions: Vec<String>,
        languages: Vec<String>,
        path_patterns: Vec<String>,
    ) -> Result<Vec<SearchResult>> {
        // These filters are applied AFTER the search, so the candidate pool has to be big
        // enough that the surviving rows can actually fill `limit`. With the old `limit * 3`
        // this silently returned nothing whenever the wanted rows were a small minority of
        // the corpus -- search_git_history is exactly that shape: a few hundred commits
        // sharing a table with tens of thousands of code chunks, so no commit ever reached
        // the top-N and the language filter then emptied the list every time.
        //
        // The proper fix is predicate pushdown into the LanceDB query; until then, widen
        // the pool when a filter is actually present. The keyword arm helps here too now
        // that fusion works: a commit whose message contains the query terms is surfaced by
        // BM25 directly rather than having to win on vector distance.
        let filtered =
            !file_extensions.is_empty() || !languages.is_empty() || !path_patterns.is_empty();
        let search_limit = if filtered {
            (limit * 20).max(200)
        } else {
            limit * 3
        };

        // Do basic search with hybrid support
        let mut results = self
            .search(
                query_vector,
                query_text,
                search_limit,
                min_score,
                project.clone(),
                root_path.clone(),
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

            // Filter by path pattern using proper glob matching
            if !path_patterns.is_empty() {
                if !glob_utils::matches_any_pattern(&result.file_path, &path_patterns) {
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
        // Delete from all per-project BM25 indexes
        // Must be done in a scope to drop lock before await
        {
            let bm25_indexes = self
                .bm25_indexes
                .read()
                .map_err(|e| anyhow::anyhow!("Failed to acquire BM25 read lock: {}", e))?;

            for (root_hash, bm25) in bm25_indexes.iter() {
                bm25.delete_by_file_path(file_path)
                    .context("Failed to delete from BM25 index")?;
                tracing::debug!(
                    "Deleted BM25 entries for file: {} in index: {}",
                    file_path,
                    root_hash
                );
            }
        } // bm25_indexes dropped here

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
        // Drop and recreate table (empty namespace array for default namespace)
        self.connection
            .drop_table(&self.table_name, &[])
            .await
            .context("Failed to drop table")?;

        // Clear all per-project BM25 indexes
        let bm25_indexes = self
            .bm25_indexes
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire BM25 read lock: {}", e))?;

        for (root_hash, bm25) in bm25_indexes.iter() {
            bm25.clear().context("Failed to clear BM25 index")?;
            tracing::info!("Cleared BM25 index for root hash: {}", root_hash);
        }
        drop(bm25_indexes);

        tracing::info!("Cleared all embeddings and all per-project BM25 indexes");
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
                "file_path".to_string(),
                "root_path".to_string(),
            ]))
            .execute()
            .await
            .context("Failed to query languages")?;

        let query_result: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .context("Failed to collect language data")?;

        let mut chunk_counts: HashMap<String, usize> = HashMap::new();
        let mut files_by_language: HashMap<String, HashSet<(String, String)>> = HashMap::new();
        let mut all_files: HashSet<(String, String)> = HashSet::new();

        for batch in query_result {
            let language_array = batch
                .column_by_name("language")
                .context("Missing language column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid language type")?;

            let file_path_array = batch
                .column_by_name("file_path")
                .context("Missing file_path column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid file_path type")?;

            let root_path_array = batch
                .column_by_name("root_path")
                .context("Missing root_path column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid root_path type")?;

            for i in 0..batch.num_rows() {
                let language = language_array.value(i);

                // file_path is stored relative to root_path, so only the pair
                // identifies a file: the same relative path can exist under
                // two different indexed roots.
                let root = if root_path_array.is_null(i) {
                    String::new()
                } else {
                    root_path_array.value(i).to_string()
                };
                let file_key = (root, file_path_array.value(i).to_string());

                *chunk_counts.entry(language.to_string()).or_insert(0) += 1;
                files_by_language
                    .entry(language.to_string())
                    .or_default()
                    .insert(file_key.clone());
                all_files.insert(file_key);
            }
        }

        let mut language_breakdown: Vec<LanguageBreakdown> = chunk_counts
            .into_iter()
            .map(|(language, chunk_count)| {
                let file_count = files_by_language.get(&language).map_or(0, |f| f.len());
                LanguageBreakdown {
                    language,
                    file_count,
                    chunk_count,
                }
            })
            .collect();
        language_breakdown.sort_by(|a, b| b.chunk_count.cmp(&a.chunk_count));

        Ok(DatabaseStats {
            total_files: all_files.len(),
            total_points: count_result,
            total_vectors: count_result,
            database_size_bytes: directory_size_bytes(Path::new(&self.db_path)),
            language_breakdown,
        })
    }

    async fn flush(&self) -> Result<()> {
        // LanceDB persists automatically, no explicit flush needed
        Ok(())
    }

    async fn count_by_root_path(&self, root_path: &str) -> Result<usize> {
        let table = self.get_table().await?;

        // Use SQL-like filter to count rows with matching root_path
        let filter = format!("root_path = '{}'", root_path);
        let count = table
            .count_rows(Some(filter))
            .await
            .context("Failed to count rows by root path")?;

        Ok(count)
    }

    async fn get_indexed_files(&self, root_path: &str) -> Result<Vec<String>> {
        let table = self.get_table().await?;

        // Query file_path column filtered by root_path
        let filter = format!("root_path = '{}'", root_path);
        let stream = table
            .query()
            .only_if(filter)
            .select(lancedb::query::Select::Columns(vec![
                "file_path".to_string(),
            ]))
            .execute()
            .await
            .context("Failed to query indexed files")?;

        let results: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .context("Failed to collect file paths")?;

        // Extract unique file paths
        let mut file_paths = std::collections::HashSet::new();

        for batch in results {
            let file_path_array = batch
                .column_by_name("file_path")
                .context("Missing file_path column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid file_path type")?;

            for i in 0..batch.num_rows() {
                file_paths.insert(file_path_array.value(i).to_string());
            }
        }

        Ok(file_paths.into_iter().collect())
    }
}

#[cfg(test)]
mod tests;
