use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::bm25_search::BM25Search;
use crate::types::{ChunkMetadata, SearchResult};
use crate::vector_db::{DatabaseStats, VectorDatabase};

/// USearch-based vector database implementation with BM25 hybrid search
///
/// USearch provides 10x faster indexing than LanceDB using HNSW algorithm
/// with excellent memory efficiency (16x better than standard hnswlib).
/// Tantivy provides BM25 keyword search for hybrid retrieval.
pub struct USearchDB {
    /// The USearch index (thread-safe)
    index: Arc<RwLock<Option<Index>>>,
    /// BM25 search index for keyword matching
    bm25_index: Arc<RwLock<Option<BM25Search>>>,
    /// Database path for persistence
    db_path: PathBuf,
    /// Embedding dimension (uses AtomicUsize for thread-safe interior mutability)
    dimension: AtomicUsize,
    /// Metadata storage (indexed by point ID)
    metadata: Arc<RwLock<HashMap<u64, ChunkMetadata>>>,
    /// Content storage (indexed by point ID)
    contents: Arc<RwLock<HashMap<u64, String>>>,
    /// Next available ID for insertion
    next_id: Arc<RwLock<u64>>,
    /// Current collection name
    collection_name: String,
}

impl USearchDB {
    pub fn new<P: AsRef<Path>>(db_path: P, collection_name: &str) -> Result<Self> {
        let db_path = db_path.as_ref().to_path_buf();

        // Initialize BM25 index
        let bm25_path = db_path.join(format!("{}_bm25", collection_name));
        let bm25_index = BM25Search::new(&bm25_path)
            .context("Failed to initialize BM25 index")?;

        Ok(Self {
            index: Arc::new(RwLock::new(None)),
            bm25_index: Arc::new(RwLock::new(Some(bm25_index))),
            db_path,
            dimension: AtomicUsize::new(0),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            contents: Arc::new(RwLock::new(HashMap::new())),
            next_id: Arc::new(RwLock::new(0)),
            collection_name: collection_name.to_string(),
        })
    }

    /// Save index and metadata to disk
    fn save(&self) -> Result<()> {
        // Save the index
        let index_path = self.db_path.join(format!("{}.usearch", self.collection_name));
        let metadata_path = self.db_path.join(format!("{}.meta", self.collection_name));
        let contents_path = self.db_path.join(format!("{}.contents", self.collection_name));

        // Create directory if it doesn't exist
        std::fs::create_dir_all(&self.db_path)
            .context("Failed to create database directory")?;

        // Save index
        if let Some(ref index) = *self.index.read().unwrap() {
            index.save(&index_path.to_string_lossy())
                .context("Failed to save USearch index")?;
        }

        // Save metadata
        let metadata = self.metadata.read().unwrap();
        let metadata_json = serde_json::to_string(&*metadata)
            .context("Failed to serialize metadata")?;
        std::fs::write(metadata_path, metadata_json)
            .context("Failed to write metadata file")?;

        // Save contents
        let contents = self.contents.read().unwrap();
        let contents_json = serde_json::to_string(&*contents)
            .context("Failed to serialize contents")?;
        std::fs::write(contents_path, contents_json)
            .context("Failed to write contents file")?;

        // Save next_id
        let id_path = self.db_path.join(format!("{}.nextid", self.collection_name));
        let next_id = *self.next_id.read().unwrap();
        std::fs::write(id_path, next_id.to_string())
            .context("Failed to write next_id")?;

        Ok(())
    }

    /// Load index and metadata from disk
    fn load(&self) -> Result<()> {
        let index_path = self.db_path.join(format!("{}.usearch", self.collection_name));
        let metadata_path = self.db_path.join(format!("{}.meta", self.collection_name));
        let contents_path = self.db_path.join(format!("{}.contents", self.collection_name));
        let id_path = self.db_path.join(format!("{}.nextid", self.collection_name));

        // Load index if it exists
        if index_path.exists() {
            let mut options = IndexOptions::default();
            options.dimensions = self.dimension.load(Ordering::Relaxed);
            options.metric = MetricKind::Cos;
            options.quantization = ScalarKind::F32;

            let index = Index::new(&options)
                .context("Failed to create USearch index")?;

            index.load(&index_path.to_string_lossy())
                .context("Failed to load USearch index")?;

            *self.index.write().unwrap() = Some(index);
        }

        // Load metadata if it exists
        if metadata_path.exists() {
            let metadata_json = std::fs::read_to_string(metadata_path)
                .context("Failed to read metadata file")?;
            let metadata: HashMap<u64, ChunkMetadata> = serde_json::from_str(&metadata_json)
                .context("Failed to deserialize metadata")?;
            *self.metadata.write().unwrap() = metadata;
        }

        // Load contents if it exists
        if contents_path.exists() {
            let contents_json = std::fs::read_to_string(contents_path)
                .context("Failed to read contents file")?;
            let contents: HashMap<u64, String> = serde_json::from_str(&contents_json)
                .context("Failed to deserialize contents")?;
            *self.contents.write().unwrap() = contents;
        }

        // Load next_id if it exists
        if id_path.exists() {
            let next_id_str = std::fs::read_to_string(id_path)
                .context("Failed to read next_id file")?;
            let next_id: u64 = next_id_str.trim().parse()
                .context("Failed to parse next_id")?;
            *self.next_id.write().unwrap() = next_id;
        }

        Ok(())
    }

    /// Convert score from distance to similarity (0-1 range where 1 is perfect match)
    fn distance_to_similarity(&self, distance: f32) -> f32 {
        // For cosine distance: similarity = 1 - distance
        // Cosine distance range is [0, 2], so we normalize to [0, 1]
        (1.0 - (distance / 2.0)).max(0.0).min(1.0)
    }
}

#[async_trait::async_trait]
impl VectorDatabase for USearchDB {
    async fn initialize(&self, dimension: usize) -> Result<()> {
        // Store dimension (uses AtomicUsize for thread-safe interior mutability)
        self.dimension.store(dimension, Ordering::Relaxed);

        // Try to load existing index
        if let Err(_) = self.load() {
            // If load fails, create a new index
            let mut options = IndexOptions::default();
            options.dimensions = dimension;
            options.metric = MetricKind::Cos;  // Cosine similarity
            options.quantization = ScalarKind::F32;

            let index = Index::new(&options)
                .context("Failed to create USearch index")?;

            *self.index.write().unwrap() = Some(index);
        }

        Ok(())
    }

    async fn store_embeddings(
        &self,
        embeddings: Vec<Vec<f32>>,
        metadata: Vec<ChunkMetadata>,
        contents: Vec<String>,
    ) -> Result<usize> {
        let mut index_lock = self.index.write().unwrap();
        let index = index_lock.as_mut()
            .context("Index not initialized")?;

        let mut metadata_lock = self.metadata.write().unwrap();
        let mut contents_lock = self.contents.write().unwrap();
        let mut next_id_lock = self.next_id.write().unwrap();

        let count = embeddings.len();

        // Prepare documents for BM25 indexing
        let mut bm25_docs = Vec::new();

        for (i, embedding) in embeddings.iter().enumerate() {
            let id = *next_id_lock;

            // Add to USearch index
            index.add(id, embedding)
                .context("Failed to add vector to index")?;

            // Store metadata and content separately
            metadata_lock.insert(id, metadata[i].clone());
            contents_lock.insert(id, contents[i].clone());

            // Prepare for BM25 index
            bm25_docs.push((id, contents[i].clone()));

            *next_id_lock += 1;
        }

        drop(index_lock);
        drop(metadata_lock);
        drop(contents_lock);
        drop(next_id_lock);

        // Add documents to BM25 index
        if let Some(bm25) = self.bm25_index.read().unwrap().as_ref() {
            bm25.add_documents(bm25_docs)
                .context("Failed to add documents to BM25 index")?;
        }

        // Save to disk after batch insert
        self.save().context("Failed to save index after insertion")?;

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
        let index_lock = self.index.read().unwrap();
        let index = index_lock.as_ref()
            .context("Index not initialized")?;

        if hybrid {
            // Hybrid search: combine vector and BM25 results with RRF
            // Get more results from each source for RRF to combine
            let search_limit = limit * 3;

            // Vector search
            let matches = index.search(&query_vector, search_limit)
                .context("Failed to search vector index")?;

            let mut vector_results = Vec::new();
            for (key, distance) in matches.keys.iter().zip(matches.distances.iter()) {
                let score = self.distance_to_similarity(*distance);
                vector_results.push((*key, score));
            }

            // BM25 keyword search
            let bm25_results = if let Some(bm25) = self.bm25_index.read().unwrap().as_ref() {
                bm25.search(query_text, search_limit)
                    .context("Failed to search BM25 index")?
            } else {
                Vec::new()
            };

            // Combine results with Reciprocal Rank Fusion
            let combined = crate::bm25_search::reciprocal_rank_fusion(
                vector_results,
                bm25_results,
                limit,
            );

            // Build final results
            let metadata_lock = self.metadata.read().unwrap();
            let contents_lock = self.contents.read().unwrap();
            let mut results = Vec::new();

            for (key, score) in combined {
                // Filter by min_score
                if score < min_score {
                    continue;
                }

                // Get metadata
                if let Some(meta) = metadata_lock.get(&key) {
                    // Filter by project if specified
                    if let Some(ref project_filter) = project {
                        if meta.project.as_ref() != Some(project_filter) {
                            continue;
                        }
                    }

                    let content = contents_lock.get(&key).cloned().unwrap_or_default();

                    results.push(SearchResult {
                        file_path: meta.file_path.clone(),
                        content,
                        score,
                        vector_score: score,
                        keyword_score: Some(score), // RRF combines both
                        start_line: meta.start_line,
                        end_line: meta.end_line,
                        language: meta.language.clone().unwrap_or_else(|| "Unknown".to_string()),
                        project: meta.project.clone(),
                    });
                }
            }

            Ok(results)
        } else {
            // Pure vector search
            let matches = index.search(&query_vector, limit)
                .context("Failed to search index")?;

            let metadata_lock = self.metadata.read().unwrap();
            let contents_lock = self.contents.read().unwrap();
            let mut results = Vec::new();

            for match_result in matches.keys.iter().zip(matches.distances.iter()) {
                let (key, distance) = match_result;
                let score = self.distance_to_similarity(*distance);

                // Filter by min_score
                if score < min_score {
                    continue;
                }

                // Get metadata
                if let Some(meta) = metadata_lock.get(key) {
                    // Filter by project if specified
                    if let Some(ref project_filter) = project {
                        if meta.project.as_ref() != Some(project_filter) {
                            continue;
                        }
                    }

                    let content = contents_lock.get(key).cloned().unwrap_or_default();

                    results.push(SearchResult {
                        file_path: meta.file_path.clone(),
                        content,
                        score,
                        vector_score: score,
                        keyword_score: None,
                        start_line: meta.start_line,
                        end_line: meta.end_line,
                        language: meta.language.clone().unwrap_or_else(|| "Unknown".to_string()),
                        project: meta.project.clone(),
                    });
                }
            }

            Ok(results)
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

        // Do basic search
        let mut results = self.search(
            query_vector,
            query_text,
            search_limit,
            min_score,
            project,
            hybrid,
        ).await?;

        // Post-process filtering
        results.retain(|result| {
            // Filter by file extension
            if !file_extensions.is_empty() {
                let has_extension = file_extensions.iter().any(|ext| {
                    result.file_path.ends_with(&format!(".{}", ext))
                });
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
                let matches_pattern = path_patterns.iter().any(|pattern| {
                    result.file_path.contains(pattern)
                });
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
        let mut metadata_lock = self.metadata.write().unwrap();
        let index_lock = self.index.read().unwrap();
        let index = index_lock.as_ref()
            .context("Index not initialized")?;

        // Find all IDs for this file
        let ids_to_delete: Vec<u64> = metadata_lock
            .iter()
            .filter(|(_, meta)| meta.file_path == file_path)
            .map(|(id, _)| *id)
            .collect();

        let count = ids_to_delete.len();

        // Remove from vector index
        for id in &ids_to_delete {
            index.remove(*id)
                .context("Failed to remove vector from index")?;
        }

        // Remove from BM25 index
        if let Some(bm25) = self.bm25_index.read().unwrap().as_ref() {
            for id in &ids_to_delete {
                let _ = bm25.delete_by_id(*id); // Ignore errors for missing documents
            }
        }

        // Remove from metadata and contents
        for id in &ids_to_delete {
            metadata_lock.remove(id);
            self.contents.write().unwrap().remove(id);
        }

        drop(index_lock);
        drop(metadata_lock);

        // Save changes
        self.save().context("Failed to save after deletion")?;

        Ok(count)
    }

    async fn clear(&self) -> Result<()> {
        // Clear the vector index by creating a new one
        let mut options = IndexOptions::default();
        options.dimensions = self.dimension.load(Ordering::Relaxed);
        options.metric = MetricKind::Cos;
        options.quantization = ScalarKind::F32;

        let new_index = Index::new(&options)
            .context("Failed to create new USearch index")?;

        *self.index.write().unwrap() = Some(new_index);

        // Clear BM25 index
        if let Some(bm25) = self.bm25_index.read().unwrap().as_ref() {
            bm25.clear().context("Failed to clear BM25 index")?;
        }

        // Clear metadata and contents
        self.metadata.write().unwrap().clear();
        self.contents.write().unwrap().clear();

        // Reset next_id
        *self.next_id.write().unwrap() = 0;

        // Delete saved files
        let index_path = self.db_path.join(format!("{}.usearch", self.collection_name));
        let metadata_path = self.db_path.join(format!("{}.meta", self.collection_name));
        let contents_path = self.db_path.join(format!("{}.contents", self.collection_name));
        let id_path = self.db_path.join(format!("{}.nextid", self.collection_name));

        let _ = std::fs::remove_file(index_path);
        let _ = std::fs::remove_file(metadata_path);
        let _ = std::fs::remove_file(contents_path);
        let _ = std::fs::remove_file(id_path);

        Ok(())
    }

    async fn get_statistics(&self) -> Result<DatabaseStats> {
        let metadata_lock = self.metadata.read().unwrap();
        let total_chunks = metadata_lock.len();

        // Count files and languages
        let mut files = std::collections::HashSet::new();
        let mut language_counts: HashMap<String, usize> = HashMap::new();

        for meta in metadata_lock.values() {
            files.insert(meta.file_path.clone());

            let lang = meta.language.as_deref().unwrap_or("Unknown");
            *language_counts.entry(lang.to_string()).or_insert(0) += 1;
        }

        // Convert HashMap to Vec<(String, usize)>
        let language_breakdown: Vec<(String, usize)> = language_counts.into_iter().collect();

        Ok(DatabaseStats {
            total_points: files.len(),
            total_vectors: total_chunks,
            language_breakdown,
        })
    }
}
