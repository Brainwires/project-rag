use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{Index, IndexWriter, ReloadPolicy, TantivyDocument, doc};

/// BM25-based keyword search using Tantivy
pub struct BM25Search {
    index: Index,
    id_field: Field,
    content_field: Field,
    file_path_field: Field,
    /// Path to the index directory (needed for lock cleanup)
    index_path: std::path::PathBuf,
    /// Mutex to ensure only one IndexWriter is created at a time
    writer_lock: Mutex<()>,
}

/// Search result from BM25
#[derive(Debug, Clone)]
pub struct BM25Result {
    pub id: u64,
    pub score: f32,
}

impl BM25Search {
    /// Create a new BM25 search index
    pub fn new<P: AsRef<Path>>(index_path: P) -> Result<Self> {
        let index_path = index_path.as_ref().to_path_buf();

        // Create schema with ID, content, and file_path fields
        let mut schema_builder = Schema::builder();
        let id_field = schema_builder.add_u64_field("id", STORED | INDEXED);
        let content_field = schema_builder.add_text_field("content", TEXT);
        let file_path_field = schema_builder.add_text_field("file_path", STRING | STORED);
        let schema = schema_builder.build();

        // Create or open index
        std::fs::create_dir_all(&index_path).context("Failed to create BM25 index directory")?;

        let index = if index_path.join("meta.json").exists() {
            Index::open_in_dir(&index_path).context("Failed to open existing BM25 index")?
        } else {
            Index::create_in_dir(&index_path, schema.clone())
                .context("Failed to create BM25 index")?
        };

        Ok(Self {
            index,
            id_field,
            content_field,
            file_path_field,
            index_path,
            writer_lock: Mutex::new(()),
        })
    }

    /// Check if a lock file is stale (older than 5 minutes with no recent activity)
    fn is_lock_stale(lock_path: &Path) -> bool {
        if !lock_path.exists() {
            return false;
        }

        // Check file modification time
        if let Ok(metadata) = std::fs::metadata(lock_path)
            && let Ok(modified) = metadata.modified()
            && let Ok(elapsed) = modified.elapsed()
        {
            // Consider lock stale if older than 5 minutes
            return elapsed.as_secs() > 300;
        }

        false
    }

    /// Try to clean up stale lock files only if they appear to be from crashed processes
    fn try_cleanup_stale_locks(index_path: &Path) -> Result<bool> {
        let writer_lock = index_path.join(".tantivy-writer.lock");
        let meta_lock = index_path.join(".tantivy-meta.lock");

        let writer_stale = Self::is_lock_stale(&writer_lock);
        let meta_stale = Self::is_lock_stale(&meta_lock);

        if !writer_stale && !meta_stale {
            return Ok(false); // Locks appear to be active
        }

        if writer_stale && writer_lock.exists() {
            tracing::warn!(
                "Removing stale Tantivy writer lock file (>5min old): {:?}",
                writer_lock
            );
            std::fs::remove_file(&writer_lock)
                .context("Failed to remove stale writer lock file")?;
        }

        if meta_stale && meta_lock.exists() {
            tracing::warn!(
                "Removing stale Tantivy meta lock file (>5min old): {:?}",
                meta_lock
            );
            std::fs::remove_file(&meta_lock).context("Failed to remove stale meta lock file")?;
        }

        Ok(true) // Cleaned up stale locks
    }

    /// Add documents to the index
    ///
    /// Arguments:
    /// * `documents` - Vec of (id, content, file_path) tuples
    pub fn add_documents(&self, documents: Vec<(u64, String, String)>) -> Result<()> {
        // Lock to ensure only one writer at a time (within this process)
        let _guard = self
            .writer_lock
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire writer lock: {}", e))?;

        // Try to create the index writer
        let mut index_writer: IndexWriter<TantivyDocument> = match self.index.writer(50_000_000) {
            Ok(writer) => writer,
            Err(e) => {
                // Check if this is a lock error
                let error_msg = format!("{}", e);
                if error_msg.contains("lock") || error_msg.contains("Lock") {
                    tracing::warn!(
                        "Index writer creation failed (possibly locked), checking for stale locks..."
                    );

                    // Try to cleanup stale locks
                    match Self::try_cleanup_stale_locks(&self.index_path) {
                        Ok(true) => {
                            // Stale locks were cleaned up, retry once
                            tracing::info!("Stale locks cleaned up, retrying writer creation...");
                            self.index.writer(50_000_000).context(
                                "Failed to create index writer after cleaning stale locks",
                            )?
                        }
                        Ok(false) => {
                            // Locks exist but are not stale (another process is actively using the index)
                            return Err(anyhow::anyhow!(
                                "BM25 index is currently being used by another process. Please wait and try again later."
                            ));
                        }
                        Err(cleanup_err) => {
                            // Failed to cleanup locks
                            return Err(anyhow::anyhow!(
                                "Failed to create index writer (locked) and failed to cleanup stale locks: {}. Original error: {}",
                                cleanup_err,
                                e
                            ));
                        }
                    }
                } else {
                    // Not a lock error, propagate original error
                    return Err(e).context("Failed to create index writer");
                }
            }
        };

        for (id, content, file_path) in documents {
            let doc = doc!(
                self.id_field => id,
                self.content_field => content,
                self.file_path_field => file_path,
            );
            index_writer
                .add_document(doc)
                .context("Failed to add document")?;
        }

        index_writer
            .commit()
            .context("Failed to commit documents")?;

        Ok(())
    }

    /// Search the index with BM25 scoring
    pub fn search(&self, query_text: &str, limit: usize) -> Result<Vec<BM25Result>> {
        let reader = self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .context("Failed to create index reader")?;

        let searcher = reader.searcher();

        // Parse query using lenient mode to handle special characters like :: in code
        // (e.g., "Tool::new" would fail strict parsing since : is a field separator)
        let query_parser = QueryParser::for_index(&self.index, vec![self.content_field]);
        let (query, _errors) = query_parser.parse_query_lenient(query_text);

        // Search with BM25
        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(limit))
            .context("Failed to execute search")?;

        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = searcher
                .doc(doc_address)
                .context("Failed to retrieve document")?;

            if let Some(id_value) = retrieved_doc.get_first(self.id_field)
                && let Some(id) = id_value.as_u64()
            {
                results.push(BM25Result { id, score });
            }
        }

        Ok(results)
    }

    /// Delete all documents for a specific ID
    pub fn delete_by_id(&self, id: u64) -> Result<()> {
        // Lock to ensure only one writer at a time
        let _guard = self
            .writer_lock
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire writer lock: {}", e))?;

        let mut index_writer: IndexWriter<TantivyDocument> = self
            .index
            .writer(50_000_000)
            .context("Failed to create index writer")?;

        let term = Term::from_field_u64(self.id_field, id);
        index_writer.delete_term(term);

        index_writer.commit().context("Failed to commit deletion")?;

        Ok(())
    }

    /// Delete all documents with a specific file_path
    ///
    /// This is used for incremental updates when files are deleted or modified.
    pub fn delete_by_file_path(&self, file_path: &str) -> Result<usize> {
        // Lock to ensure only one writer at a time
        let _guard = self
            .writer_lock
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire writer lock: {}", e))?;

        let mut index_writer: IndexWriter<TantivyDocument> = self
            .index
            .writer(50_000_000)
            .context("Failed to create index writer")?;

        let term = Term::from_field_text(self.file_path_field, file_path);
        index_writer.delete_term(term);

        index_writer
            .commit()
            .context("Failed to commit file_path deletion")?;

        // Note: Tantivy doesn't return count of deleted documents
        // Return 0 as placeholder
        Ok(0)
    }

    /// Clear the entire index
    pub fn clear(&self) -> Result<()> {
        // Lock to ensure only one writer at a time
        let _guard = self
            .writer_lock
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire writer lock: {}", e))?;

        let mut index_writer: IndexWriter<TantivyDocument> = self
            .index
            .writer(50_000_000)
            .context("Failed to create index writer")?;

        index_writer
            .delete_all_documents()
            .context("Failed to delete all documents")?;

        index_writer.commit().context("Failed to commit clear")?;

        Ok(())
    }

    /// Get index statistics
    pub fn get_stats(&self) -> Result<BM25Stats> {
        let reader = self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .context("Failed to create index reader")?;

        let searcher = reader.searcher();
        let total_docs = searcher.num_docs() as usize;

        Ok(BM25Stats {
            total_documents: total_docs,
        })
    }
}

/// Statistics about the BM25 index
#[derive(Debug, Clone)]
pub struct BM25Stats {
    pub total_documents: usize,
}

/// Standard RRF constant (60.0 is the commonly used value from the RRF paper)
pub const RRF_K_CONSTANT: f32 = 60.0;

/// Reciprocal Rank Fusion (RRF) for combining vector and BM25 results
pub fn reciprocal_rank_fusion(
    vector_results: Vec<(u64, f32)>,
    bm25_results: Vec<BM25Result>,
    k: usize,
) -> Vec<(u64, f32)> {
    let mut score_map: HashMap<u64, f32> = HashMap::new();

    // Add vector scores with RRF formula: 1 / (k + rank)
    for (rank, (id, _score)) in vector_results.iter().enumerate() {
        let rrf_score = 1.0 / (RRF_K_CONSTANT + (rank + 1) as f32);
        *score_map.entry(*id).or_insert(0.0) += rrf_score;
    }

    // Add BM25 scores with RRF formula
    for (rank, result) in bm25_results.iter().enumerate() {
        let rrf_score = 1.0 / (RRF_K_CONSTANT + (rank + 1) as f32);
        *score_map.entry(result.id).or_insert(0.0) += rrf_score;
    }

    // Convert to vec and sort by combined score
    let mut combined: Vec<(u64, f32)> = score_map.into_iter().collect();
    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Take top k results
    combined.truncate(k);

    combined
}

/// Generic Reciprocal Rank Fusion (RRF) for combining arbitrary ranked lists
///
/// This is a generic version that works with any type that implements Eq + Hash + Clone.
/// Useful for combining results from different search systems.
///
/// # Arguments
/// * `ranked_lists` - Iterator of ranked result lists, each containing (id, original_score)
/// * `limit` - Maximum results to return
///
/// # Returns
/// Vec of (id, combined_rrf_score) sorted by score descending
pub fn reciprocal_rank_fusion_generic<T, I, L>(ranked_lists: I, limit: usize) -> Vec<(T, f32)>
where
    T: Eq + std::hash::Hash + Clone,
    I: IntoIterator<Item = L>,
    L: IntoIterator<Item = (T, f32)>,
{
    let mut score_map: HashMap<T, f32> = HashMap::new();

    for list in ranked_lists {
        for (rank, (id, _score)) in list.into_iter().enumerate() {
            let rrf_score = 1.0 / (RRF_K_CONSTANT + (rank + 1) as f32);
            *score_map.entry(id).or_insert(0.0) += rrf_score;
        }
    }

    let mut combined: Vec<(T, f32)> = score_map.into_iter().collect();
    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    combined.truncate(limit);

    combined
}
