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
    chunk_id_field: Field,
    content_field: Field,
    file_path_field: Field,
    /// Path to the index directory (needed for lock cleanup)
    index_path: std::path::PathBuf,
    /// Mutex to ensure only one IndexWriter is created at a time
    writer_lock: Mutex<()>,
}

/// Search result from BM25
///
/// `chunk_id` is the chunk's stable `file_path:start_line` identifier -- the same value
/// stored in the vector table's `id` column. Both retrieval arms must key on it, or
/// Reciprocal Rank Fusion silently fuses nothing. See the note on `add_documents`.
#[derive(Debug, Clone)]
pub struct BM25Result {
    pub chunk_id: String,
    pub score: f32,
}

impl BM25Search {
    /// Create a new BM25 search index
    pub fn new<P: AsRef<Path>>(index_path: P) -> Result<Self> {
        let index_path = index_path.as_ref().to_path_buf();

        // Schema keyed by the chunk's stable `file_path:start_line` id.
        let build_schema = || {
            let mut b = Schema::builder();
            b.add_text_field("chunk_id", STRING | STORED);
            b.add_text_field("content", TEXT);
            b.add_text_field("file_path", STRING | STORED);
            b.build()
        };

        // Create or open index
        std::fs::create_dir_all(&index_path).context("Failed to create BM25 index directory")?;

        let mut index = if index_path.join("meta.json").exists() {
            Index::open_in_dir(&index_path).context("Failed to open existing BM25 index")?
        } else {
            Index::create_in_dir(&index_path, build_schema())
                .context("Failed to create BM25 index")?
        };

        // Older builds keyed documents by a u64 `id` holding a table row number. Field
        // handles are positional, so opening one of those directories with the new schema
        // would read a u64 field as text and corrupt every lookup. Detect and rebuild --
        // the BM25 index is a derived artifact, so throwing it away costs only a re-index.
        if index.schema().get_field("chunk_id").is_err() {
            tracing::warn!(
                "BM25 index at {:?} predates chunk_id keying; rebuilding it",
                index_path
            );
            drop(index);
            std::fs::remove_dir_all(&index_path)
                .context("Failed to remove stale BM25 index directory")?;
            std::fs::create_dir_all(&index_path)
                .context("Failed to recreate BM25 index directory")?;
            index = Index::create_in_dir(&index_path, build_schema())
                .context("Failed to recreate BM25 index")?;
        }

        // Take handles from the schema actually on disk, never from a locally built one.
        let schema = index.schema();
        let chunk_id_field = schema
            .get_field("chunk_id")
            .context("BM25 schema is missing the chunk_id field")?;
        let content_field = schema
            .get_field("content")
            .context("BM25 schema is missing the content field")?;
        let file_path_field = schema
            .get_field("file_path")
            .context("BM25 schema is missing the file_path field")?;

        Ok(Self {
            index,
            chunk_id_field,
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
    /// * `documents` - Vec of (chunk_id, content, file_path) tuples
    ///
    /// `chunk_id` MUST be the same `file_path:start_line` value stored in the vector
    /// table's `id` column. It used to be a `count_rows()`-derived row number, which
    /// matched the vector arm's batch-relative index only for the first insert into a
    /// fresh table -- which is exactly the shape the unit tests create, so the mismatch
    /// never showed up there while fusion was dead in every real index.
    pub fn add_documents(&self, documents: Vec<(String, String, String)>) -> Result<()> {
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

        for (chunk_id, content, file_path) in documents {
            let doc = doc!(
                self.chunk_id_field => chunk_id,
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

            if let Some(id_value) = retrieved_doc.get_first(self.chunk_id_field)
                && let Some(chunk_id) = id_value.as_str()
            {
                results.push(BM25Result {
                    chunk_id: chunk_id.to_string(),
                    score,
                });
            }
        }

        Ok(results)
    }

    /// Delete all documents for a specific chunk ID
    #[allow(dead_code)]
    pub fn delete_by_chunk_id(&self, chunk_id: &str) -> Result<()> {
        // Lock to ensure only one writer at a time
        let _guard = self
            .writer_lock
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire writer lock: {}", e))?;

        let mut index_writer: IndexWriter<TantivyDocument> = self
            .index
            .writer(50_000_000)
            .context("Failed to create index writer")?;

        let term = Term::from_field_text(self.chunk_id_field, chunk_id);
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
///
/// This is a convenience wrapper around `reciprocal_rank_fusion_generic` for the common case
/// of combining vector search results (u64 IDs) with BM25 results.
pub fn reciprocal_rank_fusion(
    vector_results: Vec<(String, f32)>,
    bm25_results: Vec<BM25Result>,
    k: usize,
) -> Vec<(String, f32)> {
    // Convert BM25 results to the same format as vector results
    let bm25_tuples: Vec<(String, f32)> = bm25_results
        .into_iter()
        .map(|r| (r.chunk_id, r.score))
        .collect();

    // Use the generic implementation
    reciprocal_rank_fusion_generic([vector_results, bm25_tuples], k)
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
