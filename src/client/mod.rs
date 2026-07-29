//! Core library client for project-rag
//!
//! This module provides the main client interface for using project-rag
//! as a library in your own Rust applications.

use crate::cache::HashCache;
use crate::config::Config;
use crate::embedding::{EmbeddingProvider, FastEmbedManager};
use crate::git_cache::GitCache;
use crate::indexer::{CodeChunker, FileInfo, detect_language};
use crate::relations::{
    DefinitionResult, HybridRelationsProvider, ReferenceResult, RelationsProvider,
};
use crate::types::*;
use crate::vector_db::VectorDatabase;

// Conditionally import the appropriate vector database backend
#[cfg(feature = "qdrant-backend")]
use crate::vector_db::QdrantVectorDB;

#[cfg(not(feature = "qdrant-backend"))]
use crate::vector_db::LanceVectorDB;

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tokio::sync::broadcast;

// Filesystem locking for cross-process coordination
mod fs_lock;
pub(crate) use fs_lock::FsLockGuard;

// Index locking mechanism (uses fs_lock for cross-process, broadcast for in-process)
mod index_lock;
pub(crate) use index_lock::{IndexLockGuard, IndexLockResult, IndexingOperation};

/// Main client for interacting with the RAG system
///
/// This client provides a high-level API for indexing codebases and performing
/// semantic searches. It contains all the core functionality and can be used
/// directly as a library or wrapped by the MCP server.
///
/// # Example
///
/// ```no_run
/// use project_rag::{RagClient, IndexRequest, QueryRequest};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     // Create client with default configuration
///     let client = RagClient::new().await?;
///
///     // Index a codebase
///     let index_req = IndexRequest {
///         path: "/path/to/code".to_string(),
///         project: Some("my-project".to_string()),
///         include_patterns: vec!["**/*.rs".to_string()],
///         exclude_patterns: vec!["**/target/**".to_string()],
///         max_file_size: 1_048_576,
///     };
///     let response = client.index_codebase(index_req).await?;
///     println!("Indexed {} files", response.files_indexed);
///
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct RagClient {
    pub(crate) embedding_provider: Arc<FastEmbedManager>,
    #[cfg(feature = "qdrant-backend")]
    pub(crate) vector_db: Arc<QdrantVectorDB>,
    #[cfg(not(feature = "qdrant-backend"))]
    pub(crate) vector_db: Arc<LanceVectorDB>,
    pub(crate) chunker: Arc<CodeChunker>,
    // Persistent hash cache for incremental updates
    pub(crate) hash_cache: Arc<RwLock<HashCache>>,
    pub(crate) cache_path: PathBuf,
    // Git cache for git history indexing
    pub(crate) git_cache: Arc<RwLock<GitCache>>,
    pub(crate) git_cache_path: PathBuf,
    // Configuration (for accessing batch sizes, timeouts, etc.)
    pub(crate) config: Arc<Config>,
    // In-progress indexing operations (prevents concurrent indexing and allows result sharing)
    pub(crate) indexing_ops: Arc<RwLock<HashMap<String, IndexingOperation>>>,
    // Relations provider for code navigation (find definition, references, call graph)
    pub(crate) relations_provider: Arc<HybridRelationsProvider>,
}

impl RagClient {
    /// Create a new RAG client with default configuration
    ///
    /// This will initialize the embedding model, vector database, and load
    /// any existing caches from disk.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Configuration cannot be loaded
    /// - Embedding model cannot be initialized
    /// - Vector database cannot be initialized
    pub async fn new() -> Result<Self> {
        let config = Config::new().context("Failed to load configuration")?;
        Self::with_config(config).await
    }

    /// Create a new RAG client with custom configuration
    ///
    /// # Example
    ///
    /// ```no_run
    /// use project_rag::{RagClient, Config};
    ///
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    ///     let mut config = Config::default();
    ///     config.embedding.model_name = "BAAI/bge-small-en-v1.5".to_string();
    ///
    ///     let client = RagClient::with_config(config).await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn with_config(config: Config) -> Result<Self> {
        tracing::info!("Initializing RAG client with configuration");
        tracing::debug!("Vector DB backend: {}", config.vector_db.backend);
        tracing::debug!("Embedding model: {}", config.embedding.model_name);
        tracing::debug!("Chunk size: {}", config.indexing.chunk_size);

        // Initialize embedding provider with configured model
        let embedding_provider = Arc::new(
            FastEmbedManager::from_model_name(&config.embedding.model_name)
                .context("Failed to initialize embedding provider")?,
        );

        // Initialize the appropriate vector database backend
        #[cfg(feature = "qdrant-backend")]
        let vector_db = {
            tracing::info!(
                "Using Qdrant vector database backend at {}",
                config.vector_db.qdrant_url
            );
            Arc::new(
                QdrantVectorDB::with_url(&config.vector_db.qdrant_url)
                    .await
                    .context("Failed to initialize Qdrant vector database")?,
            )
        };

        #[cfg(not(feature = "qdrant-backend"))]
        let vector_db = {
            tracing::info!(
                "Using LanceDB vector database backend at {}",
                config.vector_db.lancedb_path.display()
            );
            Arc::new(
                LanceVectorDB::with_path(&config.vector_db.lancedb_path.to_string_lossy())
                    .await
                    .context("Failed to initialize LanceDB vector database")?,
            )
        };

        // Initialize the database with the embedding dimension
        vector_db
            .initialize(embedding_provider.dimension())
            .await
            .context("Failed to initialize vector database collections")?;

        // Create chunker with configured chunk size
        let chunker = Arc::new(CodeChunker::default_strategy());

        // Load persistent hash cache
        let cache_path = config.cache.hash_cache_path.clone();
        let hash_cache = HashCache::load(&cache_path).unwrap_or_else(|e| {
            tracing::warn!("Failed to load hash cache: {}, starting fresh", e);
            HashCache::default()
        });

        tracing::info!("Using hash cache file: {:?}", cache_path);

        // Load persistent git cache
        let git_cache_path = config.cache.git_cache_path.clone();
        let git_cache = GitCache::load(&git_cache_path).unwrap_or_else(|e| {
            tracing::warn!("Failed to load git cache: {}, starting fresh", e);
            GitCache::default()
        });

        tracing::info!("Using git cache file: {:?}", git_cache_path);

        // Initialize relations provider for code navigation
        let relations_provider = Arc::new(
            HybridRelationsProvider::new(false) // stack-graphs disabled by default
                .context("Failed to initialize relations provider")?,
        );

        Ok(Self {
            embedding_provider,
            vector_db,
            chunker,
            hash_cache: Arc::new(RwLock::new(hash_cache)),
            cache_path,
            git_cache: Arc::new(RwLock::new(git_cache)),
            git_cache_path,
            config: Arc::new(config),
            indexing_ops: Arc::new(RwLock::new(HashMap::new())),
            relations_provider,
        })
    }

    /// Create a new client with custom database path (for testing)
    #[cfg(test)]
    pub async fn new_with_db_path(db_path: &str, cache_path: PathBuf) -> Result<Self> {
        // Create a test config with custom paths
        let mut config = Config::default();
        config.vector_db.lancedb_path = PathBuf::from(db_path);
        config.cache.hash_cache_path = cache_path.clone();
        config.cache.git_cache_path = cache_path.parent().unwrap().join("git_cache.json");

        Self::with_config(config).await
    }

    /// Create FileInfo from a file path for relations analysis
    fn create_file_info(&self, file_path: &str, project: Option<String>) -> Result<FileInfo> {
        use std::path::Path;

        let path = Path::new(file_path);
        let canonical = std::fs::canonicalize(path)
            .with_context(|| format!("Failed to canonicalize path: {}", file_path))?;

        let content = std::fs::read_to_string(&canonical)
            .with_context(|| format!("Failed to read file: {}", file_path))?;

        let extension = canonical
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_string());

        let language = extension.as_ref().and_then(|ext| {
            detect_language(ext)
        });

        // Compute file hash
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let hash = format!("{:x}", hasher.finalize());

        // Determine root path (parent directory)
        let root_path = canonical
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "/".to_string());

        let relative_path = canonical
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| file_path.to_string());

        Ok(FileInfo {
            path: canonical,
            relative_path,
            root_path,
            project,
            extension,
            language,
            content,
            hash,
        })
    }

    /// Normalize a path to a canonical absolute form for consistent cache lookups
    pub fn normalize_path(path: &str) -> Result<String> {
        let path_buf = PathBuf::from(path);
        let canonical = std::fs::canonicalize(&path_buf)
            .with_context(|| format!("Failed to canonicalize path: {}", path))?;
        Ok(canonical.to_string_lossy().to_string())
    }

    /// Check if a specific path's index is dirty (incomplete/corrupted)
    ///
    /// Returns true if the path is marked as dirty, meaning a previous indexing
    /// operation was interrupted and the data may be inconsistent.
    pub async fn is_index_dirty(&self, path: &str) -> bool {
        if let Ok(normalized) = Self::normalize_path(path) {
            let cache = self.hash_cache.read().await;
            cache.is_dirty(&normalized)
        } else {
            false
        }
    }

    /// Check if any indexed paths are dirty
    ///
    /// Returns a list of paths that have dirty indexes.
    pub async fn get_dirty_paths(&self) -> Vec<String> {
        let cache = self.hash_cache.read().await;
        cache.get_dirty_roots().keys().cloned().collect()
    }

    /// Check if searching on a specific path should be blocked due to dirty state
    ///
    /// Returns an error if the path is dirty, otherwise Ok(())
    async fn check_path_not_dirty(&self, path: Option<&str>) -> Result<()> {
        if let Some(p) = path {
            if self.is_index_dirty(p).await {
                anyhow::bail!(
                    "Index for '{}' is dirty (previous indexing was interrupted). \
                    Please re-run index_codebase to rebuild the index before querying.",
                    p
                );
            }
        }
        Ok(())
    }

    /// Try to acquire an indexing lock for a given path
    ///
    /// This uses a two-layer locking strategy:
    /// 1. Filesystem lock (flock) for cross-process coordination
    /// 2. In-memory lock for broadcasting results to waiters in the same process
    ///
    /// Returns either:
    /// - `IndexLockResult::Acquired(guard)` if we should perform the indexing
    /// - `IndexLockResult::WaitForResult(receiver)` if another task in THIS process is indexing
    /// - `IndexLockResult::WaitForFilesystemLock(path)` if ANOTHER PROCESS is indexing
    ///
    /// The lock is automatically released when the returned guard is dropped.
    pub(crate) async fn try_acquire_index_lock(&self, path: &str) -> Result<IndexLockResult> {
        use std::sync::atomic::Ordering;
        use std::time::Instant;

        // Normalize the path to ensure consistent locking across different path formats
        let normalized_path = Self::normalize_path(path)?;

        // STEP 1: Try to acquire filesystem lock first (cross-process coordination)
        // This must happen BEFORE checking in-memory state to prevent race conditions
        let fs_lock = {
            let path_clone = normalized_path.clone();
            tokio::task::spawn_blocking(move || FsLockGuard::try_acquire(&path_clone))
                .await
                .context("Filesystem lock task panicked")??
        };

        // If we couldn't get the filesystem lock, another PROCESS is indexing
        let fs_lock = match fs_lock {
            Some(lock) => lock,
            None => {
                tracing::info!(
                    "Another process is indexing {} - returning WaitForFilesystemLock",
                    normalized_path
                );
                return Ok(IndexLockResult::WaitForFilesystemLock(normalized_path));
            }
        };

        // STEP 2: We have the filesystem lock, now check in-memory state
        // This handles the case where another task in THIS process is indexing

        // Acquire write lock on the ops map
        let mut ops = self.indexing_ops.write().await;

        // Check if an operation is already in progress for this path (in this process)
        if let Some(existing_op) = ops.get(&normalized_path) {
            // Check if the operation is stale (timed out or crashed)
            if existing_op.is_stale() {
                tracing::warn!(
                    "Removing stale indexing lock for {} (operation timed out after {:?})",
                    normalized_path,
                    existing_op.started_at.elapsed()
                );
                ops.remove(&normalized_path);
            } else if existing_op.active.load(Ordering::Acquire) {
                // Operation is still active and not stale, subscribe to receive the result
                // Note: We drop the filesystem lock here since we won't be indexing
                drop(fs_lock);
                let receiver = existing_op.result_tx.subscribe();
                tracing::info!(
                    "Indexing already in progress in this process for {} (started {:?} ago), waiting for result",
                    normalized_path,
                    existing_op.started_at.elapsed()
                );
                return Ok(IndexLockResult::WaitForResult(receiver));
            } else {
                // Operation completed but cleanup hasn't happened yet
                tracing::debug!(
                    "Removing completed indexing lock for {} (cleanup pending)",
                    normalized_path
                );
                ops.remove(&normalized_path);
            }
        }

        // STEP 3: We have both locks, register the operation

        // Create a new broadcast channel for this operation
        // Capacity of 1 is enough since we only send one result
        let (result_tx, _) = broadcast::channel(1);

        // Create the active flag - starts as true (active)
        let active_flag = Arc::new(std::sync::atomic::AtomicBool::new(true));

        // Register this operation with timestamp
        ops.insert(
            normalized_path.clone(),
            IndexingOperation {
                result_tx: result_tx.clone(),
                active: active_flag.clone(),
                started_at: Instant::now(),
            },
        );

        // Drop the write lock on the map
        drop(ops);

        Ok(IndexLockResult::Acquired(IndexLockGuard::new(
            normalized_path,
            self.indexing_ops.clone(),
            result_tx,
            active_flag,
            fs_lock,
        )))
    }

    /// Index a codebase directory
    ///
    /// This automatically performs full indexing for new codebases or incremental
    /// updates for previously indexed codebases.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use project_rag::{RagClient, IndexRequest};
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let client = RagClient::new().await?;
    ///
    /// let request = IndexRequest {
    ///     path: "/path/to/code".to_string(),
    ///     project: Some("my-project".to_string()),
    ///     include_patterns: vec!["**/*.rs".to_string()],
    ///     exclude_patterns: vec!["**/target/**".to_string()],
    ///     max_file_size: 1_048_576,
    /// };
    ///
    /// let response = client.index_codebase(request).await?;
    /// println!("Indexed {} files in {} ms",
    ///          response.files_indexed,
    ///          response.duration_ms);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn index_codebase(&self, request: IndexRequest) -> Result<IndexResponse> {
        // Validate request
        request.validate().map_err(|e| anyhow::anyhow!(e))?;

        // Use the smart indexing logic without progress notifications
        // Default cancellation token - not cancellable from this API
        let cancel_token = tokio_util::sync::CancellationToken::new();
        indexing::do_index_smart(
            self,
            request.path,
            request.project,
            request.include_patterns,
            request.exclude_patterns,
            request.max_file_size,
            None, // No peer
            None, // No progress token
            cancel_token,
        )
        .await
    }

    /// Query the indexed codebase using semantic search
    ///
    /// # Example
    ///
    /// ```no_run
    /// use project_rag::{RagClient, QueryRequest};
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let client = RagClient::new().await?;
    ///
    /// let request = QueryRequest {
    ///     query: "authentication logic".to_string(),
    ///     project: Some("my-project".to_string()),
    ///     limit: 10,
    ///     min_score: 0.7,
    ///     hybrid: true,
    /// };
    ///
    /// let response = client.query_codebase(request).await?;
    /// for result in response.results {
    ///     println!("Found in {}: {:.2}", result.file_path, result.score);
    ///     println!("{}", result.content);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn query_codebase(&self, request: QueryRequest) -> Result<QueryResponse> {
        request.validate().map_err(|e| anyhow::anyhow!(e))?;

        // Check if the target path is dirty (if path filter is specified)
        self.check_path_not_dirty(request.path.as_deref()).await?;

        let start = Instant::now();

        let query_embedding = self
            .embedding_provider
            .embed_batch(vec![request.query.clone()])
            .context("Failed to generate query embedding")?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding generated"))?;

        let original_threshold = request.min_score;
        let mut threshold_used = original_threshold;
        let mut threshold_lowered = false;

        let mut results = self
            .vector_db
            .search(
                query_embedding.clone(),
                &request.query,
                request.limit,
                threshold_used,
                request.project.clone(),
                request.path.clone(),
                request.hybrid,
            )
            .await
            .context("Failed to search")?;

        if results.is_empty() && original_threshold > 0.3 {
            let fallback_thresholds = [0.6, 0.5, 0.4, 0.3];

            for &threshold in &fallback_thresholds {
                if threshold >= original_threshold {
                    continue;
                }

                results = self
                    .vector_db
                    .search(
                        query_embedding.clone(),
                        &request.query,
                        request.limit,
                        threshold,
                        request.project.clone(),
                        request.path.clone(),
                        request.hybrid,
                    )
                    .await
                    .context("Failed to search")?;

                if !results.is_empty() {
                    threshold_used = threshold;
                    threshold_lowered = true;
                    break;
                }
            }
        }

        Ok(QueryResponse {
            results,
            duration_ms: start.elapsed().as_millis() as u64,
            threshold_used,
            threshold_lowered,
        })
    }

    /// Advanced search with filters for file type, language, and path patterns
    pub async fn search_with_filters(
        &self,
        request: AdvancedSearchRequest,
    ) -> Result<QueryResponse> {
        request.validate().map_err(|e| anyhow::anyhow!(e))?;

        // Check if the target path is dirty (if path filter is specified)
        self.check_path_not_dirty(request.path.as_deref()).await?;

        let start = Instant::now();

        let query_embedding = self
            .embedding_provider
            .embed_batch(vec![request.query.clone()])
            .context("Failed to generate query embedding")?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding generated"))?;

        let original_threshold = request.min_score;
        let mut threshold_used = original_threshold;
        let mut threshold_lowered = false;

        let mut results = self
            .vector_db
            .search_filtered(
                query_embedding.clone(),
                &request.query,
                request.limit,
                threshold_used,
                request.project.clone(),
                request.path.clone(),
                true,
                request.file_extensions.clone(),
                request.languages.clone(),
                request.path_patterns.clone(),
            )
            .await
            .context("Failed to search with filters")?;

        // Adaptive threshold lowering if no results found
        if results.is_empty() && original_threshold > 0.3 {
            let fallback_thresholds = [0.6, 0.5, 0.4, 0.3];

            for &threshold in &fallback_thresholds {
                if threshold >= original_threshold {
                    continue;
                }

                results = self
                    .vector_db
                    .search_filtered(
                        query_embedding.clone(),
                        &request.query,
                        request.limit,
                        threshold,
                        request.project.clone(),
                        request.path.clone(),
                        true,
                        request.file_extensions.clone(),
                        request.languages.clone(),
                        request.path_patterns.clone(),
                    )
                    .await
                    .context("Failed to search with filters")?;

                if !results.is_empty() {
                    threshold_used = threshold;
                    threshold_lowered = true;
                    break;
                }
            }
        }

        Ok(QueryResponse {
            results,
            duration_ms: start.elapsed().as_millis() as u64,
            threshold_used,
            threshold_lowered,
        })
    }

    /// Get statistics about the indexed codebase
    pub async fn get_statistics(&self) -> Result<StatisticsResponse> {
        let stats = self
            .vector_db
            .get_statistics()
            .await
            .context("Failed to get statistics")?;

        let language_breakdown = stats
            .language_breakdown
            .into_iter()
            .map(|entry| LanguageStats {
                language: entry.language,
                file_count: entry.file_count,
                chunk_count: entry.chunk_count,
            })
            .collect();

        Ok(StatisticsResponse {
            total_files: stats.total_files,
            total_chunks: stats.total_points,
            total_embeddings: stats.total_vectors,
            database_size_bytes: stats.database_size_bytes,
            language_breakdown,
        })
    }

    /// Clear all indexed data from the vector database and hash cache
    pub async fn clear_index(&self) -> Result<ClearResponse> {
        match self.vector_db.clear().await {
            Ok(_) => {
                // Clear hash cache (both roots and dirty_roots)
                let mut cache = self.hash_cache.write().await;
                cache.roots.clear();
                cache.dirty_roots.clear();

                // Delete cache file directly for robustness (in case save fails)
                if self.cache_path.exists() {
                    if let Err(e) = std::fs::remove_file(&self.cache_path) {
                        tracing::warn!("Failed to delete hash cache file: {}", e);
                    } else {
                        tracing::info!("Deleted hash cache file: {:?}", self.cache_path);
                    }
                }

                // Save empty cache (recreates the file with empty state)
                if let Err(e) = cache.save(&self.cache_path) {
                    tracing::warn!("Failed to save cleared cache: {}", e);
                }

                // Also clear git cache
                let mut git_cache = self.git_cache.write().await;
                git_cache.repos.clear();
                if self.git_cache_path.exists() {
                    if let Err(e) = std::fs::remove_file(&self.git_cache_path) {
                        tracing::warn!("Failed to delete git cache file: {}", e);
                    } else {
                        tracing::info!("Deleted git cache file: {:?}", self.git_cache_path);
                    }
                }
                if let Err(e) = git_cache.save(&self.git_cache_path) {
                    tracing::warn!("Failed to save cleared git cache: {}", e);
                }

                if let Err(e) = self
                    .vector_db
                    .initialize(self.embedding_provider.dimension())
                    .await
                {
                    Ok(ClearResponse {
                        success: false,
                        message: format!("Cleared but failed to reinitialize: {}", e),
                    })
                } else {
                    Ok(ClearResponse {
                        success: true,
                        message: "Successfully cleared all indexed data and cache".to_string(),
                    })
                }
            }
            Err(e) => Ok(ClearResponse {
                success: false,
                message: format!("Failed to clear index: {}", e),
            }),
        }
    }

    /// Search git commit history using semantic search
    ///
    /// # Example
    ///
    /// ```no_run
    /// use project_rag::{RagClient, SearchGitHistoryRequest};
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let client = RagClient::new().await?;
    ///
    /// let request = SearchGitHistoryRequest {
    ///     query: "bug fix authentication".to_string(),
    ///     path: "/path/to/repo".to_string(),
    ///     project: None,
    ///     branch: None,
    ///     max_commits: 100,
    ///     limit: 10,
    ///     min_score: 0.7,
    ///     author: None,
    ///     since: None,
    ///     until: None,
    ///     file_pattern: None,
    /// };
    ///
    /// let response = client.search_git_history(request).await?;
    /// for result in response.results {
    ///     println!("Commit {}: {}", result.commit_hash, result.commit_message);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_git_history(
        &self,
        request: SearchGitHistoryRequest,
    ) -> Result<SearchGitHistoryResponse> {
        // Validate request
        request.validate().map_err(|e| anyhow::anyhow!(e))?;

        // Forward to git indexing implementation
        git_indexing::do_search_git_history(
            self.embedding_provider.clone(),
            self.vector_db.clone(),
            self.git_cache.clone(),
            &self.git_cache_path,
            request,
        )
        .await
    }

    /// Get the configuration used by this client
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Get the embedding dimension used by this client
    pub fn embedding_dimension(&self) -> usize {
        self.embedding_provider.dimension()
    }

    /// Find the definition of a symbol at a given file location
    ///
    /// This method looks up the symbol at the specified location and returns
    /// its definition information if found.
    ///
    /// # Arguments
    ///
    /// * `request` - The find definition request containing file path, line, and column
    ///
    /// # Returns
    ///
    /// A response containing the definition if found, along with precision info
    /// The identifier token sitting at a 1-based `line` / 0-based `column`.
    ///
    /// Symbol resolution used to work purely by range containment: take the first
    /// definition whose start..end spans the line. On a call site that is always the
    /// ENCLOSING function, so asking about `SendNotification2(...)` inside
    /// `ProcessDeviceNotifyCache` resolved to `ProcessDeviceNotifyCache`. Reading the
    /// actual token under the cursor is what the caller meant by "the symbol here".
    fn identifier_at(content: &str, line: usize, column: usize) -> Option<String> {
        let text = content.lines().nth(line.checked_sub(1)?)?;
        let bytes = text.as_bytes();
        let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';

        if bytes.is_empty() {
            return None;
        }
        // Clamp into the line; a column past the end just anchors at the last character.
        let mut idx = column.min(bytes.len() - 1);
        // A cursor resting just after a token (or on its opening delimiter) should still
        // resolve that token.
        if !is_ident(bytes[idx]) && idx > 0 && is_ident(bytes[idx - 1]) {
            idx -= 1;
        }
        if !is_ident(bytes[idx]) {
            return None;
        }

        let mut start = idx;
        while start > 0 && is_ident(bytes[start - 1]) {
            start -= 1;
        }
        let mut end = idx;
        while end + 1 < bytes.len() && is_ident(bytes[end + 1]) {
            end += 1;
        }
        Some(text[start..=end].to_string())
    }

    /// Resolve which definition a cursor position refers to.
    ///
    /// Preference order:
    ///   1. the identifier under the cursor, if it names a definition in this file
    ///   2. the INNERMOST definition whose range contains the line
    ///
    /// Step 2 used to be "the first definition that contains the line", which picked
    /// whichever happened to be earliest in extraction order -- normally the enclosing
    /// class or function rather than the nested one being asked about.
    fn resolve_symbol_at<'a>(
        definitions: &'a [crate::relations::Definition],
        content: &str,
        line: usize,
        column: usize,
        callable_only: bool,
    ) -> Option<&'a crate::relations::Definition> {
        let is_candidate = |def: &crate::relations::Definition| {
            !callable_only
                || matches!(
                    def.symbol_id.kind,
                    crate::relations::SymbolKind::Function | crate::relations::SymbolKind::Method
                )
        };

        if let Some(name) = Self::identifier_at(content, line, column) {
            let exact = definitions
                .iter()
                .filter(|d| d.symbol_id.name == name && is_candidate(d))
                .min_by_key(|d| d.end_line.saturating_sub(d.symbol_id.start_line));
            if exact.is_some() {
                return exact;
            }
        }

        definitions
            .iter()
            .filter(|d| {
                is_candidate(d) && line >= d.symbol_id.start_line && line <= d.end_line
            })
            .min_by_key(|d| d.end_line.saturating_sub(d.symbol_id.start_line))
    }

    /// Files that plausibly mention `symbol`, newest-ranked first.
    ///
    /// References live wherever the identifier appears, which is generally NOT the file
    /// that defines it -- the previous implementation only ever scanned the definition's
    /// own file, so any cross-file reference was invisible. Rather than parse the whole
    /// corpus, shortlist with keyword search: a reference must contain the literal token,
    /// so BM25 surfaces exactly the right files and tree-sitter only runs on those.
    async fn files_mentioning(
        &self,
        symbol: &str,
        project: Option<String>,
        limit: usize,
    ) -> Result<Vec<std::path::PathBuf>> {
        let embedding = self
            .embedding_provider
            .embed_batch(vec![symbol.to_string()])
            .context("Failed to embed symbol name")?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding generated for symbol"))?;

        let results = self
            .vector_db
            .search(embedding, symbol, limit, 0.0, project, None, true)
            .await
            .context("Failed to search for candidate files")?;

        let mut seen = std::collections::HashSet::new();
        let mut files = Vec::new();
        for r in results {
            let full = match &r.root_path {
                Some(root) => std::path::Path::new(root).join(&r.file_path),
                None => std::path::PathBuf::from(&r.file_path),
            };
            if seen.insert(full.clone()) {
                files.push(full);
            }
        }
        Ok(files)
    }

    pub async fn find_definition(&self, request: FindDefinitionRequest) -> Result<FindDefinitionResponse> {
        let start = Instant::now();

        // Validate request
        request.validate().map_err(|e| anyhow::anyhow!(e))?;

        // Create FileInfo for the file
        let file_info = self.create_file_info(&request.file_path, request.project.clone())?;

        // Get precision level for this language
        let language = file_info.language.as_deref().unwrap_or("Unknown");
        let precision = self.relations_provider.precision_level(language);

        // Extract definitions from the file
        let definitions = self
            .relations_provider
            .extract_definitions(&file_info)
            .context("Failed to extract definitions")?;

        // Find the definition at the requested position
        let result = Self::resolve_symbol_at(
            &definitions,
            &file_info.content,
            request.line,
            request.column,
            false,
        )
        .map(DefinitionResult::from);

        Ok(FindDefinitionResponse {
            definition: result,
            precision: format!("{:?}", precision).to_lowercase(),
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Find all references to a symbol at a given file location
    ///
    /// This method finds all locations where the symbol at the given position
    /// is referenced throughout the indexed codebase.
    ///
    /// # Arguments
    ///
    /// * `request` - The find references request containing file path, line, column, and limit
    ///
    /// # Returns
    ///
    /// A response containing the list of references found
    pub async fn find_references(&self, request: FindReferencesRequest) -> Result<FindReferencesResponse> {
        let start = Instant::now();

        // Validate request
        request.validate().map_err(|e| anyhow::anyhow!(e))?;

        // Create FileInfo for the file
        let file_info = self.create_file_info(&request.file_path, request.project.clone())?;

        // Get precision level for this language
        let language = file_info.language.as_deref().unwrap_or("Unknown");
        let precision = self.relations_provider.precision_level(language);

        // Extract definitions from the file to find the symbol at the position
        let definitions = self
            .relations_provider
            .extract_definitions(&file_info)
            .context("Failed to extract definitions")?;

        // Find the symbol at the requested position
        let target_symbol = Self::resolve_symbol_at(
            &definitions,
            &file_info.content,
            request.line,
            request.column,
            false,
        );

        let symbol_name = target_symbol.map(|def| def.symbol_id.name.clone());

        // If no symbol found at position, return empty result
        if symbol_name.is_none() {
            return Ok(FindReferencesResponse {
                symbol_name: None,
                references: Vec::new(),
                total_count: 0,
                precision: format!("{:?}", precision).to_lowercase(),
                duration_ms: start.elapsed().as_millis() as u64,
            });
        }

        let symbol_name_str = symbol_name.clone().unwrap();

        // Index ONLY the target symbol. ReferenceFinder matches identifiers against this
        // map, so restricting it keeps the scan of other files cheap and on-topic.
        let target_defs: Vec<crate::relations::Definition> = definitions
            .iter()
            .filter(|d| d.symbol_id.name == symbol_name_str)
            .cloned()
            .collect();
        let mut symbol_index: std::collections::HashMap<String, Vec<crate::relations::Definition>> =
            std::collections::HashMap::new();
        if !target_defs.is_empty() {
            symbol_index.insert(symbol_name_str.clone(), target_defs);
        }

        // Scan the defining file plus every other file the index says mentions the symbol.
        // Searching only the defining file is why this returned nothing for anything called
        // from elsewhere, which is the normal case for a public API.
        let mut scan_targets: Vec<std::path::PathBuf> = vec![file_info.path.clone()];
        match self
            .files_mentioning(&symbol_name_str, request.project.clone(), request.limit.max(20))
            .await
        {
            Ok(found) => {
                for f in found {
                    if !scan_targets.iter().any(|p| p == &f) {
                        scan_targets.push(f);
                    }
                }
            }
            Err(e) => tracing::warn!("Candidate lookup failed, scanning defining file only: {}", e),
        }

        let mut matching_refs: Vec<ReferenceResult> = Vec::new();
        for target in &scan_targets {
            if matching_refs.len() >= request.limit {
                break;
            }
            let scan_info = if target == &file_info.path {
                file_info.clone()
            } else {
                match self.create_file_info(&target.to_string_lossy(), request.project.clone()) {
                    Ok(fi) => fi,
                    Err(e) => {
                        tracing::debug!("Skipping unreadable candidate {:?}: {}", target, e);
                        continue;
                    }
                }
            };

            let references = match self
                .relations_provider
                .extract_references(&scan_info, &symbol_index)
            {
                Ok(r) => r,
                Err(e) => {
                    tracing::debug!("Reference extraction failed for {:?}: {}", target, e);
                    continue;
                }
            };

            for r in references.iter() {
                if matching_refs.len() >= request.limit {
                    break;
                }
                if r.target_symbol_id.contains(&symbol_name_str) {
                    matching_refs.push(ReferenceResult::from(r));
                }
            }
        }

        let total_count = matching_refs.len();

        Ok(FindReferencesResponse {
            symbol_name,
            references: matching_refs,
            total_count,
            precision: format!("{:?}", precision).to_lowercase(),
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Control-flow and cast keywords that are followed by a parenthesis but are not
    /// calls. Without this, `if (` is reported as a callee whenever some file happens
    /// to carry a bogus definition of that name.
    fn is_call_like_keyword(name: &str) -> bool {
        matches!(
            name,
            "if" | "for" | "while" | "switch" | "catch" | "return" | "sizeof"
                | "do" | "else" | "new" | "delete" | "throw" | "defined"
                | "static_cast" | "dynamic_cast" | "reinterpret_cast" | "const_cast"
        )
    }
    /// Identifiers that appear immediately before an opening parenthesis inside the
    /// given 1-based line span -- that is, plausible call sites.
    ///
    /// Used to widen the callee symbol index beyond the defining file. Deliberately
    /// crude: over-reporting costs one extra lookup, under-reporting loses a callee.
    fn call_identifiers_in_span(content: &str, start_line: usize, end_line: usize) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (idx, line) in content.lines().enumerate() {
            let n = idx + 1;
            if n < start_line || n > end_line {
                continue;
            }
            let bytes = line.as_bytes();
            let mut i = 0usize;
            while i < bytes.len() {
                if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
                    let start = i;
                    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                        i += 1;
                    }
                    let mut j = i;
                    while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                        j += 1;
                    }
                    if j < bytes.len() && bytes[j] == b'(' {
                        let name = &line[start..i];
                        if name.len() > 1
                            && !Self::is_call_like_keyword(name)
                            && seen.insert(name.to_string())
                        {
                            out.push(name.to_string());
                        }
                    }
                } else {
                    i += 1;
                }
            }
        }
        out
    }
    /// Get the call graph for a function at a given file location
    ///
    /// This method returns the callers (incoming calls) and callees (outgoing calls)
    /// for the function at the specified location.
    ///
    /// # Arguments
    ///
    /// * `request` - The call graph request containing file path, line, column, and depth
    ///
    /// # Returns
    ///
    /// A response containing the root symbol and its call graph
    pub async fn get_call_graph(&self, request: GetCallGraphRequest) -> Result<GetCallGraphResponse> {
        let start = Instant::now();

        // Validate request
        request.validate().map_err(|e| anyhow::anyhow!(e))?;

        // Create FileInfo for the file
        let file_info = self.create_file_info(&request.file_path, request.project.clone())?;

        // Get precision level for this language
        let language = file_info.language.as_deref().unwrap_or("Unknown");
        let precision = self.relations_provider.precision_level(language);

        // Extract definitions from the file to find the function at the position
        let definitions = self
            .relations_provider
            .extract_definitions(&file_info)
            .context("Failed to extract definitions")?;

        // Find the function at the requested position
        let target_function = Self::resolve_symbol_at(
            &definitions,
            &file_info.content,
            request.line,
            request.column,
            true,
        );

        // If no function found at position, return empty result
        let root_symbol = match target_function {
            Some(func) => crate::relations::SymbolInfo {
                name: func.symbol_id.name.clone(),
                kind: func.symbol_id.kind.clone(),
                file_path: request.file_path.clone(),
                start_line: func.symbol_id.start_line,
                end_line: func.end_line,
                signature: func.signature.clone(),
            },
            None => {
                return Ok(GetCallGraphResponse {
                    root_symbol: None,
                    callers: Vec::new(),
                    callees: Vec::new(),
                    precision: format!("{:?}", precision).to_lowercase(),
                    duration_ms: start.elapsed().as_millis() as u64,
                });
            }
        };

        let function_name = root_symbol.name.clone();

        // Build symbol index from definitions
        let mut symbol_index: std::collections::HashMap<String, Vec<crate::relations::Definition>> =
            std::collections::HashMap::new();
        for def in &definitions {
            symbol_index
                .entry(def.symbol_id.name.clone())
                .or_default()
                .push(def.clone());
        }

        // References in this file, used for the callee side.
        //
        // The index handed to extract_references decides what is even visible: a
        // reference is emitted only when its identifier is a key in that map. Building
        // it from this file alone therefore drops every callee defined in another
        // translation unit, and TForm1 alone is spread over five .cpp files. So widen
        // it first with definitions of the names actually called inside the target
        // span. Bounded on both axes so a large function cannot fan out forever.
        const MAX_CALLEE_PROBES: usize = 40;
        const FILES_PER_NAME: usize = 5;

        let mut callee_index = symbol_index.clone();
        let mut probed_files: std::collections::HashSet<std::path::PathBuf> =
            std::collections::HashSet::from([file_info.path.clone()]);

        let called_names = Self::call_identifiers_in_span(
            &file_info.content,
            root_symbol.start_line,
            root_symbol.end_line,
        );

        for name in called_names
            .iter()
            .filter(|n| !symbol_index.contains_key(*n))
            .take(MAX_CALLEE_PROBES)
        {
            let candidates = match self
                .files_mentioning(name, request.project.clone(), FILES_PER_NAME)
                .await
            {
                Ok(f) => f,
                Err(e) => {
                    tracing::debug!("Callee candidate lookup failed for {}: {}", name, e);
                    continue;
                }
            };
            for f in candidates {
                if !probed_files.insert(f.clone()) {
                    continue;
                }
                let fi = match self.create_file_info(&f.to_string_lossy(), request.project.clone()) {
                    Ok(fi) => fi,
                    Err(e) => {
                        tracing::debug!("Skipping unreadable callee candidate {:?}: {}", f, e);
                        continue;
                    }
                };
                match self.relations_provider.extract_definitions(&fi) {
                    Ok(defs) => {
                        for d in defs {
                            callee_index.entry(d.symbol_id.name.clone()).or_default().push(d);
                        }
                    }
                    Err(e) => tracing::debug!("Definition extraction failed for {:?}: {}", f, e),
                }
            }
        }

        let references = self
            .relations_provider
            .extract_references(&file_info, &callee_index)
            .context("Failed to extract references")?;

        // Callers can live anywhere, so scan the defining file plus every file the index
        // says mentions the function. Restricting this to the defining file is why the
        // caller list came back empty for anything with an external call site.
        let caller_index: std::collections::HashMap<String, Vec<crate::relations::Definition>> =
            std::collections::HashMap::from([(
                function_name.clone(),
                definitions
                    .iter()
                    .filter(|d| d.symbol_id.name == function_name)
                    .cloned()
                    .collect(),
            )]);

        let mut scan_targets: Vec<std::path::PathBuf> = vec![file_info.path.clone()];
        match self
            .files_mentioning(&function_name, request.project.clone(), 20)
            .await
        {
            Ok(found) => {
                for f in found {
                    if !scan_targets.iter().any(|p| p == &f) {
                        scan_targets.push(f);
                    }
                }
            }
            Err(e) => tracing::warn!("Candidate lookup failed, scanning defining file only: {}", e),
        }

        let mut seen_callers = std::collections::HashSet::new();
        let mut callers: Vec<crate::relations::CallGraphNode> = Vec::new();

        for target in &scan_targets {
            let (scan_info, scan_defs) = if target == &file_info.path {
                (file_info.clone(), definitions.clone())
            } else {
                match self.create_file_info(&target.to_string_lossy(), request.project.clone()) {
                    Ok(fi) => {
                        let defs = self
                            .relations_provider
                            .extract_definitions(&fi)
                            .unwrap_or_default();
                        (fi, defs)
                    }
                    Err(e) => {
                        tracing::debug!("Skipping unreadable candidate {:?}: {}", target, e);
                        continue;
                    }
                }
            };

            let refs = match self
                .relations_provider
                .extract_references(&scan_info, &caller_index)
            {
                Ok(r) => r,
                Err(e) => {
                    tracing::debug!("Reference extraction failed for {:?}: {}", target, e);
                    continue;
                }
            };

            for r in refs.iter().filter(|r| {
                r.reference_kind == crate::relations::ReferenceKind::Call
                    && r.target_symbol_id.contains(&function_name)
            }) {
                // Attribute the call to the innermost function containing it, in the file
                // the call was actually found in.
                let enclosing = scan_defs
                    .iter()
                    .filter(|def| {
                        matches!(
                            def.symbol_id.kind,
                            crate::relations::SymbolKind::Function
                                | crate::relations::SymbolKind::Method
                        ) && r.start_line >= def.symbol_id.start_line
                            && r.start_line <= def.end_line
                    })
                    .min_by_key(|def| def.end_line.saturating_sub(def.symbol_id.start_line));

                if let Some(def) = enclosing
                    && seen_callers.insert((
                        scan_info.relative_path.clone(),
                        def.symbol_id.name.clone(),
                    ))
                {
                    callers.push(crate::relations::CallGraphNode {
                        name: def.symbol_id.name.clone(),
                        kind: def.symbol_id.kind.clone(),
                        file_path: scan_info.relative_path.clone(),
                        line: def.symbol_id.start_line,
                        children: Vec::new(),
                    });
                }
            }
        }

        // Find callees (calls made from within our function)
        let target_func = target_function.unwrap();
        let mut seen_callees = std::collections::HashSet::new();
        let callees: Vec<crate::relations::CallGraphNode> = references
            .iter()
            .filter(|r| {
                r.reference_kind == crate::relations::ReferenceKind::Call
                    && r.start_line >= target_func.symbol_id.start_line
                    && r.start_line <= target_func.end_line
            })
            .filter_map(|r| {
                // Extract the called function name from target_symbol_id.
                // target_symbol_id is a Definition id -- `def:<file>:<name>:<line>` --
                // NOT a SymbolId id (`<file>:<name>:<line>:<col>`). Parsing it with the
                // wrong layout, or with a forward split that yields the file path, is why
                // callees were always empty and assumed unimplemented.
                crate::relations::Definition::name_from_storage_id(&r.target_symbol_id)
                    .map(|s| s.to_string())
            })
            .filter(|name| !Self::is_call_like_keyword(name))
            .filter(|name| seen_callees.insert(name.clone()))
            .filter_map(|name| {
                // Resolve against the widened index so a callee defined in another
                // translation unit still resolves to a definition.
                callee_index.get(&name).and_then(|defs| defs.first()).cloned()
            })
            .map(|def| crate::relations::CallGraphNode {
                name: def.symbol_id.name.clone(),
                kind: def.symbol_id.kind.clone(),
                // The definition own file, not the requested one: a cross-TU callee
                // does not live in request.file_path.
                file_path: def.symbol_id.file_path.clone(),
                line: def.symbol_id.start_line,
                children: Vec::new(),
            })
            .collect();

        Ok(GetCallGraphResponse {
            root_symbol: Some(root_symbol),
            callers,
            callees,
            precision: format!("{:?}", precision).to_lowercase(),
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// List every symbol defined in a single file.
    ///
    /// Returns definitions only -- name, kind, line span, signature -- and never chunk
    /// content, so enumerating a large file stays cheap. This is the enumeration
    /// primitive the other tools lack: query_codebase and search_by_filters are
    /// relevance-ranked with a limit, and find_definition / find_references need a
    /// position the caller already has.
    pub async fn list_symbols(&self, request: ListSymbolsRequest) -> Result<ListSymbolsResponse> {
        let start = Instant::now();

        request.validate().map_err(|e| anyhow::anyhow!(e))?;

        let file_info = self.create_file_info(&request.file_path, request.project.clone())?;
        let language = file_info.language.as_deref().unwrap_or("Unknown");
        let precision = self.relations_provider.precision_level(language);

        let definitions = self
            .relations_provider
            .extract_definitions(&file_info)
            .context("Failed to extract definitions")?;

        let wanted: Vec<String> = request.kinds.iter().map(|k| k.to_lowercase()).collect();
        let mut symbols: Vec<crate::relations::SymbolInfo> = definitions
            .iter()
            .filter(|d| {
                wanted.is_empty()
                    || wanted.contains(&format!("{:?}", d.symbol_id.kind).to_lowercase())
            })
            .map(|d| crate::relations::SymbolInfo {
                name: d.symbol_id.name.clone(),
                kind: d.symbol_id.kind.clone(),
                file_path: file_info.relative_path.clone(),
                start_line: d.symbol_id.start_line,
                end_line: d.end_line,
                signature: d.signature.clone(),
            })
            .collect();
        symbols.sort_by_key(|s| s.start_line);

        Ok(ListSymbolsResponse {
            file_path: file_info.relative_path.clone(),
            total_count: symbols.len(),
            symbols,
            precision: format!("{:?}", precision).to_lowercase(),
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }
}

// Indexing operations module
pub(crate) mod indexing;
// Git indexing operations module
pub(crate) mod git_indexing;

#[cfg(test)]
mod tests;
