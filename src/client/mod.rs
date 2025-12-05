//! Core library client for project-rag
//!
//! This module provides the main client interface for using project-rag
//! as a library in your own Rust applications.

use crate::cache::HashCache;
use crate::config::Config;
use crate::embedding::{EmbeddingProvider, FastEmbedManager};
use crate::git_cache::GitCache;
use crate::indexer::CodeChunker;
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

/// State for an in-progress indexing operation
pub(crate) struct IndexingOperation {
    /// Sender to broadcast the result to all waiters
    result_tx: broadcast::Sender<IndexResponse>,
}

/// Result of trying to acquire an index lock
pub(crate) enum IndexLockResult {
    /// We acquired the lock and should perform indexing
    Acquired(IndexLockGuard),
    /// Another operation is in progress, wait for its result
    WaitForResult(broadcast::Receiver<IndexResponse>),
}

/// Guard for index locks that cleans up the lock when dropped
pub(crate) struct IndexLockGuard {
    path: String,
    locks_map: Arc<RwLock<HashMap<String, IndexingOperation>>>,
    /// Sender to broadcast the result when indexing completes
    pub(crate) result_tx: broadcast::Sender<IndexResponse>,
}

impl IndexLockGuard {
    /// Broadcast the indexing result to all waiters
    pub(crate) fn broadcast_result(&self, result: &IndexResponse) {
        // Ignore send errors (no receivers is fine)
        let _ = self.result_tx.send(result.clone());
    }
}

impl Drop for IndexLockGuard {
    fn drop(&mut self) {
        // Cleanup the lock from the map when the guard is dropped
        let path = self.path.clone();
        let locks_map = self.locks_map.clone();

        // Spawn a task to clean up the lock asynchronously
        tokio::spawn(async move {
            let mut locks = locks_map.write().await;
            locks.remove(&path);
        });
    }
}

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
        // Migrate from old project-rag directories if they exist
        if let Err(e) = crate::paths::PlatformPaths::migrate_from_project_rag() {
            eprintln!("Warning: Migration failed: {}", e);
        }

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

    /// Normalize a path to a canonical absolute form for consistent cache lookups
    pub fn normalize_path(path: &str) -> Result<String> {
        let path_buf = PathBuf::from(path);
        let canonical = std::fs::canonicalize(&path_buf)
            .with_context(|| format!("Failed to canonicalize path: {}", path))?;
        Ok(canonical.to_string_lossy().to_string())
    }

    /// Try to acquire an indexing lock for a given path
    ///
    /// Returns either:
    /// - `IndexLockResult::Acquired(guard)` if we should perform the indexing
    /// - `IndexLockResult::WaitForResult(receiver)` if another operation is in progress
    ///
    /// The lock is automatically released when the returned guard is dropped.
    pub(crate) async fn try_acquire_index_lock(&self, path: &str) -> Result<IndexLockResult> {
        // Normalize the path to ensure consistent locking across different path formats
        let normalized_path = Self::normalize_path(path)?;

        // Acquire write lock on the ops map
        let mut ops = self.indexing_ops.write().await;

        // Check if an operation is already in progress for this path
        if let Some(existing_op) = ops.get(&normalized_path) {
            // Check if the operation is still active by looking at receiver count
            // If receiver_count() == 0, the indexing has completed but cleanup hasn't
            // happened yet (the async Drop task is pending). In this case, treat it
            // as if no operation exists and start a new one.
            if existing_op.result_tx.receiver_count() > 0 {
                // Operation is still active, subscribe to receive the result
                let receiver = existing_op.result_tx.subscribe();
                tracing::info!(
                    "Indexing already in progress for {}, waiting for result",
                    normalized_path
                );
                return Ok(IndexLockResult::WaitForResult(receiver));
            } else {
                // Stale entry - remove it and proceed to create a new operation
                tracing::debug!(
                    "Removing stale indexing lock for {} (operation completed)",
                    normalized_path
                );
                ops.remove(&normalized_path);
            }
        }

        // Create a new broadcast channel for this operation
        // Capacity of 1 is enough since we only send one result
        let (result_tx, _) = broadcast::channel(1);

        // Register this operation
        ops.insert(
            normalized_path.clone(),
            IndexingOperation {
                result_tx: result_tx.clone(),
            },
        );

        // Drop the write lock on the map
        drop(ops);

        Ok(IndexLockResult::Acquired(IndexLockGuard {
            path: normalized_path,
            locks_map: self.indexing_ops.clone(),
            result_tx,
        }))
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
        indexing::do_index_smart(
            self,
            request.path,
            request.project,
            request.include_patterns,
            request.exclude_patterns,
            request.max_file_size,
            None, // No peer
            None, // No progress token
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

        let start = Instant::now();

        let query_embedding = self
            .embedding_provider
            .embed_batch(vec![request.query.clone()])
            .context("Failed to generate query embedding")?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding generated"))?;

        let results = self
            .vector_db
            .search_filtered(
                query_embedding,
                &request.query,
                request.limit,
                request.min_score,
                request.project,
                request.path,
                true,
                request.file_extensions,
                request.languages,
                request.path_patterns,
            )
            .await
            .context("Failed to search with filters")?;

        Ok(QueryResponse {
            results,
            duration_ms: start.elapsed().as_millis() as u64,
            threshold_used: request.min_score,
            threshold_lowered: false,
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
            .map(|(language, count)| LanguageStats {
                language,
                file_count: count,
                chunk_count: count,
            })
            .collect();

        Ok(StatisticsResponse {
            total_files: stats.total_points,
            total_chunks: stats.total_vectors,
            total_embeddings: stats.total_vectors,
            database_size_bytes: 0,
            language_breakdown,
        })
    }

    /// Clear all indexed data from the vector database
    pub async fn clear_index(&self) -> Result<ClearResponse> {
        match self.vector_db.clear().await {
            Ok(_) => {
                let mut cache = self.hash_cache.write().await;
                cache.roots.clear();

                if let Err(e) = cache.save(&self.cache_path) {
                    tracing::warn!("Failed to save cleared cache: {}", e);
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
}

// Indexing operations module
pub(crate) mod indexing;
// Git indexing operations module
pub(crate) mod git_indexing;

#[cfg(test)]
mod tests;
