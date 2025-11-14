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
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

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

        Ok(Self {
            embedding_provider,
            vector_db,
            chunker,
            hash_cache: Arc::new(RwLock::new(hash_cache)),
            cache_path,
            git_cache: Arc::new(RwLock::new(git_cache)),
            git_cache_path,
            config: Arc::new(config),
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
mod tests {
    use super::*;
    use tempfile::TempDir;

    // Helper to create a test client
    async fn create_test_client() -> (RagClient, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
        let cache_path = temp_dir.path().join("cache.json");
        let client = RagClient::new_with_db_path(&db_path, cache_path)
            .await
            .unwrap();
        (client, temp_dir)
    }

    // ===== Client Initialization Tests =====

    #[tokio::test]
    async fn test_new_with_db_path() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
        let cache_path = temp_dir.path().join("cache.json");

        let result = RagClient::new_with_db_path(&db_path, cache_path).await;
        assert!(result.is_ok());

        let client = result.unwrap();
        assert_eq!(client.embedding_dimension(), 384);
    }

    #[tokio::test]
    async fn test_client_clone() {
        let (client, _temp_dir) = create_test_client().await;
        let _cloned = client.clone();
        // Should compile and not panic
    }

    #[tokio::test]
    async fn test_config_accessor() {
        let (client, _temp_dir) = create_test_client().await;
        let config = client.config();
        assert!(config.indexing.chunk_size > 0);
    }

    #[tokio::test]
    async fn test_embedding_dimension_accessor() {
        let (client, _temp_dir) = create_test_client().await;
        let dimension = client.embedding_dimension();
        assert_eq!(dimension, 384); // all-MiniLM-L6-v2 has 384 dimensions
    }

    // ===== normalize_path Tests =====

    #[test]
    fn test_normalize_path_valid() {
        let result = RagClient::normalize_path(".");
        assert!(result.is_ok());
        let normalized = result.unwrap();
        assert!(!normalized.is_empty());
    }

    #[test]
    fn test_normalize_path_nonexistent() {
        let result = RagClient::normalize_path("/nonexistent/path/12345");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to canonicalize"));
    }

    #[test]
    fn test_normalize_path_absolute() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_string_lossy().to_string();

        let result = RagClient::normalize_path(&path);
        assert!(result.is_ok());
        let normalized = result.unwrap();
        assert!(normalized.starts_with('/'));
    }

    // ===== index_codebase Tests =====

    #[tokio::test]
    async fn test_index_codebase_empty_directory() {
        let (client, temp_dir) = create_test_client().await;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();

        let request = IndexRequest {
            path: data_dir.to_string_lossy().to_string(),
            project: None,
            include_patterns: vec![],
            exclude_patterns: vec![],
            max_file_size: 1024 * 1024,
        };

        let result = client.index_codebase(request).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.files_indexed, 0);
    }

    #[tokio::test]
    async fn test_index_codebase_with_single_file() {
        let (client, temp_dir) = create_test_client().await;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

        let request = IndexRequest {
            path: data_dir.to_string_lossy().to_string(),
            project: Some("test-project".to_string()),
            include_patterns: vec![],
            exclude_patterns: vec![],
            max_file_size: 1024 * 1024,
        };

        let result = client.index_codebase(request).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.files_indexed, 1);
        assert!(response.chunks_created > 0);
        assert!(response.embeddings_generated > 0);
    }

    #[tokio::test]
    async fn test_index_codebase_validation_failure() {
        let (client, _temp_dir) = create_test_client().await;

        let request = IndexRequest {
            path: "/nonexistent/path".to_string(),
            project: None,
            include_patterns: vec![],
            exclude_patterns: vec![],
            max_file_size: 1024 * 1024,
        };

        let result = client.index_codebase(request).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    // ===== query_codebase Tests =====

    #[tokio::test]
    async fn test_query_codebase_empty_index() {
        let (client, _temp_dir) = create_test_client().await;

        let request = QueryRequest {
            query: "test query".to_string(),
            project: None,
            limit: 10,
            min_score: 0.7,
            hybrid: true,
        };

        let result = client.query_codebase(request).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.results.len(), 0);
        assert_eq!(response.threshold_used, 0.7);
        assert!(!response.threshold_lowered);
    }

    #[tokio::test]
    async fn test_query_codebase_with_data() {
        let (client, temp_dir) = create_test_client().await;

        // Index some data first
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        std::fs::write(
            data_dir.join("test.rs"),
            "fn authenticate_user() { /* authentication logic */ }",
        )
        .unwrap();

        let index_req = IndexRequest {
            path: data_dir.to_string_lossy().to_string(),
            project: Some("test-project".to_string()),
            include_patterns: vec![],
            exclude_patterns: vec![],
            max_file_size: 1024 * 1024,
        };
        client.index_codebase(index_req).await.unwrap();

        // Now query
        let query_req = QueryRequest {
            query: "authentication".to_string(),
            project: Some("test-project".to_string()),
            limit: 10,
            min_score: 0.3,
            hybrid: true,
        };

        let result = client.query_codebase(query_req).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(response.results.len() > 0);
        assert!(response.duration_ms > 0);
    }

    #[tokio::test]
    async fn test_query_codebase_adaptive_threshold() {
        let (client, temp_dir) = create_test_client().await;

        // Index some data
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        std::fs::write(data_dir.join("test.rs"), "fn hello() {}").unwrap();

        let index_req = IndexRequest {
            path: data_dir.to_string_lossy().to_string(),
            project: None,
            include_patterns: vec![],
            exclude_patterns: vec![],
            max_file_size: 1024 * 1024,
        };
        client.index_codebase(index_req).await.unwrap();

        // Query with high threshold (might trigger adaptive lowering)
        let query_req = QueryRequest {
            query: "completely unrelated query about databases".to_string(),
            project: None,
            limit: 10,
            min_score: 0.9, // Very high threshold
            hybrid: true,
        };

        let result = client.query_codebase(query_req).await;
        assert!(result.is_ok());
        // Adaptive threshold may or may not lower depending on similarity
    }

    #[tokio::test]
    async fn test_query_codebase_validation_failure() {
        let (client, _temp_dir) = create_test_client().await;

        let request = QueryRequest {
            query: "   ".to_string(), // Empty query
            project: None,
            limit: 10,
            min_score: 0.7,
            hybrid: true,
        };

        let result = client.query_codebase(request).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));
    }

    // ===== search_with_filters Tests =====

    #[tokio::test]
    async fn test_search_with_filters_empty_index() {
        let (client, _temp_dir) = create_test_client().await;

        let request = AdvancedSearchRequest {
            query: "test".to_string(),
            project: None,
            limit: 10,
            min_score: 0.7,
            file_extensions: vec!["rs".to_string()],
            languages: vec!["Rust".to_string()],
            path_patterns: vec!["src/**".to_string()],
        };

        let result = client.search_with_filters(request).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.results.len(), 0);
    }

    #[tokio::test]
    async fn test_search_with_filters_validation_failure() {
        let (client, _temp_dir) = create_test_client().await;

        let request = AdvancedSearchRequest {
            query: "test".to_string(),
            project: None,
            limit: 10,
            min_score: 0.7,
            file_extensions: vec!["".to_string()], // Invalid
            languages: vec![],
            path_patterns: vec![],
        };

        let result = client.search_with_filters(request).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("file extension cannot be empty"));
    }

    // ===== get_statistics Tests =====

    #[tokio::test]
    async fn test_get_statistics_empty() {
        let (client, _temp_dir) = create_test_client().await;

        let result = client.get_statistics().await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.total_files, 0);
        assert_eq!(response.total_chunks, 0);
        assert_eq!(response.total_embeddings, 0);
    }

    #[tokio::test]
    async fn test_get_statistics_with_data() {
        let (client, temp_dir) = create_test_client().await;

        // Index some data
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

        let request = IndexRequest {
            path: data_dir.to_string_lossy().to_string(),
            project: None,
            include_patterns: vec![],
            exclude_patterns: vec![],
            max_file_size: 1024 * 1024,
        };
        client.index_codebase(request).await.unwrap();

        // Get statistics
        let result = client.get_statistics().await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(response.total_files > 0);
        assert!(response.total_chunks > 0);
        assert!(response.total_embeddings > 0);
    }

    // ===== clear_index Tests =====

    #[tokio::test]
    async fn test_clear_index_empty() {
        let (client, _temp_dir) = create_test_client().await;

        let result = client.clear_index().await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(response.success);
    }

    #[tokio::test]
    async fn test_clear_index_with_data() {
        let (client, temp_dir) = create_test_client().await;

        // Index some data
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

        let request = IndexRequest {
            path: data_dir.to_string_lossy().to_string(),
            project: None,
            include_patterns: vec![],
            exclude_patterns: vec![],
            max_file_size: 1024 * 1024,
        };
        client.index_codebase(request).await.unwrap();

        // Clear the index
        let result = client.clear_index().await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(response.success);
        assert!(response.message.contains("Successfully cleared"));

        // Verify it's empty
        let stats = client.get_statistics().await.unwrap();
        assert_eq!(stats.total_files, 0);
    }

    // ===== search_git_history Tests =====

    #[tokio::test]
    async fn test_search_git_history_validation_failure() {
        let (client, _temp_dir) = create_test_client().await;

        let request = SearchGitHistoryRequest {
            query: "  ".to_string(), // Empty query
            path: ".".to_string(),
            project: None,
            branch: None,
            max_commits: 10,
            limit: 10,
            min_score: 0.7,
            author: None,
            since: None,
            until: None,
            file_pattern: None,
        };

        let result = client.search_git_history(request).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));
    }

    #[tokio::test]
    async fn test_search_git_history_nonexistent_path() {
        let (client, _temp_dir) = create_test_client().await;

        let request = SearchGitHistoryRequest {
            query: "test".to_string(),
            path: "/nonexistent/path".to_string(),
            project: None,
            branch: None,
            max_commits: 10,
            limit: 10,
            min_score: 0.7,
            author: None,
            since: None,
            until: None,
            file_pattern: None,
        };

        let result = client.search_git_history(request).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    // ===== Integration Tests =====

    #[tokio::test]
    async fn test_full_workflow_index_query_clear() {
        let (client, temp_dir) = create_test_client().await;

        // Step 1: Index
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        std::fs::write(
            data_dir.join("math.rs"),
            "fn add(a: i32, b: i32) -> i32 { a + b }",
        )
        .unwrap();

        let index_req = IndexRequest {
            path: data_dir.to_string_lossy().to_string(),
            project: Some("math-lib".to_string()),
            include_patterns: vec![],
            exclude_patterns: vec![],
            max_file_size: 1024 * 1024,
        };
        let index_resp = client.index_codebase(index_req).await.unwrap();
        assert_eq!(index_resp.files_indexed, 1);

        // Step 2: Query
        let query_req = QueryRequest {
            query: "addition function".to_string(),
            project: Some("math-lib".to_string()),
            limit: 5,
            min_score: 0.3,
            hybrid: true,
        };
        let query_resp = client.query_codebase(query_req).await.unwrap();
        assert!(query_resp.results.len() > 0);

        // Step 3: Statistics
        let stats = client.get_statistics().await.unwrap();
        assert!(stats.total_files > 0);

        // Step 4: Clear
        let clear_resp = client.clear_index().await.unwrap();
        assert!(clear_resp.success);

        // Step 5: Verify empty
        let stats_after = client.get_statistics().await.unwrap();
        assert_eq!(stats_after.total_files, 0);
    }

    #[tokio::test]
    async fn test_project_isolation() {
        let (client, temp_dir) = create_test_client().await;

        // Index for project A
        let data_dir_a = temp_dir.path().join("project_a");
        std::fs::create_dir(&data_dir_a).unwrap();
        std::fs::write(data_dir_a.join("a.rs"), "fn project_a() {}").unwrap();

        let req_a = IndexRequest {
            path: data_dir_a.to_string_lossy().to_string(),
            project: Some("project-a".to_string()),
            include_patterns: vec![],
            exclude_patterns: vec![],
            max_file_size: 1024 * 1024,
        };
        client.index_codebase(req_a).await.unwrap();

        // Index for project B
        let data_dir_b = temp_dir.path().join("project_b");
        std::fs::create_dir(&data_dir_b).unwrap();
        std::fs::write(data_dir_b.join("b.rs"), "fn project_b() {}").unwrap();

        let req_b = IndexRequest {
            path: data_dir_b.to_string_lossy().to_string(),
            project: Some("project-b".to_string()),
            include_patterns: vec![],
            exclude_patterns: vec![],
            max_file_size: 1024 * 1024,
        };
        client.index_codebase(req_b).await.unwrap();

        // Query only project A
        let query_a = QueryRequest {
            query: "project".to_string(),
            project: Some("project-a".to_string()),
            limit: 10,
            min_score: 0.3,
            hybrid: true,
        };
        let results_a = client.query_codebase(query_a).await.unwrap();

        // Results should only be from project A
        for result in results_a.results {
            assert_eq!(result.project, Some("project-a".to_string()));
        }
    }
}
