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

// Index locking mechanism
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

        Ok(IndexLockResult::Acquired(IndexLockGuard::new(
            normalized_path,
            self.indexing_ops.clone(),
            result_tx,
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
        let definition = definitions.into_iter().find(|def| {
            request.line >= def.symbol_id.start_line
                && request.line <= def.end_line
                && (request.column == 0 || request.column >= def.symbol_id.start_col)
        });

        let result = definition.map(|def| DefinitionResult::from(&def));

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
        let target_symbol = definitions.iter().find(|def| {
            request.line >= def.symbol_id.start_line
                && request.line <= def.end_line
                && (request.column == 0 || request.column >= def.symbol_id.start_col)
        });

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

        // Build symbol index from definitions
        let mut symbol_index: std::collections::HashMap<String, Vec<crate::relations::Definition>> =
            std::collections::HashMap::new();
        for def in definitions {
            symbol_index
                .entry(def.symbol_id.name.clone())
                .or_default()
                .push(def);
        }

        // Find references in the same file
        let references = self
            .relations_provider
            .extract_references(&file_info, &symbol_index)
            .context("Failed to extract references")?;

        // Filter to references matching our target symbol
        let matching_refs: Vec<ReferenceResult> = references
            .iter()
            .filter(|r| {
                // Check if this reference points to our target symbol
                r.target_symbol_id.contains(&symbol_name_str)
            })
            .take(request.limit)
            .map(|r| ReferenceResult::from(r))
            .collect();

        let total_count = matching_refs.len();

        Ok(FindReferencesResponse {
            symbol_name,
            references: matching_refs,
            total_count,
            precision: format!("{:?}", precision).to_lowercase(),
            duration_ms: start.elapsed().as_millis() as u64,
        })
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
        let target_function = definitions.iter().find(|def| {
            // Only consider functions/methods
            matches!(
                def.symbol_id.kind,
                crate::relations::SymbolKind::Function | crate::relations::SymbolKind::Method
            ) && request.line >= def.symbol_id.start_line
                && request.line <= def.end_line
                && (request.column == 0 || request.column >= def.symbol_id.start_col)
        });

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

        // Find references in the same file to identify callers
        let references = self
            .relations_provider
            .extract_references(&file_info, &symbol_index)
            .context("Failed to extract references")?;

        // Find callers (references with Call kind pointing to our function)
        let mut seen_callers = std::collections::HashSet::new();
        let callers: Vec<crate::relations::CallGraphNode> = references
            .iter()
            .filter(|r| {
                r.reference_kind == crate::relations::ReferenceKind::Call
                    && r.target_symbol_id.contains(&function_name)
            })
            .filter_map(|r| {
                // Try to find which function contains this call
                definitions.iter().find(|def| {
                    matches!(
                        def.symbol_id.kind,
                        crate::relations::SymbolKind::Function | crate::relations::SymbolKind::Method
                    ) && r.start_line >= def.symbol_id.start_line
                        && r.start_line <= def.end_line
                })
            })
            .filter(|def| seen_callers.insert(def.symbol_id.name.clone()))
            .map(|def| crate::relations::CallGraphNode {
                name: def.symbol_id.name.clone(),
                kind: def.symbol_id.kind.clone(),
                file_path: request.file_path.clone(),
                line: def.symbol_id.start_line,
                children: Vec::new(),
            })
            .collect();

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
                // Extract the called function name from target_symbol_id
                let parts: Vec<&str> = r.target_symbol_id.split(':').collect();
                if parts.len() >= 2 {
                    Some(parts[1].to_string())
                } else {
                    None
                }
            })
            .filter(|name| seen_callees.insert(name.clone()))
            .filter_map(|name| {
                // Find the definition of the called function
                symbol_index.get(&name).and_then(|defs| defs.first()).cloned()
            })
            .map(|def| crate::relations::CallGraphNode {
                name: def.symbol_id.name.clone(),
                kind: def.symbol_id.kind.clone(),
                file_path: request.file_path.clone(),
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
}

// Indexing operations module
pub(crate) mod indexing;
// Git indexing operations module
pub(crate) mod git_indexing;

#[cfg(test)]
mod tests;
