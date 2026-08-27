//! Core library client for project-rag
//!
//! This module provides the main client interface for using project-rag
//! as a library in your own Rust applications.

use crate::cache::HashCache;
use crate::config::Config;
use crate::embedding::{EmbeddingProvider, FastEmbedManager};
use crate::git_cache::GitCache;
use crate::indexer::{CodeChunker, FileInfo, detect_language};
use crate::relations::storage::{LanceRelationsStore, RelationsStore};
use crate::relations::{
    DefinitionResult, HybridRelationsProvider, ReferenceKind, ReferenceResult, RelationsProvider,
    ResolutionStatus,
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

// read_file/edit_file: single-file read and write-then-reindex operations
mod file_ops;

// find_unused: unused import and dead-symbol candidate detection
mod find_unused;

const SEARCH_MAX_RESULTS: usize = 100;
const SEARCH_CONTEXT_LINES: usize = 15;
const SEARCH_MAX_CHARS_PER_RESULT: usize = 4_000;
const SEARCH_MAX_TOTAL_CHARS: usize = 20_000;

fn apply_search_budget(
    query: &str,
    results: Vec<SearchResult>,
    requested_results: usize,
) -> (Vec<SearchResult>, usize, bool) {
    let total_matches = results.len();
    let result_budget = requested_results.min(SEARCH_MAX_RESULTS);
    let query_terms = query
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|term| term.len() >= 2)
        .map(str::to_lowercase)
        .collect::<Vec<_>>();

    let mut returned = Vec::new();
    let mut total_chars = 0usize;

    for mut result in results.into_iter().take(result_budget) {
        if total_chars >= SEARCH_MAX_TOTAL_CHARS {
            break;
        }

        let full_start = result.start_line;
        let full_end = result.end_line;
        result.full_start_line = full_start;
        result.full_end_line = full_end;

        let lines = result.content.lines().collect::<Vec<_>>();
        if !lines.is_empty() {
            let anchor = lines
                .iter()
                .position(|line| {
                    let lower = line.to_lowercase();
                    query_terms.iter().any(|term| lower.contains(term))
                })
                .unwrap_or(0);
            let window_start = anchor.saturating_sub(SEARCH_CONTEXT_LINES);
            let window_end = (anchor + SEARCH_CONTEXT_LINES + 1).min(lines.len());
            let mut snippet = lines[window_start..window_end].join("\n");
            result.start_line = full_start + window_start;
            result.end_line = result.start_line + window_end - window_start - 1;
            result.content_truncated = window_start > 0 || window_end < lines.len();

            let per_result_budget =
                SEARCH_MAX_CHARS_PER_RESULT.min(SEARCH_MAX_TOTAL_CHARS.saturating_sub(total_chars));
            if snippet.len() > per_result_budget {
                let end = crate::git::floor_char_boundary(&snippet, per_result_budget);
                snippet.truncate(end);
                let returned_lines = snippet.lines().count();
                result.end_line = if returned_lines == 0 {
                    result.start_line
                } else {
                    result.start_line + returned_lines - 1
                };
                result.content_truncated = true;
            }
            result.content = snippet;
        }

        total_chars += result.content.len();
        returned.push(result);
    }

    let results_truncated = total_matches > returned.len()
        || requested_results > SEARCH_MAX_RESULTS
        || returned.iter().any(|result| result.content_truncated);
    (returned, total_matches, results_truncated)
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
    // Relations provider for code navigation (find definition, references, call graph)
    pub(crate) relations_provider: Arc<HybridRelationsProvider>,
    // Persistent store for extracted definitions/references (shares the LanceDB directory)
    pub(crate) relations_store: Arc<LanceRelationsStore>,
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

        // Relations store lives in the same LanceDB directory as the embeddings
        // table, so one database directory holds everything the index knows.
        let relations_store = Arc::new(
            LanceRelationsStore::new(config.vector_db.lancedb_path.clone())
                .await
                .context("Failed to initialize relations store")?,
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
            relations_store,
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

    /// Resolve a transport path against exactly one indexed project root.
    /// Relative inputs are project-relative, never process-CWD-relative.
    pub(crate) async fn resolve_project_path(
        &self,
        file_path: &str,
        project_id: Option<&str>,
        for_write: bool,
    ) -> Result<(crate::project_path::ResolvedProjectPath, String)> {
        let roots = {
            let cache = self.hash_cache.read().await;
            cache
                .roots
                .keys()
                .filter(|root| {
                    project_id.is_none_or(|wanted| cache.project_id(root) == Some(wanted))
                })
                .cloned()
                .collect::<Vec<_>>()
        };
        if roots.is_empty() {
            anyhow::bail!("No indexed project root matches this request; run index_codebase first");
        }

        let mut matches = Vec::new();
        for root in roots {
            let resolver = match crate::project_path::ProjectPathResolver::new(&root) {
                Ok(resolver) => resolver,
                Err(_) => continue,
            };
            let resolved = if for_write {
                resolver.resolve_for_write(file_path)
            } else {
                resolver.resolve_existing(file_path)
            };
            if let Ok(resolved) = resolved {
                matches.push((resolved, root));
            }
        }

        match matches.len() {
            1 => Ok(matches.remove(0)),
            0 => anyhow::bail!(
                "'{}' does not resolve to a file inside an indexed project root",
                file_path
            ),
            _ => anyhow::bail!(
                "Relative path '{}' is ambiguous across indexed projects; specify project or use an absolute alias",
                file_path
            ),
        }
    }

    /// Create FileInfo through the canonical project path layer.
    async fn create_file_info(&self, file_path: &str, project: Option<String>) -> Result<FileInfo> {
        let (resolved, root) = self
            .resolve_project_path(file_path, project.as_deref(), false)
            .await?;
        let project_id = self
            .hash_cache
            .read()
            .await
            .project_id(&root)
            .map(str::to_string)
            .or(project);
        Self::build_file_info_in_root(&resolved.absolute, &root, project_id)
    }

    /// Associated form used inside blocking analysis closures when the explicit
    /// project root is already known.
    pub(crate) fn build_file_info_in_root(
        file_path: &std::path::Path,
        root: &str,
        project: Option<String>,
    ) -> Result<FileInfo> {
        let resolver = crate::project_path::ProjectPathResolver::new(root)?;
        let resolved = resolver.resolve_existing(&file_path.to_string_lossy())?;
        let canonical = resolved.absolute;

        let content = std::fs::read_to_string(&canonical)
            .with_context(|| format!("Failed to read file: {}", canonical.display()))?;

        let extension = canonical
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_string());

        let language = extension.as_ref().and_then(|ext| detect_language(ext));

        // Compute file hash
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let hash = format!("{:x}", hasher.finalize());

        Ok(FileInfo {
            path: canonical,
            relative_path: resolved.relative,
            root_path: resolver.root().to_string_lossy().to_string(),
            project,
            extension,
            language,
            content,
            hash,
        })
    }

    /// Normalize a path to a canonical absolute form for consistent cache lookups
    pub fn normalize_path(path: &str) -> Result<String> {
        let path_buf = crate::project_path::normalize_transport_path(path);
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
    ///     path: None,
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
        let search_root = request
            .path
            .as_deref()
            .map(Self::normalize_path)
            .transpose()?;

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

        let probe_limit = request.limit.min(SEARCH_MAX_RESULTS).saturating_add(1);
        let mut results = self
            .vector_db
            .search(
                query_embedding.clone(),
                &request.query,
                probe_limit,
                threshold_used,
                request.project.clone(),
                search_root.clone(),
                request.hybrid,
                RecordOrigin::Current,
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
                        probe_limit,
                        threshold,
                        request.project.clone(),
                        search_root.clone(),
                        request.hybrid,
                        RecordOrigin::Current,
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

        let (results, total_matches, results_truncated) =
            apply_search_budget(&request.query, results, request.limit);
        Ok(QueryResponse {
            returned_matches: results.len(),
            results,
            duration_ms: start.elapsed().as_millis() as u64,
            threshold_used,
            threshold_lowered,
            total_matches,
            results_truncated,
            next_cursor: None,
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
        let search_root = request
            .path
            .as_deref()
            .map(Self::normalize_path)
            .transpose()?;

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

        let probe_limit = request.limit.min(SEARCH_MAX_RESULTS).saturating_add(1);
        let mut results = self
            .vector_db
            .search_filtered(
                query_embedding.clone(),
                &request.query,
                probe_limit,
                threshold_used,
                request.project.clone(),
                search_root.clone(),
                true,
                request.file_extensions.clone(),
                request.languages.clone(),
                request.path_patterns.clone(),
                RecordOrigin::Current,
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
                        probe_limit,
                        threshold,
                        request.project.clone(),
                        search_root.clone(),
                        true,
                        request.file_extensions.clone(),
                        request.languages.clone(),
                        request.path_patterns.clone(),
                        RecordOrigin::Current,
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

        let (results, total_matches, results_truncated) =
            apply_search_budget(&request.query, results, request.limit);
        Ok(QueryResponse {
            returned_matches: results.len(),
            results,
            duration_ms: start.elapsed().as_millis() as u64,
            threshold_used,
            threshold_lowered,
            total_matches,
            results_truncated,
            next_cursor: None,
        })
    }

    /// Get statistics about the indexed codebase
    pub async fn get_statistics(&self) -> Result<StatisticsResponse> {
        let stats = self
            .vector_db
            .get_statistics()
            .await
            .context("Failed to get statistics")?;

        // Relations counts are best-effort: statistics must not fail just
        // because the relations tables are unreadable.
        let relations_stats = self.relations_store.get_stats().await.unwrap_or_else(|e| {
            tracing::warn!("Failed to get relations statistics: {:#}", e);
            Default::default()
        });

        let language_breakdown = stats
            .language_breakdown
            .into_iter()
            .map(|entry| LanguageStats {
                language: entry.language,
                file_count: entry.file_count,
                chunk_count: entry.chunk_count,
            })
            .collect();

        let cache = self.hash_cache.read().await;
        let git_cache = self.git_cache.read().await;
        let mut index_diagnostics = cache.diagnostics.clone();
        index_diagnostics.extend(git_cache.diagnostics.clone());

        Ok(StatisticsResponse {
            total_files: stats.total_files,
            total_chunks: stats.total_points,
            total_embeddings: stats.total_vectors,
            database_size_bytes: stats.database_size_bytes,
            language_breakdown,
            total_definitions: relations_stats.definition_count,
            total_references: relations_stats.reference_count,
            code_reference_count: relations_stats.code_reference_count,
            files_with_definitions: relations_stats.files_with_definitions,
            index_schema_version: crate::cache::INDEX_SCHEMA_VERSION,
            invalid_history_records: git_cache.invalid_records,
            index_diagnostics,
        })
    }

    /// Clear all indexed data from the vector database and hash cache
    pub async fn clear_index(&self) -> Result<ClearResponse> {
        match self.vector_db.clear().await {
            Ok(_) => {
                // Clear hash cache (both roots and dirty_roots)
                let mut cache = self.hash_cache.write().await;
                cache.roots.clear();
                cache.project_ids.clear();
                cache.dirty_roots.clear();
                cache.diagnostics.clear();

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
                git_cache.clear();
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

                // Also clear stored definitions/references
                if let Err(e) = self.relations_store.clear().await {
                    tracing::warn!("Failed to clear relations store: {}", e);
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
                    crate::relations::SymbolKind::Function
                        | crate::relations::SymbolKind::Method
                        | crate::relations::SymbolKind::Constructor
                        | crate::relations::SymbolKind::Destructor
                )
        };

        if let Some(name) = Self::identifier_at(content, line, column) {
            // Prefer a real definition over an import binding of the same name:
            // `use foo::helper;` plus `fn helper()` in one file must resolve to the
            // function. Import defs span one line, so on span alone they would win.
            let exact = definitions
                .iter()
                .filter(|d| d.symbol_id.name == name && is_candidate(d))
                .min_by_key(|d| {
                    (
                        d.symbol_id.kind == crate::relations::SymbolKind::Import,
                        d.end_line.saturating_sub(d.symbol_id.start_line),
                    )
                });
            // An identifier which has no definition in this file is a reference
            // candidate, not permission to return its enclosing function as the
            // identifier's definition.
            return exact;
        }

        definitions
            .iter()
            .filter(|d| is_candidate(d) && line >= d.symbol_id.start_line && line <= d.end_line)
            .min_by_key(|d| d.end_line.saturating_sub(d.symbol_id.start_line))
    }

    /// Files that plausibly mention `symbol`, newest-ranked first.
    ///
    /// This remains a discovery-only helper for conservative unused analysis. It
    /// must never be used to establish a resolved symbol edge.
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
            .search(
                embedding,
                symbol,
                limit,
                0.0,
                project,
                None,
                true,
                RecordOrigin::Current,
            )
            .await
            .context("Failed to search for candidate files")?;
        let mut seen = std::collections::HashSet::new();
        let mut files = Vec::new();
        for result in results {
            let full = match &result.root_path {
                Some(root) => std::path::Path::new(root).join(&result.file_path),
                None => std::path::PathBuf::from(&result.file_path),
            };
            if seen.insert(full.clone()) {
                files.push(full);
            }
        }
        Ok(files)
    }

    pub async fn find_definition(
        &self,
        request: FindDefinitionRequest,
    ) -> Result<FindDefinitionResponse> {
        let start = Instant::now();

        // Validate request
        request.validate().map_err(|e| anyhow::anyhow!(e))?;

        // Create FileInfo for the file
        let file_info = self
            .create_file_info(&request.file_path, request.project.clone())
            .await?;

        // Get precision level for this language
        let language = file_info.language.as_deref().unwrap_or("Unknown");
        let precision = self.relations_provider.precision_level(language);

        // Extract definitions from the file
        let definitions = self
            .relations_provider
            .extract_definitions(&file_info)
            .context("Failed to extract definitions")?;

        // Find the definition at the requested position
        let local_definition = Self::resolve_symbol_at(
            &definitions,
            &file_info.content,
            request.line,
            request.column,
            false,
        );
        let (result, resolution_status, evidence_kind, candidates) = if let Some(definition) =
            local_definition
        {
            (
                Some(DefinitionResult::from(definition)),
                crate::relations::ResolutionStatus::Resolved,
                crate::relations::EvidenceKind::Syntactic,
                vec![crate::relations::ReferenceCandidate {
                    symbol_id: definition.to_storage_id(),
                    reason: "cursor is on a parser-extracted declaration/definition".to_string(),
                }],
            )
        } else if let Some(reference) = self
            .relations_store
            .find_reference_at_in_root(
                &file_info.relative_path,
                &file_info.root_path,
                request.line,
                request.column,
            )
            .await?
        {
            let resolved_definition = if reference.resolution_status
                == crate::relations::ResolutionStatus::Resolved
                && !reference.target_symbol_id.is_empty()
            {
                let mut locations = self
                    .relations_store
                    .find_definitions_by_symbol_id_in_root(
                        &reference.target_symbol_id,
                        &file_info.root_path,
                    )
                    .await?;
                locations.sort_by_key(|definition| {
                    definition.location.role != crate::relations::LocationRole::Definition
                });
                locations.first().map(DefinitionResult::from)
            } else {
                None
            };
            (
                resolved_definition,
                reference.resolution_status,
                reference.evidence_kind,
                reference.candidates,
            )
        } else {
            (
                None,
                crate::relations::ResolutionStatus::Unresolved,
                crate::relations::EvidenceKind::Heuristic,
                Vec::new(),
            )
        };

        Ok(FindDefinitionResponse {
            definition: result,
            resolution_status,
            evidence_kind,
            candidates,
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
    pub async fn find_references(
        &self,
        request: FindReferencesRequest,
    ) -> Result<FindReferencesResponse> {
        let start = Instant::now();

        // Validate request
        request.validate().map_err(|e| anyhow::anyhow!(e))?;

        // Create FileInfo for the file
        let file_info = self
            .create_file_info(&request.file_path, request.project.clone())
            .await?;

        // Get precision level for this language
        let language = file_info.language.as_deref().unwrap_or("Unknown");
        let precision = self.relations_provider.precision_level(language);

        // Extract definitions from the file to find the symbol at the position
        let definitions = self
            .relations_provider
            .extract_definitions(&file_info)
            .context("Failed to extract definitions")?;

        // Find the symbol at the requested position
        let mut target_symbol = Self::resolve_symbol_at(
            &definitions,
            &file_info.content,
            request.line,
            request.column,
            false,
        )
        .cloned();
        let mut target_resolution_status = crate::relations::ResolutionStatus::Resolved;
        let mut target_evidence_kind = crate::relations::EvidenceKind::Syntactic;
        let mut occurrence_name = None;
        let mut target_candidates = target_symbol
            .as_ref()
            .map(|definition| {
                vec![crate::relations::ReferenceCandidate {
                    symbol_id: definition.to_storage_id(),
                    reason: "cursor is on a parser-extracted declaration/definition".to_string(),
                }]
            })
            .unwrap_or_default();

        if target_symbol.is_none()
            && let Some(reference) = self
                .relations_store
                .find_reference_at_in_root(
                    &file_info.relative_path,
                    &file_info.root_path,
                    request.line,
                    request.column,
                )
                .await?
        {
            occurrence_name = Some(reference.target_name.clone());
            target_resolution_status = reference.resolution_status;
            target_evidence_kind = reference.evidence_kind;
            target_candidates = reference.candidates.clone();
            if reference.resolution_status == crate::relations::ResolutionStatus::Resolved
                && !reference.target_symbol_id.is_empty()
            {
                target_symbol = self
                    .relations_store
                    .find_definitions_by_symbol_id_in_root(
                        &reference.target_symbol_id,
                        &file_info.root_path,
                    )
                    .await?
                    .into_iter()
                    .next();
            }
        }

        let symbol_name = target_symbol
            .as_ref()
            .map(|def| def.symbol_id.name.clone())
            .or(occurrence_name);

        // If no symbol found at position, return empty result
        if target_symbol.is_none() {
            return Ok(FindReferencesResponse {
                symbol_name,
                target_resolution_status,
                target_evidence_kind,
                target_candidates,
                references: Vec::new(),
                total_count: 0,
                total_matches: 0,
                returned_matches: 0,
                results_truncated: false,
                next_cursor: None,
                statistics: ReferenceStatistics {
                    code_reference_count: 0,
                    resolved_count: 0,
                    ambiguous_count: 0,
                    unresolved_count: 0,
                },
                precision: format!("{:?}", precision).to_lowercase(),
                duration_ms: start.elapsed().as_millis() as u64,
            });
        }

        let symbol_name_str = symbol_name.clone().unwrap();
        let target_id = target_symbol.expect("checked above").to_storage_id();
        let path_filter = if let Some(path) = request.path_filter.as_deref() {
            Some(
                crate::project_path::ProjectPathResolver::new(&file_info.root_path)?
                    .resolve_existing(path)?
                    .relative,
            )
        } else {
            None
        };
        let mut persisted = self
            .relations_store
            .find_references_by_name_in_root(&symbol_name_str, &file_info.root_path)
            .await
            .context("Failed to query persisted reference source of truth")?;
        persisted.retain(|reference| {
            let points_to_target = reference.target_symbol_id == target_id
                || reference
                    .candidates
                    .iter()
                    .any(|candidate| candidate.symbol_id == target_id);
            let definition_allowed = request.include_definition
                || !matches!(
                    reference.reference_kind,
                    crate::relations::ReferenceKind::Definition
                        | crate::relations::ReferenceKind::Declaration
                );
            let kind_allowed = if request.reference_kinds.is_empty() {
                request.include_non_code || reference.reference_kind.is_code()
            } else {
                request.reference_kinds.contains(&reference.reference_kind)
            };
            let language_allowed = request
                .language
                .as_ref()
                .is_none_or(|wanted| reference.language.eq_ignore_ascii_case(wanted));
            let path_allowed = path_filter
                .as_ref()
                .is_none_or(|wanted| &reference.file_path == wanted);
            let resolution_allowed = request.resolution_statuses.is_empty()
                || request
                    .resolution_statuses
                    .contains(&reference.resolution_status);
            let evidence_allowed = request.evidence_kinds.is_empty()
                || request.evidence_kinds.contains(&reference.evidence_kind);
            let configuration_allowed = crate::build_config::configuration_scope_matches(
                &reference.configuration_states,
                &request.configurations,
            );
            points_to_target
                && definition_allowed
                && kind_allowed
                && language_allowed
                && path_allowed
                && resolution_allowed
                && evidence_allowed
                && configuration_allowed
        });
        persisted.sort_by(|a, b| {
            (&a.file_path, a.start_line, a.start_col, &a.location_id).cmp(&(
                &b.file_path,
                b.start_line,
                b.start_col,
                &b.location_id,
            ))
        });

        let total_matches = persisted.len();
        let resolved_count = persisted
            .iter()
            .filter(|r| r.resolution_status == crate::relations::ResolutionStatus::Resolved)
            .count();
        let ambiguous_count = persisted
            .iter()
            .filter(|r| r.resolution_status == crate::relations::ResolutionStatus::Ambiguous)
            .count();
        let unresolved_count = persisted
            .iter()
            .filter(|r| r.resolution_status == crate::relations::ResolutionStatus::Unresolved)
            .count();
        let page: Vec<_> = persisted
            .iter()
            .skip(request.cursor)
            .take(request.limit)
            .map(ReferenceResult::from)
            .collect();
        let returned_matches = page.len();
        let next_offset = request.cursor.saturating_add(returned_matches);
        let next_cursor = (next_offset < total_matches).then_some(next_offset);

        Ok(FindReferencesResponse {
            symbol_name,
            target_resolution_status,
            target_evidence_kind,
            target_candidates,
            references: page,
            total_count: total_matches,
            total_matches,
            returned_matches,
            results_truncated: next_cursor.is_some(),
            next_cursor,
            statistics: ReferenceStatistics {
                code_reference_count: total_matches,
                resolved_count,
                ambiguous_count,
                unresolved_count,
            },
            precision: format!("{:?}", precision).to_lowercase(),
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Control-flow and cast keywords that are followed by a parenthesis but are not
    /// calls. Without this, `if (` is reported as a callee whenever some file happens
    /// to carry a bogus definition of that name.
    #[cfg(test)]
    fn is_call_like_keyword(name: &str) -> bool {
        matches!(
            name,
            "if" | "for"
                | "while"
                | "switch"
                | "catch"
                | "return"
                | "sizeof"
                | "do"
                | "else"
                | "new"
                | "delete"
                | "throw"
                | "defined"
                | "static_cast"
                | "dynamic_cast"
                | "reinterpret_cast"
                | "const_cast"
        )
    }
    /// Identifiers that appear immediately before an opening parenthesis inside the
    /// given 1-based line span -- that is, plausible call sites.
    ///
    /// Used to widen the callee symbol index beyond the defining file. Deliberately
    /// crude: over-reporting costs one extra lookup, under-reporting loses a callee.
    #[cfg(test)]
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
                    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
                    {
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
    pub async fn get_call_graph(
        &self,
        request: GetCallGraphRequest,
    ) -> Result<GetCallGraphResponse> {
        let start = Instant::now();

        // Validate request
        request.validate().map_err(|e| anyhow::anyhow!(e))?;

        // Create FileInfo for the file
        let file_info = self
            .create_file_info(&request.file_path, request.project.clone())
            .await?;

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

        let edge_kinds = if request.edge_kinds.is_empty() {
            vec![
                ReferenceKind::Call,
                ReferenceKind::MethodCall,
                ReferenceKind::ConstructorCall,
                ReferenceKind::ObjectConstruction,
            ]
        } else {
            request.edge_kinds.clone()
        };
        let resolution_statuses = if request.resolution_statuses.is_empty() {
            vec![ResolutionStatus::Resolved]
        } else {
            request.resolution_statuses.clone()
        };

        let Some(root_definition) = target_function.cloned() else {
            return Ok(GetCallGraphResponse {
                root_symbol: None,
                nodes: Vec::new(),
                edges: Vec::new(),
                requested_depth: request.depth,
                graph_truncated: false,
                returned_nodes: 0,
                returned_edges: 0,
                estimated_or_known_total: GraphTotals {
                    nodes: 0,
                    edges: 0,
                    exact: true,
                },
                continuation: None,
                applied_edge_kinds: edge_kinds,
                applied_resolution_statuses: resolution_statuses,
                precision: format!("{:?}", precision).to_lowercase(),
                duration_ms: start.elapsed().as_millis() as u64,
            });
        };
        let root_symbol = crate::relations::SymbolInfo {
            symbol_id: root_definition.to_storage_id(),
            location_id: root_definition.location.to_storage_id(),
            name: root_definition.symbol_id.name.clone(),
            qualified_name: root_definition.symbol_id.qualified_name.clone(),
            kind: root_definition.symbol_id.kind,
            file_path: root_definition.symbol_id.file_path.clone(),
            start_line: root_definition.symbol_id.start_line,
            end_line: root_definition.end_line,
            signature: root_definition.signature.clone(),
            language: root_definition.symbol_id.language.clone(),
            location_role: root_definition.location.role,
        };
        let options = crate::relations::graph::GraphTraversalOptions {
            depth: request.depth,
            include_incoming: request.include_callers,
            include_outgoing: request.include_callees,
            max_nodes: request.max_nodes,
            max_edges: request.max_edges,
            edge_kinds: edge_kinds.clone(),
            resolution_statuses: resolution_statuses.clone(),
            language_filters: request.language_filters,
            path_filters: request.path_filters,
            configurations: request.configurations,
        };
        let graph = crate::relations::graph::traverse_dependency_graph(
            self.relations_store.as_ref(),
            &root_definition,
            &file_info.root_path,
            &options,
        )
        .await?;
        let returned_nodes = graph.nodes.len();
        let returned_edges = graph.edges.len();

        Ok(GetCallGraphResponse {
            root_symbol: Some(root_symbol),
            nodes: graph.nodes,
            edges: graph.edges,
            requested_depth: request.depth,
            graph_truncated: graph.graph_truncated,
            returned_nodes,
            returned_edges,
            estimated_or_known_total: graph.estimated_or_known_total,
            continuation: graph.continuation,
            applied_edge_kinds: edge_kinds,
            applied_resolution_statuses: resolution_statuses,
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

        let file_info = self
            .create_file_info(&request.file_path, request.project.clone())
            .await?;
        let language = file_info.language.as_deref().unwrap_or("Unknown");
        let precision = self.relations_provider.precision_level(language);

        let (definitions, skipped) = self
            .relations_provider
            .extract_definitions_reporting(&file_info)
            .context("Failed to extract definitions")?;

        let wanted: Vec<String> = request.kinds.iter().map(|k| k.to_lowercase()).collect();
        let mut symbols: Vec<crate::relations::SymbolInfo> = definitions
            .iter()
            .filter(|d| {
                wanted.is_empty()
                    || wanted.contains(&format!("{:?}", d.symbol_id.kind).to_lowercase())
            })
            .map(|d| crate::relations::SymbolInfo {
                symbol_id: d.to_storage_id(),
                location_id: d.location.to_storage_id(),
                name: d.symbol_id.name.clone(),
                qualified_name: d.symbol_id.qualified_name.clone(),
                kind: d.symbol_id.kind,
                file_path: file_info.relative_path.clone(),
                start_line: d.symbol_id.start_line,
                end_line: d.end_line,
                signature: d.signature.clone(),
                language: d.symbol_id.language.clone(),
                location_role: d.location.role,
            })
            .collect();
        symbols.sort_by_key(|s| s.start_line);

        Ok(ListSymbolsResponse {
            file_path: file_info.relative_path.clone(),
            total_count: symbols.len(),
            symbols,
            precision: format!("{:?}", precision).to_lowercase(),
            skipped,
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
