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
use rmcp::{
    ErrorData as McpError, Peer, RoleServer, ServerHandler, ServiceExt,
    handler::server::{router::prompt::PromptRouter, tool::ToolRouter, wrapper::Parameters},
    model::*,
    prompt, prompt_handler, prompt_router,
    service::RequestContext,
    tool, tool_handler, tool_router,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct RagMcpServer {
    embedding_provider: Arc<FastEmbedManager>,
    #[cfg(feature = "qdrant-backend")]
    vector_db: Arc<QdrantVectorDB>,
    #[cfg(not(feature = "qdrant-backend"))]
    vector_db: Arc<LanceVectorDB>,
    chunker: Arc<CodeChunker>,
    // Persistent hash cache for incremental updates
    hash_cache: Arc<RwLock<HashCache>>,
    cache_path: PathBuf,
    // Git cache for git history indexing
    git_cache: Arc<RwLock<GitCache>>,
    git_cache_path: PathBuf,
    // Configuration (for accessing batch sizes, timeouts, etc.)
    config: Arc<Config>,
    tool_router: ToolRouter<Self>,
    prompt_router: PromptRouter<Self>,
}

impl RagMcpServer {
    /// Create a new RAG MCP server with default configuration
    pub async fn new() -> Result<Self> {
        let config = Config::new().context("Failed to load configuration")?;
        Self::with_config(config).await
    }

    /// Create a new RAG MCP server with custom configuration
    pub async fn with_config(config: Config) -> Result<Self> {
        tracing::info!("Initializing RAG MCP server with configuration");
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
            tool_router: Self::tool_router(),
            prompt_router: Self::prompt_router(),
        })
    }

    /// Create a new server with custom database path (for testing)
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
    fn normalize_path(path: &str) -> Result<String> {
        let path_buf = PathBuf::from(path);
        let canonical = std::fs::canonicalize(&path_buf)
            .with_context(|| format!("Failed to canonicalize path: {}", path))?;
        Ok(canonical.to_string_lossy().to_string())
    }
}

// Indexing operations module
mod indexing;
// Git indexing operations module
mod git_indexing;

#[tool_router(router = tool_router)]
impl RagMcpServer {
    #[tool(
        description = "Index a codebase directory, creating embeddings for semantic search. Automatically performs full indexing for new codebases or incremental updates for previously indexed codebases."
    )]
    async fn index_codebase(
        &self,
        meta: Meta,
        peer: Peer<RoleServer>,
        Parameters(req): Parameters<IndexRequest>,
    ) -> Result<String, String> {
        // Validate request inputs
        req.validate()?;

        // Get progress token if provided
        let progress_token = meta.get_progress_token();

        let response = self
            .do_index_smart(
                req.path,
                req.project,
                req.include_patterns,
                req.exclude_patterns,
                req.max_file_size,
                Some(peer),
                progress_token,
            )
            .await
            .map_err(|e| format!("{:#}", e))?; // Use alternate display to show full error chain

        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(description = "Query the indexed codebase using semantic search")]
    async fn query_codebase(
        &self,
        Parameters(req): Parameters<QueryRequest>,
    ) -> Result<String, String> {
        // Validate request inputs
        req.validate()?;

        let start = Instant::now();

        // Generate query embedding
        let query_embedding = self
            .embedding_provider
            .embed_batch(vec![req.query.clone()])
            .map_err(|e| format!("Failed to generate query embedding: {}", e))?
            .into_iter()
            .next()
            .ok_or("No embedding generated")?;

        // Adaptive search: try with requested threshold, then progressively lower if no results
        let original_threshold = req.min_score;
        let mut threshold_used = original_threshold;
        let mut threshold_lowered = false;

        // Try original threshold first
        let mut results = self
            .vector_db
            .search(
                query_embedding.clone(),
                &req.query,
                req.limit,
                threshold_used,
                req.project.clone(),
                req.hybrid,
            )
            .await
            .map_err(|e| format!("Failed to search: {}", e))?;

        // If no results and threshold is high, try progressively lower thresholds
        if results.is_empty() && original_threshold > 0.3 {
            let fallback_thresholds = [0.6, 0.5, 0.4, 0.3];

            for &threshold in &fallback_thresholds {
                if threshold >= original_threshold {
                    continue; // Skip thresholds that are higher than or equal to what we already tried
                }

                tracing::info!(
                    "No results with threshold {}, trying {}",
                    threshold_used,
                    threshold
                );

                results = self
                    .vector_db
                    .search(
                        query_embedding.clone(),
                        &req.query,
                        req.limit,
                        threshold,
                        req.project.clone(),
                        req.hybrid,
                    )
                    .await
                    .map_err(|e| format!("Failed to search: {}", e))?;

                if !results.is_empty() {
                    threshold_used = threshold;
                    threshold_lowered = true;
                    tracing::info!(
                        "Found {} results with lowered threshold {}",
                        results.len(),
                        threshold
                    );
                    break;
                }
            }
        }

        let response = QueryResponse {
            results,
            duration_ms: start.elapsed().as_millis() as u64,
            threshold_used,
            threshold_lowered,
        };

        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(description = "Get statistics about the indexed codebase")]
    async fn get_statistics(
        &self,
        Parameters(_req): Parameters<StatisticsRequest>,
    ) -> Result<String, String> {
        let stats = self
            .vector_db
            .get_statistics()
            .await
            .map_err(|e| format!("Failed to get statistics: {}", e))?;

        let language_breakdown = stats
            .language_breakdown
            .into_iter()
            .map(|(language, count)| LanguageStats {
                language,
                file_count: count,
                chunk_count: count,
            })
            .collect();

        let response = StatisticsResponse {
            total_files: stats.total_points,
            total_chunks: stats.total_vectors,
            total_embeddings: stats.total_vectors,
            database_size_bytes: 0, // Would need to query Qdrant storage
            language_breakdown,
        };

        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(description = "Clear all indexed data from the vector database")]
    async fn clear_index(
        &self,
        Parameters(_req): Parameters<ClearRequest>,
    ) -> Result<String, String> {
        let response = match self.vector_db.clear().await {
            Ok(_) => {
                // Clear the persistent cache
                let mut cache = self.hash_cache.write().await;
                cache.roots.clear();

                // Persist to disk
                if let Err(e) = cache.save(&self.cache_path) {
                    tracing::warn!("Failed to save cleared cache: {}", e);
                }

                // Re-initialize the collection
                if let Err(e) = self
                    .vector_db
                    .initialize(self.embedding_provider.dimension())
                    .await
                {
                    ClearResponse {
                        success: false,
                        message: format!("Cleared but failed to reinitialize: {}", e),
                    }
                } else {
                    ClearResponse {
                        success: true,
                        message: "Successfully cleared all indexed data and cache".to_string(),
                    }
                }
            }
            Err(e) => ClearResponse {
                success: false,
                message: format!("Failed to clear index: {}", e),
            },
        };

        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    // #[tool(
    //     description = "DEPRECATED: Use index_codebase instead, which automatically performs incremental updates. This tool is kept for backward compatibility."
    // )]
    // async fn incremental_update(
    //     &self,
    //     meta: Meta,
    //     peer: Peer<RoleServer>,
    //     Parameters(req): Parameters<IncrementalUpdateRequest>,
    // ) -> Result<String, String> {
    //     tracing::warn!(
    //         "incremental_update is deprecated, use index_codebase instead which auto-detects the mode"
    //     );

    //     // Get progress token if provided
    //     let progress_token = meta.get_progress_token();

    //     // Convert IncrementalUpdateRequest to IndexRequest
    //     let index_req = IndexRequest {
    //         path: req.path,
    //         project: req.project,
    //         include_patterns: req.include_patterns,
    //         exclude_patterns: req.exclude_patterns,
    //         max_file_size: 1_048_576, // Use default
    //     };

    //     // Call the smart index method which will detect that this is an incremental update
    //     let response = self
    //         .do_index_smart(
    //             index_req.path,
    //             index_req.project,
    //             index_req.include_patterns,
    //             index_req.exclude_patterns,
    //             index_req.max_file_size,
    //             Some(peer),
    //             progress_token,
    //         )
    //         .await
    //         .map_err(|e| format!("{:#}", e))?;

    //     // Convert IndexResponse to IncrementalUpdateResponse for backward compatibility
    //     let compat_response = IncrementalUpdateResponse {
    //         files_added: response.files_indexed,
    //         files_updated: response.files_updated,
    //         files_removed: response.files_removed,
    //         chunks_modified: response.chunks_created,
    //         duration_ms: response.duration_ms,
    //     };

    //     serde_json::to_string_pretty(&compat_response)
    //         .map_err(|e| format!("Serialization failed: {}", e))
    // }

    #[tool(description = "Advanced search with filters for file type, language, and path patterns")]
    async fn search_by_filters(
        &self,
        Parameters(req): Parameters<AdvancedSearchRequest>,
    ) -> Result<String, String> {
        // Validate request inputs
        req.validate()?;

        let start = Instant::now();

        // Generate query embedding
        let query_embedding = self
            .embedding_provider
            .embed_batch(vec![req.query.clone()])
            .map_err(|e| format!("Failed to generate query embedding: {}", e))?
            .into_iter()
            .next()
            .ok_or("No embedding generated")?;

        // Adaptive search: try with requested threshold, then progressively lower if no results
        let original_threshold = req.min_score;
        let mut threshold_used = original_threshold;
        let mut threshold_lowered = false;

        // Try original threshold first
        let mut results = self
            .vector_db
            .search_filtered(
                query_embedding.clone(),
                &req.query,
                req.limit,
                threshold_used,
                req.project.clone(),
                true, // Always use hybrid for advanced search
                req.file_extensions.clone(),
                req.languages.clone(),
                req.path_patterns.clone(),
            )
            .await
            .map_err(|e| format!("Failed to search: {}", e))?;

        // If no results and threshold is high, try progressively lower thresholds
        if results.is_empty() && original_threshold > 0.3 {
            let fallback_thresholds = [0.6, 0.5, 0.4, 0.3];

            for &threshold in &fallback_thresholds {
                if threshold >= original_threshold {
                    continue;
                }

                tracing::info!(
                    "No filtered results with threshold {}, trying {}",
                    threshold_used,
                    threshold
                );

                results = self
                    .vector_db
                    .search_filtered(
                        query_embedding.clone(),
                        &req.query,
                        req.limit,
                        threshold,
                        req.project.clone(),
                        true,
                        req.file_extensions.clone(),
                        req.languages.clone(),
                        req.path_patterns.clone(),
                    )
                    .await
                    .map_err(|e| format!("Failed to search: {}", e))?;

                if !results.is_empty() {
                    threshold_used = threshold;
                    threshold_lowered = true;
                    tracing::info!(
                        "Found {} filtered results with lowered threshold {}",
                        results.len(),
                        threshold
                    );
                    break;
                }
            }
        }

        let response = QueryResponse {
            results,
            duration_ms: start.elapsed().as_millis() as u64,
            threshold_used,
            threshold_lowered,
        };

        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(description = "Search git commit history using semantic search with on-demand indexing")]
    async fn search_git_history(
        &self,
        Parameters(req): Parameters<SearchGitHistoryRequest>,
    ) -> Result<String, String> {
        // Validate request inputs
        req.validate()?;

        let response = git_indexing::do_search_git_history(
            self.embedding_provider.clone(),
            self.vector_db.clone(),
            self.git_cache.clone(),
            &self.git_cache_path,
            req,
        )
        .await
        .map_err(|e| format!("{:#}", e))?;

        serde_json::to_string_pretty(&response).map_err(|e| format!("Serialization failed: {}", e))
    }
}

// Prompts for slash commands
#[prompt_router]
impl RagMcpServer {
    #[prompt(
        name = "index",
        description = "Index a codebase directory to enable semantic search (automatically performs full or incremental based on existing index)"
    )]
    async fn index_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Result<GetPromptResult, McpError> {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        let messages = vec![PromptMessage::new_text(
            PromptMessageRole::User,
            format!(
                "Please index the codebase at path: '{}'. This will automatically perform a full index if this is the first time, or an incremental update if the codebase has been indexed before.",
                path
            ),
        )];

        Ok(GetPromptResult {
            description: Some(format!(
                "Index codebase at {} (auto-detects full/incremental)",
                path
            )),
            messages,
        })
    }

    #[prompt(
        name = "query",
        description = "Search the indexed codebase using semantic search"
    )]
    async fn query_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Result<Vec<PromptMessage>, McpError> {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");

        Ok(vec![PromptMessage::new_text(
            PromptMessageRole::User,
            format!("Please search the codebase for: {}", query),
        )])
    }

    #[prompt(
        name = "stats",
        description = "Get statistics about the indexed codebase"
    )]
    async fn stats_prompt(&self) -> Vec<PromptMessage> {
        vec![PromptMessage::new_text(
            PromptMessageRole::User,
            "Please get statistics about the indexed codebase.",
        )]
    }

    #[prompt(
        name = "clear",
        description = "Clear all indexed data from the vector database"
    )]
    async fn clear_prompt(&self) -> Vec<PromptMessage> {
        vec![PromptMessage::new_text(
            PromptMessageRole::User,
            "Please clear all indexed data from the vector database.",
        )]
    }

    // #[prompt(
    //     name = "update",
    //     description = "DEPRECATED: Use /index instead. This command is kept for backward compatibility."
    // )]
    // async fn update_prompt(
    //     &self,
    //     Parameters(args): Parameters<serde_json::Value>,
    // ) -> Result<Vec<PromptMessage>, McpError> {
    //     let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");

    //     Ok(vec![PromptMessage::new_text(
    //         PromptMessageRole::User,
    //         format!(
    //             "Please index the codebase at path: {}. Note: The /update command is deprecated - use /index instead, which automatically detects whether to do a full index or incremental update.",
    //             path
    //         ),
    //     )])
    // }

    #[prompt(
        name = "search",
        description = "Advanced search with filters (file type, language, path)"
    )]
    async fn search_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Result<Vec<PromptMessage>, McpError> {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");

        Ok(vec![PromptMessage::new_text(
            PromptMessageRole::User,
            format!("Please perform an advanced search for: {}", query),
        )])
    }

    #[prompt(
        name = "git-search",
        description = "Search git commit history using semantic search (automatically indexes commits on-demand)"
    )]
    async fn git_search_prompt(
        &self,
        Parameters(args): Parameters<serde_json::Value>,
    ) -> Result<Vec<PromptMessage>, McpError> {
        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        Ok(vec![PromptMessage::new_text(
            PromptMessageRole::User,
            format!(
                "Please search git commit history at path '{}' for: {}. This will automatically index commits as needed.",
                path, query
            ),
        )])
    }
}

#[tool_handler(router = self.tool_router)]
#[prompt_handler]
impl ServerHandler for RagMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::default(),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .enable_prompts()
                .build(),
            server_info: Implementation {
                name: "project".into(),
                title: Some("Project RAG - Code Understanding with Semantic Search".into()),
                version: env!("CARGO_PKG_VERSION").into(),
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "RAG-based codebase indexing and semantic search. \
                Use index_codebase to create embeddings (automatically performs full or incremental indexing), \
                query_codebase to search, and search_by_filters for advanced queries."
                    .into(),
            ),
        }
    }
}

impl RagMcpServer {
    pub async fn serve_stdio() -> Result<()> {
        tracing::info!("Starting RAG MCP server");

        let server = Self::new().await.context("Failed to create MCP server")?;

        let transport = rmcp::transport::io::stdio();

        server.serve(transport).await?.waiting().await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests;
