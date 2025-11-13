use crate::cache::HashCache;
use crate::embedding::{EmbeddingProvider, FastEmbedManager};
use crate::indexer::{CodeChunker, FileWalker};
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
use std::collections::HashMap;
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
    tool_router: ToolRouter<Self>,
    prompt_router: PromptRouter<Self>,
}

impl RagMcpServer {
    pub async fn new() -> Result<Self> {
        let embedding_provider =
            Arc::new(FastEmbedManager::new().context("Failed to initialize embedding provider")?);

        // Initialize the appropriate vector database backend
        #[cfg(feature = "qdrant-backend")]
        let vector_db = {
            tracing::info!("Using Qdrant vector database backend");
            Arc::new(
                QdrantVectorDB::new()
                    .await
                    .context("Failed to initialize Qdrant vector database")?,
            )
        };

        #[cfg(not(feature = "qdrant-backend"))]
        let vector_db = {
            tracing::info!("Using LanceDB vector database backend (default)");
            Arc::new(
                LanceVectorDB::new()
                    .await
                    .context("Failed to initialize LanceDB vector database")?,
            )
        };

        // Initialize the database with the embedding dimension
        vector_db
            .initialize(embedding_provider.dimension())
            .await
            .context("Failed to initialize vector database collections")?;

        let chunker = Arc::new(CodeChunker::default_strategy());

        // Load persistent hash cache
        let cache_path = HashCache::default_path();
        let hash_cache = HashCache::load(&cache_path).unwrap_or_else(|e| {
            tracing::warn!("Failed to load hash cache: {}, starting fresh", e);
            HashCache::default()
        });

        tracing::info!("Using cache file: {:?}", cache_path);

        Ok(Self {
            embedding_provider,
            vector_db,
            chunker,
            hash_cache: Arc::new(RwLock::new(hash_cache)),
            cache_path,
            tool_router: Self::tool_router(),
            prompt_router: Self::prompt_router(),
        })
    }

    /// Create a new server with custom database path (for testing)
    #[cfg(test)]
    pub async fn new_with_db_path(db_path: &str, cache_path: PathBuf) -> Result<Self> {
        let embedding_provider =
            Arc::new(FastEmbedManager::new().context("Failed to initialize embedding provider")?);

        #[cfg(not(feature = "qdrant-backend"))]
        let vector_db = {
            Arc::new(
                LanceVectorDB::with_path(db_path)
                    .await
                    .context("Failed to initialize LanceDB vector database")?,
            )
        };

        #[cfg(feature = "qdrant-backend")]
        let vector_db = {
            Arc::new(
                QdrantVectorDB::new()
                    .await
                    .context("Failed to initialize Qdrant vector database")?,
            )
        };

        vector_db
            .initialize(embedding_provider.dimension())
            .await
            .context("Failed to initialize vector database collections")?;

        let chunker = Arc::new(CodeChunker::default_strategy());
        let hash_cache = HashCache::default();

        Ok(Self {
            embedding_provider,
            vector_db,
            chunker,
            hash_cache: Arc::new(RwLock::new(hash_cache)),
            cache_path,
            tool_router: Self::tool_router(),
            prompt_router: Self::prompt_router(),
        })
    }

    /// Normalize a path to a canonical absolute form for consistent cache lookups
    fn normalize_path(path: &str) -> Result<String> {
        let path_buf = PathBuf::from(path);
        let canonical = std::fs::canonicalize(&path_buf)
            .with_context(|| format!("Failed to canonicalize path: {}", path))?;
        Ok(canonical.to_string_lossy().to_string())
    }

    /// Index a complete codebase
    async fn do_index(
        &self,
        path: String,
        project: Option<String>,
        include_patterns: Vec<String>,
        exclude_patterns: Vec<String>,
        max_file_size: usize,
        peer: Option<Peer<RoleServer>>,
        progress_token: Option<ProgressToken>,
    ) -> Result<IndexResponse> {
        let start = Instant::now();
        let mut errors = Vec::new();

        // Send initial progress
        if let (Some(peer), Some(token)) = (&peer, &progress_token) {
            let _ = peer
                .notify_progress(ProgressNotificationParam {
                    progress_token: token.clone(),
                    progress: 0.0,
                    total: Some(100.0),
                    message: Some("Starting file walk...".into()),
                })
                .await;
        }

        // Walk the directory (on a blocking thread since it's CPU-intensive)
        let walker = FileWalker::new(&path, max_file_size)
            .with_project(project.clone())
            .with_patterns(include_patterns.clone(), exclude_patterns.clone());

        let files = tokio::task::spawn_blocking(move || walker.walk())
            .await
            .context("Failed to spawn file walker task")?
            .context("Failed to walk directory")?;
        let files_indexed = files.len();

        // Send progress after file walk
        if let (Some(peer), Some(token)) = (&peer, &progress_token) {
            let _ = peer
                .notify_progress(ProgressNotificationParam {
                    progress_token: token.clone(),
                    progress: 20.0,
                    total: Some(100.0),
                    message: Some(format!("Found {} files, chunking...", files_indexed)),
                })
                .await;
        }

        // Chunk all files
        let mut all_chunks = Vec::new();
        for file in &files {
            all_chunks.extend(self.chunker.chunk_file(file));
        }

        let chunks_created = all_chunks.len();

        // Send progress after chunking
        if let (Some(peer), Some(token)) = (&peer, &progress_token) {
            let _ = peer
                .notify_progress(ProgressNotificationParam {
                    progress_token: token.clone(),
                    progress: 40.0,
                    total: Some(100.0),
                    message: Some(format!(
                        "Created {} chunks, generating embeddings...",
                        chunks_created
                    )),
                })
                .await;
        }

        if all_chunks.is_empty() {
            return Ok(IndexResponse {
                mode: crate::types::IndexingMode::Full,
                files_indexed: 0,
                chunks_created: 0,
                embeddings_generated: 0,
                duration_ms: start.elapsed().as_millis() as u64,
                errors: vec!["No code chunks found to index".to_string()],
                files_updated: 0,
                files_removed: 0,
            });
        }

        // Generate embeddings in batches
        let batch_size = 32;
        let mut all_embeddings = Vec::new();
        let total_batches = (all_chunks.len() + batch_size - 1) / batch_size;

        for (batch_idx, chunk_batch) in all_chunks.chunks(batch_size).enumerate() {
            let texts: Vec<String> = chunk_batch.iter().map(|c| c.content.clone()).collect();

            match self.embedding_provider.embed_batch(texts) {
                Ok(embeddings) => all_embeddings.extend(embeddings),
                Err(e) => {
                    errors.push(format!("Failed to generate embeddings: {}", e));
                    continue;
                }
            }

            // Send progress during embedding (40% to 80%)
            if let (Some(peer), Some(token)) = (&peer, &progress_token) {
                let progress = 40.0 + ((batch_idx + 1) as f64 / total_batches as f64) * 40.0;
                let _ = peer
                    .notify_progress(ProgressNotificationParam {
                        progress_token: token.clone(),
                        progress,
                        total: Some(100.0),
                        message: Some(format!(
                            "Generating embeddings... {}/{} batches",
                            batch_idx + 1,
                            total_batches
                        )),
                    })
                    .await;
            }
        }

        let embeddings_generated = all_embeddings.len();

        // Send progress before storing
        if let (Some(peer), Some(token)) = (&peer, &progress_token) {
            let _ = peer
                .notify_progress(ProgressNotificationParam {
                    progress_token: token.clone(),
                    progress: 85.0,
                    total: Some(100.0),
                    message: Some(format!(
                        "Storing {} embeddings in database...",
                        embeddings_generated
                    )),
                })
                .await;
        }

        // Store in vector database
        let metadata: Vec<ChunkMetadata> = all_chunks.iter().map(|c| c.metadata.clone()).collect();
        let contents: Vec<String> = all_chunks.iter().map(|c| c.content.clone()).collect();

        self.vector_db
            .store_embeddings(all_embeddings, metadata, contents)
            .await
            .context("Failed to store embeddings")?;

        // Send progress before saving cache
        if let (Some(peer), Some(token)) = (&peer, &progress_token) {
            let _ = peer
                .notify_progress(ProgressNotificationParam {
                    progress_token: token.clone(),
                    progress: 95.0,
                    total: Some(100.0),
                    message: Some("Saving cache...".into()),
                })
                .await;
        }

        // Save file hashes to persistent cache
        let file_hashes: HashMap<String, String> = files
            .iter()
            .map(|f| (f.relative_path.clone(), f.hash.clone()))
            .collect();

        let mut cache = self.hash_cache.write().await;
        cache.update_root(path, file_hashes);

        // Persist to disk
        if let Err(e) = cache.save(&self.cache_path) {
            tracing::warn!("Failed to save hash cache: {}", e);
        }

        // Send progress before flush
        if let (Some(peer), Some(token)) = (&peer, &progress_token) {
            let _ = peer
                .notify_progress(ProgressNotificationParam {
                    progress_token: token.clone(),
                    progress: 98.0,
                    total: Some(100.0),
                    message: Some("Flushing index to disk...".into()),
                })
                .await;
        }

        // Flush the index to disk
        self.vector_db
            .flush()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to flush index to disk: {}", e))?;

        // Send final completion progress
        if let (Some(peer), Some(token)) = (&peer, &progress_token) {
            let _ = peer
                .notify_progress(ProgressNotificationParam {
                    progress_token: token.clone(),
                    progress: 100.0,
                    total: Some(100.0),
                    message: Some("Indexing complete!".into()),
                })
                .await;
        }

        Ok(IndexResponse {
            mode: crate::types::IndexingMode::Full,
            files_indexed,
            chunks_created,
            embeddings_generated,
            duration_ms: start.elapsed().as_millis() as u64,
            errors,
            files_updated: 0,
            files_removed: 0,
        })
    }

    /// Perform incremental update (only changed files)
    async fn do_incremental_update(
        &self,
        path: String,
        project: Option<String>,
        include_patterns: Vec<String>,
        exclude_patterns: Vec<String>,
        max_file_size: usize,
        peer: Option<Peer<RoleServer>>,
        progress_token: Option<ProgressToken>,
    ) -> Result<IndexResponse> {
        let start = Instant::now();

        // Send initial progress
        if let (Some(peer), Some(token)) = (&peer, &progress_token) {
            let _ = peer
                .notify_progress(ProgressNotificationParam {
                    progress_token: token.clone(),
                    progress: 0.0,
                    total: Some(100.0),
                    message: Some("Checking for changes...".into()),
                })
                .await;
        }

        // Get existing file hashes from persistent cache
        let cache = self.hash_cache.read().await;
        let existing_hashes = cache.get_root(&path).cloned().unwrap_or_default();
        drop(cache);

        // Send progress after reading cache
        if let (Some(peer), Some(token)) = (&peer, &progress_token) {
            let _ = peer
                .notify_progress(ProgressNotificationParam {
                    progress_token: token.clone(),
                    progress: 10.0,
                    total: Some(100.0),
                    message: Some(format!(
                        "Found {} cached files, scanning directory...",
                        existing_hashes.len()
                    )),
                })
                .await;
        }

        // Walk directory to find current files (on a blocking thread)
        let walker = FileWalker::new(&path, max_file_size)
            .with_project(project.clone())
            .with_patterns(include_patterns.clone(), exclude_patterns.clone());

        let current_files = tokio::task::spawn_blocking(move || walker.walk())
            .await
            .context("Failed to spawn file walker task")?
            .context("Failed to walk directory")?;

        let mut files_added = 0;
        let mut files_updated = 0;
        let mut files_removed = 0;
        let mut chunks_modified = 0;

        // Send progress after file walk
        if let (Some(peer), Some(token)) = (&peer, &progress_token) {
            let _ = peer
                .notify_progress(ProgressNotificationParam {
                    progress_token: token.clone(),
                    progress: 30.0,
                    total: Some(100.0),
                    message: Some(format!(
                        "Found {} files, comparing with cache...",
                        current_files.len()
                    )),
                })
                .await;
        }

        // Find new and modified files
        let mut new_hashes = HashMap::new();
        let mut files_to_index = Vec::new();

        for file in current_files {
            new_hashes.insert(file.relative_path.clone(), file.hash.clone());

            match existing_hashes.get(&file.relative_path) {
                None => {
                    // New file
                    files_added += 1;
                    files_to_index.push(file);
                }
                Some(old_hash) if old_hash != &file.hash => {
                    // Modified file - delete old embeddings first
                    if let Err(e) = self.vector_db.delete_by_file(&file.relative_path).await {
                        tracing::warn!("Failed to delete old embeddings: {}", e);
                    }
                    files_updated += 1;
                    files_to_index.push(file);
                }
                _ => {
                    // Unchanged file, skip
                }
            }
        }

        // Find removed files
        for old_file in existing_hashes.keys() {
            if !new_hashes.contains_key(old_file) {
                files_removed += 1;
                if let Err(e) = self.vector_db.delete_by_file(old_file).await {
                    tracing::warn!("Failed to delete embeddings for removed file: {}", e);
                }
            }
        }

        // Send progress after identifying changes
        if let (Some(peer), Some(token)) = (&peer, &progress_token) {
            let _ = peer
                .notify_progress(ProgressNotificationParam {
                    progress_token: token.clone(),
                    progress: 50.0,
                    total: Some(100.0),
                    message: Some(format!(
                        "Processing {} changed files...",
                        files_to_index.len()
                    )),
                })
                .await;
        }

        // Index new/modified files
        let embeddings_generated = if !files_to_index.is_empty() {
            let mut all_chunks = Vec::new();
            for file in &files_to_index {
                all_chunks.extend(self.chunker.chunk_file(file));
            }

            chunks_modified = all_chunks.len();

            // Send progress after chunking
            if let (Some(peer), Some(token)) = (&peer, &progress_token) {
                let _ = peer
                    .notify_progress(ProgressNotificationParam {
                        progress_token: token.clone(),
                        progress: 60.0,
                        total: Some(100.0),
                        message: Some(format!(
                            "Created {} chunks, generating embeddings...",
                            chunks_modified
                        )),
                    })
                    .await;
            }

            // Generate embeddings in batches
            let batch_size = 32;
            let mut all_embeddings = Vec::new();
            let total_batches = (all_chunks.len() + batch_size - 1) / batch_size;

            for (batch_idx, chunk_batch) in all_chunks.chunks(batch_size).enumerate() {
                let texts: Vec<String> = chunk_batch.iter().map(|c| c.content.clone()).collect();

                let embeddings = self
                    .embedding_provider
                    .embed_batch(texts)
                    .context("Failed to generate embeddings")?;

                all_embeddings.extend(embeddings);

                // Send progress during embedding (60% to 85%)
                if let (Some(peer), Some(token)) = (&peer, &progress_token) {
                    let progress = 60.0 + ((batch_idx + 1) as f64 / total_batches as f64) * 25.0;
                    let _ = peer
                        .notify_progress(ProgressNotificationParam {
                            progress_token: token.clone(),
                            progress,
                            total: Some(100.0),
                            message: Some(format!(
                                "Generating embeddings... {}/{} batches",
                                batch_idx + 1,
                                total_batches
                            )),
                        })
                        .await;
                }
            }

            // Send progress before storing
            if let (Some(peer), Some(token)) = (&peer, &progress_token) {
                let _ = peer
                    .notify_progress(ProgressNotificationParam {
                        progress_token: token.clone(),
                        progress: 90.0,
                        total: Some(100.0),
                        message: Some(format!("Storing {} embeddings...", all_embeddings.len())),
                    })
                    .await;
            }

            // Store all embeddings
            let metadata: Vec<ChunkMetadata> =
                all_chunks.iter().map(|c| c.metadata.clone()).collect();
            let contents: Vec<String> = all_chunks.iter().map(|c| c.content.clone()).collect();

            self.vector_db
                .store_embeddings(all_embeddings.clone(), metadata, contents)
                .await
                .context("Failed to store embeddings")?;

            all_embeddings.len()
        } else {
            0
        };

        // Send progress before saving cache
        if let (Some(peer), Some(token)) = (&peer, &progress_token) {
            let _ = peer
                .notify_progress(ProgressNotificationParam {
                    progress_token: token.clone(),
                    progress: 95.0,
                    total: Some(100.0),
                    message: Some("Saving cache...".into()),
                })
                .await;
        }

        // Update persistent cache
        let mut cache = self.hash_cache.write().await;
        cache.update_root(path, new_hashes);

        // Persist to disk
        if let Err(e) = cache.save(&self.cache_path) {
            tracing::warn!("Failed to save hash cache: {}", e);
        }
        drop(cache);

        // Send progress before flush
        if let (Some(peer), Some(token)) = (&peer, &progress_token) {
            let _ = peer
                .notify_progress(ProgressNotificationParam {
                    progress_token: token.clone(),
                    progress: 98.0,
                    total: Some(100.0),
                    message: Some("Flushing index to disk...".into()),
                })
                .await;
        }

        // Flush the vector database to disk
        self.vector_db
            .flush()
            .await
            .context("Failed to flush index to disk")?;

        // Send final completion progress
        if let (Some(peer), Some(token)) = (&peer, &progress_token) {
            let _ = peer
                .notify_progress(ProgressNotificationParam {
                    progress_token: token.clone(),
                    progress: 100.0,
                    total: Some(100.0),
                    message: Some("Incremental update complete!".into()),
                })
                .await;
        }

        Ok(IndexResponse {
            mode: crate::types::IndexingMode::Incremental,
            files_indexed: files_added,
            chunks_created: chunks_modified,
            embeddings_generated,
            duration_ms: start.elapsed().as_millis() as u64,
            errors: vec![],
            files_updated,
            files_removed,
        })
    }

    /// Smart index that automatically chooses between full and incremental based on existing cache
    async fn do_index_smart(
        &self,
        path: String,
        project: Option<String>,
        include_patterns: Vec<String>,
        exclude_patterns: Vec<String>,
        max_file_size: usize,
        peer: Option<Peer<RoleServer>>,
        progress_token: Option<ProgressToken>,
    ) -> Result<IndexResponse> {
        // Normalize path to canonical form for consistent cache lookups
        let normalized_path = Self::normalize_path(&path)?;

        // Check if we have an existing cache for this path
        let cache = self.hash_cache.read().await;
        let has_existing_index = cache.get_root(&normalized_path).is_some();
        drop(cache);

        if has_existing_index {
            tracing::info!(
                "Existing index found for '{}' (normalized: '{}'), performing incremental update",
                path,
                normalized_path
            );
            self.do_incremental_update(
                normalized_path,
                project,
                include_patterns,
                exclude_patterns,
                max_file_size,
                peer,
                progress_token,
            )
            .await
        } else {
            tracing::info!(
                "No existing index found for '{}' (normalized: '{}'), performing full index",
                path,
                normalized_path
            );
            self.do_index(
                normalized_path,
                project,
                include_patterns,
                exclude_patterns,
                max_file_size,
                peer,
                progress_token,
            )
            .await
        }
    }
}

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
                "Please index the codebase at path: {}. This will automatically perform a full index if this is the first time, or an incremental update if the codebase has been indexed before.",
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
