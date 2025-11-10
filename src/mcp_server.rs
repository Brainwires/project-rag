use crate::embedding::{EmbeddingProvider, FastEmbedManager};
use crate::indexer::{CodeChunker, FileWalker};
use crate::types::*;
use crate::vector_db::{QdrantVectorDB, VectorDatabase};
use anyhow::{Context, Result};
use rmcp::{
    handler::server::{ServerHandler, tool::ToolRouter, wrapper::Parameters},
    model::{Implementation, ProtocolVersion, ServerCapabilities, ServerInfo},
    service::ServiceExt,
    tool, tool_handler, tool_router,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct RagMcpServer {
    embedding_provider: Arc<FastEmbedManager>,
    vector_db: Arc<QdrantVectorDB>,
    chunker: Arc<CodeChunker>,
    // Track indexed roots for incremental updates
    indexed_roots: Arc<RwLock<HashMap<String, HashMap<String, String>>>>,
    tool_router: ToolRouter<Self>,
}

impl RagMcpServer {
    pub async fn new() -> Result<Self> {
        let embedding_provider = Arc::new(
            FastEmbedManager::new().context("Failed to initialize embedding provider")?,
        );

        let vector_db = Arc::new(
            QdrantVectorDB::new()
                .await
                .context("Failed to initialize vector database")?,
        );

        // Initialize the database with the embedding dimension
        vector_db
            .initialize(embedding_provider.dimension())
            .await
            .context("Failed to initialize vector database collections")?;

        let chunker = Arc::new(CodeChunker::default_strategy());

        Ok(Self {
            embedding_provider,
            vector_db,
            chunker,
            indexed_roots: Arc::new(RwLock::new(HashMap::new())),
            tool_router: Self::tool_router(),
        })
    }

    /// Index a complete codebase
    async fn do_index(
        &self,
        path: String,
        include_patterns: Vec<String>,
        exclude_patterns: Vec<String>,
        max_file_size: usize,
    ) -> Result<IndexResponse> {
        let start = Instant::now();
        let mut errors = Vec::new();

        // Walk the directory
        let walker = FileWalker::new(&path, max_file_size)
            .with_patterns(include_patterns.clone(), exclude_patterns.clone());

        let files = walker.walk().context("Failed to walk directory")?;
        let files_indexed = files.len();

        // Chunk all files
        let mut all_chunks = Vec::new();
        for file in &files {
            all_chunks.extend(self.chunker.chunk_file(file));
        }

        let chunks_created = all_chunks.len();

        if all_chunks.is_empty() {
            return Ok(IndexResponse {
                files_indexed: 0,
                chunks_created: 0,
                embeddings_generated: 0,
                duration_ms: start.elapsed().as_millis() as u64,
                errors: vec!["No code chunks found to index".to_string()],
            });
        }

        // Generate embeddings in batches
        let batch_size = 32;
        let mut all_embeddings = Vec::new();

        for chunk_batch in all_chunks.chunks(batch_size) {
            let texts: Vec<String> = chunk_batch.iter().map(|c| c.content.clone()).collect();

            match self.embedding_provider.embed_batch(texts) {
                Ok(embeddings) => all_embeddings.extend(embeddings),
                Err(e) => {
                    errors.push(format!("Failed to generate embeddings: {}", e));
                    continue;
                }
            }
        }

        let embeddings_generated = all_embeddings.len();

        // Store in vector database
        let metadata: Vec<ChunkMetadata> = all_chunks
            .iter()
            .map(|c| c.metadata.clone())
            .collect();
        let contents: Vec<String> = all_chunks.iter().map(|c| c.content.clone()).collect();

        self.vector_db
            .store_embeddings(all_embeddings, metadata, contents)
            .await
            .context("Failed to store embeddings")?;

        // Save file hashes for incremental updates
        let mut indexed_roots = self.indexed_roots.write().await;
        let file_hashes: HashMap<String, String> = files
            .iter()
            .map(|f| (f.relative_path.clone(), f.hash.clone()))
            .collect();
        indexed_roots.insert(path, file_hashes);

        Ok(IndexResponse {
            files_indexed,
            chunks_created,
            embeddings_generated,
            duration_ms: start.elapsed().as_millis() as u64,
            errors,
        })
    }
}

#[tool_router(router = tool_router)]
impl RagMcpServer {
    #[tool(description = "Index a codebase directory, creating embeddings for semantic search")]
    async fn index_codebase(
        &self,
        Parameters(req): Parameters<IndexRequest>,
    ) -> Result<String, String> {
        let response = self.do_index(
            req.path,
            req.include_patterns,
            req.exclude_patterns,
            req.max_file_size,
        )
        .await
        .map_err(|e| e.to_string())?;

        serde_json::to_string_pretty(&response)
            .map_err(|e| format!("Serialization failed: {}", e))
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
            .embed_batch(vec![req.query])
            .map_err(|e| format!("Failed to generate query embedding: {}", e))?
            .into_iter()
            .next()
            .ok_or("No embedding generated")?;

        // Search vector database
        let results = self
            .vector_db
            .search(query_embedding, req.limit, req.min_score)
            .await
            .map_err(|e| format!("Failed to search: {}", e))?;

        let response = QueryResponse {
            results,
            duration_ms: start.elapsed().as_millis() as u64,
        };

        serde_json::to_string_pretty(&response)
            .map_err(|e| format!("Serialization failed: {}", e))
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

        serde_json::to_string_pretty(&response)
            .map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(description = "Clear all indexed data from the vector database")]
    async fn clear_index(&self, Parameters(_req): Parameters<ClearRequest>) -> Result<String, String> {
        let response = match self.vector_db.clear().await {
            Ok(_) => {
                // Clear the indexed roots cache
                let mut indexed_roots = self.indexed_roots.write().await;
                indexed_roots.clear();

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
                        message: "Successfully cleared all indexed data".to_string(),
                    }
                }
            }
            Err(e) => ClearResponse {
                success: false,
                message: format!("Failed to clear index: {}", e),
            },
        };

        serde_json::to_string_pretty(&response)
            .map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(description = "Incrementally update the index by processing only changed files")]
    async fn incremental_update(
        &self,
        Parameters(req): Parameters<IncrementalUpdateRequest>,
    ) -> Result<String, String> {
        let start = Instant::now();

        // Get existing file hashes
        let indexed_roots = self.indexed_roots.read().await;
        let existing_hashes = indexed_roots
            .get(&req.path)
            .cloned()
            .unwrap_or_default();
        drop(indexed_roots);

        // Walk directory to find current files
        let walker = FileWalker::new(&req.path, 1_048_576)
            .with_patterns(req.include_patterns.clone(), req.exclude_patterns.clone());

        let current_files = walker
            .walk()
            .map_err(|e| format!("Failed to walk directory: {}", e))?;

        let mut files_added = 0;
        let mut files_updated = 0;
        let mut files_removed = 0;
        let mut chunks_modified = 0;

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

        // Index new/modified files
        if !files_to_index.is_empty() {
            let mut all_chunks = Vec::new();
            for file in &files_to_index {
                all_chunks.extend(self.chunker.chunk_file(file));
            }

            chunks_modified = all_chunks.len();

            // Generate embeddings and store
            let batch_size = 32;
            for chunk_batch in all_chunks.chunks(batch_size) {
                let texts: Vec<String> = chunk_batch.iter().map(|c| c.content.clone()).collect();

                let embeddings = self
                    .embedding_provider
                    .embed_batch(texts)
                    .map_err(|e| format!("Failed to generate embeddings: {}", e))?;

                let metadata: Vec<ChunkMetadata> =
                    chunk_batch.iter().map(|c| c.metadata.clone()).collect();
                let contents: Vec<String> = chunk_batch.iter().map(|c| c.content.clone()).collect();

                self.vector_db
                    .store_embeddings(embeddings, metadata, contents)
                    .await
                    .map_err(|e| format!("Failed to store embeddings: {}", e))?;
            }
        }

        // Update cached hashes
        let mut indexed_roots = self.indexed_roots.write().await;
        indexed_roots.insert(req.path, new_hashes);

        let response = IncrementalUpdateResponse {
            files_added,
            files_updated,
            files_removed,
            chunks_modified,
            duration_ms: start.elapsed().as_millis() as u64,
        };

        serde_json::to_string_pretty(&response)
            .map_err(|e| format!("Serialization failed: {}", e))
    }

    #[tool(description = "Advanced search with filters for file type, language, and path patterns")]
    async fn search_by_filters(
        &self,
        Parameters(req): Parameters<AdvancedSearchRequest>,
    ) -> Result<String, String> {
        let start = Instant::now();

        // Generate query embedding
        let query_embedding = self
            .embedding_provider
            .embed_batch(vec![req.query])
            .map_err(|e| format!("Failed to generate query embedding: {}", e))?
            .into_iter()
            .next()
            .ok_or("No embedding generated")?;

        // Search with filters
        let results = self
            .vector_db
            .search_filtered(
                query_embedding,
                req.limit,
                req.min_score,
                req.file_extensions,
                req.languages,
                req.path_patterns,
            )
            .await
            .map_err(|e| format!("Failed to search: {}", e))?;

        let response = QueryResponse {
            results,
            duration_ms: start.elapsed().as_millis() as u64,
        };

        serde_json::to_string_pretty(&response)
            .map_err(|e| format!("Serialization failed: {}", e))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for RagMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::default(),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "project-rag".into(),
                title: Some("Project RAG - Code Understanding with Semantic Search".into()),
                version: env!("CARGO_PKG_VERSION").into(),
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "RAG-based codebase indexing and semantic search. \
                Use index_codebase to create embeddings, query_codebase to search, \
                incremental_update for changed files, and search_by_filters for advanced queries."
                    .into(),
            ),
        }
    }
}

impl RagMcpServer {
    pub async fn serve_stdio() -> Result<()> {
        tracing::info!("Starting RAG MCP server");

        let server = Self::new()
            .await
            .context("Failed to create MCP server")?;

        let transport = rmcp::transport::io::stdio();

        server.serve(transport).await?.waiting().await?;

        Ok(())
    }
}
