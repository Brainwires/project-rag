use super::RagMcpServer;
use crate::embedding::EmbeddingProvider;
use crate::indexer::FileWalker;
use crate::types::{ChunkMetadata, IndexResponse};
use crate::vector_db::VectorDatabase;
use anyhow::{Context, Result};
use rayon::prelude::*;
use rmcp::{Peer, RoleServer, model::ProgressToken, model::ProgressNotificationParam};
use std::collections::HashMap;
use std::time::Instant;

impl RagMcpServer {
    /// Index a complete codebase
    pub async fn do_index(
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

        // Chunk all files in parallel for better performance
        let chunker = self.chunker.clone();
        let all_chunks: Vec<_> = files
            .par_iter()
            .flat_map(|file| chunker.chunk_file(file))
            .collect();

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

        // Generate embeddings in batches (using config values)
        let batch_size = self.config.embedding.batch_size;
        let timeout_secs = self.config.embedding.timeout_secs;
        let mut all_embeddings = Vec::with_capacity(all_chunks.len());
        let total_batches = (all_chunks.len() + batch_size - 1) / batch_size;

        for (batch_idx, chunk_batch) in all_chunks.chunks(batch_size).enumerate() {
            let texts: Vec<String> = chunk_batch.iter().map(|c| c.content.clone()).collect();

            // Generate embeddings with timeout protection (configurable)
            let provider = self.embedding_provider.clone();
            let embed_future = tokio::task::spawn_blocking(move || provider.embed_batch(texts));

            match tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), embed_future)
                .await
            {
                Ok(Ok(Ok(embeddings))) => all_embeddings.extend(embeddings),
                Ok(Ok(Err(e))) => {
                    errors.push(format!("Failed to generate embeddings: {}", e));
                    continue;
                }
                Ok(Err(e)) => {
                    errors.push(format!("Embedding task panicked: {}", e));
                    continue;
                }
                Err(_) => {
                    errors.push(format!(
                        "Embedding generation timed out after {} seconds",
                        timeout_secs
                    ));
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
    pub(super) async fn do_incremental_update(
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
        let mut new_hashes = HashMap::with_capacity(current_files.len());
        let mut files_to_index = Vec::with_capacity(current_files.len());

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
            // Chunk files in parallel for better performance
            let chunker = self.chunker.clone();
            let all_chunks: Vec<_> = files_to_index
                .par_iter()
                .flat_map(|file| chunker.chunk_file(file))
                .collect();

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

            // Generate embeddings in batches (using config values)
            let batch_size = self.config.embedding.batch_size;
            let mut all_embeddings = Vec::with_capacity(all_chunks.len());
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
    pub(super) async fn do_index_smart(
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
