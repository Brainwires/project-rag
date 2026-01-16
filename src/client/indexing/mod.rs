use super::RagClient;
use crate::embedding::EmbeddingProvider;
use crate::indexer::{CodeChunk, FileWalker};
use crate::types::{ChunkMetadata, IndexResponse};
use crate::vector_db::VectorDatabase;
use anyhow::{Context, Result};
use rayon::prelude::*;
use rmcp::{Peer, RoleServer, model::ProgressNotificationParam, model::ProgressToken};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

/// Helper macro to check for cancellation and return early if cancelled
macro_rules! check_cancelled {
    ($cancel_token:expr) => {
        if $cancel_token.is_cancelled() {
            tracing::info!("Indexing operation cancelled");
            anyhow::bail!("Indexing was cancelled");
        }
    };
}

/// Result of embedding generation with cancellation support
struct EmbeddingResult {
    embeddings: Vec<Vec<f32>>,
    successful_chunks: Vec<CodeChunk>,
    errors: Vec<String>,
}

/// Generate embeddings for chunks with frequent cancellation checks
///
/// This function processes chunks in small batches and checks for cancellation
/// between each batch, allowing for faster response to cancellation requests.
async fn generate_embeddings_with_cancellation(
    client: &RagClient,
    chunks: &[CodeChunk],
    cancel_token: &CancellationToken,
    peer: &Option<Peer<RoleServer>>,
    progress_token: &Option<ProgressToken>,
    progress_start: f64,
    progress_end: f64,
) -> Result<EmbeddingResult> {
    let batch_size = client.config.embedding.batch_size;
    let timeout_secs = client.config.embedding.timeout_secs;
    let check_interval = if client.config.embedding.cancellation_check_interval > 0 {
        client.config.embedding.cancellation_check_interval
    } else {
        batch_size // Fall back to batch size if interval is 0
    };

    let mut all_embeddings = Vec::with_capacity(chunks.len());
    let mut successful_chunks = Vec::with_capacity(chunks.len());
    let mut errors = Vec::new();

    let total_batches = chunks.len().div_ceil(batch_size);
    let mut chunks_processed = 0;

    for (batch_idx, chunk_batch) in chunks.chunks(batch_size).enumerate() {
        // Check for cancellation at start of each batch
        if cancel_token.is_cancelled() {
            tracing::info!(
                "Embedding generation cancelled after {} chunks",
                chunks_processed
            );
            anyhow::bail!("Indexing was cancelled");
        }

        // Process batch in smaller sub-batches for more frequent cancellation checks
        let mut batch_embeddings = Vec::new();
        let mut batch_successful_chunks = Vec::new();

        for sub_batch in chunk_batch.chunks(check_interval) {
            // Check cancellation before each sub-batch
            if cancel_token.is_cancelled() {
                tracing::info!(
                    "Embedding generation cancelled during batch {} after {} chunks",
                    batch_idx,
                    chunks_processed
                );
                anyhow::bail!("Indexing was cancelled");
            }

            let texts: Vec<String> = sub_batch.iter().map(|c| c.content.clone()).collect();

            // Generate embeddings with timeout protection
            let provider = client.embedding_provider.clone();
            let embed_future = tokio::task::spawn_blocking(move || provider.embed_batch(texts));

            match tokio::time::timeout(
                std::time::Duration::from_secs(timeout_secs),
                embed_future,
            )
            .await
            {
                Ok(Ok(Ok(embeddings))) => {
                    batch_embeddings.extend(embeddings);
                    batch_successful_chunks.extend(sub_batch.iter().cloned());
                    chunks_processed += sub_batch.len();
                }
                Ok(Ok(Err(e))) => {
                    errors.push(format!(
                        "Failed to generate embeddings for sub-batch: {}",
                        e
                    ));
                    // Continue with next sub-batch
                }
                Ok(Err(e)) => {
                    errors.push(format!("Embedding task panicked: {}", e));
                    // Continue with next sub-batch
                }
                Err(_) => {
                    errors.push(format!(
                        "Embedding generation timed out after {} seconds",
                        timeout_secs
                    ));
                    // Continue with next sub-batch
                }
            }
        }

        // Add batch results to overall results
        all_embeddings.extend(batch_embeddings);
        successful_chunks.extend(batch_successful_chunks);

        // Send progress during embedding
        if let (Some(peer), Some(token)) = (peer, progress_token) {
            let progress =
                progress_start + ((batch_idx + 1) as f64 / total_batches as f64) * (progress_end - progress_start);
            let _ = peer
                .notify_progress(ProgressNotificationParam {
                    progress_token: token.clone(),
                    progress,
                    total: Some(100.0),
                    message: Some(format!(
                        "Generating embeddings... {}/{} batches ({} chunks)",
                        batch_idx + 1,
                        total_batches,
                        chunks_processed
                    )),
                })
                .await;
        }
    }

    Ok(EmbeddingResult {
        embeddings: all_embeddings,
        successful_chunks,
        errors,
    })
}

/// Index a complete codebase
#[allow(clippy::too_many_arguments)]
pub async fn do_index(
    client: &RagClient,
    path: String,
    project: Option<String>,
    include_patterns: Vec<String>,
    exclude_patterns: Vec<String>,
    max_file_size: usize,
    peer: Option<Peer<RoleServer>>,
    progress_token: Option<ProgressToken>,
    cancel_token: CancellationToken,
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
    // Create a cancellation flag for the blocking file walker
    let cancelled_flag = Arc::new(AtomicBool::new(false));
    let cancelled_flag_clone = cancelled_flag.clone();
    let cancel_token_clone = cancel_token.clone();

    // Spawn a task to set the flag when cancellation is requested
    let _cancel_watcher = tokio::spawn(async move {
        cancel_token_clone.cancelled().await;
        cancelled_flag_clone.store(true, Ordering::Relaxed);
        tracing::debug!("Cancellation flag set for file walker");
    });

    let walker = FileWalker::new(&path, max_file_size)
        .with_project(project.clone())
        .with_patterns(include_patterns.clone(), exclude_patterns.clone())
        .with_cancellation_flag(cancelled_flag);

    let files = tokio::task::spawn_blocking(move || walker.walk())
        .await
        .context("Failed to spawn file walker task")?
        .context("Failed to walk directory")?;
    let files_indexed = files.len();

    // Check for cancellation after file walk
    check_cancelled!(cancel_token);

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
    let chunker = client.chunker.clone();
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

    // Generate embeddings with frequent cancellation checks
    // Progress range: 40% to 80%
    let embed_result = generate_embeddings_with_cancellation(
        client,
        &all_chunks,
        &cancel_token,
        &peer,
        &progress_token,
        40.0,
        80.0,
    )
    .await?;

    let all_embeddings = embed_result.embeddings;
    let successful_chunks = embed_result.successful_chunks;
    errors.extend(embed_result.errors);

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

    // Store in vector database (pass normalized root path for per-project BM25)
    // Use successful_chunks to ensure metadata/contents match embeddings count
    let metadata: Vec<ChunkMetadata> = successful_chunks
        .iter()
        .map(|c| c.metadata.clone())
        .collect();
    let contents: Vec<String> = successful_chunks.iter().map(|c| c.content.clone()).collect();

    // Sanity check: ensure all arrays have the same length to prevent RecordBatch errors
    debug_assert_eq!(
        all_embeddings.len(),
        metadata.len(),
        "Embeddings and metadata count mismatch"
    );
    debug_assert_eq!(
        all_embeddings.len(),
        contents.len(),
        "Embeddings and contents count mismatch"
    );

    // Check for cancellation before storing
    check_cancelled!(cancel_token);

    if !all_embeddings.is_empty() {
        client
            .vector_db
            .store_embeddings(all_embeddings, metadata, contents, &path)
            .await
            .context("Failed to store embeddings")?;
    }

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

    let mut cache = client.hash_cache.write().await;
    cache.update_root(path, file_hashes);

    // Persist to disk
    if let Err(e) = cache.save(&client.cache_path) {
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
    client
        .vector_db
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
#[allow(clippy::too_many_arguments)]
pub async fn do_incremental_update(
    client: &RagClient,
    path: String,
    project: Option<String>,
    include_patterns: Vec<String>,
    exclude_patterns: Vec<String>,
    max_file_size: usize,
    peer: Option<Peer<RoleServer>>,
    progress_token: Option<ProgressToken>,
    cancel_token: CancellationToken,
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
    let cache = client.hash_cache.read().await;
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
    // Create a cancellation flag for the blocking file walker
    let cancelled_flag = Arc::new(AtomicBool::new(false));
    let cancelled_flag_clone = cancelled_flag.clone();
    let cancel_token_clone = cancel_token.clone();

    // Spawn a task to set the flag when cancellation is requested
    let _cancel_watcher = tokio::spawn(async move {
        cancel_token_clone.cancelled().await;
        cancelled_flag_clone.store(true, Ordering::Relaxed);
        tracing::debug!("Cancellation flag set for file walker");
    });

    let walker = FileWalker::new(&path, max_file_size)
        .with_project(project.clone())
        .with_patterns(include_patterns.clone(), exclude_patterns.clone())
        .with_cancellation_flag(cancelled_flag);

    let current_files = tokio::task::spawn_blocking(move || walker.walk())
        .await
        .context("Failed to spawn file walker task")?
        .context("Failed to walk directory")?;

    // Check for cancellation after file walk
    check_cancelled!(cancel_token);

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
                if let Err(e) = client.vector_db.delete_by_file(&file.relative_path).await {
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
            if let Err(e) = client.vector_db.delete_by_file(old_file).await {
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
    let (embeddings_generated, embed_errors) = if !files_to_index.is_empty() {
        // Chunk files in parallel for better performance
        let chunker = client.chunker.clone();
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

        // Generate embeddings with frequent cancellation checks
        // Progress range: 60% to 85%
        let embed_result = generate_embeddings_with_cancellation(
            client,
            &all_chunks,
            &cancel_token,
            &peer,
            &progress_token,
            60.0,
            85.0,
        )
        .await?;

        let all_embeddings = embed_result.embeddings;
        let successful_chunks = embed_result.successful_chunks;

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

        // Check for cancellation before storing
        check_cancelled!(cancel_token);

        // Store all embeddings (pass normalized root path for per-project BM25)
        // Use successful_chunks to ensure metadata/contents match embeddings count
        let metadata: Vec<ChunkMetadata> = successful_chunks
            .iter()
            .map(|c| c.metadata.clone())
            .collect();
        let contents: Vec<String> = successful_chunks.iter().map(|c| c.content.clone()).collect();

        if !all_embeddings.is_empty() {
            client
                .vector_db
                .store_embeddings(all_embeddings.clone(), metadata, contents, &path)
                .await
                .context("Failed to store embeddings")?;
        }

        (all_embeddings.len(), embed_result.errors)
    } else {
        (0, vec![])
    };

    // Collect any embedding errors (logged but not fatal)
    for err in embed_errors {
        tracing::warn!("Embedding error during incremental update: {}", err);
    }

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
    let mut cache = client.hash_cache.write().await;
    cache.update_root(path, new_hashes);

    // Persist to disk
    if let Err(e) = cache.save(&client.cache_path) {
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
    client
        .vector_db
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
#[allow(clippy::too_many_arguments)]
pub async fn do_index_smart(
    client: &RagClient,
    path: String,
    project: Option<String>,
    include_patterns: Vec<String>,
    exclude_patterns: Vec<String>,
    max_file_size: usize,
    peer: Option<Peer<RoleServer>>,
    progress_token: Option<ProgressToken>,
    cancel_token: CancellationToken,
) -> Result<IndexResponse> {
    use super::IndexLockResult;

    // Try to acquire indexing lock
    let lock_result = client.try_acquire_index_lock(&path).await?;

    match lock_result {
        IndexLockResult::WaitForResult(mut receiver) => {
            // Another task in THIS PROCESS is indexing, wait for its result via broadcast
            tracing::info!("Waiting for existing indexing operation in this process to complete for: {}", path);

            // Send progress notification if we have a peer
            if let (Some(peer), Some(token)) = (&peer, &progress_token) {
                let _ = peer
                    .notify_progress(ProgressNotificationParam {
                        progress_token: token.clone(),
                        progress: 0.0,
                        total: Some(100.0),
                        message: Some("Waiting for existing indexing operation to complete...".into()),
                    })
                    .await;
            }

            // Wait for the result from the other operation
            match receiver.recv().await {
                Ok(result) => {
                    tracing::info!("Received result from existing indexing operation");
                    Ok(result)
                }
                Err(e) => {
                    // The sender was dropped without sending a result (error case)
                    Err(anyhow::anyhow!(
                        "Indexing operation failed or was cancelled: {}",
                        e
                    ))
                }
            }
        }
        IndexLockResult::WaitForFilesystemLock(normalized_path) => {
            // Another PROCESS is indexing this path, wait for the filesystem lock
            tracing::info!(
                "Another process is indexing {} - waiting for filesystem lock to be released",
                normalized_path
            );

            // Send progress notification if we have a peer
            if let (Some(peer), Some(token)) = (&peer, &progress_token) {
                let _ = peer
                    .notify_progress(ProgressNotificationParam {
                        progress_token: token.clone(),
                        progress: 0.0,
                        total: Some(100.0),
                        message: Some("Waiting for another process to finish indexing...".into()),
                    })
                    .await;
            }

            // Block until we can acquire the filesystem lock (with 30 min timeout)
            // This happens when the other process finishes indexing
            use super::FsLockGuard;
            use std::time::Duration;

            let path_for_lock = normalized_path.clone();
            let fs_lock_result = tokio::task::spawn_blocking(move || {
                FsLockGuard::acquire_blocking(&path_for_lock, Duration::from_secs(30 * 60))
            })
            .await
            .context("Filesystem lock blocking task panicked")??;

            match fs_lock_result {
                Some(_lock) => {
                    // We acquired the lock! The other process finished.
                    // The database should be up-to-date from their indexing.
                    // We'll do an incremental check to be safe (will be fast if nothing changed)
                    tracing::info!(
                        "Other process finished indexing {} - performing incremental check",
                        normalized_path
                    );

                    // Drop the lock immediately - we don't need it for incremental check
                    // since we're not modifying the database
                    drop(_lock);

                    // Return a response indicating we waited and the index should be current
                    // The caller can do an incremental check if they want to verify
                    Ok(IndexResponse {
                        mode: crate::types::IndexingMode::Incremental,
                        files_indexed: 0,
                        chunks_created: 0,
                        embeddings_generated: 0,
                        duration_ms: 0,
                        errors: vec![],
                        files_updated: 0,
                        files_removed: 0,
                    })
                }
                None => {
                    // Timeout waiting for the lock - the other process took too long
                    Err(anyhow::anyhow!(
                        "Timeout waiting for another process to finish indexing {} (30 minutes)",
                        normalized_path
                    ))
                }
            }
        }
        IndexLockResult::Acquired(lock) => {
            // We acquired the lock, perform the actual indexing
            let result = do_index_smart_inner(
                client,
                path.clone(),
                project,
                include_patterns,
                exclude_patterns,
                max_file_size,
                peer,
                progress_token,
                cancel_token,
            )
            .await;

            // Broadcast the result to any waiters (even on error, so they don't hang)
            match &result {
                Ok(response) => {
                    lock.broadcast_result(response);
                }
                Err(e) => {
                    // On error, broadcast an error response so waiters don't hang
                    tracing::error!("Indexing failed for {}: {}", path, e);
                    let error_response = IndexResponse {
                        mode: crate::types::IndexingMode::Full,
                        files_indexed: 0,
                        chunks_created: 0,
                        embeddings_generated: 0,
                        duration_ms: 0,
                        errors: vec![format!("Indexing failed: {}", e)],
                        files_updated: 0,
                        files_removed: 0,
                    };
                    lock.broadcast_result(&error_response);
                }
            }

            // Release the lock synchronously to avoid race conditions
            // This ensures the lock is removed from the map before we return
            lock.release().await;

            result
        }
    }
}

/// Inner implementation of smart indexing (called when we have the lock)
#[allow(clippy::too_many_arguments)]
async fn do_index_smart_inner(
    client: &RagClient,
    path: String,
    project: Option<String>,
    include_patterns: Vec<String>,
    exclude_patterns: Vec<String>,
    max_file_size: usize,
    peer: Option<Peer<RoleServer>>,
    progress_token: Option<ProgressToken>,
    cancel_token: CancellationToken,
) -> Result<IndexResponse> {
    // Normalize path to canonical form for consistent cache lookups
    let normalized_path = RagClient::normalize_path(&path)?;

    // Check if index is dirty (previous indexing was interrupted)
    // If dirty, we MUST do a full reindex after clearing the corrupted data
    let cache = client.hash_cache.read().await;
    let is_dirty = cache.is_dirty(&normalized_path);
    let has_existing_index = cache.get_root(&normalized_path).is_some();
    drop(cache);

    // Handle dirty index: clear everything and force full reindex
    if is_dirty {
        tracing::warn!(
            "Index for '{}' is marked as dirty (previous indexing was interrupted). Clearing corrupted data and performing full reindex.",
            normalized_path
        );

        // Send progress notification about dirty state
        if let (Some(peer), Some(token)) = (&peer, &progress_token) {
            let _ = peer
                .notify_progress(ProgressNotificationParam {
                    progress_token: token.clone(),
                    progress: 0.0,
                    total: Some(100.0),
                    message: Some("Dirty index detected, clearing corrupted data...".into()),
                })
                .await;
        }

        // Clear any existing embeddings for this path from the vector database
        // We can't trust partial data, so delete everything for this path
        if let Err(e) = clear_path_data(client, &normalized_path).await {
            tracing::error!("Failed to clear corrupted index data for '{}': {}", normalized_path, e);
            // Continue anyway - the full index will overwrite
        }

        // Clear the cache entry for this path (removes from roots and dirty_roots)
        let mut cache = client.hash_cache.write().await;
        cache.remove_root(&normalized_path);
        if let Err(e) = cache.save(&client.cache_path) {
            tracing::warn!("Failed to save cache after clearing dirty state: {}", e);
        }
        drop(cache);
    }

    // Mark the index as dirty BEFORE starting (persisted immediately)
    // This ensures that if we crash/are killed, the next run knows the index is corrupted
    {
        let mut cache = client.hash_cache.write().await;
        cache.mark_dirty(&normalized_path);
        if let Err(e) = cache.save(&client.cache_path) {
            tracing::error!("Failed to save dirty flag: {}", e);
            // This is critical - if we can't persist the dirty flag, we shouldn't proceed
            anyhow::bail!("Failed to mark index as dirty before indexing: {}", e);
        }
        tracing::debug!("Marked index as dirty for: {}", normalized_path);
    }

    // Perform the actual indexing
    let result = if has_existing_index && !is_dirty {
        tracing::info!(
            "Existing index found for '{}' (normalized: '{}'), performing incremental update",
            path,
            normalized_path
        );
        do_incremental_update(
            client,
            normalized_path.clone(),
            project,
            include_patterns,
            exclude_patterns,
            max_file_size,
            peer,
            progress_token,
            cancel_token,
        )
        .await
    } else {
        tracing::info!(
            "No existing index found for '{}' (normalized: '{}'), performing full index",
            path,
            normalized_path
        );
        do_index(
            client,
            normalized_path.clone(),
            project,
            include_patterns,
            exclude_patterns,
            max_file_size,
            peer,
            progress_token,
            cancel_token,
        )
        .await
    };

    // Clear the dirty flag ONLY on successful completion
    // On error/cancellation, the dirty flag remains set
    match &result {
        Ok(_) => {
            let mut cache = client.hash_cache.write().await;
            cache.clear_dirty(&normalized_path);
            if let Err(e) = cache.save(&client.cache_path) {
                tracing::warn!("Failed to clear dirty flag after successful indexing: {}", e);
                // Don't fail the whole operation for this
            }
            tracing::debug!("Cleared dirty flag for: {}", normalized_path);
        }
        Err(e) => {
            tracing::warn!(
                "Indexing failed or was cancelled for '{}', dirty flag remains set: {}",
                normalized_path,
                e
            );
            // Dirty flag intentionally left set - next indexing will do full reindex
        }
    }

    result
}

/// Clear all indexed data for a specific path
async fn clear_path_data(client: &RagClient, normalized_path: &str) -> Result<()> {
    // Get all file paths that were indexed for this root
    let cache = client.hash_cache.read().await;
    let file_paths: Vec<String> = cache
        .get_root(normalized_path)
        .map(|hashes| hashes.keys().cloned().collect())
        .unwrap_or_default();
    drop(cache);

    // Delete embeddings for each file
    for file_path in file_paths {
        if let Err(e) = client.vector_db.delete_by_file(&file_path).await {
            tracing::warn!("Failed to delete embeddings for file '{}': {}", file_path, e);
        }
    }

    tracing::info!("Cleared indexed data for path: {}", normalized_path);
    Ok(())
}

#[cfg(test)]
mod tests;
