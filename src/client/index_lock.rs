//! Index locking mechanism for preventing concurrent indexing operations
//!
//! This module provides two layers of locking:
//! 1. Filesystem locks (cross-process) - prevents multiple processes from indexing the same path
//! 2. In-memory locks (in-process) - allows waiting tasks to receive the result via broadcast

use super::fs_lock::FsLockGuard;
use crate::types::IndexResponse;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::broadcast;
use tokio::sync::RwLock;

/// Maximum time an indexing operation can run before being considered stale (30 minutes)
/// This handles cases where the process crashes or panics without proper cleanup
const MAX_LOCK_DURATION: Duration = Duration::from_secs(30 * 60);

/// State for an in-progress indexing operation
pub(crate) struct IndexingOperation {
    /// Sender to broadcast the result to all waiters
    pub(crate) result_tx: broadcast::Sender<IndexResponse>,
    /// Flag indicating the operation is still active (set to false when complete)
    /// This is used to distinguish between "just created" and "completed but not cleaned up"
    pub(crate) active: Arc<AtomicBool>,
    /// Timestamp when this operation started (for stale detection)
    pub(crate) started_at: Instant,
}

impl IndexingOperation {
    /// Check if this operation is stale (running too long or crashed)
    pub(crate) fn is_stale(&self) -> bool {
        // If not active, it's completed (not stale, just needs cleanup)
        if !self.active.load(Ordering::Acquire) {
            return false;
        }
        // If active but running too long, consider it stale
        self.started_at.elapsed() > MAX_LOCK_DURATION
    }
}

/// Result of trying to acquire an index lock
pub(crate) enum IndexLockResult {
    /// We acquired the lock and should perform indexing
    Acquired(IndexLockGuard),
    /// Another operation in the SAME PROCESS is in progress, wait for its result
    WaitForResult(broadcast::Receiver<IndexResponse>),
    /// Another PROCESS is indexing this path, need to wait for filesystem lock
    WaitForFilesystemLock(String),
}

/// Guard for index locks that cleans up the lock when released
///
/// This guard holds both the filesystem lock (cross-process) and the in-memory
/// lock state (for broadcasting results to waiters in the same process).
pub(crate) struct IndexLockGuard {
    path: String,
    locks_map: Arc<RwLock<HashMap<String, IndexingOperation>>>,
    /// Sender to broadcast the result when indexing completes
    pub(crate) result_tx: broadcast::Sender<IndexResponse>,
    /// Shared active flag - set to false when operation completes
    active_flag: Arc<AtomicBool>,
    /// Flag to track if the lock has been properly released
    released: bool,
    /// Filesystem lock guard - dropped automatically when IndexLockGuard is dropped
    /// This ensures cross-process coordination
    #[allow(dead_code)]
    fs_lock: FsLockGuard,
}

impl IndexLockGuard {
    /// Create a new IndexLockGuard with both filesystem and in-memory locks
    pub(crate) fn new(
        path: String,
        locks_map: Arc<RwLock<HashMap<String, IndexingOperation>>>,
        result_tx: broadcast::Sender<IndexResponse>,
        active_flag: Arc<AtomicBool>,
        fs_lock: FsLockGuard,
    ) -> Self {
        Self {
            path,
            locks_map,
            result_tx,
            active_flag,
            released: false,
            fs_lock,
        }
    }

    /// Broadcast the indexing result to all waiters
    pub(crate) fn broadcast_result(&self, result: &IndexResponse) {
        // Mark the operation as no longer active BEFORE broadcasting
        // This allows cleanup to happen if there are no waiters
        self.active_flag.store(false, Ordering::Release);
        // Ignore send errors (no receivers is fine)
        let _ = self.result_tx.send(result.clone());
    }

    /// Release the lock explicitly - MUST be called after broadcasting result
    /// This ensures synchronous cleanup before the guard is dropped
    pub(crate) async fn release(mut self) {
        let mut locks = self.locks_map.write().await;
        locks.remove(&self.path);
        self.released = true;
        // Drop self here, but released=true prevents the Drop impl from spawning cleanup
    }
}

impl Drop for IndexLockGuard {
    fn drop(&mut self) {
        if !self.released {
            // Lock wasn't properly released - this is a fallback for error cases
            // (panic, early return without calling release(), etc.)

            // CRITICAL: Mark as inactive so waiters don't hang forever
            // This must happen synchronously before we spawn the cleanup task
            self.active_flag.store(false, Ordering::Release);

            // Broadcast an error response to any waiters so they don't hang
            let error_response = IndexResponse {
                mode: crate::types::IndexingMode::Full,
                files_indexed: 0,
                chunks_created: 0,
                embeddings_generated: 0,
                duration_ms: 0,
                errors: vec!["Indexing operation was interrupted (panic or early return)".to_string()],
                files_updated: 0,
                files_removed: 0,
            };
            let _ = self.result_tx.send(error_response);

            // Spawn a task to clean up the lock from the map asynchronously
            let path = self.path.clone();
            let locks_map = self.locks_map.clone();

            tracing::warn!(
                "IndexLockGuard for '{}' dropped without explicit release - spawning cleanup task",
                path
            );

            tokio::spawn(async move {
                let mut locks = locks_map.write().await;
                locks.remove(&path);
            });
        }
    }
}
