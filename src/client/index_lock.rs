//! Index locking mechanism for preventing concurrent indexing operations

use crate::types::IndexResponse;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::sync::broadcast;

/// State for an in-progress indexing operation
pub(crate) struct IndexingOperation {
    /// Sender to broadcast the result to all waiters
    pub(crate) result_tx: broadcast::Sender<IndexResponse>,
}

/// Result of trying to acquire an index lock
pub(crate) enum IndexLockResult {
    /// We acquired the lock and should perform indexing
    Acquired(IndexLockGuard),
    /// Another operation is in progress, wait for its result
    WaitForResult(broadcast::Receiver<IndexResponse>),
}

/// Guard for index locks that cleans up the lock when released
pub(crate) struct IndexLockGuard {
    path: String,
    locks_map: Arc<RwLock<HashMap<String, IndexingOperation>>>,
    /// Sender to broadcast the result when indexing completes
    pub(crate) result_tx: broadcast::Sender<IndexResponse>,
    /// Flag to track if the lock has been properly released
    released: bool,
}

impl IndexLockGuard {
    /// Create a new IndexLockGuard
    pub(crate) fn new(
        path: String,
        locks_map: Arc<RwLock<HashMap<String, IndexingOperation>>>,
        result_tx: broadcast::Sender<IndexResponse>,
    ) -> Self {
        Self {
            path,
            locks_map,
            result_tx,
            released: false,
        }
    }

    /// Broadcast the indexing result to all waiters
    pub(crate) fn broadcast_result(&self, result: &IndexResponse) {
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
            // Spawn a task to clean up the lock asynchronously
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
