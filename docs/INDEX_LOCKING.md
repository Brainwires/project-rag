# Index Locking System

This document describes the concurrent indexing protection system used by project-rag to prevent duplicate indexing operations and ensure safe parallel access.

## Overview

When multiple AI agents or clients try to index the same codebase simultaneously, only one should perform the actual indexing work. The others should wait and receive the same result when it completes. This is implemented using a lock-and-broadcast pattern.

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         RagClient                                │
│                                                                  │
│  indexing_ops: Arc<RwLock<HashMap<String, IndexingOperation>>>  │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              IndexingOperation                           │   │
│  │  - result_tx: broadcast::Sender<IndexResponse>          │   │
│  │  - active: Arc<AtomicBool>                              │   │
│  │  - started_at: Instant                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              IndexLockGuard                              │   │
│  │  - path: String (normalized)                            │   │
│  │  - locks_map: Arc<RwLock<...>>                          │   │
│  │  - result_tx: broadcast::Sender<IndexResponse>          │   │
│  │  - active_flag: Arc<AtomicBool>                         │   │
│  │  - released: bool                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Types

- **IndexingOperation**: Tracks an in-progress indexing operation in the global map
- **IndexLockResult**: Enum returned when trying to acquire a lock
  - `Acquired(IndexLockGuard)`: We should perform indexing
  - `WaitForResult(Receiver)`: Another operation is in progress, wait for result
- **IndexLockGuard**: RAII guard that manages the lock lifecycle

## Lock Acquisition Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Agent 1    │     │   Agent 2    │     │   Agent 3    │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │ try_acquire_lock   │                    │
       │──────────────────▶ │                    │
       │                    │                    │
       │ Acquired(guard)    │                    │
       │◀────────────────── │                    │
       │                    │                    │
       │ starts indexing    │ try_acquire_lock   │
       │                    │──────────────────▶ │
       │                    │                    │
       │                    │ WaitForResult(rx)  │ try_acquire_lock
       │                    │◀────────────────── │───────────────▶
       │                    │                    │
       │                    │                    │ WaitForResult(rx)
       │                    │                    │◀───────────────
       │                    │                    │
       │ broadcast_result   │                    │
       │───────────────────▶│───────────────────▶│
       │                    │                    │
       │ release()          │ receives result    │ receives result
       │                    │                    │
```

## Path Normalization

All paths are normalized to canonical absolute form before being used as lock keys. This ensures that different representations of the same path share the same lock:

```rust
"/home/user/project"        → "/home/user/project"
"/home/user/project/../project" → "/home/user/project"
"./project"                 → "/home/user/project" (from cwd)
```

## Stale Lock Detection

### Problem

Locks can become stale if:
1. The process crashes during indexing
2. A panic occurs without proper cleanup
3. An operation hangs indefinitely

### Solution

Each `IndexingOperation` tracks:
- `active: Arc<AtomicBool>` - Set to `true` on creation, `false` on completion
- `started_at: Instant` - Timestamp when operation started

Stale detection logic:
```rust
fn is_stale(&self) -> bool {
    if !self.active.load(Ordering::Acquire) {
        return false;  // Completed, not stale
    }
    self.started_at.elapsed() > MAX_LOCK_DURATION  // 30 minutes
}
```

### Detection Points

Stale locks are detected and removed when:
1. A new lock acquisition is attempted for the same path
2. The operation has been running longer than `MAX_LOCK_DURATION` (30 minutes)

## Panic/Error Handling

The `IndexLockGuard` implements `Drop` to handle abnormal termination:

```rust
impl Drop for IndexLockGuard {
    fn drop(&mut self) {
        if !self.released {
            // 1. Mark as inactive (prevents new waiters)
            self.active_flag.store(false, Ordering::Release);

            // 2. Broadcast error to existing waiters (prevents hanging)
            let error_response = IndexResponse {
                errors: vec!["Indexing operation was interrupted".to_string()],
                ...
            };
            let _ = self.result_tx.send(error_response);

            // 3. Spawn async cleanup task
            tokio::spawn(async move {
                locks_map.write().await.remove(&path);
            });
        }
    }
}
```

This ensures:
- Waiters receive a response (won't hang forever)
- Lock is cleaned up from the map
- Subsequent operations can acquire the lock

## Thread Safety

The locking system uses several synchronization primitives:

| Component | Type | Purpose |
|-----------|------|---------|
| `indexing_ops` | `Arc<RwLock<HashMap<...>>>` | Protects the operation registry |
| `active` | `Arc<AtomicBool>` | Lock-free status check |
| `result_tx` | `broadcast::Sender` | Multi-consumer result distribution |

### Memory Ordering

- `Ordering::Acquire` when reading `active` flag (ensures visibility of prior writes)
- `Ordering::Release` when writing `active` flag (ensures writes are visible to readers)

## Usage Example

```rust
// Smart indexing automatically handles locking
pub async fn do_index_smart(client: &RagClient, path: String, ...) -> Result<IndexResponse> {
    let lock_result = client.try_acquire_index_lock(&path).await?;

    match lock_result {
        IndexLockResult::WaitForResult(mut receiver) => {
            // Another operation is in progress, wait for its result
            receiver.recv().await.map_err(|e| anyhow!("Failed: {}", e))
        }
        IndexLockResult::Acquired(lock) => {
            // We have the lock, perform indexing
            let result = do_index_inner(client, path, ...).await;

            // Broadcast result to waiters
            match &result {
                Ok(response) => lock.broadcast_result(response),
                Err(e) => lock.broadcast_result(&error_response(e)),
            }

            // Release lock (removes from map)
            lock.release().await;

            result
        }
    }
}
```

## Configuration

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_LOCK_DURATION` | 30 minutes | Maximum time before a lock is considered stale |
| Broadcast channel capacity | 1 | Only one result is ever sent |

## Test Coverage

The locking system has comprehensive tests:

- `test_index_lock_prevents_duplicate_indexing` - Verifies second caller waits
- `test_index_lock_waiters_receive_result` - Verifies broadcast works
- `test_index_lock_path_normalization` - Verifies equivalent paths share locks
- `test_index_lock_released_after_completion` - Verifies lock can be reacquired
- `test_concurrent_index_calls_share_result` - End-to-end concurrent test
- `test_index_lock_drop_without_release_broadcasts_error` - Verifies panic handling
- `test_index_lock_can_reacquire_after_drop_without_release` - Verifies recovery

## File Locations

- Lock implementation: `src/client/index_lock.rs`
- Lock acquisition: `src/client/mod.rs` (`try_acquire_index_lock`)
- Smart indexing: `src/client/indexing/mod.rs` (`do_index_smart`)
- Tests: `src/client/tests.rs`
