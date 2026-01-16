//! Filesystem-based locking for cross-process coordination
//!
//! This module provides filesystem locks using flock() to prevent multiple
//! processes from indexing the same codebase simultaneously. This complements
//! the in-process locking in index_lock.rs.

use anyhow::{Context, Result};
use fs2::FileExt;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Get the directory for lock files
fn lock_dir() -> PathBuf {
    // Use the same data directory as the rest of project-rag
    #[cfg(feature = "alt-folder-name")]
    let folder_name = "brainwires";
    #[cfg(not(feature = "alt-folder-name"))]
    let folder_name = "project-rag";

    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(folder_name)
        .join("locks")
}

/// Get the lock file path for a given normalized codebase path
fn lock_file_path(normalized_path: &str) -> PathBuf {
    // Hash the path to create a safe filename
    let mut hasher = Sha256::new();
    hasher.update(normalized_path.as_bytes());
    let hash = format!("{:x}", hasher.finalize());
    // Use first 16 chars of hash for brevity while maintaining uniqueness
    lock_dir().join(format!("{}.lock", &hash[..16]))
}

/// Guard that holds an exclusive filesystem lock
///
/// The lock is automatically released when this guard is dropped.
/// If the process crashes, the OS automatically releases the flock.
pub struct FsLockGuard {
    _file: File,
    _path: PathBuf,
}

impl FsLockGuard {
    /// Try to acquire an exclusive filesystem lock, non-blocking
    ///
    /// Returns:
    /// - `Ok(Some(guard))` if the lock was acquired
    /// - `Ok(None)` if another process holds the lock
    /// - `Err(...)` on IO errors
    pub fn try_acquire(normalized_path: &str) -> Result<Option<Self>> {
        let lock_path = lock_file_path(normalized_path);

        tracing::debug!(
            "Attempting to acquire filesystem lock: path={}, lock_file={:?}",
            normalized_path,
            lock_path
        );

        // Ensure lock directory exists
        if let Some(parent) = lock_path.parent() {
            fs::create_dir_all(parent).context("Failed to create lock directory")?;
        }

        // Open/create lock file
        let file = File::create(&lock_path).context("Failed to create lock file")?;

        // Try non-blocking exclusive lock
        match file.try_lock_exclusive() {
            Ok(()) => {
                tracing::debug!(
                    "Acquired filesystem lock for: {} (lock_file={:?})",
                    normalized_path,
                    lock_path
                );
                Ok(Some(Self {
                    _file: file,
                    _path: lock_path,
                }))
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                tracing::debug!(
                    "Filesystem lock blocked (another holder) for: {} (lock_file={:?})",
                    normalized_path,
                    lock_path
                );
                Ok(None)
            }
            Err(e) => Err(e).context("Failed to acquire filesystem lock"),
        }
    }

    /// Acquire lock, blocking until available (with timeout)
    ///
    /// This polls the lock with a sleep interval until either:
    /// - The lock is acquired (returns `Ok(Some(guard))`)
    /// - The timeout expires (returns `Ok(None)`)
    /// - An IO error occurs (returns `Err(...)`)
    pub fn acquire_blocking(normalized_path: &str, timeout: Duration) -> Result<Option<Self>> {
        let start = Instant::now();
        let sleep_interval = Duration::from_millis(500);

        tracing::info!(
            "Waiting for filesystem lock on {} (timeout: {:?})",
            normalized_path,
            timeout
        );

        loop {
            match Self::try_acquire(normalized_path)? {
                Some(guard) => {
                    tracing::info!(
                        "Acquired filesystem lock after {:?}",
                        start.elapsed()
                    );
                    return Ok(Some(guard));
                }
                None => {
                    if start.elapsed() >= timeout {
                        tracing::warn!(
                            "Timeout waiting for filesystem lock on {} after {:?}",
                            normalized_path,
                            timeout
                        );
                        return Ok(None);
                    }
                    std::thread::sleep(sleep_interval);
                }
            }
        }
    }
}

impl Drop for FsLockGuard {
    fn drop(&mut self) {
        // The lock is automatically released when the file is closed,
        // but we log for debugging purposes
        tracing::debug!("Releasing filesystem lock");
        // Note: we don't delete the lock file - it can be reused
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_acquire_and_release() {
        let path = "/test/path/for/locking";

        // Acquire lock
        let guard = FsLockGuard::try_acquire(path).unwrap();
        assert!(guard.is_some());

        // Drop to release
        drop(guard);

        // Should be able to acquire again
        let guard2 = FsLockGuard::try_acquire(path).unwrap();
        assert!(guard2.is_some());
    }

    #[test]
    fn test_concurrent_lock_fails() {
        let path = "/test/path/for/concurrent/locking";

        // Acquire lock in main thread
        let guard1 = FsLockGuard::try_acquire(path).unwrap();
        assert!(guard1.is_some());

        // Try to acquire from another thread - should fail
        let path_clone = path.to_string();
        let handle = thread::spawn(move || FsLockGuard::try_acquire(&path_clone).unwrap());

        let result = handle.join().unwrap();
        assert!(result.is_none(), "Second lock should fail");

        // Release first lock
        drop(guard1);

        // Now second thread should succeed
        let guard2 = FsLockGuard::try_acquire(path).unwrap();
        assert!(guard2.is_some());
    }

    #[test]
    fn test_blocking_acquire_with_timeout() {
        let path = "/test/path/for/blocking/timeout";

        // Acquire lock
        let _guard = FsLockGuard::try_acquire(path).unwrap().unwrap();

        // Try blocking acquire with very short timeout from another thread
        let path_clone = path.to_string();
        let handle = thread::spawn(move || {
            FsLockGuard::acquire_blocking(&path_clone, Duration::from_millis(100)).unwrap()
        });

        let result = handle.join().unwrap();
        assert!(result.is_none(), "Should timeout waiting for lock");
    }

    #[test]
    fn test_lock_file_path_uniqueness() {
        let path1 = "/path/to/project1";
        let path2 = "/path/to/project2";
        let path1_dup = "/path/to/project1";

        let lock1 = lock_file_path(path1);
        let lock2 = lock_file_path(path2);
        let lock1_dup = lock_file_path(path1_dup);

        assert_ne!(lock1, lock2, "Different paths should have different lock files");
        assert_eq!(lock1, lock1_dup, "Same path should have same lock file");
    }
}

    #[tokio::test]
    async fn test_concurrent_lock_fails_async() {
        let path = "/test/path/for/async/concurrent/locking";

        // Acquire lock in spawn_blocking (simulating what RagClient does)
        let path1 = path.to_string();
        let guard1 = tokio::task::spawn_blocking(move || {
            FsLockGuard::try_acquire(&path1).unwrap()
        }).await.unwrap();
        
        assert!(guard1.is_some(), "First lock should succeed");
        
        // Hold the guard in this task
        let _held_guard = guard1.unwrap();

        // Try to acquire again from spawn_blocking
        let path2 = path.to_string();
        let guard2 = tokio::task::spawn_blocking(move || {
            FsLockGuard::try_acquire(&path2).unwrap()
        }).await.unwrap();

        assert!(guard2.is_none(), "Second lock should fail because first is held");
    }
