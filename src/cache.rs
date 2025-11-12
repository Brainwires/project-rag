use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Cache for file hashes to support incremental updates
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HashCache {
    /// Map of root path -> (file path -> hash)
    pub roots: HashMap<String, HashMap<String, String>>,
}

impl HashCache {
    /// Load cache from disk
    pub fn load(cache_path: &Path) -> Result<Self> {
        if !cache_path.exists() {
            tracing::debug!("Cache file not found, starting with empty cache");
            return Ok(Self::default());
        }

        let content = fs::read_to_string(cache_path).context("Failed to read cache file")?;

        let cache: HashCache =
            serde_json::from_str(&content).context("Failed to parse cache file")?;

        tracing::info!("Loaded cache with {} indexed roots", cache.roots.len());
        Ok(cache)
    }

    /// Save cache to disk
    pub fn save(&self, cache_path: &Path) -> Result<()> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent).context("Failed to create cache directory")?;
        }

        let content = serde_json::to_string_pretty(self).context("Failed to serialize cache")?;

        fs::write(cache_path, content).context("Failed to write cache file")?;

        tracing::debug!("Saved cache to {:?}", cache_path);
        Ok(())
    }

    /// Get file hashes for a root path
    pub fn get_root(&self, root: &str) -> Option<&HashMap<String, String>> {
        self.roots.get(root)
    }

    /// Update file hashes for a root path
    pub fn update_root(&mut self, root: String, hashes: HashMap<String, String>) {
        self.roots.insert(root, hashes);
    }

    /// Remove a root path from the cache
    pub fn remove_root(&mut self, root: &str) {
        self.roots.remove(root);
    }

    /// Get default cache path (in user's cache directory)
    pub fn default_path() -> PathBuf {
        let cache_dir = if cfg!(target_os = "windows") {
            PathBuf::from(std::env::var("LOCALAPPDATA").unwrap_or_else(|_| ".".to_string()))
        } else if cfg!(target_os = "macos") {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()))
                .join("Library/Caches")
        } else {
            // Linux/Unix
            PathBuf::from(std::env::var("XDG_CACHE_HOME").unwrap_or_else(|_| {
                format!(
                    "{}/.cache",
                    std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
                )
            }))
        };

        cache_dir.join("project-rag").join("hash_cache.json")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_cache_serialization() {
        let mut cache = HashCache::default();
        let mut hashes = HashMap::new();
        hashes.insert("file1.rs".to_string(), "hash1".to_string());
        hashes.insert("file2.rs".to_string(), "hash2".to_string());
        cache.update_root("/test/path".to_string(), hashes);

        let json = serde_json::to_string(&cache).unwrap();
        let deserialized: HashCache = serde_json::from_str(&json).unwrap();

        assert_eq!(cache.roots.len(), deserialized.roots.len());
        assert_eq!(
            cache.roots.get("/test/path"),
            deserialized.roots.get("/test/path")
        );
    }

    #[test]
    fn test_cache_save_load() {
        let temp_file = NamedTempFile::new().unwrap();
        let cache_path = temp_file.path().to_path_buf();

        // Create and save cache
        let mut cache = HashCache::default();
        let mut hashes = HashMap::new();
        hashes.insert("file1.rs".to_string(), "hash1".to_string());
        cache.update_root("/test/path".to_string(), hashes);

        cache.save(&cache_path).unwrap();

        // Load cache
        let loaded = HashCache::load(&cache_path).unwrap();
        assert_eq!(cache.roots.len(), loaded.roots.len());
        assert_eq!(
            cache.roots.get("/test/path"),
            loaded.roots.get("/test/path")
        );
    }

    #[test]
    fn test_cache_operations() {
        let mut cache = HashCache::default();

        // Update root
        let mut hashes = HashMap::new();
        hashes.insert("file1.rs".to_string(), "hash1".to_string());
        cache.update_root("/test/path".to_string(), hashes);

        // Get root
        assert!(cache.get_root("/test/path").is_some());
        assert!(cache.get_root("/nonexistent").is_none());

        // Remove root
        cache.remove_root("/test/path");
        assert!(cache.get_root("/test/path").is_none());
    }

    #[test]
    fn test_load_nonexistent_cache() {
        let result = HashCache::load(Path::new("/nonexistent/path/cache.json"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().roots.len(), 0);
    }

    #[test]
    fn test_load_corrupted_cache() {
        let temp_file = NamedTempFile::new().unwrap();
        let cache_path = temp_file.path().to_path_buf();

        // Write invalid JSON
        fs::write(&cache_path, "{ invalid json }").unwrap();

        let result = HashCache::load(&cache_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_save_creates_parent_directory() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cache_path = temp_dir.path().join("subdir").join("cache.json");

        let cache = HashCache::default();
        cache.save(&cache_path).unwrap();

        assert!(cache_path.exists());
    }

    #[test]
    fn test_default_path() {
        let path = HashCache::default_path();
        assert!(path.to_string_lossy().contains("project-rag"));
        assert!(path.to_string_lossy().contains("hash_cache.json"));
    }

    #[test]
    fn test_update_root_replaces_existing() {
        let mut cache = HashCache::default();

        // Add first set of hashes
        let mut hashes1 = HashMap::new();
        hashes1.insert("file1.rs".to_string(), "hash1".to_string());
        cache.update_root("/test/path".to_string(), hashes1);

        // Replace with new set
        let mut hashes2 = HashMap::new();
        hashes2.insert("file2.rs".to_string(), "hash2".to_string());
        cache.update_root("/test/path".to_string(), hashes2);

        let root_hashes = cache.get_root("/test/path").unwrap();
        assert_eq!(root_hashes.len(), 1);
        assert!(root_hashes.contains_key("file2.rs"));
        assert!(!root_hashes.contains_key("file1.rs"));
    }

    #[test]
    fn test_multiple_roots() {
        let mut cache = HashCache::default();

        let mut hashes1 = HashMap::new();
        hashes1.insert("file1.rs".to_string(), "hash1".to_string());
        cache.update_root("/path1".to_string(), hashes1);

        let mut hashes2 = HashMap::new();
        hashes2.insert("file2.rs".to_string(), "hash2".to_string());
        cache.update_root("/path2".to_string(), hashes2);

        assert_eq!(cache.roots.len(), 2);
        assert!(cache.get_root("/path1").is_some());
        assert!(cache.get_root("/path2").is_some());
    }

    #[test]
    fn test_empty_cache_operations() {
        let cache = HashCache::default();
        assert!(cache.get_root("/any/path").is_none());
        assert_eq!(cache.roots.len(), 0);
    }

    #[test]
    fn test_remove_root_nonexistent() {
        let mut cache = HashCache::default();
        cache.remove_root("/nonexistent");
        // Should not panic
        assert_eq!(cache.roots.len(), 0);
    }
}
