use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Information about a dirty (in-progress) indexing operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirtyInfo {
    /// Unix timestamp when the dirty flag was set
    pub timestamp: u64,
    /// Number of files that were expected to be indexed (if known)
    pub expected_files: Option<usize>,
}

impl DirtyInfo {
    /// Create a new dirty info with the current timestamp
    pub fn new() -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            timestamp,
            expected_files: None,
        }
    }

    /// Create a new dirty info with expected file count
    pub fn with_expected_files(expected_files: usize) -> Self {
        let mut info = Self::new();
        info.expected_files = Some(expected_files);
        info
    }

    /// Check if this dirty flag is stale (older than or equal to the given duration in seconds)
    pub fn is_stale(&self, max_age_secs: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        now.saturating_sub(self.timestamp) >= max_age_secs
    }

    /// Get the age of this dirty flag in seconds
    pub fn age_secs(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        now.saturating_sub(self.timestamp)
    }
}

impl Default for DirtyInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache for file hashes to support incremental updates
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HashCache {
    /// Map of root path -> (file path -> hash)
    pub roots: HashMap<String, HashMap<String, String>>,
    /// Map of root paths that are currently being indexed (dirty state) with metadata
    /// If a root is in this map, its index may be incomplete/corrupted
    #[serde(default)]
    pub dirty_roots: HashMap<String, DirtyInfo>,
}

/// Legacy cache format for migration (dirty_roots was a HashSet)
#[derive(Debug, Deserialize)]
struct LegacyHashCache {
    roots: HashMap<String, HashMap<String, String>>,
    #[serde(default)]
    dirty_roots: std::collections::HashSet<String>,
}

impl HashCache {
    /// Load cache from disk
    /// Handles migration from old format (dirty_roots as HashSet) to new format (dirty_roots as HashMap with DirtyInfo)
    pub fn load(cache_path: &Path) -> Result<Self> {
        if !cache_path.exists() {
            tracing::debug!("Cache file not found, starting with empty cache");
            return Ok(Self::default());
        }

        let content = fs::read_to_string(cache_path).context("Failed to read cache file")?;

        // Try to parse as new format first
        if let Ok(cache) = serde_json::from_str::<HashCache>(&content) {
            tracing::info!("Loaded cache with {} indexed roots", cache.roots.len());
            return Ok(cache);
        }

        // Try to parse as legacy format and migrate
        if let Ok(legacy) = serde_json::from_str::<LegacyHashCache>(&content) {
            tracing::info!(
                "Migrating cache from legacy format ({} roots, {} dirty roots)",
                legacy.roots.len(),
                legacy.dirty_roots.len()
            );

            // Migrate dirty_roots from HashSet to HashMap with default DirtyInfo
            let dirty_roots: HashMap<String, DirtyInfo> = legacy
                .dirty_roots
                .into_iter()
                .map(|root| (root, DirtyInfo::new()))
                .collect();

            let cache = HashCache {
                roots: legacy.roots,
                dirty_roots,
            };

            // Save the migrated cache immediately
            if let Err(e) = cache.save(cache_path) {
                tracing::warn!("Failed to save migrated cache: {}", e);
            } else {
                tracing::info!("Successfully migrated cache to new format");
            }

            return Ok(cache);
        }

        // Neither format worked
        anyhow::bail!("Failed to parse cache file as either new or legacy format")
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
        self.dirty_roots.remove(root);
    }

    /// Mark a root path as dirty (indexing in progress)
    /// This should be called BEFORE indexing starts and the cache saved immediately
    pub fn mark_dirty(&mut self, root: &str) {
        self.dirty_roots.insert(root.to_string(), DirtyInfo::new());
    }

    /// Mark a root path as dirty with expected file count
    pub fn mark_dirty_with_info(&mut self, root: &str, expected_files: usize) {
        self.dirty_roots.insert(
            root.to_string(),
            DirtyInfo::with_expected_files(expected_files),
        );
    }

    /// Clear the dirty flag for a root path (indexing completed successfully)
    /// This should be called AFTER indexing completes and the cache saved immediately
    pub fn clear_dirty(&mut self, root: &str) {
        self.dirty_roots.remove(root);
    }

    /// Check if a root path is marked as dirty
    pub fn is_dirty(&self, root: &str) -> bool {
        self.dirty_roots.contains_key(root)
    }

    /// Get dirty info for a root path
    pub fn get_dirty_info(&self, root: &str) -> Option<&DirtyInfo> {
        self.dirty_roots.get(root)
    }

    /// Get all dirty root paths
    pub fn get_dirty_roots(&self) -> &HashMap<String, DirtyInfo> {
        &self.dirty_roots
    }

    /// Check if any roots are dirty
    pub fn has_dirty_roots(&self) -> bool {
        !self.dirty_roots.is_empty()
    }

    /// Check if a dirty flag is stale (older than the given duration)
    /// Returns true if the root is dirty AND the flag is older than max_age_secs
    pub fn is_dirty_stale(&self, root: &str, max_age_secs: u64) -> bool {
        self.dirty_roots
            .get(root)
            .is_some_and(|info| info.is_stale(max_age_secs))
    }

    /// Get the age of a dirty flag in seconds, or None if not dirty
    pub fn dirty_age_secs(&self, root: &str) -> Option<u64> {
        self.dirty_roots.get(root).map(|info| info.age_secs())
    }

    /// Clear stale dirty flags (older than max_age_secs)
    /// Returns the number of stale flags cleared
    pub fn clear_stale_dirty_flags(&mut self, max_age_secs: u64) -> usize {
        let stale_roots: Vec<String> = self
            .dirty_roots
            .iter()
            .filter(|(_, info)| info.is_stale(max_age_secs))
            .map(|(root, _)| root.clone())
            .collect();

        let count = stale_roots.len();
        for root in stale_roots {
            tracing::info!(
                "Clearing stale dirty flag for '{}' (age: {} seconds)",
                root,
                self.dirty_roots
                    .get(&root)
                    .map(|i| i.age_secs())
                    .unwrap_or(0)
            );
            self.dirty_roots.remove(&root);
        }
        count
    }

    /// Get default cache path (in user's cache directory)
    pub fn default_path() -> PathBuf {
        crate::paths::PlatformPaths::default_hash_cache_path()
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

    #[test]
    fn test_dirty_flag_operations() {
        let mut cache = HashCache::default();

        // Initially not dirty
        assert!(!cache.is_dirty("/test/path"));
        assert!(!cache.has_dirty_roots());
        assert!(cache.get_dirty_roots().is_empty());

        // Mark as dirty
        cache.mark_dirty("/test/path");
        assert!(cache.is_dirty("/test/path"));
        assert!(cache.has_dirty_roots());
        assert!(cache.get_dirty_roots().contains_key("/test/path"));

        // Verify dirty info has timestamp
        let info = cache.get_dirty_info("/test/path").unwrap();
        assert!(info.timestamp > 0);
        assert!(info.expected_files.is_none());

        // Clear dirty flag
        cache.clear_dirty("/test/path");
        assert!(!cache.is_dirty("/test/path"));
        assert!(!cache.has_dirty_roots());
    }

    #[test]
    fn test_dirty_flag_with_expected_files() {
        let mut cache = HashCache::default();

        cache.mark_dirty_with_info("/test/path", 100);
        assert!(cache.is_dirty("/test/path"));

        let info = cache.get_dirty_info("/test/path").unwrap();
        assert_eq!(info.expected_files, Some(100));
    }

    #[test]
    fn test_dirty_flag_staleness() {
        let mut cache = HashCache::default();

        cache.mark_dirty("/test/path");

        // A fresh dirty flag should not be stale (with a 1 hour timeout)
        assert!(!cache.is_dirty_stale("/test/path", 3600));

        // Get the age (should be very small, just created)
        let age = cache.dirty_age_secs("/test/path").unwrap();
        assert!(age < 5); // Should be less than 5 seconds

        // With a 0 second timeout, it should be considered stale
        assert!(cache.is_dirty_stale("/test/path", 0));
    }

    #[test]
    fn test_clear_stale_dirty_flags() {
        let mut cache = HashCache::default();

        cache.mark_dirty("/path1");
        cache.mark_dirty("/path2");

        // With 0 second timeout, all should be cleared
        let cleared = cache.clear_stale_dirty_flags(0);
        assert_eq!(cleared, 2);
        assert!(!cache.has_dirty_roots());
    }

    #[test]
    fn test_dirty_flag_persistence() {
        let temp_file = NamedTempFile::new().unwrap();
        let cache_path = temp_file.path().to_path_buf();

        // Create cache with dirty flag
        let mut cache = HashCache::default();
        cache.mark_dirty("/test/path");
        cache.save(&cache_path).unwrap();

        // Load and verify dirty flag persisted
        let loaded = HashCache::load(&cache_path).unwrap();
        assert!(loaded.is_dirty("/test/path"));
        assert!(loaded.has_dirty_roots());
    }

    #[test]
    fn test_remove_root_clears_dirty() {
        let mut cache = HashCache::default();

        // Add root with files and mark as dirty
        let mut hashes = HashMap::new();
        hashes.insert("file1.rs".to_string(), "hash1".to_string());
        cache.update_root("/test/path".to_string(), hashes);
        cache.mark_dirty("/test/path");

        assert!(cache.is_dirty("/test/path"));
        assert!(cache.get_root("/test/path").is_some());

        // Remove root - should also clear dirty
        cache.remove_root("/test/path");
        assert!(!cache.is_dirty("/test/path"));
        assert!(cache.get_root("/test/path").is_none());
    }

    #[test]
    fn test_multiple_dirty_roots() {
        let mut cache = HashCache::default();

        cache.mark_dirty("/path1");
        cache.mark_dirty("/path2");
        cache.mark_dirty("/path3");

        assert!(cache.is_dirty("/path1"));
        assert!(cache.is_dirty("/path2"));
        assert!(cache.is_dirty("/path3"));
        assert_eq!(cache.get_dirty_roots().len(), 3);

        cache.clear_dirty("/path2");
        assert!(cache.is_dirty("/path1"));
        assert!(!cache.is_dirty("/path2"));
        assert!(cache.is_dirty("/path3"));
        assert_eq!(cache.get_dirty_roots().len(), 2);
    }

    #[test]
    fn test_dirty_flag_idempotent() {
        let mut cache = HashCache::default();

        // Marking same path multiple times should be idempotent
        cache.mark_dirty("/test/path");
        cache.mark_dirty("/test/path");
        cache.mark_dirty("/test/path");
        assert_eq!(cache.get_dirty_roots().len(), 1);

        // Clearing same path multiple times should be safe
        cache.clear_dirty("/test/path");
        cache.clear_dirty("/test/path");
        assert!(!cache.is_dirty("/test/path"));
    }

    #[test]
    fn test_dirty_flag_with_old_cache_format() {
        // Test that loading a cache without dirty_roots field works (backwards compatibility)
        let temp_file = NamedTempFile::new().unwrap();
        let cache_path = temp_file.path().to_path_buf();

        // Write old format JSON (without dirty_roots)
        let old_format = r#"{"roots":{"/test/path":{"file1.rs":"hash1"}}}"#;
        fs::write(&cache_path, old_format).unwrap();

        // Load should succeed with empty dirty_roots
        let loaded = HashCache::load(&cache_path).unwrap();
        assert!(loaded.get_root("/test/path").is_some());
        assert!(!loaded.has_dirty_roots());
        assert!(!loaded.is_dirty("/test/path"));
    }

    #[test]
    fn test_dirty_flag_migration_from_hashset() {
        // Test that loading a cache with old HashSet dirty_roots format works
        // This handles migration from the old format (HashSet<String>) to new (HashMap<String, DirtyInfo>)
        let temp_file = NamedTempFile::new().unwrap();
        let cache_path = temp_file.path().to_path_buf();

        // Write old format JSON with HashSet dirty_roots (array format)
        let old_format =
            r#"{"roots":{"/test/path":{"file1.rs":"hash1"}},"dirty_roots":["/test/path"]}"#;
        fs::write(&cache_path, old_format).unwrap();

        // Load should successfully migrate the old format
        let loaded = HashCache::load(&cache_path).unwrap();

        // Verify the migration worked
        assert!(loaded.get_root("/test/path").is_some());
        assert!(loaded.is_dirty("/test/path"));
        assert!(loaded.has_dirty_roots());

        // Verify the dirty info has a timestamp
        let info = loaded.get_dirty_info("/test/path").unwrap();
        assert!(info.timestamp > 0);

        // Verify the file was updated to new format
        let reloaded = HashCache::load(&cache_path).unwrap();
        assert!(reloaded.is_dirty("/test/path"));
    }

    #[test]
    fn test_dirty_info_default() {
        let info = DirtyInfo::default();
        assert!(info.timestamp > 0);
        assert!(info.expected_files.is_none());
    }

    #[test]
    fn test_dirty_info_with_expected_files() {
        let info = DirtyInfo::with_expected_files(50);
        assert!(info.timestamp > 0);
        assert_eq!(info.expected_files, Some(50));
    }
}
