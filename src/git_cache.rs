use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Cache for indexed git commits to support incremental updates
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GitCache {
    /// Map of repository path -> set of indexed commit hashes
    pub repos: HashMap<String, HashSet<String>>,
}

impl GitCache {
    /// Get the default cache file path
    pub fn default_path() -> PathBuf {
        crate::paths::PlatformPaths::default_git_cache_path()
    }

    /// Load cache from disk
    pub fn load(cache_path: &Path) -> Result<Self> {
        if !cache_path.exists() {
            tracing::debug!("Git cache file not found, starting with empty cache");
            return Ok(Self::default());
        }

        let content = fs::read_to_string(cache_path).context("Failed to read git cache file")?;

        let cache: GitCache =
            serde_json::from_str(&content).context("Failed to parse git cache file")?;

        tracing::info!(
            "Loaded git cache with {} indexed repositories",
            cache.repos.len()
        );
        Ok(cache)
    }

    /// Save cache to disk
    pub fn save(&self, cache_path: &Path) -> Result<()> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent).context("Failed to create git cache directory")?;
        }

        let content =
            serde_json::to_string_pretty(self).context("Failed to serialize git cache")?;

        fs::write(cache_path, content).context("Failed to write git cache file")?;

        tracing::debug!("Saved git cache to {:?}", cache_path);
        Ok(())
    }

    /// Check if a commit is already indexed
    pub fn has_commit(&self, repo_path: &str, commit_hash: &str) -> bool {
        self.repos
            .get(repo_path)
            .map(|commits| commits.contains(commit_hash))
            .unwrap_or(false)
    }

    /// Get all indexed commits for a repository
    pub fn get_repo(&self, repo_path: &str) -> Option<&HashSet<String>> {
        self.repos.get(repo_path)
    }

    /// Get the count of indexed commits for a repository
    pub fn commit_count(&self, repo_path: &str) -> usize {
        self.repos
            .get(repo_path)
            .map(|commits| commits.len())
            .unwrap_or(0)
    }

    /// Add indexed commits for a repository
    pub fn add_commits(&mut self, repo_path: String, commit_hashes: HashSet<String>) {
        self.repos
            .entry(repo_path)
            .or_insert_with(HashSet::new)
            .extend(commit_hashes);
    }

    /// Update commits for a repository (replaces existing)
    pub fn update_repo(&mut self, repo_path: String, commit_hashes: HashSet<String>) {
        self.repos.insert(repo_path, commit_hashes);
    }

    /// Remove a repository from cache
    pub fn remove_repo(&mut self, repo_path: &str) -> bool {
        self.repos.remove(repo_path).is_some()
    }

    /// Clear all cached repositories
    pub fn clear(&mut self) {
        self.repos.clear();
    }

    /// Get total number of indexed commits across all repos
    pub fn total_commits(&self) -> usize {
        self.repos.values().map(|commits| commits.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_default() {
        let cache = GitCache::default();
        assert_eq!(cache.repos.len(), 0);
    }

    #[test]
    fn test_has_commit() {
        let mut cache = GitCache::default();
        let mut commits = HashSet::new();
        commits.insert("abc123".to_string());
        cache.repos.insert("/repo/path".to_string(), commits);

        assert!(cache.has_commit("/repo/path", "abc123"));
        assert!(!cache.has_commit("/repo/path", "def456"));
        assert!(!cache.has_commit("/other/path", "abc123"));
    }

    #[test]
    fn test_add_commits() {
        let mut cache = GitCache::default();
        let mut commits = HashSet::new();
        commits.insert("abc123".to_string());

        cache.add_commits("/repo/path".to_string(), commits);
        assert_eq!(cache.commit_count("/repo/path"), 1);

        let mut more_commits = HashSet::new();
        more_commits.insert("def456".to_string());
        cache.add_commits("/repo/path".to_string(), more_commits);
        assert_eq!(cache.commit_count("/repo/path"), 2);
    }

    #[test]
    fn test_update_repo() {
        let mut cache = GitCache::default();
        let mut commits1 = HashSet::new();
        commits1.insert("abc123".to_string());
        cache.add_commits("/repo/path".to_string(), commits1);

        let mut commits2 = HashSet::new();
        commits2.insert("def456".to_string());
        cache.update_repo("/repo/path".to_string(), commits2);

        assert_eq!(cache.commit_count("/repo/path"), 1);
        assert!(!cache.has_commit("/repo/path", "abc123"));
        assert!(cache.has_commit("/repo/path", "def456"));
    }

    #[test]
    fn test_remove_repo() {
        let mut cache = GitCache::default();
        let mut commits = HashSet::new();
        commits.insert("abc123".to_string());
        cache.add_commits("/repo/path".to_string(), commits);

        assert!(cache.remove_repo("/repo/path"));
        assert!(!cache.remove_repo("/repo/path"));
        assert_eq!(cache.commit_count("/repo/path"), 0);
    }

    #[test]
    fn test_clear() {
        let mut cache = GitCache::default();
        let mut commits = HashSet::new();
        commits.insert("abc123".to_string());
        cache.add_commits("/repo1".to_string(), commits.clone());
        cache.add_commits("/repo2".to_string(), commits);

        cache.clear();
        assert_eq!(cache.repos.len(), 0);
        assert_eq!(cache.total_commits(), 0);
    }

    #[test]
    fn test_total_commits() {
        let mut cache = GitCache::default();
        let mut commits1 = HashSet::new();
        commits1.insert("abc123".to_string());
        commits1.insert("abc124".to_string());

        let mut commits2 = HashSet::new();
        commits2.insert("def456".to_string());

        cache.add_commits("/repo1".to_string(), commits1);
        cache.add_commits("/repo2".to_string(), commits2);

        assert_eq!(cache.total_commits(), 3);
    }

    #[test]
    fn test_save_load() {
        let dir = tempdir().unwrap();
        let cache_path = dir.path().join("git_cache.json");

        let mut cache = GitCache::default();
        let mut commits = HashSet::new();
        commits.insert("abc123".to_string());
        commits.insert("def456".to_string());
        cache.add_commits("/repo/path".to_string(), commits);

        cache.save(&cache_path).unwrap();
        assert!(cache_path.exists());

        let loaded = GitCache::load(&cache_path).unwrap();
        assert_eq!(loaded.commit_count("/repo/path"), 2);
        assert!(loaded.has_commit("/repo/path", "abc123"));
        assert!(loaded.has_commit("/repo/path", "def456"));
    }

    #[test]
    fn test_load_nonexistent() {
        let dir = tempdir().unwrap();
        let cache_path = dir.path().join("nonexistent.json");

        let cache = GitCache::load(&cache_path).unwrap();
        assert_eq!(cache.repos.len(), 0);
    }

    #[test]
    fn test_save_creates_directory() {
        let dir = tempdir().unwrap();
        let cache_path = dir.path().join("subdir/git_cache.json");

        let cache = GitCache::default();
        cache.save(&cache_path).unwrap();
        assert!(cache_path.exists());
    }
}
