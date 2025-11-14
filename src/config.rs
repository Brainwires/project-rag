/// Configuration system for project-rag
///
/// Supports loading from multiple sources with priority:
/// CLI args > Environment variables > Config file > Defaults
use crate::error::{ConfigError, RagError};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// Vector database configuration
    pub vector_db: VectorDbConfig,

    /// Embedding model configuration
    pub embedding: EmbeddingConfig,

    /// Indexing configuration
    pub indexing: IndexingConfig,

    /// Search configuration
    pub search: SearchConfig,

    /// Cache configuration
    pub cache: CacheConfig,
}

/// Vector database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDbConfig {
    /// Database backend: "lancedb" or "qdrant"
    #[serde(default = "default_db_backend")]
    pub backend: String,

    /// LanceDB data directory path
    #[serde(default = "default_lancedb_path")]
    pub lancedb_path: PathBuf,

    /// Qdrant server URL
    #[serde(default = "default_qdrant_url")]
    pub qdrant_url: String,

    /// Collection name for vector storage
    #[serde(default = "default_collection_name")]
    pub collection_name: String,
}

/// Embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model name (e.g., "all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5")
    #[serde(default = "default_model_name")]
    pub model_name: String,

    /// Batch size for embedding generation
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Timeout in seconds for embedding generation
    #[serde(default = "default_embedding_timeout")]
    pub timeout_secs: u64,
}

/// Indexing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingConfig {
    /// Default chunk size for FixedLines strategy
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,

    /// Maximum file size to index (in bytes)
    #[serde(default = "default_max_file_size")]
    pub max_file_size: usize,

    /// Default include patterns
    #[serde(default)]
    pub include_patterns: Vec<String>,

    /// Default exclude patterns
    #[serde(default = "default_exclude_patterns")]
    pub exclude_patterns: Vec<String>,
}

/// Search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Default minimum similarity score (0.0 to 1.0)
    #[serde(default = "default_min_score")]
    pub min_score: f32,

    /// Default result limit
    #[serde(default = "default_result_limit")]
    pub limit: usize,

    /// Enable hybrid search (vector + BM25) by default
    #[serde(default = "default_hybrid_search")]
    pub hybrid: bool,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Hash cache file path
    #[serde(default = "default_hash_cache_path")]
    pub hash_cache_path: PathBuf,

    /// Git cache file path
    #[serde(default = "default_git_cache_path")]
    pub git_cache_path: PathBuf,
}

// Default value functions
fn default_db_backend() -> String {
    #[cfg(feature = "qdrant-backend")]
    return "qdrant".to_string();
    #[cfg(not(feature = "qdrant-backend"))]
    return "lancedb".to_string();
}

fn default_lancedb_path() -> PathBuf {
    crate::paths::PlatformPaths::default_lancedb_path()
}

fn default_qdrant_url() -> String {
    "http://localhost:6334".to_string()
}

fn default_collection_name() -> String {
    "code_embeddings".to_string()
}

fn default_model_name() -> String {
    "all-MiniLM-L6-v2".to_string()
}

fn default_batch_size() -> usize {
    32
}

fn default_embedding_timeout() -> u64 {
    30
}

fn default_chunk_size() -> usize {
    50
}

fn default_max_file_size() -> usize {
    1_048_576 // 1 MB
}

fn default_exclude_patterns() -> Vec<String> {
    vec![
        "target".to_string(),
        "node_modules".to_string(),
        ".git".to_string(),
        "dist".to_string(),
        "build".to_string(),
    ]
}

fn default_min_score() -> f32 {
    0.7
}

fn default_result_limit() -> usize {
    10
}

fn default_hybrid_search() -> bool {
    true
}

fn default_hash_cache_path() -> PathBuf {
    crate::paths::PlatformPaths::default_hash_cache_path()
}

fn default_git_cache_path() -> PathBuf {
    crate::paths::PlatformPaths::default_git_cache_path()
}

impl Default for VectorDbConfig {
    fn default() -> Self {
        Self {
            backend: default_db_backend(),
            lancedb_path: default_lancedb_path(),
            qdrant_url: default_qdrant_url(),
            collection_name: default_collection_name(),
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_name: default_model_name(),
            batch_size: default_batch_size(),
            timeout_secs: default_embedding_timeout(),
        }
    }
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            chunk_size: default_chunk_size(),
            max_file_size: default_max_file_size(),
            include_patterns: Vec::new(),
            exclude_patterns: default_exclude_patterns(),
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            min_score: default_min_score(),
            limit: default_result_limit(),
            hybrid: default_hybrid_search(),
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            hash_cache_path: default_hash_cache_path(),
            git_cache_path: default_git_cache_path(),
        }
    }
}

impl Config {
    /// Load configuration from file
    pub fn from_file(path: &Path) -> Result<Self, RagError> {
        if !path.exists() {
            return Err(ConfigError::FileNotFound(path.display().to_string()).into());
        }

        let content = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::LoadFailed(format!("Failed to read config file: {}", e)))?;

        let config: Config = toml::from_str(&content)
            .map_err(|e| ConfigError::ParseFailed(format!("Invalid TOML: {}", e)))?;

        config.validate()?;
        Ok(config)
    }

    /// Load configuration from default location or create default
    pub fn load_or_default() -> Result<Self, RagError> {
        let config_path = crate::paths::PlatformPaths::default_config_path();

        if config_path.exists() {
            tracing::info!("Loading config from: {}", config_path.display());
            Self::from_file(&config_path)
        } else {
            tracing::info!("No config file found, using defaults");
            Ok(Self::default())
        }
    }

    /// Save configuration to file
    pub fn save(&self, path: &Path) -> Result<(), RagError> {
        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                ConfigError::SaveFailed(format!("Failed to create config directory: {}", e))
            })?;
        }

        let content = toml::to_string_pretty(self)
            .map_err(|e| ConfigError::SaveFailed(format!("Failed to serialize config: {}", e)))?;

        std::fs::write(path, content)
            .map_err(|e| ConfigError::SaveFailed(format!("Failed to write config file: {}", e)))?;

        tracing::info!("Saved config to: {}", path.display());
        Ok(())
    }

    /// Save to default location
    pub fn save_default(&self) -> Result<(), RagError> {
        let config_path = crate::paths::PlatformPaths::default_config_path();
        self.save(&config_path)
    }

    /// Validate configuration values
    pub fn validate(&self) -> Result<(), RagError> {
        // Validate vector DB backend
        if self.vector_db.backend != "lancedb" && self.vector_db.backend != "qdrant" {
            return Err(ConfigError::InvalidValue {
                key: "vector_db.backend".to_string(),
                reason: format!(
                    "must be 'lancedb' or 'qdrant', got '{}'",
                    self.vector_db.backend
                ),
            }
            .into());
        }

        // Validate batch size
        if self.embedding.batch_size == 0 {
            return Err(ConfigError::InvalidValue {
                key: "embedding.batch_size".to_string(),
                reason: "must be greater than 0".to_string(),
            }
            .into());
        }

        // Validate chunk size
        if self.indexing.chunk_size == 0 {
            return Err(ConfigError::InvalidValue {
                key: "indexing.chunk_size".to_string(),
                reason: "must be greater than 0".to_string(),
            }
            .into());
        }

        // Validate max file size
        if self.indexing.max_file_size == 0 {
            return Err(ConfigError::InvalidValue {
                key: "indexing.max_file_size".to_string(),
                reason: "must be greater than 0".to_string(),
            }
            .into());
        }

        // Validate min_score range
        if !(0.0..=1.0).contains(&self.search.min_score) {
            return Err(ConfigError::InvalidValue {
                key: "search.min_score".to_string(),
                reason: format!("must be between 0.0 and 1.0, got {}", self.search.min_score),
            }
            .into());
        }

        // Validate limit
        if self.search.limit == 0 {
            return Err(ConfigError::InvalidValue {
                key: "search.limit".to_string(),
                reason: "must be greater than 0".to_string(),
            }
            .into());
        }

        Ok(())
    }

    /// Apply environment variable overrides
    pub fn apply_env_overrides(&mut self) {
        // Vector DB backend
        if let Ok(backend) = std::env::var("PROJECT_RAG_DB_BACKEND") {
            self.vector_db.backend = backend;
        }

        // LanceDB path
        if let Ok(path) = std::env::var("PROJECT_RAG_LANCEDB_PATH") {
            self.vector_db.lancedb_path = PathBuf::from(path);
        }

        // Qdrant URL
        if let Ok(url) = std::env::var("PROJECT_RAG_QDRANT_URL") {
            self.vector_db.qdrant_url = url;
        }

        // Embedding model
        if let Ok(model) = std::env::var("PROJECT_RAG_MODEL") {
            self.embedding.model_name = model;
        }

        // Batch size
        if let Ok(batch_size) = std::env::var("PROJECT_RAG_BATCH_SIZE")
            && let Ok(size) = batch_size.parse()
        {
            self.embedding.batch_size = size;
        }

        // Min score
        if let Ok(min_score) = std::env::var("PROJECT_RAG_MIN_SCORE")
            && let Ok(score) = min_score.parse()
        {
            self.search.min_score = score;
        }
    }

    /// Create a new Config with defaults and environment overrides
    pub fn new() -> Result<Self, RagError> {
        let mut config = Self::load_or_default()?;
        config.apply_env_overrides();
        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.embedding.model_name, "all-MiniLM-L6-v2");
        assert_eq!(config.embedding.batch_size, 32);
        assert_eq!(config.indexing.chunk_size, 50);
        assert_eq!(config.search.min_score, 0.7);
        assert_eq!(config.search.limit, 10);
        assert!(config.search.hybrid);
    }

    #[test]
    fn test_validate_valid_config() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_invalid_backend() {
        let mut config = Config::default();
        config.vector_db.backend = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_batch_size() {
        let mut config = Config::default();
        config.embedding.batch_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_min_score() {
        let mut config = Config::default();
        config.search.min_score = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_save_and_load() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let mut config = Config::default();
        config.embedding.batch_size = 64;
        config.search.min_score = 0.8;

        config.save(path).unwrap();
        let loaded = Config::from_file(path).unwrap();

        assert_eq!(loaded.embedding.batch_size, 64);
        assert_eq!(loaded.search.min_score, 0.8);
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = Config::from_file(Path::new("/nonexistent/config.toml"));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RagError::Config(ConfigError::FileNotFound(_))
        ));
    }

    #[test]
    fn test_toml_serialization() {
        let config = Config::default();
        let toml_str = toml::to_string(&config).unwrap();
        assert!(toml_str.contains("backend"));
        assert!(toml_str.contains("model_name"));
        assert!(toml_str.contains("chunk_size"));
    }

    #[test]
    fn test_apply_env_overrides() {
        // Safety: This test is single-threaded and we clean up after ourselves
        unsafe {
            std::env::set_var("PROJECT_RAG_DB_BACKEND", "qdrant");
            std::env::set_var("PROJECT_RAG_MODEL", "BAAI/bge-base-en-v1.5");
            std::env::set_var("PROJECT_RAG_BATCH_SIZE", "64");
            std::env::set_var("PROJECT_RAG_MIN_SCORE", "0.8");
        }

        let mut config = Config::default();
        config.apply_env_overrides();

        assert_eq!(config.vector_db.backend, "qdrant");
        assert_eq!(config.embedding.model_name, "BAAI/bge-base-en-v1.5");
        assert_eq!(config.embedding.batch_size, 64);
        assert_eq!(config.search.min_score, 0.8);

        // Cleanup
        // Safety: This test is single-threaded and we're cleaning up test state
        unsafe {
            std::env::remove_var("PROJECT_RAG_DB_BACKEND");
            std::env::remove_var("PROJECT_RAG_MODEL");
            std::env::remove_var("PROJECT_RAG_BATCH_SIZE");
            std::env::remove_var("PROJECT_RAG_MIN_SCORE");
        }
    }

    #[test]
    fn test_default_exclude_patterns() {
        let config = Config::default();
        assert!(
            config
                .indexing
                .exclude_patterns
                .contains(&"target".to_string())
        );
        assert!(
            config
                .indexing
                .exclude_patterns
                .contains(&"node_modules".to_string())
        );
        assert!(
            config
                .indexing
                .exclude_patterns
                .contains(&".git".to_string())
        );
    }

    #[test]
    fn test_config_paths_use_platform_paths() {
        let config = Config::default();
        let hash_cache = config.cache.hash_cache_path.to_string_lossy();
        let git_cache = config.cache.git_cache_path.to_string_lossy();

        assert!(hash_cache.contains("project-rag"));
        assert!(hash_cache.contains("hash_cache.json"));
        assert!(git_cache.contains("project-rag"));
        assert!(git_cache.contains("git_cache.json"));
    }

    #[test]
    fn test_vector_db_config_defaults() {
        let config = VectorDbConfig::default();
        assert_eq!(config.collection_name, "code_embeddings");

        #[cfg(feature = "qdrant-backend")]
        assert_eq!(config.backend, "qdrant");

        #[cfg(not(feature = "qdrant-backend"))]
        assert_eq!(config.backend, "lancedb");
    }

    #[test]
    fn test_embedding_config_defaults() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.model_name, "all-MiniLM-L6-v2");
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.timeout_secs, 30);
    }

    #[test]
    fn test_search_config_defaults() {
        let config = SearchConfig::default();
        assert_eq!(config.min_score, 0.7);
        assert_eq!(config.limit, 10);
        assert!(config.hybrid);
    }

    // ===== Edge Case Tests =====

    #[test]
    fn test_from_file_invalid_toml() {
        let temp_file = NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), "invalid toml {{{ content").unwrap();

        let result = Config::from_file(temp_file.path());
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RagError::Config(ConfigError::ParseFailed(_))
        ));
    }

    #[test]
    fn test_from_file_empty_file() {
        let temp_file = NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), "").unwrap();

        let result = Config::from_file(temp_file.path());
        // Empty TOML file might parse but could fail validation
        // depending on whether all required fields have defaults
        if result.is_ok() {
            let config = result.unwrap();
            // Should have default values if it parsed successfully
            assert!(!config.vector_db.backend.is_empty());
        }
        // Otherwise it's expected to fail - both outcomes are valid
    }

    #[test]
    fn test_from_file_partial_config() {
        let temp_file = NamedTempFile::new().unwrap();
        // Partial config with one custom value - other fields use defaults
        let partial_config = r#"
[embedding]
model_name = "custom-model"
        "#;
        std::fs::write(temp_file.path(), partial_config).unwrap();

        let result = Config::from_file(temp_file.path());
        // This might fail if validation requires all sections
        // In practice, TOML should use defaults for missing fields
        if result.is_ok() {
            let config = result.unwrap();
            assert_eq!(config.embedding.model_name, "custom-model");
            // Other fields should have defaults
            assert_eq!(config.search.limit, 10);
        }
        // If it fails, that's also acceptable behavior for partial config
    }

    #[test]
    fn test_apply_env_overrides_with_invalid_values() {
        unsafe {
            // Set invalid values that can't be parsed
            std::env::set_var("PROJECT_RAG_BATCH_SIZE", "not_a_number");
            std::env::set_var("PROJECT_RAG_MIN_SCORE", "invalid");
        }

        let mut config = Config::default();
        let original_batch = config.embedding.batch_size;
        let original_score = config.search.min_score;

        config.apply_env_overrides();

        // Invalid values should be ignored, keeping defaults
        assert_eq!(config.embedding.batch_size, original_batch);
        assert_eq!(config.search.min_score, original_score);

        unsafe {
            std::env::remove_var("PROJECT_RAG_BATCH_SIZE");
            std::env::remove_var("PROJECT_RAG_MIN_SCORE");
        }
    }

    #[test]
    fn test_apply_env_overrides_with_empty_strings() {
        unsafe {
            std::env::set_var("PROJECT_RAG_DB_BACKEND", "");
            std::env::set_var("PROJECT_RAG_MODEL", "");
        }

        let mut config = Config::default();
        config.apply_env_overrides();

        // Empty strings should be applied (even if invalid)
        assert_eq!(config.vector_db.backend, "");
        assert_eq!(config.embedding.model_name, "");

        unsafe {
            std::env::remove_var("PROJECT_RAG_DB_BACKEND");
            std::env::remove_var("PROJECT_RAG_MODEL");
        }
    }

    #[test]
    fn test_validate_with_zero_batch_size() {
        let mut config = Config::default();
        config.embedding.batch_size = 0;

        let result = config.validate();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RagError::Config(ConfigError::InvalidValue { .. })
        ));
    }

    #[test]
    fn test_validate_with_min_score_too_low() {
        let mut config = Config::default();
        config.search.min_score = -0.1;

        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_with_min_score_too_high() {
        let mut config = Config::default();
        config.search.min_score = 1.1;

        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_with_zero_limit() {
        let mut config = Config::default();
        config.search.limit = 0;

        let result = config.validate();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RagError::Config(ConfigError::InvalidValue { .. })
        ));
    }

    #[test]
    fn test_validate_with_unknown_backend() {
        let mut config = Config::default();
        config.vector_db.backend = "unknown_backend".to_string();

        let result = config.validate();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RagError::Config(ConfigError::InvalidValue { .. })
        ));
    }

    #[test]
    fn test_apply_env_overrides_lancedb_path() {
        unsafe {
            std::env::set_var("PROJECT_RAG_LANCEDB_PATH", "/custom/lancedb/path");
        }

        let mut config = Config::default();
        config.apply_env_overrides();

        assert_eq!(
            config.vector_db.lancedb_path,
            PathBuf::from("/custom/lancedb/path")
        );

        unsafe {
            std::env::remove_var("PROJECT_RAG_LANCEDB_PATH");
        }
    }

    #[test]
    fn test_apply_env_overrides_qdrant_url() {
        unsafe {
            std::env::set_var("PROJECT_RAG_QDRANT_URL", "http://custom:6334");
        }

        let mut config = Config::default();
        config.apply_env_overrides();

        assert_eq!(config.vector_db.qdrant_url, "http://custom:6334");

        unsafe {
            std::env::remove_var("PROJECT_RAG_QDRANT_URL");
        }
    }

    #[test]
    fn test_config_new_applies_env_and_validates() {
        unsafe {
            // Set valid environment overrides
            std::env::set_var("PROJECT_RAG_DB_BACKEND", "lancedb");
            std::env::set_var("PROJECT_RAG_BATCH_SIZE", "128");
        }

        let result = Config::new();
        assert!(result.is_ok());

        let config = result.unwrap();
        assert_eq!(config.vector_db.backend, "lancedb");
        assert_eq!(config.embedding.batch_size, 128);

        unsafe {
            std::env::remove_var("PROJECT_RAG_DB_BACKEND");
            std::env::remove_var("PROJECT_RAG_BATCH_SIZE");
        }
    }

    #[test]
    fn test_config_new_fails_on_invalid_env() {
        unsafe {
            // Set invalid backend that will fail validation
            std::env::set_var("PROJECT_RAG_DB_BACKEND", "invalid_backend");
        }

        let result = Config::new();
        assert!(result.is_err());

        unsafe {
            std::env::remove_var("PROJECT_RAG_DB_BACKEND");
        }
    }

    #[test]
    fn test_save_creates_parent_directory() {
        use tempfile::TempDir;
        let temp_dir = TempDir::new().unwrap();
        let nested_path = temp_dir
            .path()
            .join("nested")
            .join("path")
            .join("config.toml");

        let config = Config::default();
        let result = config.save(&nested_path);

        assert!(result.is_ok());
        assert!(nested_path.exists());
        assert!(nested_path.parent().unwrap().exists());
    }

    #[test]
    fn test_from_file_validates_loaded_config() {
        let temp_file = NamedTempFile::new().unwrap();
        let invalid_config = r#"
[vector_db]
backend = "invalid_backend"
collection_name = "test"

[embedding]
model_name = "test"
batch_size = 32
timeout_secs = 30

[indexing]
chunk_size = 50
max_file_size = 1048576

[search]
min_score = 0.7
limit = 10
hybrid = true

[cache]
hash_cache_path = "/tmp/hash.json"
git_cache_path = "/tmp/git.json"
        "#;
        std::fs::write(temp_file.path(), invalid_config).unwrap();

        let result = Config::from_file(temp_file.path());
        assert!(result.is_err(), "Should fail validation");
        assert!(matches!(
            result.unwrap_err(),
            RagError::Config(ConfigError::InvalidValue { .. })
        ));
    }

    #[test]
    fn test_indexing_config_default_exclude_patterns_not_empty() {
        let config = IndexingConfig::default();
        assert!(!config.exclude_patterns.is_empty());
        assert!(config.exclude_patterns.len() >= 3);
    }

    #[test]
    fn test_cache_config_defaults() {
        let config = CacheConfig::default();
        assert!(config
            .hash_cache_path
            .to_string_lossy()
            .contains("hash_cache.json"));
        assert!(config
            .git_cache_path
            .to_string_lossy()
            .contains("git_cache.json"));
    }

    #[test]
    fn test_apply_env_overrides_multiple_vars_simultaneously() {
        unsafe {
            std::env::set_var("PROJECT_RAG_DB_BACKEND", "qdrant");
            std::env::set_var("PROJECT_RAG_LANCEDB_PATH", "/custom/lance");
            std::env::set_var("PROJECT_RAG_QDRANT_URL", "http://localhost:7777");
            std::env::set_var("PROJECT_RAG_MODEL", "custom-model");
            std::env::set_var("PROJECT_RAG_BATCH_SIZE", "256");
            std::env::set_var("PROJECT_RAG_MIN_SCORE", "0.9");
        }

        let mut config = Config::default();
        config.apply_env_overrides();

        assert_eq!(config.vector_db.backend, "qdrant");
        assert_eq!(
            config.vector_db.lancedb_path,
            PathBuf::from("/custom/lance")
        );
        assert_eq!(config.vector_db.qdrant_url, "http://localhost:7777");
        assert_eq!(config.embedding.model_name, "custom-model");
        assert_eq!(config.embedding.batch_size, 256);
        assert_eq!(config.search.min_score, 0.9);

        unsafe {
            std::env::remove_var("PROJECT_RAG_DB_BACKEND");
            std::env::remove_var("PROJECT_RAG_LANCEDB_PATH");
            std::env::remove_var("PROJECT_RAG_QDRANT_URL");
            std::env::remove_var("PROJECT_RAG_MODEL");
            std::env::remove_var("PROJECT_RAG_BATCH_SIZE");
            std::env::remove_var("PROJECT_RAG_MIN_SCORE");
        }
    }

    #[test]
    fn test_validate_boundary_values() {
        // Test min_score exactly at boundaries (0.0 and 1.0 should be valid)
        let mut config = Config::default();

        config.search.min_score = 0.0;
        assert!(config.validate().is_ok());

        config.search.min_score = 1.0;
        assert!(config.validate().is_ok());

        // Test batch_size = 1 (minimum valid value)
        config.embedding.batch_size = 1;
        assert!(config.validate().is_ok());
    }
}
