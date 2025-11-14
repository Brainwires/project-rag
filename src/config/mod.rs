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
mod tests;
