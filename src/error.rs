/// Centralized error types for project-rag using thiserror
///
/// Provides domain-specific error types for better error handling and user-facing messages.
use thiserror::Error;

/// Main error type for the RAG system
#[derive(Error, Debug)]
pub enum RagError {
    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),

    #[error("Vector database error: {0}")]
    VectorDb(#[from] VectorDbError),

    #[error("Indexing error: {0}")]
    Indexing(#[from] IndexingError),

    #[error("Chunking error: {0}")]
    Chunking(#[from] ChunkingError),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),

    #[error("Git error: {0}")]
    Git(#[from] GitError),

    #[error("Cache error: {0}")]
    Cache(#[from] CacheError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

/// Errors related to embedding generation
#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Failed to initialize embedding model: {0}")]
    InitializationFailed(String),

    #[error("Failed to generate embeddings: {0}")]
    GenerationFailed(String),

    #[error("Embedding batch is empty")]
    EmptyBatch,

    #[error("Embedding generation timed out after {0} seconds")]
    Timeout(u64),

    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Model lock was poisoned: {0}")]
    LockPoisoned(String),
}

/// Errors related to vector database operations
#[derive(Error, Debug)]
pub enum VectorDbError {
    #[error("Failed to initialize vector database: {0}")]
    InitializationFailed(String),

    #[error("Failed to connect to vector database: {0}")]
    ConnectionFailed(String),

    #[error("Failed to create collection '{collection}': {reason}")]
    CollectionCreationFailed { collection: String, reason: String },

    #[error("Collection '{0}' not found")]
    CollectionNotFound(String),

    #[error("Failed to store embeddings: {0}")]
    StoreFailed(String),

    #[error("Failed to search embeddings: {0}")]
    SearchFailed(String),

    #[error("Failed to delete embeddings: {0}")]
    DeleteFailed(String),

    #[error("Failed to get statistics: {0}")]
    StatisticsFailed(String),

    #[error("Failed to clear database: {0}")]
    ClearFailed(String),

    #[error("Invalid search parameters: {0}")]
    InvalidSearchParams(String),

    #[error("Database is not initialized")]
    NotInitialized,
}

/// Errors related to file indexing
#[derive(Error, Debug)]
pub enum IndexingError {
    #[error("Directory not found: {0}")]
    DirectoryNotFound(String),

    #[error("Path is not a directory: {0}")]
    NotADirectory(String),

    #[error("Failed to walk directory: {0}")]
    WalkFailed(String),

    #[error("Failed to read file '{file}': {reason}")]
    FileReadFailed { file: String, reason: String },

    #[error("File is not valid UTF-8: {0}")]
    InvalidUtf8(String),

    #[error("File is binary and cannot be indexed: {0}")]
    BinaryFile(String),

    #[error("File size exceeds maximum: {size} > {max}")]
    FileTooLarge { size: usize, max: usize },

    #[error("Failed to calculate file hash: {0}")]
    HashCalculationFailed(String),

    #[error("No files found to index")]
    NoFilesFound,

    #[error("Indexing was cancelled")]
    Cancelled,
}

/// Errors related to code chunking
#[derive(Error, Debug)]
pub enum ChunkingError {
    #[error("Failed to parse code: {0}")]
    ParseFailed(String),

    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),

    #[error("Invalid chunk size: {0}")]
    InvalidChunkSize(String),

    #[error("No chunks generated from file: {0}")]
    NoChunksGenerated(String),

    #[error("AST parsing failed: {0}")]
    AstParsingFailed(String),
}

/// Errors related to configuration
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to load configuration file: {0}")]
    LoadFailed(String),

    #[error("Failed to parse configuration: {0}")]
    ParseFailed(String),

    #[error("Invalid configuration value for '{key}': {reason}")]
    InvalidValue { key: String, reason: String },

    #[error("Missing required configuration: {0}")]
    MissingRequired(String),

    #[error("Failed to save configuration: {0}")]
    SaveFailed(String),

    #[error("Configuration file not found: {0}")]
    FileNotFound(String),
}

/// Errors related to input validation
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Path does not exist: {0}")]
    PathNotFound(String),

    #[error("Path is not absolute: {0}")]
    PathNotAbsolute(String),

    #[error("Invalid path: {0}")]
    InvalidPath(String),

    #[error("Invalid project name: {0}")]
    InvalidProjectName(String),

    #[error("Invalid pattern: {0}")]
    InvalidPattern(String),

    #[error("{field} must be {constraint}, got {actual}")]
    ConstraintViolation {
        field: String,
        constraint: String,
        actual: String,
    },

    #[error("Invalid value for {0}: {1}")]
    InvalidValue(String, String),

    #[error("Empty {0}")]
    Empty(String),
}

/// Errors related to git operations
#[derive(Error, Debug)]
pub enum GitError {
    #[error("Git repository not found at: {0}")]
    RepoNotFound(String),

    #[error("Failed to open git repository: {0}")]
    OpenFailed(String),

    #[error("Failed to get git reference: {0}")]
    RefNotFound(String),

    #[error("Failed to iterate commits: {0}")]
    IterFailed(String),

    #[error("Invalid commit hash: {0}")]
    InvalidCommitHash(String),

    #[error("Failed to parse commit: {0}")]
    ParseFailed(String),

    #[error("Branch not found: {0}")]
    BranchNotFound(String),

    #[error("No commits found matching criteria")]
    NoCommitsFound,
}

/// Errors related to cache operations
#[derive(Error, Debug)]
pub enum CacheError {
    #[error("Failed to load cache from '{path}': {reason}")]
    LoadFailed { path: String, reason: String },

    #[error("Failed to save cache to '{path}': {reason}")]
    SaveFailed { path: String, reason: String },

    #[error("Failed to parse cache file: {0}")]
    ParseFailed(String),

    #[error("Cache is corrupted: {0}")]
    Corrupted(String),

    #[error("Failed to create cache directory: {0}")]
    DirectoryCreationFailed(String),
}

// Conversion from anyhow::Error to RagError
impl From<anyhow::Error> for RagError {
    fn from(err: anyhow::Error) -> Self {
        RagError::Other(format!("{:#}", err))
    }
}

// Helper methods for RagError
impl RagError {
    /// Create a new error from a string message
    pub fn other(msg: impl Into<String>) -> Self {
        RagError::Other(msg.into())
    }

    /// Convert to a user-facing error string suitable for MCP responses
    pub fn to_user_string(&self) -> String {
        format!("{}", self)
    }

    /// Check if this is a user error (validation, not found) vs system error
    pub fn is_user_error(&self) -> bool {
        matches!(
            self,
            RagError::Validation(_) | RagError::Config(ConfigError::InvalidValue { .. })
        )
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            RagError::VectorDb(VectorDbError::ConnectionFailed(_))
                | RagError::Embedding(EmbeddingError::Timeout(_))
                | RagError::Io(_)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = RagError::Validation(ValidationError::PathNotFound("/test".to_string()));
        assert_eq!(
            err.to_string(),
            "Validation error: Path does not exist: /test"
        );
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let rag_err: RagError = io_err.into();
        assert!(matches!(rag_err, RagError::Io(_)));
    }

    #[test]
    fn test_error_from_anyhow() {
        let anyhow_err = anyhow::anyhow!("test error");
        let rag_err: RagError = anyhow_err.into();
        assert!(matches!(rag_err, RagError::Other(_)));
    }

    #[test]
    fn test_is_user_error() {
        let user_err = RagError::Validation(ValidationError::InvalidPath("test".to_string()));
        assert!(user_err.is_user_error());

        let system_err = RagError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "test"));
        assert!(!system_err.is_user_error());
    }

    #[test]
    fn test_is_retryable() {
        let retryable = RagError::VectorDb(VectorDbError::ConnectionFailed("test".to_string()));
        assert!(retryable.is_retryable());

        let not_retryable = RagError::Validation(ValidationError::InvalidPath("test".to_string()));
        assert!(!not_retryable.is_retryable());
    }

    #[test]
    fn test_embedding_error_timeout() {
        let err = EmbeddingError::Timeout(30);
        assert_eq!(
            err.to_string(),
            "Embedding generation timed out after 30 seconds"
        );
    }

    #[test]
    fn test_embedding_error_dimension_mismatch() {
        let err = EmbeddingError::DimensionMismatch {
            expected: 384,
            actual: 512,
        };
        assert_eq!(
            err.to_string(),
            "Invalid embedding dimension: expected 384, got 512"
        );
    }

    #[test]
    fn test_vector_db_error_collection_creation() {
        let err = VectorDbError::CollectionCreationFailed {
            collection: "test_collection".to_string(),
            reason: "already exists".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Failed to create collection 'test_collection': already exists"
        );
    }

    #[test]
    fn test_indexing_error_file_too_large() {
        let err = IndexingError::FileTooLarge {
            size: 1000000,
            max: 500000,
        };
        assert_eq!(
            err.to_string(),
            "File size exceeds maximum: 1000000 > 500000"
        );
    }

    #[test]
    fn test_validation_error_constraint() {
        let err = ValidationError::ConstraintViolation {
            field: "max_file_size".to_string(),
            constraint: "less than 100MB".to_string(),
            actual: "200MB".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "max_file_size must be less than 100MB, got 200MB"
        );
    }

    #[test]
    fn test_config_error_invalid_value() {
        let err = ConfigError::InvalidValue {
            key: "port".to_string(),
            reason: "must be between 1-65535".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Invalid configuration value for 'port': must be between 1-65535"
        );
    }

    #[test]
    fn test_cache_error_load_failed() {
        let err = CacheError::LoadFailed {
            path: "/tmp/cache.json".to_string(),
            reason: "permission denied".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Failed to load cache from '/tmp/cache.json': permission denied"
        );
    }

    #[test]
    fn test_rag_error_other() {
        let err = RagError::other("custom error message");
        assert_eq!(err.to_string(), "custom error message");
    }

    #[test]
    fn test_error_chain() {
        let embedding_err = EmbeddingError::GenerationFailed("model error".to_string());
        let rag_err: RagError = embedding_err.into();
        assert!(matches!(rag_err, RagError::Embedding(_)));
        assert_eq!(
            rag_err.to_string(),
            "Embedding error: Failed to generate embeddings: model error"
        );
    }
}
