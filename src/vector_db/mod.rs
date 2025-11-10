mod qdrant_client;

pub use qdrant_client::QdrantVectorDB;

use crate::types::{ChunkMetadata, SearchResult};
use anyhow::Result;

/// Trait for vector database operations
pub trait VectorDatabase: Send + Sync {
    /// Initialize the database and create collections if needed
    async fn initialize(&self, dimension: usize) -> Result<()>;

    /// Store embeddings with metadata
    async fn store_embeddings(
        &self,
        embeddings: Vec<Vec<f32>>,
        metadata: Vec<ChunkMetadata>,
        contents: Vec<String>,
    ) -> Result<usize>;

    /// Search for similar vectors
    async fn search(
        &self,
        query_vector: Vec<f32>,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<SearchResult>>;

    /// Search with filters
    async fn search_filtered(
        &self,
        query_vector: Vec<f32>,
        limit: usize,
        min_score: f32,
        file_extensions: Vec<String>,
        languages: Vec<String>,
        path_patterns: Vec<String>,
    ) -> Result<Vec<SearchResult>>;

    /// Delete embeddings for a specific file
    async fn delete_by_file(&self, file_path: &str) -> Result<usize>;

    /// Clear all embeddings
    async fn clear(&self) -> Result<()>;

    /// Get statistics
    async fn get_statistics(&self) -> Result<DatabaseStats>;
}

#[derive(Debug, Clone)]
pub struct DatabaseStats {
    pub total_points: usize,
    pub total_vectors: usize,
    pub language_breakdown: Vec<(String, usize)>,
}
