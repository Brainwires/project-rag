use super::{DatabaseStats, VectorDatabase};
use crate::types::{ChunkMetadata, SearchResult};
use anyhow::{Context, Result};
use qdrant_client::{Payload, Qdrant};
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, PointStruct,
    SearchPointsBuilder, UpsertPointsBuilder, VectorParams, VectorsConfig,
};
use serde_json::json;

const COLLECTION_NAME: &str = "code_embeddings";

pub struct QdrantVectorDB {
    client: Qdrant,
}

impl QdrantVectorDB {
    /// Create a new Qdrant client with default local configuration
    pub async fn new() -> Result<Self> {
        Self::with_url("http://localhost:6334").await
    }

    /// Create a new Qdrant client with a custom URL
    pub async fn with_url(url: &str) -> Result<Self> {
        tracing::info!("Connecting to Qdrant at {}", url);

        let client = Qdrant::from_url(url)
            .build()
            .context("Failed to create Qdrant client")?;

        Ok(Self { client })
    }

    /// Check if collection exists
    async fn collection_exists(&self) -> Result<bool> {
        let collections = self
            .client
            .list_collections()
            .await
            .context("Failed to list collections")?;

        Ok(collections
            .collections
            .iter()
            .any(|c| c.name == COLLECTION_NAME))
    }
}

impl VectorDatabase for QdrantVectorDB {
    async fn initialize(&self, dimension: usize) -> Result<()> {
        if self.collection_exists().await? {
            tracing::info!("Collection '{}' already exists", COLLECTION_NAME);
            return Ok(());
        }

        tracing::info!(
            "Creating collection '{}' with dimension {}",
            COLLECTION_NAME,
            dimension
        );

        self.client
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION_NAME)
                    .vectors_config(VectorsConfig {
                        config: Some(Config::Params(VectorParams {
                            size: dimension as u64,
                            distance: Distance::Cosine.into(),
                            ..Default::default()
                        })),
                    })
            )
            .await
            .context("Failed to create collection")?;

        Ok(())
    }

    async fn store_embeddings(
        &self,
        embeddings: Vec<Vec<f32>>,
        metadata: Vec<ChunkMetadata>,
        contents: Vec<String>,
    ) -> Result<usize> {
        if embeddings.is_empty() {
            return Ok(0);
        }

        let count = embeddings.len();
        tracing::debug!("Storing {} embeddings", count);

        let points: Vec<PointStruct> = embeddings
            .into_iter()
            .zip(metadata.into_iter())
            .zip(contents.into_iter())
            .enumerate()
            .map(|(idx, ((embedding, meta), content))| {
                let payload: Payload = json!({
                    "file_path": meta.file_path,
                    "start_line": meta.start_line,
                    "end_line": meta.end_line,
                    "language": meta.language,
                    "extension": meta.extension,
                    "file_hash": meta.file_hash,
                    "indexed_at": meta.indexed_at,
                    "content": content,
                })
                .try_into()
                .unwrap();

                PointStruct::new(idx as u64, embedding, payload)
            })
            .collect();

        self.client
            .upsert_points(
                UpsertPointsBuilder::new(COLLECTION_NAME, points)
            )
            .await
            .context("Failed to upsert points")?;

        Ok(count)
    }

    async fn search(
        &self,
        query_vector: Vec<f32>,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<SearchResult>> {
        self.search_filtered(query_vector, limit, min_score, vec![], vec![], vec![])
            .await
    }

    async fn search_filtered(
        &self,
        query_vector: Vec<f32>,
        limit: usize,
        min_score: f32,
        file_extensions: Vec<String>,
        languages: Vec<String>,
        path_patterns: Vec<String>,
    ) -> Result<Vec<SearchResult>> {
        tracing::debug!(
            "Searching with limit={}, min_score={}, filters: ext={:?}, lang={:?}, path={:?}",
            limit,
            min_score,
            file_extensions,
            languages,
            path_patterns
        );

        let mut filter = Filter::default();
        let mut must_conditions = vec![];

        // Add file extension filter
        if !file_extensions.is_empty() {
            must_conditions.push(Condition::matches(
                "extension",
                file_extensions.into_iter().collect::<Vec<_>>(),
            ));
        }

        // Add language filter
        if !languages.is_empty() {
            must_conditions.push(Condition::matches(
                "language",
                languages.into_iter().collect::<Vec<_>>(),
            ));
        }

        // Note: Path pattern filtering would require more complex logic
        // For now, we'll do post-filtering in memory for path patterns

        if !must_conditions.is_empty() {
            filter.must = must_conditions;
        }

        let mut search_builder = SearchPointsBuilder::new(COLLECTION_NAME, query_vector, limit as u64)
            .score_threshold(min_score)
            .with_payload(true);

        if !filter.must.is_empty() {
            search_builder = search_builder.filter(filter);
        }

        let search_result = self
            .client
            .search_points(search_builder)
            .await
            .context("Failed to search points")?;

        let mut results: Vec<SearchResult> = search_result
            .result
            .into_iter()
            .filter_map(|point| {
                let payload = point.payload;
                let score = point.score;

                Some(SearchResult {
                    file_path: payload.get("file_path")?.as_str()?.to_string(),
                    content: payload.get("content")?.as_str()?.to_string(),
                    score,
                    start_line: payload.get("start_line")?.as_integer()? as usize,
                    end_line: payload.get("end_line")?.as_integer()? as usize,
                    language: payload
                        .get("language")
                        .and_then(|v| v.as_str().map(String::from)),
                })
            })
            .collect();

        // Post-filter by path patterns if needed
        if !path_patterns.is_empty() {
            results.retain(|r| {
                path_patterns
                    .iter()
                    .any(|pattern| r.file_path.contains(pattern))
            });
        }

        Ok(results)
    }

    async fn delete_by_file(&self, file_path: &str) -> Result<usize> {
        tracing::debug!("Deleting embeddings for file: {}", file_path);

        let filter = Filter::must([Condition::matches("file_path", file_path.to_string())]);

        self.client
            .delete_points(
                DeletePointsBuilder::new(COLLECTION_NAME)
                    .points(filter)
            )
            .await
            .context("Failed to delete points")?;

        // Note: Qdrant doesn't return the count of deleted points directly
        // We return 0 as a placeholder
        Ok(0)
    }

    async fn clear(&self) -> Result<()> {
        tracing::info!("Clearing all embeddings from collection");

        self.client
            .delete_collection(COLLECTION_NAME)
            .await
            .context("Failed to delete collection")?;

        Ok(())
    }

    async fn get_statistics(&self) -> Result<DatabaseStats> {
        let collection_info = self
            .client
            .collection_info(COLLECTION_NAME)
            .await
            .context("Failed to get collection info")?;

        let points_count = collection_info.result.and_then(|r| r.points_count).unwrap_or(0);

        // For language breakdown, we'd need to scroll through all points
        // For now, return a simplified version
        Ok(DatabaseStats {
            total_points: points_count as usize,
            total_vectors: points_count as usize,
            language_breakdown: vec![],
        })
    }
}

impl Default for QdrantVectorDB {
    fn default() -> Self {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(Self::new())
            .expect("Failed to create default Qdrant client")
    }
}
