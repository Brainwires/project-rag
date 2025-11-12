use super::{DatabaseStats, VectorDatabase};
use crate::types::{ChunkMetadata, SearchResult};
use anyhow::{Context, Result};
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, PointStruct,
    SearchPointsBuilder, UpsertPointsBuilder, VectorParams, VectorsConfig,
};
use qdrant_client::{Payload, Qdrant};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

const COLLECTION_NAME: &str = "code_embeddings";

/// Document frequency statistics for IDF calculation
#[derive(Debug, Clone, Default)]
struct IdfStats {
    /// Total number of documents in corpus
    total_docs: usize,
    /// Term -> number of documents containing that term
    doc_frequencies: HashMap<String, usize>,
}

pub struct QdrantVectorDB {
    client: Qdrant,
    /// IDF statistics for BM25 calculation
    idf_stats: Arc<RwLock<IdfStats>>,
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

        let db = Self {
            client,
            idf_stats: Arc::new(RwLock::new(IdfStats::default())),
        };

        // Initialize IDF stats by scanning existing documents
        if let Err(e) = db.refresh_idf_stats().await {
            tracing::warn!("Failed to initialize IDF stats: {}", e);
        }

        Ok(db)
    }

    /// Refresh IDF statistics by scanning the entire corpus
    async fn refresh_idf_stats(&self) -> Result<()> {
        use qdrant_client::qdrant::ScrollPointsBuilder;

        tracing::info!("Refreshing IDF statistics...");

        let mut doc_frequencies: HashMap<String, usize> = HashMap::new();
        let mut total_docs = 0;
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;

        loop {
            let mut builder = ScrollPointsBuilder::new(COLLECTION_NAME)
                .with_payload(true)
                .limit(100);

            if let Some(ref point_id) = offset {
                builder = builder.offset(point_id.clone());
            }

            let scroll_result = match self.client.scroll(builder).await {
                Ok(result) => result,
                Err(_) => break, // Collection might not exist yet
            };

            if scroll_result.result.is_empty() {
                break;
            }

            for point in &scroll_result.result {
                let payload = &point.payload;
                if let Some(content) = payload.get("content").and_then(|v| v.as_str()) {
                    total_docs += 1;

                    // Extract unique terms from this document
                    let terms = Self::tokenize(content);
                    let unique_terms: std::collections::HashSet<String> =
                        terms.into_iter().collect();

                    for term in unique_terms {
                        *doc_frequencies.entry(term).or_insert(0) += 1;
                    }
                }
            }

            offset = scroll_result.next_page_offset;
            if offset.is_none() {
                break;
            }
        }

        let mut stats = self.idf_stats.write().await;
        stats.total_docs = total_docs;
        stats.doc_frequencies = doc_frequencies;

        tracing::info!(
            "IDF stats refreshed: {} documents, {} unique terms",
            total_docs,
            stats.doc_frequencies.len()
        );

        Ok(())
    }

    /// Tokenize text into terms
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(String::from)
            .collect()
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

    /// Calculate full BM25 score with IDF for a query against content
    async fn calculate_bm25_score(&self, query: &str, content: &str) -> f32 {
        let query_terms = Self::tokenize(query);
        if query_terms.is_empty() {
            return 0.0;
        }

        let content_terms = Self::tokenize(content);
        let content_len = content_terms.len() as f32;

        let stats = self.idf_stats.read().await;
        let total_docs = stats.total_docs as f32;

        // BM25 parameters
        let k1 = 1.5;
        let b = 0.75;
        let avg_doc_len = 100.0; // Approximate, could be calculated from stats

        let mut score = 0.0;

        for term in &query_terms {
            // Term frequency in document
            let tf = content_terms.iter().filter(|t| t == &term).count() as f32;

            if tf > 0.0 {
                // Calculate IDF
                let doc_freq = stats.doc_frequencies.get(term).copied().unwrap_or(1) as f32;
                let idf = ((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln();

                // BM25 formula
                let norm = 1.0 - b + b * (content_len / avg_doc_len);
                let term_score = idf * (tf * (k1 + 1.0)) / (tf + k1 * norm);
                score += term_score;
            }
        }

        // Normalize by number of query terms
        let normalized_score = score / query_terms.len() as f32;

        // Clamp to [0, 1]
        normalized_score.min(1.0).max(0.0)
    }
}

#[async_trait::async_trait]
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
                CreateCollectionBuilder::new(COLLECTION_NAME).vectors_config(VectorsConfig {
                    config: Some(Config::Params(VectorParams {
                        size: dimension as u64,
                        distance: Distance::Cosine.into(),
                        ..Default::default()
                    })),
                }),
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
                    "project": meta.project,
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
            .upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME, points))
            .await
            .context("Failed to upsert points")?;

        // Refresh IDF statistics after adding new documents
        if let Err(e) = self.refresh_idf_stats().await {
            tracing::warn!("Failed to refresh IDF stats after indexing: {}", e);
        }

        Ok(count)
    }

    async fn search(
        &self,
        query_vector: Vec<f32>,
        query_text: &str,
        limit: usize,
        min_score: f32,
        project: Option<String>,
        hybrid: bool,
    ) -> Result<Vec<SearchResult>> {
        self.search_filtered(
            query_vector,
            query_text,
            limit,
            min_score,
            project,
            hybrid,
            vec![],
            vec![],
            vec![],
        )
        .await
    }

    async fn search_filtered(
        &self,
        query_vector: Vec<f32>,
        query_text: &str,
        limit: usize,
        min_score: f32,
        project: Option<String>,
        hybrid: bool,
        file_extensions: Vec<String>,
        languages: Vec<String>,
        path_patterns: Vec<String>,
    ) -> Result<Vec<SearchResult>> {
        tracing::debug!(
            "Searching with limit={}, min_score={}, project={:?}, hybrid={}, filters: ext={:?}, lang={:?}, path={:?}",
            limit,
            min_score,
            project,
            hybrid,
            file_extensions,
            languages,
            path_patterns
        );

        let mut filter = Filter::default();
        let mut must_conditions = vec![];

        // Add project filter
        if let Some(proj) = project {
            must_conditions.push(Condition::matches("project", proj));
        }

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

        let mut search_builder =
            SearchPointsBuilder::new(COLLECTION_NAME, query_vector, limit as u64)
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

        // Collect results with async BM25 scoring
        let mut results: Vec<SearchResult> = Vec::new();

        for point in search_result.result {
            let payload = point.payload;
            let vector_score = point.score;
            let content = match payload.get("content").and_then(|v| v.as_str()) {
                Some(c) => c.to_string(),
                None => continue,
            };

            // Calculate keyword score if hybrid search is enabled
            let (final_score, keyword_score) = if hybrid {
                let kw_score = self.calculate_bm25_score(query_text, &content).await;
                // Combine scores: 70% vector + 30% keyword
                let combined = (vector_score * 0.7) + (kw_score * 0.3);
                (combined, Some(kw_score))
            } else {
                (vector_score, None)
            };

            let file_path = match payload.get("file_path").and_then(|v| v.as_str()) {
                Some(p) => p.to_string(),
                None => continue,
            };

            let start_line = match payload.get("start_line").and_then(|v| v.as_integer()) {
                Some(l) => l as usize,
                None => continue,
            };

            let end_line = match payload.get("end_line").and_then(|v| v.as_integer()) {
                Some(l) => l as usize,
                None => continue,
            };

            let language = payload
                .get("language")
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_else(|| "Unknown".to_string());

            let project = payload
                .get("project")
                .and_then(|v| v.as_str().map(String::from));

            results.push(SearchResult {
                file_path,
                content,
                score: final_score,
                vector_score,
                keyword_score,
                start_line,
                end_line,
                language,
                project,
            });
        }

        // Re-sort by combined score if hybrid
        if hybrid {
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

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
            .delete_points(DeletePointsBuilder::new(COLLECTION_NAME).points(filter))
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

        // Clear IDF stats
        let mut stats = self.idf_stats.write().await;
        stats.total_docs = 0;
        stats.doc_frequencies.clear();

        Ok(())
    }

    async fn get_statistics(&self) -> Result<DatabaseStats> {
        let collection_info = self
            .client
            .collection_info(COLLECTION_NAME)
            .await
            .context("Failed to get collection info")?;

        let points_count = collection_info
            .result
            .and_then(|r| r.points_count)
            .unwrap_or(0);

        // For language breakdown, we'd need to scroll through all points
        // For now, return a simplified version
        Ok(DatabaseStats {
            total_points: points_count as usize,
            total_vectors: points_count as usize,
            language_breakdown: vec![],
        })
    }

    async fn flush(&self) -> Result<()> {
        // Qdrant persists automatically, no explicit flush needed
        Ok(())
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
