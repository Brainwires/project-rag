use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{Index, IndexWriter, ReloadPolicy, TantivyDocument, doc};

/// BM25-based keyword search using Tantivy
pub struct BM25Search {
    index: Index,
    id_field: Field,
    content_field: Field,
    /// Mutex to ensure only one IndexWriter is created at a time
    writer_lock: Mutex<()>,
}

/// Search result from BM25
#[derive(Debug, Clone)]
pub struct BM25Result {
    pub id: u64,
    pub score: f32,
}

impl BM25Search {
    /// Create a new BM25 search index
    pub fn new<P: AsRef<Path>>(index_path: P) -> Result<Self> {
        let index_path = index_path.as_ref().to_path_buf();

        // Create schema with ID and content fields
        let mut schema_builder = Schema::builder();
        let id_field = schema_builder.add_u64_field("id", STORED | INDEXED);
        let content_field = schema_builder.add_text_field("content", TEXT);
        let schema = schema_builder.build();

        // Create or open index
        std::fs::create_dir_all(&index_path).context("Failed to create BM25 index directory")?;

        let index = if index_path.join("meta.json").exists() {
            Index::open_in_dir(&index_path).context("Failed to open existing BM25 index")?
        } else {
            Index::create_in_dir(&index_path, schema.clone())
                .context("Failed to create BM25 index")?
        };

        Ok(Self {
            index,
            id_field,
            content_field,
            writer_lock: Mutex::new(()),
        })
    }

    /// Add documents to the index
    pub fn add_documents(&self, documents: Vec<(u64, String)>) -> Result<()> {
        // Lock to ensure only one writer at a time
        let _guard = self
            .writer_lock
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire writer lock: {}", e))?;

        let mut index_writer: IndexWriter<TantivyDocument> = self
            .index
            .writer(50_000_000)
            .context("Failed to create index writer")?;

        for (id, content) in documents {
            let doc = doc!(
                self.id_field => id,
                self.content_field => content,
            );
            index_writer
                .add_document(doc)
                .context("Failed to add document")?;
        }

        index_writer
            .commit()
            .context("Failed to commit documents")?;

        Ok(())
    }

    /// Search the index with BM25 scoring
    pub fn search(&self, query_text: &str, limit: usize) -> Result<Vec<BM25Result>> {
        let reader = self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .context("Failed to create index reader")?;

        let searcher = reader.searcher();

        // Parse query
        let query_parser = QueryParser::for_index(&self.index, vec![self.content_field]);
        let query = query_parser
            .parse_query(query_text)
            .context("Failed to parse query")?;

        // Search with BM25
        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(limit))
            .context("Failed to execute search")?;

        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = searcher
                .doc(doc_address)
                .context("Failed to retrieve document")?;

            if let Some(id_value) = retrieved_doc.get_first(self.id_field) {
                if let Some(id) = id_value.as_u64() {
                    results.push(BM25Result { id, score });
                }
            }
        }

        Ok(results)
    }

    /// Delete all documents for a specific ID
    pub fn delete_by_id(&self, id: u64) -> Result<()> {
        // Lock to ensure only one writer at a time
        let _guard = self
            .writer_lock
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire writer lock: {}", e))?;

        let mut index_writer: IndexWriter<TantivyDocument> = self
            .index
            .writer(50_000_000)
            .context("Failed to create index writer")?;

        let term = Term::from_field_u64(self.id_field, id);
        index_writer.delete_term(term);

        index_writer.commit().context("Failed to commit deletion")?;

        Ok(())
    }

    /// Clear the entire index
    pub fn clear(&self) -> Result<()> {
        // Lock to ensure only one writer at a time
        let _guard = self
            .writer_lock
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to acquire writer lock: {}", e))?;

        let mut index_writer: IndexWriter<TantivyDocument> = self
            .index
            .writer(50_000_000)
            .context("Failed to create index writer")?;

        index_writer
            .delete_all_documents()
            .context("Failed to delete all documents")?;

        index_writer.commit().context("Failed to commit clear")?;

        Ok(())
    }

    /// Get index statistics
    pub fn get_stats(&self) -> Result<BM25Stats> {
        let reader = self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .context("Failed to create index reader")?;

        let searcher = reader.searcher();
        let total_docs = searcher.num_docs() as usize;

        Ok(BM25Stats {
            total_documents: total_docs,
        })
    }
}

/// Statistics about the BM25 index
#[derive(Debug, Clone)]
pub struct BM25Stats {
    pub total_documents: usize,
}

/// Reciprocal Rank Fusion (RRF) for combining vector and BM25 results
pub fn reciprocal_rank_fusion(
    vector_results: Vec<(u64, f32)>,
    bm25_results: Vec<BM25Result>,
    k: usize,
) -> Vec<(u64, f32)> {
    const K_CONSTANT: f32 = 60.0; // Standard RRF constant

    let mut score_map: HashMap<u64, f32> = HashMap::new();

    // Add vector scores with RRF formula: 1 / (k + rank)
    for (rank, (id, _score)) in vector_results.iter().enumerate() {
        let rrf_score = 1.0 / (K_CONSTANT + (rank + 1) as f32);
        *score_map.entry(*id).or_insert(0.0) += rrf_score;
    }

    // Add BM25 scores with RRF formula
    for (rank, result) in bm25_results.iter().enumerate() {
        let rrf_score = 1.0 / (K_CONSTANT + (rank + 1) as f32);
        *score_map.entry(result.id).or_insert(0.0) += rrf_score;
    }

    // Convert to vec and sort by combined score
    let mut combined: Vec<(u64, f32)> = score_map.into_iter().collect();
    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Take top k results
    combined.truncate(k);

    combined
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_bm25_search() {
        let temp_dir = tempdir().unwrap();
        let bm25 = BM25Search::new(temp_dir.path()).unwrap();

        // Add test documents
        let docs = vec![
            (1, "authentication using JWT tokens".to_string()),
            (2, "database connection pool management".to_string()),
            (3, "user authentication and authorization".to_string()),
        ];
        bm25.add_documents(docs).unwrap();

        // Search for "authentication"
        let results = bm25.search("authentication", 10).unwrap();

        // Should return docs 1 and 3 (both contain "authentication")
        assert!(results.len() >= 2);
        assert!(results.iter().any(|r| r.id == 1));
        assert!(results.iter().any(|r| r.id == 3));
    }

    #[test]
    fn test_reciprocal_rank_fusion() {
        let vector_results = vec![(1, 0.9), (2, 0.8), (3, 0.7)];
        let bm25_results = vec![
            BM25Result { id: 3, score: 10.0 },
            BM25Result { id: 1, score: 8.0 },
            BM25Result { id: 4, score: 6.0 },
        ];

        let combined = reciprocal_rank_fusion(vector_results, bm25_results, 5);

        // Should have combined scores with ID 1 and 3 ranking higher
        // (they appear in both lists)
        assert!(!combined.is_empty());
        assert!(combined.iter().any(|r| r.0 == 1));
        assert!(combined.iter().any(|r| r.0 == 3));
    }

    #[test]
    fn test_bm25_delete_by_id() {
        let temp_dir = tempdir().unwrap();
        let bm25 = BM25Search::new(temp_dir.path()).unwrap();

        // Add documents
        let docs = vec![
            (1, "test document one".to_string()),
            (2, "test document two".to_string()),
            (3, "test document three".to_string()),
        ];
        bm25.add_documents(docs).unwrap();

        // Search should find all documents with "test"
        let results = bm25.search("test", 10).unwrap();
        assert_eq!(results.len(), 3);

        // Delete document 2
        bm25.delete_by_id(2).unwrap();

        // Search again - should only find 2 documents now
        let results = bm25.search("test", 10).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|r| r.id == 1));
        assert!(results.iter().any(|r| r.id == 3));
        assert!(!results.iter().any(|r| r.id == 2));
    }

    #[test]
    fn test_bm25_clear() {
        let temp_dir = tempdir().unwrap();
        let bm25 = BM25Search::new(temp_dir.path()).unwrap();

        // Add documents
        let docs = vec![
            (1, "content one".to_string()),
            (2, "content two".to_string()),
        ];
        bm25.add_documents(docs).unwrap();

        // Verify documents exist
        let results = bm25.search("content", 10).unwrap();
        assert_eq!(results.len(), 2);

        // Clear the index
        bm25.clear().unwrap();

        // Search should return no results
        let results = bm25.search("content", 10).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_bm25_statistics() {
        let temp_dir = tempdir().unwrap();
        let bm25 = BM25Search::new(temp_dir.path()).unwrap();

        // Initially empty
        let stats = bm25.get_stats().unwrap();
        assert_eq!(stats.total_documents, 0);

        // Add documents
        let docs = vec![
            (1, "document one".to_string()),
            (2, "document two".to_string()),
            (3, "document three".to_string()),
        ];
        bm25.add_documents(docs).unwrap();

        // Check statistics
        let stats = bm25.get_stats().unwrap();
        assert_eq!(stats.total_documents, 3);
    }

    #[test]
    fn test_bm25_empty_query() {
        let temp_dir = tempdir().unwrap();
        let bm25 = BM25Search::new(temp_dir.path()).unwrap();

        let docs = vec![(1, "test content".to_string())];
        bm25.add_documents(docs).unwrap();

        // Empty query should not crash
        let results = bm25.search("", 10);
        assert!(results.is_ok());
    }

    #[test]
    fn test_reciprocal_rank_fusion_edge_cases() {
        // Test with empty vector results
        let vector_results = vec![];
        let bm25_results = vec![
            BM25Result { id: 1, score: 10.0 },
            BM25Result { id: 2, score: 8.0 },
        ];
        let combined = reciprocal_rank_fusion(vector_results, bm25_results, 5);
        assert_eq!(combined.len(), 2);

        // Test with empty BM25 results
        let vector_results = vec![(1, 0.9), (2, 0.8)];
        let bm25_results = vec![];
        let combined = reciprocal_rank_fusion(vector_results, bm25_results, 5);
        assert_eq!(combined.len(), 2);

        // Test with both empty
        let vector_results = vec![];
        let bm25_results = vec![];
        let combined = reciprocal_rank_fusion(vector_results, bm25_results, 5);
        assert_eq!(combined.len(), 0);
    }

    #[test]
    fn test_reciprocal_rank_fusion_scoring() {
        // Test that items appearing in both lists get higher scores
        let vector_results = vec![(1, 0.9), (2, 0.5)];
        let bm25_results = vec![
            BM25Result { id: 1, score: 10.0 },
            BM25Result { id: 3, score: 8.0 },
        ];
        let combined = reciprocal_rank_fusion(vector_results, bm25_results, 5);

        // ID 1 appears in both lists (rank 1 in both), should score highest
        // ID 2 appears only in vector (rank 2)
        // ID 3 appears only in BM25 (rank 2)
        assert!(combined[0].0 == 1, "ID 1 should rank first");
        assert!(
            combined[0].1 > combined[1].1,
            "ID 1 should have highest score"
        );
    }
}
