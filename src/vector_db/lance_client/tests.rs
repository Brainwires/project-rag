mod tests {
    use crate::types::ChunkMetadata;
    use crate::vector_db::{LanceVectorDB, VectorDatabase};
    use tempfile::{TempDir, tempdir};

    fn create_test_metadata(file_path: &str, start_line: usize, end_line: usize) -> ChunkMetadata {
        ChunkMetadata {
            root_path: None,
            file_path: file_path.to_string(),
            project: Some("test-project".to_string()),
            start_line,
            end_line,
            language: Some("Rust".to_string()),
            extension: Some("rs".to_string()),
            file_hash: "test_hash_123".to_string(),
            indexed_at: 1234567890,
        }
    }

    #[tokio::test]
    async fn test_new_creates_instance() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();

        let db = LanceVectorDB::with_path(&db_path).await;
        assert!(db.is_ok());

        let db = db.unwrap();
        assert_eq!(db.table_name, "code_embeddings");
        assert_eq!(db.db_path, db_path);
    }

    #[tokio::test]
    async fn test_default_path() {
        let path = LanceVectorDB::default_lancedb_path();
        assert!(path.contains("project-rag"));
        assert!(path.contains("lancedb"));
    }

    #[tokio::test]
    async fn test_initialize_creates_table() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();

        // Initialize with dimension 384
        let result = db.initialize(384).await;
        assert!(result.is_ok());

        // Table should now exist
        let table_names = db.connection.table_names().execute().await.unwrap();
        assert!(table_names.contains(&"code_embeddings".to_string()));
    }

    #[tokio::test]
    async fn test_initialize_idempotent() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();

        // Initialize twice
        db.initialize(384).await.unwrap();
        let result = db.initialize(384).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_store_embeddings_empty() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        let result = db
            .store_embeddings(vec![], vec![], vec![], "/test/root")
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_store_and_retrieve_embeddings() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Create test embeddings
        let embeddings = vec![vec![0.1; 384], vec![0.2; 384]];
        let metadata = vec![
            create_test_metadata("test1.rs", 1, 10),
            create_test_metadata("test2.rs", 20, 30),
        ];
        let contents = vec!["fn main() {}".to_string(), "fn test() {}".to_string()];

        let count = db
            .store_embeddings(embeddings.clone(), metadata, contents, "/test/root")
            .await
            .unwrap();
        assert_eq!(count, 2);

        // Verify storage by searching
        let query = vec![0.1; 384];
        let results = db
            .search(query, "main", 10, 0.0, None, None, false)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_search_pure_vector() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings
        let embeddings = vec![vec![0.1; 384]];
        let metadata = vec![create_test_metadata("test.rs", 1, 10)];
        let contents = vec!["fn main() {}".to_string()];
        db.store_embeddings(embeddings, metadata, contents, "/test/root")
            .await
            .unwrap();

        // Search with pure vector (hybrid=false)
        let query = vec![0.1; 384];
        let results = db
            .search(query, "main", 10, 0.0, None, None, false)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, "test.rs");
        assert_eq!(results[0].start_line, 1);
        assert_eq!(results[0].end_line, 10);
        assert!(results[0].keyword_score.is_none());
    }

    #[tokio::test]
    async fn test_search_hybrid() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings
        let embeddings = vec![vec![0.1; 384]];
        let metadata = vec![create_test_metadata("test.rs", 1, 10)];
        let contents = vec!["fn main() { println!(\"hello\"); }".to_string()];
        db.store_embeddings(embeddings, metadata, contents, "/test/root")
            .await
            .unwrap();

        // Search with hybrid (hybrid=true)
        let query = vec![0.1; 384];
        let results = db
            .search(query, "println", 10, 0.0, None, None, true)
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].file_path, "test.rs");
        // Hybrid search should have keyword score
        assert!(results[0].keyword_score.is_some());
    }

    /// Regression: hybrid search across MORE THAN ONE insert batch.
    ///
    /// `test_search_hybrid` above stores a single batch into a fresh table, and that is
    /// precisely why it never caught the bug this test exists for. The BM25 arm used to key
    /// documents by `count_rows() + i` (a table row number) while the vector arm keyed by
    /// position within the query's result batches. For the FIRST insert into an empty table
    /// both are 0..n, so fusion appeared to work; from the second insert onward the two key
    /// spaces diverged and RRF fused nothing -- every score collapsed to 1/(60+rank),
    /// keyword_score was never populated, and keyword-only hits were dropped entirely.
    ///
    /// Both arms now key on the chunk's stable `file_path:start_line`.
    #[tokio::test]
    async fn test_search_hybrid_across_multiple_batches() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // The keyword target goes in FIRST, so under the old scheme it held row id 0 -- but
        // its vector is deliberately far from the query, so the vector arm ranks it LAST.
        // That is what breaks the accidental agreement: insertion order and distance order
        // must differ, or row ids and batch positions coincide and the bug hides.
        db.store_embeddings(
            vec![vec![0.9; 384]],
            vec![create_test_metadata("target.rs", 100, 110)],
            vec!["fn zzzuniquesymbol() { /* distinctive */ }".to_string()],
            "/test/root",
        )
        .await
        .unwrap();

        // Two rows close to the query vector, inserted second. The vector arm returns these
        // at positions 0 and 1, so the old code fused BM25's id 0 (target.rs) with whatever
        // sat at position 0 -- a different document entirely.
        db.store_embeddings(
            vec![vec![0.1; 384], vec![0.1; 384]],
            vec![
                create_test_metadata("near1.rs", 1, 10),
                create_test_metadata("near2.rs", 20, 30),
            ],
            vec!["fn alpha() {}".to_string(), "fn beta() {}".to_string()],
            "/test/root",
        )
        .await
        .unwrap();

        // Query text matches ONLY target.rs; query vector is closest to the near*.rs rows.
        let results = db
            .search(
                vec![0.1; 384],
                "zzzuniquesymbol",
                10,
                0.0,
                None,
                None,
                true,
            )
            .await
            .unwrap();

        let hit = results
            .iter()
            .find(|r| r.file_path == "target.rs")
            .expect("keyword hit must survive fusion and be materialised as the RIGHT row");

        assert!(
            hit.keyword_score.is_some(),
            "keyword_score must be populated for a BM25 match; None means the two arms \
             keyed on different id spaces again"
        );
        assert_eq!(hit.start_line, 100);
        assert_eq!(hit.end_line, 110);
    }

    /// Git commits all share `file_path = git://<repo>` and `start_line = 0`, so a fusion
    /// key of path+line alone collapses an entire history onto one id and the BM25 index
    /// holds a single document for it. `file_hash` (the commit hash) is what separates
    /// them. Guards that chunks differing ONLY by file_hash stay individually retrievable.
    #[tokio::test]
    async fn test_commit_chunks_are_not_collapsed_by_shared_path() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        let commit_meta = |hash: &str| {
            let mut m = create_test_metadata("git://repo", 0, 0);
            m.file_hash = hash.to_string();
            m.language = Some("git-commit".to_string());
            m
        };

        db.store_embeddings(
            vec![vec![0.5; 384], vec![0.5; 384]],
            vec![commit_meta("aaaa1111"), commit_meta("bbbb2222")],
            vec![
                "Commit Message:\nfix the alpha subsystem".to_string(),
                "Commit Message:\nrewrite zzzuniquecommit handling".to_string(),
            ],
            "/test/repo",
        )
        .await
        .unwrap();

        let results = db
            .search(
                vec![0.5; 384],
                "zzzuniquecommit",
                10,
                0.0,
                None,
                None,
                true,
            )
            .await
            .unwrap();

        assert_eq!(
            results.len(),
            2,
            "both commits must remain distinct rows, not one collapsed id"
        );
        let matched = results
            .iter()
            .find(|r| r.content.contains("zzzuniquecommit"))
            .expect("the commit matching the query text must be retrievable");
        assert!(
            matched.keyword_score.is_some(),
            "commit chunks must participate in keyword fusion"
        );
    }

    #[tokio::test]
    async fn test_search_with_min_score() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings
        let embeddings = vec![vec![0.1; 384]];
        let metadata = vec![create_test_metadata("test.rs", 1, 10)];
        let contents = vec!["fn main() {}".to_string()];
        db.store_embeddings(embeddings, metadata, contents, "/test/root")
            .await
            .unwrap();

        // Search with high min_score (should filter out results)
        let query = vec![0.9; 384]; // Very different from stored embedding
        let results = db
            .search(query, "main", 10, 0.99, None, None, false)
            .await
            .unwrap();

        // Expect fewer or no results due to high threshold
        assert!(results.len() <= 1);
    }

    #[tokio::test]
    async fn test_search_with_project_filter() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings with different projects
        let embeddings = vec![vec![0.1; 384], vec![0.2; 384]];
        let mut meta1 = create_test_metadata("test1.rs", 1, 10);
        meta1.project = Some("project-a".to_string());
        let mut meta2 = create_test_metadata("test2.rs", 20, 30);
        meta2.project = Some("project-b".to_string());
        let metadata = vec![meta1, meta2];
        let contents = vec!["fn main() {}".to_string(), "fn test() {}".to_string()];

        db.store_embeddings(embeddings, metadata, contents, "/test/root")
            .await
            .unwrap();

        // Search with project filter
        let query = vec![0.15; 384];
        let results = db
            .search(
                query,
                "main",
                10,
                0.0,
                Some("project-a".to_string()),
                None,
                false,
            )
            .await
            .unwrap();

        // Should only get results from project-a
        for result in results {
            assert_eq!(result.project, Some("project-a".to_string()));
        }
    }

    #[tokio::test]
    async fn test_search_filtered_by_extension() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings with different file types
        let embeddings = vec![vec![0.1; 384], vec![0.2; 384]];
        let metadata = vec![
            create_test_metadata("test.rs", 1, 10),
            create_test_metadata("test.toml", 20, 30),
        ];
        let contents = vec!["fn main() {}".to_string(), "[package]".to_string()];

        db.store_embeddings(embeddings, metadata, contents, "/test/root")
            .await
            .unwrap();

        // Search filtered by .rs extension
        let query = vec![0.15; 384];
        let results = db
            .search_filtered(
                query,
                "main",
                10,
                0.0,
                None,
                None,
                false,
                vec!["rs".to_string()],
                vec![],
                vec![],
            )
            .await
            .unwrap();

        // Should only get .rs files
        for result in results {
            assert!(result.file_path.ends_with(".rs"));
        }
    }

    #[tokio::test]
    async fn test_search_filtered_by_language() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings with different languages
        let embeddings = vec![vec![0.1; 384]];
        let metadata = vec![create_test_metadata("test.rs", 1, 10)];
        let contents = vec!["fn main() {}".to_string()];

        db.store_embeddings(embeddings, metadata, contents, "/test/root")
            .await
            .unwrap();

        // Search filtered by Rust language
        let query = vec![0.1; 384];
        let results = db
            .search_filtered(
                query,
                "main",
                10,
                0.0,
                None,
                None,
                false,
                vec![],
                vec!["Rust".to_string()],
                vec![],
            )
            .await
            .unwrap();

        // Should only get Rust files
        for result in results {
            assert_eq!(result.language, "Rust");
        }
    }

    #[tokio::test]
    async fn test_search_filtered_by_path_pattern() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings with different paths
        let embeddings = vec![vec![0.1; 384], vec![0.2; 384]];
        let metadata = vec![
            create_test_metadata("src/main.rs", 1, 10),
            create_test_metadata("tests/test.rs", 20, 30),
        ];
        let contents = vec!["fn main() {}".to_string(), "fn test() {}".to_string()];

        db.store_embeddings(embeddings, metadata, contents, "/test/root")
            .await
            .unwrap();

        // Search filtered by path pattern
        let query = vec![0.15; 384];
        let results = db
            .search_filtered(
                query,
                "main",
                10,
                0.0,
                None,
                None,
                false,
                vec![],
                vec![],
                vec!["src/".to_string()],
            )
            .await
            .unwrap();

        // Should only get files in src/
        for result in results {
            assert!(result.file_path.contains("src/"));
        }
    }

    #[tokio::test]
    async fn test_delete_by_file() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings
        let embeddings = vec![vec![0.1; 384], vec![0.2; 384]];
        let metadata = vec![
            create_test_metadata("test1.rs", 1, 10),
            create_test_metadata("test2.rs", 20, 30),
        ];
        let contents = vec!["fn main() {}".to_string(), "fn test() {}".to_string()];

        db.store_embeddings(embeddings, metadata, contents, "/test/root")
            .await
            .unwrap();

        // Delete one file
        let result = db.delete_by_file("test1.rs").await;
        assert!(result.is_ok());

        // Verify deletion
        let query = vec![0.15; 384];
        let results = db
            .search(query, "main", 10, 0.0, None, None, false)
            .await
            .unwrap();

        // Should not contain deleted file
        for result in &results {
            assert_ne!(result.file_path, "test1.rs");
        }
    }

    #[tokio::test]
    async fn test_clear() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings
        let embeddings = vec![vec![0.1; 384]];
        let metadata = vec![create_test_metadata("test.rs", 1, 10)];
        let contents = vec!["fn main() {}".to_string()];

        db.store_embeddings(embeddings, metadata, contents, "/test/root")
            .await
            .unwrap();

        // Clear database
        let result = db.clear().await;
        assert!(result.is_ok());

        // Table should be gone
        let table_names = db.connection.table_names().execute().await.unwrap();
        assert!(!table_names.contains(&"code_embeddings".to_string()));
    }

    #[tokio::test]
    async fn test_get_statistics_empty() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        let stats = db.get_statistics().await.unwrap();
        assert_eq!(stats.total_points, 0);
        assert_eq!(stats.total_vectors, 0);
        assert_eq!(stats.language_breakdown.len(), 0);
    }

    #[tokio::test]
    async fn test_get_statistics_with_data() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings with different languages
        let embeddings = vec![vec![0.1; 384], vec![0.2; 384], vec![0.3; 384]];
        let mut meta1 = create_test_metadata("test1.rs", 1, 10);
        meta1.language = Some("Rust".to_string());
        let mut meta2 = create_test_metadata("test2.rs", 20, 30);
        meta2.language = Some("Rust".to_string());
        let mut meta3 = create_test_metadata("test3.py", 40, 50);
        meta3.language = Some("Python".to_string());

        let metadata = vec![meta1, meta2, meta3];
        let contents = vec![
            "fn main() {}".to_string(),
            "fn test() {}".to_string(),
            "def main(): pass".to_string(),
        ];

        db.store_embeddings(embeddings, metadata, contents, "/test/root")
            .await
            .unwrap();

        let stats = db.get_statistics().await.unwrap();
        assert_eq!(stats.total_points, 3);
        assert_eq!(stats.total_vectors, 3);
        assert_eq!(stats.language_breakdown.len(), 2);

        // Verify language counts (sorted by count descending)
        assert_eq!(stats.language_breakdown[0].0, "Rust");
        assert_eq!(stats.language_breakdown[0].1, 2);
        assert_eq!(stats.language_breakdown[1].0, "Python");
        assert_eq!(stats.language_breakdown[1].1, 1);
    }

    #[tokio::test]
    async fn test_flush() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();

        // Flush should succeed (no-op for LanceDB)
        let result = db.flush().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_create_schema() {
        let schema = LanceVectorDB::create_schema(384);

        // Verify schema has expected fields (12 fields including root_path)
        assert_eq!(schema.fields().len(), 12);
        assert_eq!(schema.field(0).name(), "vector");
        assert_eq!(schema.field(1).name(), "id");
        assert_eq!(schema.field(2).name(), "file_path");
        assert_eq!(schema.field(3).name(), "root_path");
        assert_eq!(schema.field(4).name(), "start_line");
        assert_eq!(schema.field(5).name(), "end_line");
        assert_eq!(schema.field(6).name(), "language");
        assert_eq!(schema.field(7).name(), "extension");
        assert_eq!(schema.field(8).name(), "file_hash");
        assert_eq!(schema.field(9).name(), "indexed_at");
        assert_eq!(schema.field(10).name(), "content");
        assert_eq!(schema.field(11).name(), "project");
    }

    #[tokio::test]
    async fn test_create_record_batch() {
        let embeddings = vec![vec![0.1; 384], vec![0.2; 384]];
        let metadata = vec![
            create_test_metadata("test1.rs", 1, 10),
            create_test_metadata("test2.rs", 20, 30),
        ];
        let contents = vec!["fn main() {}".to_string(), "fn test() {}".to_string()];
        let schema = LanceVectorDB::create_schema(384);

        let batch = LanceVectorDB::create_record_batch(embeddings, metadata, contents, schema);
        assert!(batch.is_ok());

        let batch = batch.unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 12); // 12 columns including root_path
    }

    #[tokio::test]
    async fn test_multiple_searches() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Store embeddings
        let embeddings = vec![vec![0.1; 384]];
        let metadata = vec![create_test_metadata("test.rs", 1, 10)];
        let contents = vec!["fn main() {}".to_string()];
        db.store_embeddings(embeddings, metadata, contents, "/test/root")
            .await
            .unwrap();

        // Perform multiple searches
        for _ in 0..3 {
            let query = vec![0.1; 384];
            let results = db
                .search(query, "main", 10, 0.0, None, None, false)
                .await
                .unwrap();
            assert_eq!(results.len(), 1);
        }
    }

    #[tokio::test]
    async fn test_per_project_bm25_isolation() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir
            .path()
            .join("lancedb")
            .to_string_lossy()
            .to_string();
        let db = LanceVectorDB::with_path(&db_path).await.unwrap();
        db.initialize(384).await.unwrap();

        // Index first project
        let project1_embeddings = vec![vec![0.1; 384], vec![0.2; 384]];
        let project1_metadata = vec![
            create_test_metadata("/project1/main.rs", 1, 10),
            create_test_metadata("/project1/lib.rs", 1, 20),
        ];
        let project1_contents = vec![
            "fn main() { println!(\"Project 1\"); }".to_string(),
            "pub fn lib_func() {}".to_string(),
        ];
        db.store_embeddings(
            project1_embeddings,
            project1_metadata,
            project1_contents,
            "/normalized/project1",
        )
        .await
        .unwrap();

        // Index second project with different root
        let project2_embeddings = vec![vec![0.3; 384], vec![0.4; 384]];
        let project2_metadata = vec![
            create_test_metadata("/project2/main.rs", 1, 10),
            create_test_metadata("/project2/utils.rs", 1, 15),
        ];
        let project2_contents = vec![
            "fn main() { println!(\"Project 2\"); }".to_string(),
            "pub fn util_func() {}".to_string(),
        ];
        db.store_embeddings(
            project2_embeddings,
            project2_metadata,
            project2_contents,
            "/normalized/project2",
        )
        .await
        .unwrap();

        // Verify both projects can be searched (hybrid search across all BM25 indexes)
        let query = vec![0.15; 384];
        let results = db
            .search(query.clone(), "main", 10, 0.0, None, None, true)
            .await
            .unwrap();

        // Should find results from both projects
        assert!(results.len() >= 2, "Should find results from both projects");

        // Verify BM25 indexes were created for both projects
        let bm25_indexes = db.bm25_indexes.read().unwrap();
        assert_eq!(bm25_indexes.len(), 2, "Should have 2 separate BM25 indexes");

        // Verify the hashes are different for different root paths
        let hash1 = LanceVectorDB::hash_root_path("/normalized/project1");
        let hash2 = LanceVectorDB::hash_root_path("/normalized/project2");
        assert_ne!(
            hash1, hash2,
            "Different root paths should have different hashes"
        );

        assert!(
            bm25_indexes.contains_key(&hash1),
            "Should have index for project1"
        );
        assert!(
            bm25_indexes.contains_key(&hash2),
            "Should have index for project2"
        );
    }
}
