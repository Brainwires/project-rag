/// Benchmarks for indexing and search performance
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use project_rag::config::Config;
use project_rag::mcp_server::RagMcpServer;
use tempfile::TempDir;
use tokio::runtime::Runtime;

/// Helper to create test files
fn create_test_files(dir: &TempDir, count: usize) -> anyhow::Result<()> {
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir)?;

    for i in 0..count {
        let content = format!(
            r#"
/// Module {i}
pub mod module_{i} {{
    pub fn function_{i}(x: i32) -> i32 {{
        x * {}
    }}

    pub struct Data{i} {{
        pub value: i32,
        pub name: String,
    }}

    impl Data{i} {{
        pub fn new(value: i32) -> Self {{
            Self {{
                value,
                name: format!("data_{{}}", value),
            }}
        }}

        pub fn process(&self) -> i32 {{
            self.value * 2
        }}
    }}

    #[cfg(test)]
    mod tests {{
        use super::*;

        #[test]
        fn test_function_{i}() {{
            assert_eq!(function_{i}(2), {});
        }}

        #[test]
        fn test_data{i}() {{
            let data = Data{i}::new(5);
            assert_eq!(data.process(), 10);
        }}
    }}
}}
"#,
            i + 1,
            (i + 1) * 2
        );
        std::fs::write(src_dir.join(format!("module_{}.rs", i)), content)?;
    }

    Ok(())
}

fn benchmark_indexing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("indexing");

    for file_count in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_files", file_count)),
            file_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let codebase_dir = TempDir::new().unwrap();
                        let db_dir = TempDir::new().unwrap();
                        let cache_dir = TempDir::new().unwrap();

                        create_test_files(&codebase_dir, count).unwrap();

                        let mut config = Config::default();
                        config.vector_db.lancedb_path = db_dir.path().to_path_buf();
                        config.cache.hash_cache_path = cache_dir.path().join("hash_cache.json");
                        config.cache.git_cache_path = cache_dir.path().join("git_cache.json");

                        let server = RagMcpServer::with_config(config).await.unwrap();

                        let normalized_path =
                            RagMcpServer::normalize_path(&codebase_dir.path().to_string_lossy())
                                .unwrap();

                        server
                            .do_index(
                                black_box(normalized_path),
                                None,
                                vec![],
                                vec![],
                                1_048_576,
                                None,
                                None,
                            )
                            .await
                            .unwrap()
                    })
                });
            },
        );
    }

    group.finish();
}

fn benchmark_chunking(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("chunking");

    for file_count in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_files", file_count)),
            file_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let codebase_dir = TempDir::new().unwrap();
                        create_test_files(&codebase_dir, count).unwrap();

                        let walker = project_rag::indexer::FileWalker::new(
                            codebase_dir.path().to_string_lossy().to_string(),
                            1_048_576,
                        );

                        let files = walker.walk().unwrap();

                        // Benchmark parallel chunking
                        use rayon::prelude::*;
                        let chunker = project_rag::indexer::CodeChunker::default_strategy();
                        let _chunks: Vec<_> = files
                            .par_iter()
                            .flat_map(|file| chunker.chunk_file(black_box(file)))
                            .collect();
                    })
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_indexing, benchmark_chunking);
criterion_main!(benches);
