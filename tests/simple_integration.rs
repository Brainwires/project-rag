/// Simple integration tests for basic server functionality
use anyhow::Result;
use project_rag::config::Config;
use project_rag::mcp_server::RagMcpServer;
use tempfile::TempDir;

#[tokio::test]
async fn test_server_creation_with_config() -> Result<()> {
    let db_dir = TempDir::new()?;
    let cache_dir = TempDir::new()?;

    let mut config = Config::default();
    config.vector_db.lancedb_path = db_dir.path().to_path_buf();
    config.cache.hash_cache_path = cache_dir.path().join("hash_cache.json");
    config.cache.git_cache_path = cache_dir.path().join("git_cache.json");

    let server = RagMcpServer::with_config(config).await?;

    // Verify server was created successfully
    assert!(std::mem::size_of_val(&server) > 0);

    Ok(())
}

#[tokio::test]
async fn test_server_creation_respects_required_project_path() -> Result<()> {
    // LanceDB deliberately has no process-global default: each MCP project must
    // provide its own storage directory. Exercise `new()` in both supported host
    // states without mutating the process environment shared by parallel tests.
    let server = RagMcpServer::new().await;
    if std::env::var_os("PROJECT_RAG_LANCEDB_PATH").is_some() {
        assert!(server.is_ok());
    } else {
        match server {
            Ok(_) => panic!("server unexpectedly started without a project-local database path"),
            Err(error) => assert!(format!("{error:#}").contains("PROJECT_RAG_LANCEDB_PATH")),
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_path_normalization() -> Result<()> {
    // Test path normalization with current directory
    let normalized = RagMcpServer::normalize_path(".")?;
    assert!(normalized.len() > 1);
    // Canonicalization must yield an absolute path. Checked via Path rather
    // than string shape: on Windows the result is the verbatim form
    // (`\\?\C:\...`), which has neither a leading '/' nor ':' at index 1.
    assert!(std::path::Path::new(&normalized).is_absolute());

    Ok(())
}

#[tokio::test]
async fn test_config_with_custom_batch_size() -> Result<()> {
    let db_dir = TempDir::new()?;
    let cache_dir = TempDir::new()?;

    let mut config = Config::default();
    config.vector_db.lancedb_path = db_dir.path().to_path_buf();
    config.cache.hash_cache_path = cache_dir.path().join("hash_cache.json");
    config.cache.git_cache_path = cache_dir.path().join("git_cache.json");
    config.embedding.batch_size = 64;
    config.embedding.timeout_secs = 60;

    let server = RagMcpServer::with_config(config).await?;

    // Verify server was created with custom config
    assert!(std::mem::size_of_val(&server) > 0);

    Ok(())
}

#[tokio::test]
async fn test_full_indexing_workflow() -> Result<()> {
    let codebase_dir = TempDir::new()?;
    let db_dir = TempDir::new()?;
    let cache_dir = TempDir::new()?;

    // Create a simple test file
    let src_dir = codebase_dir.path().join("src");
    std::fs::create_dir_all(&src_dir)?;
    std::fs::write(src_dir.join("test.rs"), "fn main() { println!(\"test\"); }")?;

    // Create server
    let mut config = Config::default();
    config.vector_db.lancedb_path = db_dir.path().to_path_buf();
    config.cache.hash_cache_path = cache_dir.path().join("hash_cache.json");
    config.cache.git_cache_path = cache_dir.path().join("git_cache.json");

    let server = RagMcpServer::with_config(config).await?;

    // Test path normalization
    let normalized_path = RagMcpServer::normalize_path(&codebase_dir.path().to_string_lossy())?;
    assert!(!normalized_path.is_empty());

    // Test indexing (using the public do_index method)
    let index_response = server
        .do_index(
            normalized_path,
            Some("test_project".to_string()),
            vec![],
            vec![],
            1_048_576,
            None,
            None,
            None, // cancel_token
        )
        .await?;

    // Verify basic indexing worked
    assert!(index_response.files_indexed > 0);
    assert!(index_response.chunks_created > 0);

    Ok(())
}
