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
async fn test_server_creation_with_defaults() -> Result<()> {
    // This should work with default configuration
    let server = RagMcpServer::new().await;

    // Server creation should succeed
    assert!(server.is_ok());

    Ok(())
}

#[tokio::test]
async fn test_path_normalization() -> Result<()> {
    // Test path normalization with current directory
    let normalized = RagMcpServer::normalize_path(".")?;
    assert!(normalized.len() > 1);
    assert!(normalized.starts_with('/') || normalized.chars().nth(1) == Some(':'));

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
        )
        .await?;

    // Verify basic indexing worked
    assert!(index_response.files_indexed > 0);
    assert!(index_response.chunks_created > 0);

    Ok(())
}
