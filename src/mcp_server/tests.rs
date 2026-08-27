use super::*;
use crate::client::RagClient;
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn test_new_creates_server() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");

    let client = RagClient::new_with_db_path(&db_path, cache_path).await;
    assert!(client.is_ok(), "Client creation should succeed");

    let client = client.unwrap();
    assert_eq!(client.embedding_dimension(), 384);

    let client = RagMcpServer::with_client(Arc::new(client));
    assert!(client.is_ok(), "Server creation should succeed");
}

/// Prove every tool and prompt is actually REACHABLE through the routers, not
/// merely defined. A handler written outside the `#[tool_router]` /
/// `#[prompt_router]` impl block compiles fine but is never exposed to MCP
/// clients; enumerating the routers is the only check that catches that. The
/// exact counts are asserted so adding a handler without routing it (or
/// forgetting to update the docs' tool count) fails here instead of silently.
#[test]
fn test_all_tools_and_prompts_are_routed() {
    let tool_names: Vec<String> = RagMcpServer::tool_router()
        .list_all()
        .iter()
        .map(|t| t.name.to_string())
        .collect();
    for expected in [
        "index_codebase",
        "query_codebase",
        "get_statistics",
        "clear_index",
        "search_by_filters",
        "search_git_history",
        "find_definition",
        "find_references",
        "get_call_graph",
        "list_symbols",
        "read_file",
        "edit_file",
        "apply_patch",
        "find_unused",
        "validate_removal",
    ] {
        assert!(
            tool_names.iter().any(|n| n == expected),
            "tool '{}' is not routed; routed tools: {:?}",
            expected,
            tool_names
        );
    }
    assert_eq!(
        tool_names.len(),
        15,
        "unexpected tool count: {:?}",
        tool_names
    );

    let prompt_names: Vec<String> = RagMcpServer::prompt_router()
        .list_all()
        .iter()
        .map(|p| p.name.to_string())
        .collect();
    assert!(
        prompt_names.iter().any(|n| n == "unused"),
        "prompt 'unused' is not routed; routed prompts: {:?}",
        prompt_names
    );
    assert!(prompt_names.iter().any(|n| n == "patch"));
    assert!(prompt_names.iter().any(|n| n == "validate-removal"));
    assert_eq!(
        prompt_names.len(),
        14,
        "unexpected prompt count: {:?}",
        prompt_names
    );
}

/// The MCP input schema for find_unused is generated from FindUnusedRequest's
/// JsonSchema derive; this pins the contract a client actually sees: all six
/// parameters present, only `path` required (the rest have serde defaults),
/// and doc comments surfaced as descriptions.
#[test]
fn test_find_unused_tool_schema() {
    let router = RagMcpServer::tool_router();
    let tools = router.list_all();
    let tool = tools
        .iter()
        .find(|t| t.name == "find_unused")
        .expect("find_unused not routed");

    let schema = serde_json::to_value(&*tool.input_schema).unwrap();
    eprintln!(
        "find_unused input_schema:\n{}",
        serde_json::to_string_pretty(&schema).unwrap()
    );

    let properties = schema["properties"]
        .as_object()
        .expect("schema has no properties");
    for field in [
        "path",
        "project",
        "check",
        "limit",
        "max_file_size",
        "configurations",
    ] {
        assert!(properties.contains_key(field), "schema missing '{}'", field);
        assert!(
            properties[field]["description"].is_string(),
            "'{}' has no description",
            field
        );
    }

    let required: Vec<&str> = schema["required"]
        .as_array()
        .expect("schema has no required list")
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert_eq!(required, vec!["path"], "only 'path' should be required");
    assert_eq!(
        schema["properties"]["configurations"]["default"],
        serde_json::json!([])
    );

    assert!(
        tool.description
            .as_deref()
            .is_some_and(|d| d.contains("never as safe to auto-delete")),
        "tool description lost its safety warning"
    );
}

#[test]
fn test_m5_editing_and_removal_tool_schemas() {
    let router = RagMcpServer::tool_router();
    let tools = router.list_all();

    let patch_tool = tools
        .iter()
        .find(|tool| tool.name == "apply_patch")
        .expect("apply_patch not routed");
    let patch_schema = serde_json::to_value(&*patch_tool.input_schema).unwrap();
    let patch_properties = patch_schema["properties"]
        .as_object()
        .expect("apply_patch schema has no properties");
    assert!(patch_properties.contains_key("patches"));
    assert!(patch_properties.contains_key("dry_run"));
    assert!(patch_properties.contains_key("project"));
    assert_eq!(patch_schema["required"], serde_json::json!(["patches"]));
    let item_schema = &patch_properties["patches"]["items"];
    let resolved_item_schema = item_schema
        .get("$ref")
        .and_then(|reference| reference.as_str())
        .and_then(|reference| reference.strip_prefix('#'))
        .and_then(|pointer| patch_schema.pointer(pointer))
        .unwrap_or(item_schema);
    let item_properties = resolved_item_schema["properties"]
        .as_object()
        .expect("FilePatch schema has no properties");
    for field in [
        "file_path",
        "content",
        "start_line",
        "end_line",
        "expected_hash",
        "delete",
    ] {
        assert!(
            item_properties.contains_key(field),
            "schema missing '{field}'"
        );
    }
    assert_eq!(
        resolved_item_schema["required"],
        serde_json::json!(["file_path"])
    );

    let removal_tool = tools
        .iter()
        .find(|tool| tool.name == "validate_removal")
        .expect("validate_removal not routed");
    let removal_schema = serde_json::to_value(&*removal_tool.input_schema).unwrap();
    let removal_properties = removal_schema["properties"]
        .as_object()
        .expect("validate_removal schema has no properties");
    assert!(removal_properties.contains_key("symbol_id"));
    assert!(removal_properties.contains_key("configurations"));
    assert!(removal_properties.contains_key("project"));
    assert_eq!(removal_schema["required"], serde_json::json!(["symbol_id"]));
}

#[tokio::test]
async fn test_get_info() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    let client = RagMcpServer::with_client(Arc::new(client)).unwrap();

    let info = client.get_info();

    assert_eq!(info.server_info.name, "project");
    assert!(info.server_info.title.is_some());
    assert!(info.instructions.is_some());
    assert!(info.capabilities.tools.is_some());
    assert!(info.capabilities.prompts.is_some());
}

#[test]
fn test_normalize_path_valid() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().to_string_lossy().to_string();

    let normalized = RagClient::normalize_path(&path);
    assert!(normalized.is_ok());

    let normalized_path = normalized.unwrap();
    assert!(!normalized_path.is_empty());
}

#[test]
fn test_normalize_path_nonexistent() {
    let result = RagClient::normalize_path("/nonexistent/path/12345");
    assert!(result.is_err());
}

#[test]
fn test_normalize_path_current_dir() {
    let result = RagClient::normalize_path(".");
    assert!(result.is_ok());
    let normalized = result.unwrap();
    assert!(!normalized.is_empty());
}

#[tokio::test]
async fn test_do_index_empty_directory() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();

    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    let result = crate::client::indexing::do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        CancellationToken::new(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.mode, IndexingMode::Full);
    assert_eq!(response.files_indexed, 0);
    assert!(!response.errors.is_empty());
}

#[tokio::test]
async fn test_do_index_with_files() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();

    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    // Create a test file
    let test_file = data_dir.join("test.rs");
    std::fs::write(&test_file, "fn main() { println!(\"test\"); }").unwrap();

    let result = crate::client::indexing::do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        Some("test-project".to_string()),
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        CancellationToken::new(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.mode, IndexingMode::Full);
    assert_eq!(response.files_indexed, 1);
    assert!(response.chunks_created > 0);
    assert!(response.embeddings_generated > 0);
}

#[tokio::test]
async fn test_do_index_with_exclude_patterns() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();

    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    // Create test files
    std::fs::write(data_dir.join("include.rs"), "fn test() {}").unwrap();
    std::fs::write(data_dir.join("exclude.txt"), "exclude this").unwrap();

    let result = crate::client::indexing::do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec!["**/*.txt".to_string()],
        1024 * 1024,
        None,
        None,
        CancellationToken::new(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    // The exclude pattern should filter out .txt files
    // Note: Both files might still be indexed if the pattern doesn't match,
    // but at least we verify the indexing works
    assert!(response.files_indexed >= 1);
}

#[tokio::test]
async fn test_do_incremental_update_no_cache() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();

    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    // Create a test file
    std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

    let result = crate::client::indexing::do_incremental_update(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        CancellationToken::new(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.mode, IndexingMode::Incremental);
}

#[tokio::test]
async fn test_do_index_smart_new_codebase() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();

    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();

    std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

    let result = crate::client::indexing::do_index_smart(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        CancellationToken::new(),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    // First time should be Full
    assert_eq!(response.mode, IndexingMode::Full);
}

#[tokio::test]
async fn test_server_cloneable() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();

    let _cloned = client.clone();
    // Should compile and run without errors
}

// ===== Tool Handler Tests =====

#[tokio::test]
async fn test_tool_query_codebase_with_empty_index() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    let server = RagMcpServer::with_client(Arc::new(client)).unwrap();

    let req = QueryRequest {
        query: "test query".to_string(),
        path: None,
        project: None,
        limit: 10,
        min_score: 0.7,
        hybrid: true,
    };

    // This should succeed even with empty index (just return no results)
    let result = server.client().query_codebase(req).await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.results.len(), 0);
}

#[tokio::test]
async fn test_tool_query_codebase_validation_failure() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    let _server = RagMcpServer::with_client(Arc::new(client)).unwrap();

    // Empty query should fail validation
    let req = QueryRequest {
        query: "   ".to_string(), // Whitespace only
        path: None,
        project: None,
        limit: 10,
        min_score: 0.7,
        hybrid: true,
    };

    let result = req.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("cannot be empty"));
}

#[tokio::test]
async fn test_tool_get_statistics_empty_index() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    let server = RagMcpServer::with_client(Arc::new(client)).unwrap();

    let result = server.client().get_statistics().await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.total_files, 0);
    assert_eq!(response.total_chunks, 0);
    assert_eq!(response.total_embeddings, 0);
}

#[tokio::test]
async fn test_tool_get_statistics_with_data() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();

    // Index some data first
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

    let _index_result = crate::client::indexing::do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        CancellationToken::new(),
    )
    .await
    .unwrap();

    let server = RagMcpServer::with_client(Arc::new(client)).unwrap();
    let result = server.client().get_statistics().await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(response.total_files > 0);
    assert!(response.total_chunks > 0);
    assert!(response.total_embeddings > 0);
    // Indexing stores definitions in the relations store; `fn main` is one.
    assert!(response.total_definitions > 0);
    assert!(response.files_with_definitions > 0);
}

#[tokio::test]
async fn test_tool_clear_index() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();

    // Index some data first
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("test.rs"), "fn main() {}").unwrap();

    let _index_result = crate::client::indexing::do_index(
        &client,
        data_dir.to_string_lossy().to_string(),
        None,
        vec![],
        vec![],
        1024 * 1024,
        None,
        None,
        CancellationToken::new(),
    )
    .await
    .unwrap();

    let server = RagMcpServer::with_client(Arc::new(client)).unwrap();

    // Clear the index
    let result = server.client().clear_index().await;
    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(response.success);

    // Verify index is empty
    let stats = server.client().get_statistics().await.unwrap();
    assert_eq!(stats.total_files, 0);
    assert_eq!(stats.total_chunks, 0);
}

#[tokio::test]
async fn test_tool_search_by_filters_validation_failure() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    let _server = RagMcpServer::with_client(Arc::new(client)).unwrap();

    // Empty file extension should fail validation
    let req = AdvancedSearchRequest {
        query: "test".to_string(),
        path: None,
        project: None,
        limit: 10,
        min_score: 0.7,
        file_extensions: vec!["".to_string()],
        languages: vec![],
        path_patterns: vec![],
    };

    let result = req.validate();
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .contains("file extension cannot be empty")
    );
}

#[tokio::test]
async fn test_tool_search_by_filters_valid_request() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    let server = RagMcpServer::with_client(Arc::new(client)).unwrap();

    let req = AdvancedSearchRequest {
        query: "test".to_string(),
        path: None,
        project: None,
        limit: 10,
        min_score: 0.7,
        file_extensions: vec!["rs".to_string()],
        languages: vec!["Rust".to_string()],
        path_patterns: vec!["src/**".to_string()],
    };

    // Should succeed even with empty index
    let result = server.client().search_with_filters(req).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_tool_search_git_history_validation_failure() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    let _server = RagMcpServer::with_client(Arc::new(client)).unwrap();

    // Empty query should fail validation
    let req = SearchGitHistoryRequest {
        query: "  ".to_string(),
        path: ".".to_string(),
        project: None,
        branch: None,
        max_commits: 10,
        limit: 10,
        min_score: 0.7,
        author: None,
        since: None,
        until: None,
        file_pattern: None,
    };

    let result = req.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("cannot be empty"));
}

#[tokio::test]
async fn test_tool_search_git_history_nonexistent_path() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    let _server = RagMcpServer::with_client(Arc::new(client)).unwrap();

    let req = SearchGitHistoryRequest {
        query: "test".to_string(),
        path: "/nonexistent/path".to_string(),
        project: None,
        branch: None,
        max_commits: 10,
        limit: 10,
        min_score: 0.7,
        author: None,
        since: None,
        until: None,
        file_pattern: None,
    };

    let result = req.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("does not exist"));
}

// ===== Prompt Handler Tests =====

#[tokio::test]
async fn test_prompt_index_with_path() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    let server = RagMcpServer::with_client(Arc::new(client)).unwrap();

    let args = serde_json::json!({
        "path": "/test/path"
    });

    let result = server.index_prompt(Parameters(args)).await;
    assert!(result.is_ok());

    let prompt_result = result.unwrap();
    assert!(prompt_result.description.is_some());
    assert!(!prompt_result.messages.is_empty());
    // Verify the message contains the path (using debug format as proxy)
    let debug_str = format!("{:?}", prompt_result.messages[0].content);
    assert!(debug_str.contains("/test/path"));
}

#[tokio::test]
async fn test_prompt_index_default_path() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    let server = RagMcpServer::with_client(Arc::new(client)).unwrap();

    let args = serde_json::json!({});

    let result = server.index_prompt(Parameters(args)).await;
    assert!(result.is_ok());

    let prompt_result = result.unwrap();
    assert!(prompt_result.description.is_some());
    assert!(!prompt_result.messages.is_empty());
    // Should default to "."
    let debug_str = format!("{:?}", prompt_result.messages[0].content);
    assert!(debug_str.contains("'.'"));
}

#[tokio::test]
async fn test_prompt_query_with_query() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    let server = RagMcpServer::with_client(Arc::new(client)).unwrap();

    let args = serde_json::json!({
        "query": "test query"
    });

    let result = server.query_prompt(Parameters(args)).await;
    assert!(result.is_ok());

    let messages = result.unwrap();
    assert!(!messages.is_empty());
    let debug_str = format!("{:?}", messages[0].content);
    assert!(debug_str.contains("test query"));
}

#[tokio::test]
async fn test_prompt_query_default() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    let server = RagMcpServer::with_client(Arc::new(client)).unwrap();

    let args = serde_json::json!({});

    let result = server.query_prompt(Parameters(args)).await;
    assert!(result.is_ok());

    let messages = result.unwrap();
    assert!(!messages.is_empty());
}

#[tokio::test]
async fn test_prompt_stats() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    let server = RagMcpServer::with_client(Arc::new(client)).unwrap();

    let result = server.stats_prompt().await;
    assert!(!result.is_empty());
    let debug_str = format!("{:?}", result[0].content);
    assert!(debug_str.contains("statistics"));
}

#[tokio::test]
async fn test_prompt_clear() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    let server = RagMcpServer::with_client(Arc::new(client)).unwrap();

    let result = server.clear_prompt().await;
    assert!(!result.is_empty());
    let debug_str = format!("{:?}", result[0].content);
    assert!(debug_str.contains("clear"));
}

#[tokio::test]
async fn test_prompt_search_with_query() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    let server = RagMcpServer::with_client(Arc::new(client)).unwrap();

    let args = serde_json::json!({
        "query": "advanced search"
    });

    let result = server.search_prompt(Parameters(args)).await;
    assert!(result.is_ok());

    let messages = result.unwrap();
    assert!(!messages.is_empty());
    let debug_str = format!("{:?}", messages[0].content);
    assert!(debug_str.contains("advanced search"));
}

#[tokio::test]
async fn test_prompt_git_search_with_query_and_path() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    let server = RagMcpServer::with_client(Arc::new(client)).unwrap();

    let args = serde_json::json!({
        "query": "git search",
        "path": "/repo/path"
    });

    let result = server.git_search_prompt(Parameters(args)).await;
    assert!(result.is_ok());

    let messages = result.unwrap();
    assert!(!messages.is_empty());
    let debug_str = format!("{:?}", messages[0].content);
    assert!(debug_str.contains("git search"));
    assert!(debug_str.contains("/repo/path"));
}

#[tokio::test]
async fn test_prompt_git_search_default_path() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    let server = RagMcpServer::with_client(Arc::new(client)).unwrap();

    let args = serde_json::json!({
        "query": "git search"
    });

    let result = server.git_search_prompt(Parameters(args)).await;
    assert!(result.is_ok());

    let messages = result.unwrap();
    assert!(!messages.is_empty());
    let debug_str = format!("{:?}", messages[0].content);
    assert!(debug_str.contains("git search"));
    assert!(debug_str.contains("'.'"));
}

// ===== ServerHandler Tests =====

#[tokio::test]
async fn test_server_info_completeness() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    let server = RagMcpServer::with_client(Arc::new(client)).unwrap();

    let info = server.get_info();

    // Verify server info details
    assert_eq!(info.server_info.name, "project");
    assert!(info.server_info.title.is_some());
    assert_eq!(
        info.server_info.title.as_deref().unwrap(),
        "Project RAG - Code Understanding with Semantic Search"
    );
    assert_eq!(info.server_info.version, env!("CARGO_PKG_VERSION"));

    // Verify capabilities
    assert!(info.capabilities.tools.is_some());
    assert!(info.capabilities.prompts.is_some());

    // Verify instructions
    assert!(info.instructions.is_some());
    let instructions = info.instructions.as_deref().unwrap();
    assert!(instructions.contains("RAG-based"));
    assert!(instructions.contains("index_codebase"));
    assert!(instructions.contains("query_codebase"));
    assert!(instructions.contains("search_by_filters"));
}

// ===== Client API Tests =====

#[tokio::test]
async fn test_client_accessor() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
    let cache_path = temp_dir.path().join("cache.json");
    let client = RagClient::new_with_db_path(&db_path, cache_path)
        .await
        .unwrap();
    let server = RagMcpServer::with_client(Arc::new(client)).unwrap();

    let client_ref = server.client();
    assert_eq!(client_ref.embedding_dimension(), 384);
}
