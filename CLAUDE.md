# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Project RAG is a Rust-based Model Context Protocol (MCP) server that provides AI assistants with RAG (Retrieval-Augmented Generation) capabilities for understanding massive codebases. It uses FastEmbed for local embeddings and Qdrant for vector storage, enabling semantic search across indexed codebases.

**Key Technology Stack:**
- Rust 2024 edition with async/await (Tokio)
- MCP protocol via `rmcp` crate (v0.8) with macros
- FastEmbed (all-MiniLM-L6-v2 model, 384 dimensions)
- Qdrant vector database (localhost:6334)
- File walking with .gitignore support via `ignore` crate

## Essential Commands

### Building and Running
```bash
# Build debug version
cargo build

# Build optimized release version
cargo build --release

# Quick compile check without building
cargo check

# Run the MCP server over stdio
cargo run
# Or directly:
./target/release/project-rag
```

### Testing
```bash
# Run all unit tests (10 tests in types.rs and chunker.rs)
cargo test --lib

# Run tests for specific module
cargo test --lib types::tests
cargo test --lib chunker::tests

# Run with verbose output
cargo test --lib -- --nocapture

# Run tests with debug logging
RUST_LOG=debug cargo test --lib -- --nocapture
```

### Code Quality
```bash
# Format code (required before commits)
cargo fmt

# Check lints with clippy
cargo clippy

# Auto-fix clippy suggestions
cargo clippy --fix
```

### Debugging
```bash
# Run with debug logging
RUST_LOG=debug cargo run

# Run with trace-level logging
RUST_LOG=trace cargo run
```

### Qdrant Management
```bash
# Start Qdrant via Docker (required for operation)
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_data:/qdrant/storage \
    qdrant/qdrant

# Check Qdrant health
curl http://localhost:6334/health

# View Qdrant logs
docker logs <container-id>
```

## Architecture

### Core Design Principles

1. **Modular Trait-Based Design**: Each major component is defined by a trait (EmbeddingProvider, VectorDatabase) with concrete implementations, enabling easy swapping of backends.

2. **MCP Protocol Integration**: Uses `rmcp` macros (`#[tool]`, `#[prompt]`, `#[tool_router]`, `#[prompt_router]`) to define 6 MCP tools and 6 slash commands. The server communicates over stdio following MCP spec.

3. **Async-First Architecture**: Built on Tokio runtime with async traits. File walking runs on blocking threads via `tokio::task::spawn_blocking` to avoid blocking the async runtime.

4. **Incremental Updates**: Tracks file hashes (SHA256) in memory (`indexed_roots: Arc<RwLock<HashMap<String, HashMap<String, String>>>>`) to detect changes and only re-index modified files.

### Module Structure

```
src/
├── mcp_server.rs           # Main MCP server with 6 tools + 6 prompts
│   ├── RagMcpServer        # Server state (embedding provider, vector DB, chunker)
│   ├── Tool handlers       # index_codebase, query_codebase, get_statistics, etc.
│   └── Prompt handlers     # Slash commands for each tool
├── embedding/              # Embedding generation abstraction
│   ├── mod.rs              # EmbeddingProvider trait
│   └── fastembed_manager.rs # FastEmbed implementation (unsafe workaround for mutability)
├── vector_db/              # Vector database abstraction
│   ├── mod.rs              # VectorDatabase trait
│   └── qdrant_client.rs    # Qdrant implementation using builder patterns
├── indexer/                # File processing and chunking
│   ├── file_walker.rs      # Directory traversal with .gitignore support
│   └── chunker.rs          # Code chunking (FixedLines, SlidingWindow strategies)
├── types.rs                # MCP request/response types with JSON schema
├── main.rs                 # Binary entry point (calls RagMcpServer::serve_stdio)
└── lib.rs                  # Library root with module exports
```

### Critical Implementation Details

**1. MCP Server Pattern (mcp_server.rs)**
- Uses `#[tool_router]` and `#[prompt_router]` macros to generate routers
- Tools return `Result<String, String>` (JSON-serialized responses)
- Prompts return `Vec<PromptMessage>` for slash command expansion
- Server implements `ServerHandler` trait with `#[tool_handler]` and `#[prompt_handler]`

**2. Embedding Provider (embedding/fastembed_manager.rs)**
- Wraps FastEmbed's `TextEmbedding` model (all-MiniLM-L6-v2)
- **UNSAFE WORKAROUND**: Uses `unsafe { &mut *(self as *const Self as *mut Self) }` to get mutable access for model initialization
- **TODO**: Should be refactored to use `Arc<Mutex<TextEmbedding>>` for safe mutability
- Batch embedding: 32 chunks per batch (configurable)

**3. Vector Database (vector_db/qdrant_client.rs)**
- Uses builder patterns: `UpsertPointsBuilder`, `SearchPointsBuilder`, `DeletePointsBuilder`
- Collection: "code_embeddings" (auto-created with dimension from embedding provider)
- Payload stores: file_path, start_line, end_line, language, extension, file_hash, indexed_at, content
- Search uses cosine similarity with configurable min_score threshold

**4. File Walking (indexer/file_walker.rs)**
- **CRITICAL**: Runs on blocking thread via `spawn_blocking` (CPU-intensive I/O)
- Uses `ignore` crate's `WalkBuilder` with gitignore support
- Binary detection: 30% non-printable byte threshold
- Skips files that fail UTF-8 validation (logged and ignored)
- Supports include/exclude patterns (simple substring matching, not glob)

**5. Code Chunking (indexer/chunker.rs)**
- Default: FixedLines(50) - 50 lines per chunk
- Alternative: SlidingWindow with configurable overlap
- Skips empty chunks (whitespace-only)
- Metadata tracks: file_path, line ranges, language, extension, hash, timestamp

**6. Incremental Updates**
- Compares current file hashes with cached hashes in `indexed_roots`
- Detects: new files (no old hash), modified files (hash changed), deleted files (in cache but not on disk)
- Deletes old embeddings before re-indexing modified files
- Updates cache after successful indexing

## Development Guidelines

### Source File Size Constraint
**CRITICAL**: All source files must stay under 600 lines. This is enforced in the project. If adding features, break large files into submodules.

### Error Handling
- Use `anyhow::Result` for functions that can fail
- Add context with `.context("Descriptive error message")`
- Return formatted errors in MCP tools: `.map_err(|e| format!("{:#}", e))`
- Use alternate display (`{:#}`) to show full error chain

### Testing Requirements
- Add unit tests for all new functionality
- Tests in same file using `#[cfg(test)]` module
- Test serialization/deserialization for all request/response types
- Mock file system for file walker tests (use `create_test_file_info`)

### Async Patterns
- Use `tokio::spawn_blocking` for CPU-intensive or blocking I/O operations
- Prefer `Arc<T>` over `Arc<RwLock<T>>` when possible (immutable shared state)
- Use `Arc<RwLock<T>>` for mutable shared state (e.g., indexed_roots cache)
- Batch operations to reduce async overhead (32 chunks per embedding batch)

### MCP Tool Development
When adding new tools:
1. Define request/response types in `types.rs` with `#[derive(JsonSchema)]`
2. Add tool method in `#[tool_router]` impl block with `#[tool]` attribute
3. Add corresponding prompt in `#[prompt_router]` impl block with `#[prompt]` attribute
4. Return `Result<String, String>` from tools (serialize response to JSON)
5. Return `Vec<PromptMessage>` from prompts (user messages for slash commands)

## Known Issues and Limitations

### Technical Debt
1. **FastEmbed Mutability**: Uses unsafe workaround in `fastembed_manager.rs:40-44`. Should refactor to `Arc<Mutex<TextEmbedding>>`.

2. **Async Trait Warnings**: 9 harmless warnings about `async fn` in public traits. Cosmetic issue only.

3. **Pattern Matching**: `file_walker.rs` uses simple substring matching for include/exclude patterns, not proper glob matching.

4. **Hash Cache Persistence**: File hashes stored in memory only. Should persist to disk for true incremental updates across server restarts.

### Build Notes
- Requires Qdrant running on localhost:6334 (fails at runtime if not available)
- First run downloads ~50MB embedding model to cache (~/.cache/fastembed)
- Release builds use aggressive optimization (takes longer to compile)

### Performance Characteristics
- Indexing: ~1000 files/minute (depends on file size and CPU)
- Search latency: 20-30ms per query (HNSW index)
- Memory: ~100MB base + 50MB model + ~40MB per 10k chunks
- Storage: ~1.5KB per chunk in Qdrant

## MCP Protocol Details

### Tool Schemas
All tools return JSON responses conforming to types defined in `types.rs`. Key response types:
- `IndexResponse`: files_indexed, chunks_created, embeddings_generated, duration_ms, errors
- `QueryResponse`: results (SearchResult[]), duration_ms
- `StatisticsResponse`: total_files, total_chunks, language_breakdown
- `IncrementalUpdateResponse`: files_added, files_updated, files_removed, chunks_modified

### Prompt (Slash Command) Pattern
Prompts in `#[prompt_router]` expand to user messages that instruct the AI to call the corresponding tool. Example:
```rust
#[prompt(name = "index", description = "...")]
async fn index_prompt(&self, Parameters(args): Parameters<serde_json::Value>)
    -> Result<GetPromptResult, McpError>
```

### Server Capabilities
Defined in `ServerHandler::get_info()`:
- Tools: Enabled (6 tools available)
- Prompts: Enabled (6 slash commands available)
- Resources: Not implemented
- Sampling: Not implemented

## Common Development Tasks

### Adding a New Programming Language
1. Update `detect_language()` in `indexer/file_walker.rs:180-213`
2. Add extension to match arm (e.g., `"zig" => "Zig"`)
3. No other changes needed (chunking and embedding are language-agnostic)

### Changing Chunk Size
1. Modify default in `CodeChunker::default_strategy()` in `indexer/chunker.rs:24-26`
2. Or pass custom `ChunkStrategy` when creating chunker
3. Re-index codebase for changes to take effect

### Adding a New MCP Tool
1. Define `FooRequest` and `FooResponse` in `types.rs` with JSON schema
2. Add test module in `types.rs` for serialization
3. Implement `do_foo()` helper in `RagMcpServer` impl block
4. Add `#[tool]` method in `#[tool_router]` block calling `do_foo()`
5. Add `#[prompt]` method in `#[prompt_router]` block for slash command
6. Update server count in comments (e.g., "7 tools" instead of "6 tools")

### Debugging MCP Communication
- Run server with `RUST_LOG=debug` to see MCP messages
- Check stdio input/output (server reads from stdin, writes to stdout)
- Validate JSON-RPC format: `{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": 1}`
- Use `mcp_test_minimal.rs` as reference for working MCP server pattern

## Dependencies and Updates

### Critical Dependencies
- `rmcp` (0.8): MCP protocol SDK - breaking changes possible in future versions
- `qdrant-client` (1.15): Requires builder patterns for all operations
- `fastembed` (5.1): Model downloads from HuggingFace on first run
- `tokio` (1.43): Full feature set required for async runtime

### Dependency Update Strategy
- **rmcp**: Check changelog carefully, macro syntax may change
- **qdrant-client**: Builder patterns are stable, but check for breaking changes
- **fastembed**: Model compatibility may change between versions
- Run full test suite after updating any dependency

## Additional Documentation
- See `README.md` for user-facing documentation
- See `NOTES.md` for development notes and known issues
- See `TEST_RESULTS.md` for test suite details
- See `COVERAGE_ANALYSIS.md` for detailed test coverage
