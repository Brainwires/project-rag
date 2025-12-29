# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Project RAG is a Rust-based Model Context Protocol (MCP) server that provides AI assistants with RAG (Retrieval-Augmented Generation) capabilities for understanding massive codebases. It uses FastEmbed for local embeddings and Qdrant for vector storage, enabling semantic search across indexed codebases.

**Key Technology Stack:**
- Rust 2024 edition with async/await (Tokio)
- MCP protocol via `rmcp` crate (v0.8) with macros
- FastEmbed (all-MiniLM-L6-v2 model, 384 dimensions)
- LanceDB vector database (default, embedded) or Qdrant (optional, external server)
- Tantivy BM25 keyword search with Reciprocal Rank Fusion (RRF) for hybrid search
- Tree-sitter AST-based chunking for 12 languages
- Persistent hash cache for incremental updates across restarts
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
# Run all unit tests (386 tests across all modules)
cargo test --lib

# Run tests for specific module
cargo test --lib types::tests
cargo test --lib chunker::tests
cargo test --lib cache::tests
cargo test --lib bm25_search::tests

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

### Vector Database Management

**Default (LanceDB - Embedded)**
```bash
# No external setup required - LanceDB is embedded
# Data stored in ./.lancedb directory by default
```

**Optional (Qdrant - External Server)**
```bash
# Build with Qdrant backend
cargo build --release --no-default-features --features qdrant-backend

# Start Qdrant via Docker
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

2. **MCP Protocol Integration**: Uses `rmcp` macros (`#[tool]`, `#[prompt]`, `#[tool_router]`, `#[prompt_router]`) to define 9 MCP tools and 9 slash commands. The server communicates over stdio following MCP spec.

3. **Async-First Architecture**: Built on Tokio runtime with async traits. File walking runs on blocking threads via `tokio::task::spawn_blocking` to avoid blocking the async runtime.

4. **Smart Indexing with Auto-Detection**: The `index_codebase` tool automatically detects whether to perform full indexing (new codebase) or incremental updates (previously indexed). Tracks file hashes (SHA256) in persistent cache (`.cache/project-rag/hash_cache.json`) to detect changes across server restarts.

5. **Hybrid Search**: Combines vector similarity (semantic understanding) with BM25 keyword matching using Reciprocal Rank Fusion (RRF) for optimal search results. Tantivy provides full-text search capabilities with IDF scoring.

### Module Structure

```
src/
├── mcp_server.rs           # Main MCP server with 9 tools + 9 prompts
│   ├── RagMcpServer        # Server state (embedding provider, vector DB, chunker, hash cache)
│   ├── Tool handlers       # index_codebase (smart), query_codebase, find_definition, etc.
│   └── Prompt handlers     # Slash commands for each tool
├── client/                 # High-level client API
│   ├── mod.rs              # RagClient - unified interface for all operations
│   └── indexing/           # Indexing pipeline with progress reporting
├── embedding/              # Embedding generation abstraction
│   ├── mod.rs              # EmbeddingProvider trait
│   └── fastembed_manager.rs # FastEmbed implementation (unsafe workaround for mutability)
├── vector_db/              # Vector database abstraction
│   ├── mod.rs              # VectorDatabase trait
│   ├── lance_client.rs     # LanceDB implementation (default, embedded)
│   └── qdrant_client.rs    # Qdrant implementation (optional, external server)
├── indexer/                # File processing and chunking
│   ├── mod.rs              # Module exports
│   ├── file_walker.rs      # Directory traversal with .gitignore support
│   ├── chunker.rs          # Code chunking (FixedLines, SlidingWindow, AST-based)
│   └── ast_parser.rs       # Tree-sitter AST parsing for 12 languages
├── relations/              # Code relationship analysis (definitions, references, call graphs)
│   ├── mod.rs              # RelationsProvider trait, HybridRelationsProvider
│   ├── types.rs            # SymbolId, Definition, Reference, CallEdge types
│   ├── repomap/            # AST-based symbol extraction (fallback provider)
│   │   ├── mod.rs          # RepoMapProvider implementing RelationsProvider
│   │   ├── symbol_extractor.rs  # Extract definitions from AST nodes
│   │   └── reference_finder.rs  # Find references via identifier matching
│   ├── storage/            # Relations storage layer
│   │   ├── mod.rs          # RelationsStore trait
│   │   └── lance_store.rs  # LanceDB storage for definitions/references
│   └── stack_graphs/       # Optional: High-precision name resolution (feature-gated)
│       └── mod.rs          # StackGraphsProvider for Python, TypeScript, Java, Ruby
├── bm25_search.rs          # Tantivy BM25 keyword search with RRF fusion
├── cache.rs                # Persistent hash cache for incremental updates
├── types/                  # MCP request/response types with JSON schema
│   └── mod.rs              # All request/response types including relations
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

**3. Vector Database**
- **LanceDB (Default)**: Embedded database with columnar storage, no external dependencies
  - Hybrid search built-in: Tantivy BM25 + LanceDB vector using RRF
  - Data stored in `.lancedb` directory
  - Zero-copy memory-mapped files for fast queries
- **Qdrant (Optional)**: External server-based vector database
  - Uses builder patterns: `UpsertPointsBuilder`, `SearchPointsBuilder`, `DeletePointsBuilder`
  - Requires external Qdrant server on localhost:6334
- Collection: "code_embeddings" (auto-created with dimension from embedding provider)
- Payload stores: file_path, start_line, end_line, language, extension, file_hash, indexed_at, content, project
- Search uses cosine similarity with configurable min_score threshold

**4. File Walking (indexer/file_walker.rs)**
- **CRITICAL**: Runs on blocking thread via `spawn_blocking` (CPU-intensive I/O)
- Uses `ignore` crate's `WalkBuilder` with gitignore support
- Binary detection: 30% non-printable byte threshold
- Skips files that fail UTF-8 validation (logged and ignored)
- Supports include/exclude patterns (simple substring matching, not glob)

**5. Code Chunking (indexer/chunker.rs and ast_parser.rs)**
- Default: Hybrid AST-based with fallback to FixedLines(50)
- AST parsing uses Tree-sitter for semantic code extraction
- Supported languages (12): Rust, Python, JavaScript, TypeScript, Go, Java, Swift, C, C++, C#, Ruby, PHP
- Extracts functions, classes, methods, structs as semantic chunks
- Falls back to 50 lines per chunk for unsupported languages
- Alternative: SlidingWindow with configurable overlap
- Skips empty chunks (whitespace-only)
- Metadata tracks: file_path, line ranges, language, extension, hash, timestamp, project

**6. Smart Indexing and Incremental Updates**
- Deprecated: Legacy `incremental_update` tool (removed in favor of smart indexing)
- `index_codebase` now auto-detects: full indexing for new codebases, incremental for previously indexed
- Persistent hash cache stored in `.cache/project-rag/hash_cache.json`
- Normalizes paths to canonical absolute form for consistent cache lookups
- Skips `.git` directories automatically to avoid indexing repository metadata
- Detects: new files (no old hash), modified files (hash changed), deleted files (in cache but not on disk)
- Deletes old embeddings before re-indexing modified files
- Updates cache after successful indexing and persists to disk

**7. Hybrid Search (bm25_search.rs)**
- Combines vector similarity with BM25 keyword matching
- Uses Tantivy inverted index for full-text search with IDF scoring
- Reciprocal Rank Fusion (RRF) merges both rankings using 1/(k+rank) formula (k=60)
- Both indexes queried in parallel for fast results
- Returns combined scores from both semantic and keyword matching

**8. Code Relations (relations/)**
- **Hybrid Architecture**: Uses stack-graphs (high precision, ~95%) for Python, TypeScript, Java, Ruby when feature enabled; RepoMap fallback (~70%) for all other languages
- **RelationsProvider Trait**: Abstraction for extracting definitions and references from source files
- **SymbolExtractor**: Uses tree-sitter AST to extract function, class, method, struct definitions
- **ReferenceFinder**: Text-based identifier matching with context analysis (call, read, write, import, etc.)
- **PrecisionLevel**: High (stack-graphs), Medium (AST-based RepoMap), Low (text-based)
- **Symbol Types**: Function, Method, Class, Struct, Interface, Trait, Enum, Module, Variable, Constant, etc.
- **Reference Types**: Call, Read, Write, Import, TypeReference, Inheritance, Instantiation

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
- Current test coverage: 386 tests across 12 modules (types, chunker, cache, BM25, embedding, AST parser, file walker, relations types, symbol extractor, reference finder)

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
6. Update server count in comments (e.g., "6 tools" instead of "5 tools")

## Known Issues and Limitations

### Technical Debt
1. **FastEmbed Mutability**: Uses unsafe workaround in `fastembed_manager.rs:40-44`. Should refactor to `Arc<Mutex<TextEmbedding>>`.

2. **Async Trait Warnings**: 9 harmless warnings about `async fn` in public traits. Cosmetic issue only.

3. **Pattern Matching**: `file_walker.rs` uses simple substring matching for include/exclude patterns, not proper glob matching.

### Build Notes
- Default build uses LanceDB (embedded, no external dependencies)
- Qdrant backend requires `--no-default-features --features qdrant-backend` flag
- First run downloads ~50MB embedding model to cache (~/.cache/fastembed)
- Release builds use aggressive optimization (takes longer to compile)
- Requires `protobuf-compiler` installed (`sudo apt-get install protobuf-compiler` on Ubuntu/Debian)

### Performance Characteristics
- Indexing: ~1000 files/minute (depends on file size and CPU)
- Search latency: 20-30ms per query (HNSW index)
- Memory: ~100MB base + 50MB model + ~40MB per 10k chunks
- Storage: ~1.5KB per chunk in Qdrant

## MCP Protocol Details

### Tool Schemas
All tools return JSON responses conforming to types defined in `types.rs`. Key response types:
- `IndexResponse`: files_indexed, chunks_created, embeddings_generated, duration_ms, errors, mode (full or incremental)
- `QueryResponse`: results (SearchResult[] with vector_score, keyword_score, combined_score), duration_ms
- `StatisticsResponse`: total_files, total_chunks, language_breakdown
- `ClearResponse`: success, message
- `FindDefinitionResponse`: definitions (DefinitionResult[] with file_path, line, symbol info), precision_level
- `FindReferencesResponse`: references (ReferenceResult[] with file_path, line, reference_kind), precision_level
- `GetCallGraphResponse`: node (CallGraphNode with callers/callees), precision_level

### Prompt (Slash Command) Pattern
Prompts in `#[prompt_router]` expand to user messages that instruct the AI to call the corresponding tool. Example:
```rust
#[prompt(name = "index", description = "...")]
async fn index_prompt(&self, Parameters(args): Parameters<serde_json::Value>)
    -> Result<GetPromptResult, McpError>
```

### Server Capabilities
Defined in `ServerHandler::get_info()`:
- Tools: Enabled (9 tools available):
  - `index_codebase` - Index a codebase with smart full/incremental detection
  - `query_codebase` - Semantic search across indexed code
  - `get_statistics` - Get index statistics
  - `clear_index` - Clear all indexed data
  - `search_by_filters` - Advanced search with file type/language filters
  - `search_git_history` - Search git commit history semantically
  - `find_definition` - Find where a symbol is defined (LSP-like)
  - `find_references` - Find all references to a symbol (LSP-like)
  - `get_call_graph` - Get callers/callees for a function
- Prompts: Enabled (9 slash commands: /project:index, /project:query, /project:stats, /project:clear, /project:search, /project:git-search, /project:definition, /project:references, /project:callgraph)
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
6. Update server count in comments (e.g., "6 tools" instead of "5 tools")
7. Add unit tests for the new functionality in the relevant module

### Debugging MCP Communication
- Run server with `RUST_LOG=debug` to see MCP messages
- Check stdio input/output (server reads from stdin, writes to stdout)
- Validate JSON-RPC format: `{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": 1}`
- Use `mcp_test_minimal.rs` as reference for working MCP server pattern

## Dependencies and Updates

### Critical Dependencies
- `rmcp` (0.8): MCP protocol SDK - breaking changes possible in future versions
- `lancedb` (0.10): Default embedded vector database
- `qdrant-client` (1.15): Optional external vector database, requires builder patterns
- `fastembed` (5.1): Model downloads from HuggingFace on first run
- `tantivy` (0.22): BM25 full-text search engine
- `tokio` (1.43): Full feature set required for async runtime
- `tree-sitter` (0.25) + language parsers: AST parsing for 12 languages

### Dependency Update Strategy
- **rmcp**: Check changelog carefully, macro syntax may change
- **lancedb**: Check for API changes in query interface
- **qdrant-client**: Builder patterns are stable, but check for breaking changes
- **fastembed**: Model compatibility may change between versions
- **tantivy**: BM25 implementation is stable, but check for API changes
- **tree-sitter**: Language parser versions must be compatible with tree-sitter version
- Run full test suite after updating any dependency (386 tests must pass)

## Additional Documentation
- See `README.md` for user-facing documentation and installation instructions
- See `NOTES.md` for development notes and known issues
- See `docs/TEST_RESULTS.md` for test suite details
- See `docs/COVERAGE_ANALYSIS.md` for detailed test coverage
- See `docs/PRODUCTION_READY.md` for production deployment guide
- See `docs/SLASH_COMMANDS.md` for detailed slash command documentation
- See `BENCHMARKING.md` for performance benchmarks and optimization tips
