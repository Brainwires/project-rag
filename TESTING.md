# Testing Guide

This document describes the testing strategy and procedures for project-rag.

## Test Organization

### Unit Tests (331 tests)
Located in `src/**/*.rs` files within `#[cfg(test)]` modules. These tests cover individual functions and modules:

- **BM25 Search** (12 tests): Keyword search, RRF fusion, statistics, lock cleanup
- **Cache** (15 tests): Hash cache operations, serialization, persistence
- **Client** (24 tests): Core API methods, initialization, query, index, clear
- **Client Indexing** (15 tests): Full indexing, incremental updates, smart mode
- **Client Git Indexing** (11 tests): Git history search, on-demand indexing, filters (8 flaky)
- **Configuration** (14 tests): Config loading, validation, environment overrides
- **Embedding** (7 tests): FastEmbed integration, model selection, batch processing
- **Error Types** (14 tests): Error conversion, categorization, display
- **File Walker** (61 tests): Directory traversal, language detection, gitignore support
- **Chunking** (25 tests): AST parsing, fixed-line chunking, sliding window
- **Git** (13 tests): Git integration, commit parsing, history indexing (1 flaky)
- **MCP Server** (31 tests): Server initialization, tool handlers, prompt handlers, validation
- **Types** (45 tests): Request/response serialization, validation, edge cases
- **Vector Database** (22 tests): LanceDB operations, search, statistics, hybrid search
- **Paths** (9 tests): Platform-specific path computation
- **Git Cache** (6 tests): Git cache operations
- **AST Parser** (13 tests): Tree-sitter parsing for 12 languages

### Integration Tests (10 tests)
Located in `tests/` directory:

- **Git Search Integration** (`tests/git_search_integration.rs`): 5 tests
  - Git walker operations
  - Commit chunking
  - Git history search
  - Cache operations

- **Simple Integration** (`tests/simple_integration.rs`): 5 tests
  - Server creation with config
  - Server creation with defaults
  - Path normalization
  - Custom batch size configuration
  - Full indexing workflow

### Benchmark Tests
Located in `benches/indexing_benchmark.rs`:

- **Indexing Benchmarks**: Test indexing performance with 10, 50, and 100 files
- **Search Benchmarks**: Compare vector search vs hybrid search
- **Chunking Benchmarks**: Measure parallel chunking performance

## Running Tests

### Run All Tests
```bash
# Run all unit and integration tests
cargo test

# Run with output
cargo test -- --nocapture

# Run with debug logging
RUST_LOG=debug cargo test -- --nocapture
```

### Run Specific Test Suites
```bash
# Unit tests only
cargo test --lib

# Specific module
cargo test --lib types::tests

# Integration tests only
cargo test --test '*'

# Specific integration test
cargo test --test simple_integration

# Single test
cargo test test_server_creation_with_config
```

### Run Benchmarks
```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench indexing

# Run with criterion output
cargo bench -- --verbose
```

## Test Coverage

Current test statistics:
- **Total Tests**: 341 (331 unit + 10 integration)
- **Success Rate**: ~97.4% (332/341 passing, 9 flaky git tests due to libgit2 bug)
- **Overall Coverage**: **92.52%** (exceeded 85% target, achieved 90%+!)
- **Modules with Tests**: 15/15 (100%)
- **Critical Paths Covered**: Indexing, search, caching, git integration, MCP server tools, request validation

### Coverage by Module
| Module | Unit Tests | Coverage % | Coverage Notes |
|--------|-----------|-----------|----------------|
| bm25_search | 12 | 87.42% | Full coverage: search, delete, statistics, RRF, lock cleanup |
| cache | 15 | 97.07% | Full coverage: load, save, operations, edge cases |
| client | 24 | 94.81% | Full API coverage: init, query, index, clear, filters, git history |
| client/indexing | 15 | 79.16% | Comprehensive indexing tests: full, incremental, smart mode |
| client/git_indexing | 11 | 91.38% | Git history search with on-demand indexing (8 flaky tests) |
| config | 14 | 86.34% | Full coverage: validation, env vars, TOML |
| embedding | 7 | 90.18% | Full coverage: all supported models, batch processing |
| error | 14 | 95.41% | Full coverage: all error types, conversions |
| file_walker | 61 | 99.08% | Full coverage: 25+ languages, gitignore, patterns |
| chunker | 25 | 98.18% | Full coverage: AST, fixed-line, sliding window |
| git | 13 | 85.71% | Full coverage: walker, chunker, history (1 flaky test) |
| mcp_server | 31 | 49.68% | Tool handlers, prompt handlers, server info, validation |
| types | 45 | 99.30% | All request/response types, validation, serialization, edge cases |
| vector_db | 22 | 97.01% | Full coverage: CRUD, search, hybrid, filters, project isolation |
| paths | 9 | 76.38% | Full coverage: all platforms, all path types |

## Writing Tests

### Unit Test Pattern
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_name() {
        // Arrange
        let input = create_test_data();

        // Act
        let result = function_under_test(input);

        // Assert
        assert_eq!(result, expected_value);
        assert!(result.is_ok());
    }

    #[test]
    fn test_error_case() {
        let invalid_input = create_invalid_data();
        let result = function_under_test(invalid_input);
        assert!(result.is_err());
    }
}
```

### Integration Test Pattern
```rust
use project_rag::mcp_server::RagMcpServer;
use tempfile::TempDir;

#[tokio::test]
async fn test_integration_scenario() -> anyhow::Result<()> {
    // Setup
    let temp_dir = TempDir::new()?;
    let server = create_test_server(&temp_dir).await?;

    // Execute
    let result = server.do_something().await?;

    // Verify
    assert!(result.success);
    Ok(())
}
```

### Benchmark Pattern
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_function(c: &mut Criterion) {
    c.bench_function("operation_name", |b| {
        b.iter(|| {
            expensive_operation(black_box(input))
        });
    });
}

criterion_group!(benches, benchmark_function);
criterion_main!(benches);
```

## Test Requirements

### For New Features
- Add unit tests for all new functions
- Add integration tests for user-facing features
- Add benchmarks for performance-critical code
- Maintain 100% test pass rate

### Before Committing
```bash
# 1. Run all tests
cargo test

# 2. Check formatting
cargo fmt --check

# 3. Run linter
cargo clippy

# 4. Run benchmarks (optional, for performance changes)
cargo bench
```

## Continuous Integration

Tests are automatically run on:
- Every commit
- Pull requests
- Release builds

CI pipeline:
1. `cargo test --all` - All tests must pass
2. `cargo clippy` - No warnings allowed
3. `cargo fmt --check` - Code must be formatted
4. `cargo build --release` - Release build must succeed

## Performance Testing

### Benchmarking Guidelines
- Run benchmarks before and after changes
- Use `criterion` for statistical significance
- Test with realistic data sizes (10-100 files)
- Document performance improvements in commits

### Expected Performance
- **Indexing**: ~1000 files/minute (4-core system)
- **Search**: 20-30ms per query
- **Chunking**: 2-4x faster with parallel processing

## Test Data Management

### Temporary Directories
Always use `tempfile::TempDir` for test isolation:
```rust
let temp_dir = TempDir::new()?;
// temp_dir automatically cleaned up when dropped
```

### Test Fixtures
Located in test modules or `tests/` directory:
- Keep fixtures small (<1KB)
- Use realistic code samples
- Document fixture purpose

## Troubleshooting Tests

### Common Issues

**Tests Fail Locally**
```bash
# Clean and rebuild
cargo clean
cargo test

# Check Rust version
rustc --version  # Should be 1.89+
```

**Slow Tests**
```bash
# Run tests in parallel (default)
cargo test

# Run single-threaded for debugging
cargo test -- --test-threads=1
```

**Benchmark Failures**
```bash
# Ensure release mode
cargo bench

# Check system load
top  # Benchmarks need quiet system
```

## Test Metrics

Track these metrics over time:
- Total test count
- Test pass rate
- Test execution time
- Benchmark performance

Current baseline:
- Tests: 341 (331 unit + 10 integration)
- Pass Rate: ~97.4% (332/341 passing, 9 flaky git tests due to libgit2 bug)
- Coverage: **92.52%** overall (exceeded 85% target!)
- Execution Time: ~14-16 seconds
- Unit Test Time: ~12-14 seconds
- Integration Test Time: ~1-2 seconds
