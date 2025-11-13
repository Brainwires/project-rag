# Testing Guide

This document describes the testing strategy and procedures for project-rag.

## Test Organization

### Unit Tests (223 tests)
Located in `src/**/*.rs` files within `#[cfg(test)]` modules. These tests cover individual functions and modules:

- **BM25 Search** (9 tests): Keyword search, RRF fusion, statistics
- **Cache** (18 tests): Hash cache operations, serialization, persistence
- **Configuration** (14 tests): Config loading, validation, environment overrides
- **Embedding** (10 tests): FastEmbed integration, model selection, batch processing
- **Error Types** (14 tests): Error conversion, categorization, display
- **File Walker** (61 tests): Directory traversal, language detection, gitignore support
- **Chunking** (29 tests): AST parsing, fixed-line chunking, sliding window
- **Git** (13 tests): Git integration, commit parsing, history indexing
- **MCP Server** (11 tests): Server initialization, path normalization
- **Types** (9 tests): Request/response serialization, validation
- **Vector Database** (26 tests): LanceDB operations, search, statistics
- **Paths** (9 tests): Platform-specific path computation

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
- **Total Tests**: 233 (223 unit + 10 integration)
- **Success Rate**: 100% (233/233 passing)
- **Modules with Tests**: 13/13 (100%)
- **Critical Paths Covered**: Indexing, search, caching, git integration

### Coverage by Module
| Module | Unit Tests | Coverage Notes |
|--------|-----------|----------------|
| bm25_search | 9 | Full coverage: search, delete, statistics, RRF |
| cache | 18 | Full coverage: load, save, operations, edge cases |
| config | 14 | Full coverage: validation, env vars, TOML |
| embedding | 10 | Full coverage: all 4 supported models |
| error | 14 | Full coverage: all error types, conversions |
| file_walker | 61 | Full coverage: 25+ languages, gitignore, patterns |
| chunker | 29 | Full coverage: AST, fixed-line, sliding window |
| git | 13 | Full coverage: walker, chunker, history |
| mcp_server | 11 | Core functionality, needs more integration tests |
| types | 9 | Full coverage: all request/response types |
| vector_db | 26 | Full coverage: CRUD, search, hybrid, filters |
| paths | 9 | Full coverage: all platforms, all path types |

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
- Code coverage percentage
- Benchmark performance

Current baseline (as of Phase 4):
- Tests: 233
- Pass Rate: 100%
- Execution Time: ~5-7 seconds
- Unit Test Time: ~3-4 seconds
- Integration Test Time: ~1-2 seconds
