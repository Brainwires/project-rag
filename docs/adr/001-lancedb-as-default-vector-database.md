# ADR 001: LanceDB as Default Vector Database

## Status

Accepted

## Context

Project RAG requires a vector database for storing and searching code embeddings. We need to choose between different vector database options, balancing factors like:

- Ease of deployment (embedded vs server-based)
- Performance (indexing and search latency)
- Features (hybrid search, filtering)
- Maintenance overhead
- Storage efficiency

The initial implementation used Qdrant as the only option, which required running an external Docker container.

## Decision

We decided to make **LanceDB the default vector database**, while keeping Qdrant as an optional feature flag.

## Rationale

### LanceDB Advantages

1. **Zero External Dependencies**: LanceDB is an embedded database - no Docker containers, no separate servers to manage
2. **Columnar Storage**: Efficient storage using Apache Arrow format with memory-mapped files
3. **Built-in Hybrid Search**: Native support for vector + BM25 keyword search using Tantivy
4. **Fast Queries**: Zero-copy reads via memory mapping, ~20-30ms search latency
5. **Production Ready**: Used by companies like Midjourney, robust and well-tested
6. **Better DX**: Simpler setup for users - `cargo build && cargo run` just works

### Qdrant Trade-offs

- Requires external server (Docker)
- Additional operational complexity
- Network latency overhead
- Better for distributed scenarios (which we don't need)

### Implementation Strategy

- LanceDB: default feature (no flag needed)
- Qdrant: optional `--features qdrant-backend` for users who prefer it
- Both implement the same `VectorDatabase` trait
- Seamless switching between backends via configuration

## Consequences

### Positive

- **Simplified onboarding**: New users can start immediately without Docker setup
- **Reduced operational complexity**: No server management, health checks, or networking issues
- **Better local development**: Fast iteration without container overhead
- **Storage efficiency**: Columnar format reduces disk usage
- **Single binary**: Entire system runs as one process

### Negative

- **Two backends to maintain**: Must keep both LanceDB and Qdrant implementations in sync with trait changes
- **Feature parity**: Need to ensure both backends support the same features
- **Testing complexity**: Must test both backends (though traits help here)

### Neutral

- Users who already use Qdrant can still opt-in via feature flag
- Migration path exists for users who want to switch backends

## Alternatives Considered

### Qdrant Only

- **Rejected**: Too much operational overhead for a local development tool
- Requires Docker and network configuration
- Not suitable for simple use cases

### Multiple Embedded Options (DuckDB, SQLite-vss)

- **Not pursued**: LanceDB's Arrow-native format and built-in Tantivy integration made it the clear winner
- Other embedded options lacked hybrid search support

### Vector-only (No Database)

- **Rejected**: Need persistence, filtering, and hybrid search
- In-memory only would require re-indexing on every restart

## References

- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- Related ADR: [002-hybrid-search-with-rrf.md](002-hybrid-search-with-rrf.md)
- Implementation: `src/vector_db/lance_client.rs`, `src/vector_db/qdrant_client.rs`

## Date

2024-11 (Implemented in Phase 3 improvements)
