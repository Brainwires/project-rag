# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for Project RAG, documenting key design decisions and their rationale.

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [001](001-lancedb-as-default-vector-database.md) | LanceDB as Default Vector Database | Accepted | 2024-11 |
| [002](002-hybrid-search-with-rrf.md) | Hybrid Search with Reciprocal Rank Fusion | Accepted | 2024-11 |
| [003](003-smart-indexing-with-auto-detection.md) | Smart Indexing with Auto-Detection | Accepted | 2024-11 |

## What is an ADR?

An Architecture Decision Record (ADR) captures an important architectural decision made along with its context and consequences.

### ADR Format

Each ADR follows this structure:

- **Status**: Proposed, Accepted, Deprecated, Superseded
- **Context**: The situation that led to the decision
- **Decision**: What we decided to do
- **Rationale**: Why we made this choice
- **Consequences**: The impact (positive, negative, neutral)
- **Alternatives Considered**: What else we evaluated
- **References**: Links to code, docs, papers

## Why ADRs?

ADRs provide:

1. **Historical Context**: Understand why decisions were made
2. **Onboarding**: Help new contributors understand the system
3. **Prevent Revisiting**: Document why alternatives were rejected
4. **Architectural Visibility**: Make implicit decisions explicit

## Key Decisions Documented

### Vector Database Choice ([ADR 001](001-lancedb-as-default-vector-database.md))

**Decision**: Use LanceDB as default embedded database instead of requiring external Qdrant server.

**Impact**: Zero-setup experience, faster local development, reduced operational complexity.

### Hybrid Search Strategy ([ADR 002](002-hybrid-search-with-rrf.md))

**Decision**: Combine vector similarity with BM25 keyword search using Reciprocal Rank Fusion.

**Impact**: Better search quality for code (exact matches + semantic understanding), minimal performance overhead.

### Smart Indexing ([ADR 003](003-smart-indexing-with-auto-detection.md))

**Decision**: Auto-detect whether to do full or incremental indexing based on persistent cache.

**Impact**: Seamless UX (no manual mode selection), 7-30x faster subsequent indexing.

## Contributing

When making significant architectural decisions:

1. Create a new ADR with next number (004, 005, etc.)
2. Follow the existing format
3. Update this README index
4. Link to relevant code and related ADRs

## Further Reading

- [ADR GitHub Organization](https://adr.github.io/)
- [Michael Nygard's ADR article](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
