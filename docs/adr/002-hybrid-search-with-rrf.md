# ADR 002: Hybrid Search with Reciprocal Rank Fusion (RRF)

## Status

Accepted

## Context

Pure vector similarity search excels at semantic understanding but can miss exact keyword matches. For code search, users often want to find:

- Exact function/variable names (keyword search)
- Semantically similar code (vector search)
- Both simultaneously (hybrid)

We need a strategy to combine vector and keyword search results effectively.

## Decision

Implement **Hybrid Search using Reciprocal Rank Fusion (RRF)** to combine vector similarity (semantic) with BM25 keyword search.

## Rationale

### Why Hybrid Search?

Code search has unique requirements:

1. **Exact Matches Matter**: Finding `authenticate_user` function requires exact tokens
2. **Semantic Understanding Helps**: "how does auth work" should find authentication code
3. **Best of Both Worlds**: Hybrid search provides both precision and recall

### Why RRF Over Score Normalization?

**Reciprocal Rank Fusion formula**: `score = 1 / (k + rank)` where k=60 (standard constant)

Advantages over score normalization:
- **Scale-independent**: Works without normalizing different score ranges
- **Robust**: Less sensitive to outliers in either ranking
- **Proven**: Used successfully in multi-stage retrieval systems
- **Simple**: No tuning of weights required

### Implementation Details

```rust
// Combine vector and BM25 rankings
for (rank, (id, _score)) in vector_results.iter().enumerate() {
    let rrf_score = 1.0 / (60.0 + (rank + 1) as f32);
    *score_map.entry(*id).or_insert(0.0) += rrf_score;
}

for (rank, result) in bm25_results.iter().enumerate() {
    let rrf_score = 1.0 / (60.0 + (rank + 1) as f32);
    *score_map.entry(result.id).or_insert(0.0) += rrf_score;
}
```

Items appearing in both rankings get boosted scores (sum of both RRF scores).

## Consequences

### Positive

- **Better Search Quality**: Combines semantic understanding with exact matching
- **User Control**: `hybrid: bool` parameter lets users choose pure vector or hybrid
- **Performance**: Both searches run in parallel (async), minimal latency overhead
- **No Tuning**: RRF constant (k=60) is standard and doesn't need adjustment

### Negative

- **Complexity**: Must maintain two indexes (LanceDB + Tantivy)
- **Storage Overhead**: ~40% increase (BM25 inverted index)
- **Sync Required**: Both indexes must be updated together

### Performance Impact

- Search latency: +5-10ms (BM25 query + RRF fusion)
- Storage: +40% (Tantivy index ~10KB per 10K chunks)
- Indexing time: +15% (parallel embedding + BM25 indexing)

## Alternatives Considered

### Pure Vector Search Only

- **Rejected**: Misses exact keyword matches
- Example: Query "JWT" wouldn't reliably find "JWT token validation"

### Pure BM25 Only

- **Rejected**: Poor semantic understanding
- Example: "auth logic" wouldn't find `verify_credentials()` function

### Learned Weights (e.g., `α*vector + β*BM25`)

- **Rejected**: Requires training data and hyperparameter tuning
- RRF is parameter-free and works well out-of-the-box

### ColBERT-style Late Interaction

- **Not feasible**: Requires storing all token embeddings, 100x storage increase
- Too expensive for local embedding model

## Implementation Notes

### BM25 Index Management

- Tantivy for inverted index
- Same document IDs as vector database
- Deletion synced between both indexes

### Hybrid Search Flow

```text
User Query
    │
    ├─> Generate Embedding ────> Vector Search (LanceDB)
    │                                  │
    └─> Parse Query Text ──────> BM25 Search (Tantivy)
                                       │
                                       ├─> Both Results
                                       │
                                       └─> RRF Fusion ─> Final Ranked Results
```

## References

- [RRF Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) - Cormack et al.
- [Tantivy BM25](https://github.com/quickwit-oss/tantivy)
- Related ADR: [001-lancedb-as-default-vector-database.md](001-lancedb-as-default-vector-database.md)
- Implementation: `src/bm25_search.rs`, `src/vector_db/lance_client.rs:287-629`

## Date

2024-11 (Implemented in Phase 3 improvements)
