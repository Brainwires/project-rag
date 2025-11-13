# ADR 003: Smart Indexing with Auto-Detection

## Status

Accepted

## Context

When indexing codebases, users need two modes:

1. **Full Indexing**: Index entire codebase from scratch (first time, or after major changes)
2. **Incremental Updates**: Only re-index changed/new files (much faster for subsequent runs)

The original API had two separate tools (`index_codebase` and `incremental_update`), requiring users to manually choose which to use. This created confusion and a poor user experience.

## Decision

Implement **Smart Indexing** where `index_codebase` automatically detects whether to perform full or incremental indexing based on persistent cache state.

## Rationale

### User Experience Problem

Previous API:
```
User: "Index my codebase"
AI: "Should I do full indexing or incremental update?"
User: "Uh... I don't know?"
```

This forced the AI assistant to ask clarifying questions, breaking the conversation flow.

### Smart Detection Algorithm

```rust
pub async fn do_index_smart(&self, path: String, ...) -> Result<IndexResponse> {
    let normalized_path = Self::normalize_path(&path)?;

    // Check persistent cache
    let cache = self.hash_cache.read().await;
    let has_cache = cache.get_all_hashes(&normalized_path).map(|h| !h.is_empty()).unwrap_or(false);
    drop(cache);

    if has_cache {
        // Cache exists -> Incremental update
        self.do_incremental_update(...)
    } else {
        // No cache -> Full indexing
        self.do_index(...)
    }
}
```

### Persistent Hash Cache

- **Location**: `.cache/project-rag/hash_cache.json`
- **Format**: `{ "/abs/path/to/project": { "file1.rs": "sha256_hash", ... } }`
- **Purpose**: Track file hashes across server restarts
- **Benefits**:
  - Survives server restarts (unlike in-memory cache)
  - Enables smart detection
  - Supports multiple projects simultaneously

## Consequences

### Positive

- **Zero-Friction UX**: Users just say "index my codebase" - it works correctly
- **Optimal Performance**: Auto-selects fastest strategy
- **Transparent**: Response indicates which mode was used (`mode: "full"` or `mode: "incremental"`)
- **Persistent State**: Cache survives restarts, making subsequent indexing fast
- **Multi-Project Support**: Each path tracked independently

### Negative

- **Hidden Behavior**: Users may not understand why indexing is sometimes fast/slow
  - *Mitigation*: Response includes `mode` field explaining what happened
- **Cache Management**: Users need to clear cache if they want fresh full indexing
  - *Mitigation*: Provide `clear_index` tool
- **Disk Usage**: Cache files use ~1-5KB per 1000 files
  - *Mitigation*: Negligible compared to vector database size

### Performance Impact

**Full Indexing** (first run):
- 1000 files: ~60 seconds (file walk + chunk + embed + store)

**Incremental Update** (no changes):
- 1000 files: ~2 seconds (file walk + hash compare + skip all)

**Incremental Update** (10% changed):
- 1000 files: ~8 seconds (walk + hash + re-index 100 files)

Speedup: **7-30x faster** for subsequent runs with few changes.

## Implementation Details

### File Change Detection

```rust
// Detect changes
for current_file in current_files {
    let new_hash = sha256(&current_file.content);
    match old_hashes.get(&current_file.path) {
        Some(old_hash) if old_hash == &new_hash => {
            // Unchanged - skip
        }
        Some(_) => {
            // Modified - delete old + re-index
            modified_files.push(current_file);
        }
        None => {
            // New file - index
            new_files.push(current_file);
        }
    }
}

// Detect deletions
for (old_path, _) in old_hashes {
    if !current_paths.contains(old_path) {
        deleted_files.push(old_path);
    }
}
```

### Cache Persistence

- **Save**: After successful indexing (full or incremental)
- **Load**: On server startup and before each smart index check
- **Update**: Atomic write to prevent corruption
- **Location**: User cache directory (`~/.cache/project-rag/` on Linux)

## Alternatives Considered

### Two Separate Tools (Original Design)

- **Rejected**: Poor UX, users don't know which to use
- Requires AI to ask clarifying questions
- Breaks conversation flow

### Git-Based Change Detection

- **Not feasible**: Not all codebases use git
- Doesn't detect uncommitted changes
- Requires working git repository

### Timestamp-Based Detection

- **Rejected**: File modification times unreliable
  - Git checkout changes timestamps
  - Build tools touch files
  - Doesn't detect content changes (e.g., format-only)

### Watch-Based Incremental Updates

- **Not implemented**: Requires persistent daemon process
- MCP servers are typically short-lived (stdio-based)
- Complexity not justified for current use case

## Migration Path

**From**: Separate `index_codebase` and `incremental_update` tools
**To**: Single `index_codebase` tool with smart detection

**Backward Compatibility**:
- Old `incremental_update` tool deprecated but still works
- Existing caches automatically used by new smart indexing
- No migration required for users

## References

- Implementation: `src/mcp_server/indexing.rs:523-560` (`do_index_smart`)
- Cache implementation: `src/cache.rs`
- Related: Hash calculation uses SHA-256 for reliability
- Related ADR: [004-persistent-caching-strategy.md](004-persistent-caching-strategy.md) (if created)

## Date

2024-11 (Implemented in Phase 3 improvements)
