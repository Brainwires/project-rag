# Troubleshooting

Common issues and solutions for Project RAG.

## Index Lock Errors

**Error**: `"BM25 index is currently being used by another process"`

This means another agent or process is actively indexing. This is expected behavior to prevent index corruption.

**Solutions**:

1. **Wait**: Let the current indexing operation complete (typically seconds to minutes)
2. **Check processes**: Verify no other Claude Code/Desktop instances are running indexing operations
3. **Force cleanup** (last resort): If certain no other process is running, manually remove stale locks:
   ```bash
   rm ~/.local/share/project-rag/lancedb/lancedb_bm25/.tantivy-*.lock
   ```

The system automatically detects and cleans up stale locks (>5 minutes old) from crashed processes. Manual intervention is rarely needed.

For more details on the locking system, see [index-locking.md](index-locking.md).

## Connection Issues

### Qdrant Connection Fails

If using the Qdrant backend:

```bash
# Check if Qdrant is running
curl http://localhost:6334/health

# View Qdrant logs
docker logs <container-id>

# Restart Qdrant
docker restart <container-id>
```

## Model Download Issues

**Error**: `"Failed to download FastEmbed model"`

**Solutions**:

1. Check internet connectivity
2. Set a HuggingFace mirror:
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
3. Pre-download the model manually:
   ```bash
   python -c "from fastembed import TextEmbedding; TextEmbedding()"
   ```

## Build Issues

**Error**: `edition "2024" not recognized`

Update Rust: `rustup update stable` (requires Rust 1.83+)

**Error**: `Can't find rmcp crate`

Clear cargo cache and retry:
```bash
rm -rf ~/.cargo/registry
cargo update
cargo build --release
```

## Performance Issues

### Slow Indexing

- Check disk I/O with `iotop`
- Use SSD instead of HDD
- Reduce `max_file_size` in index request
- Add `exclude_patterns` for large vendor directories (node_modules, target, etc.)

### Slow Searches

- Use filters to narrow search space (`search_by_filters`)
- Lower similarity threshold
- Check Qdrant resource usage (if using Qdrant backend)

### Out of Memory

- Reduce `max_file_size` in IndexRequest
- Index in smaller batches using include/exclude patterns
- Increase system swap

## Runtime Errors

### MCP Server Not Responding

1. Check if server is running: `ps aux | grep project-rag`
2. Run with debug logging: `RUST_LOG=debug ./target/release/project-rag`
3. Rebuild the binary: `cargo build --release`

### Empty Search Results

- Verify the codebase is indexed: use `/project:stats`
- Lower the `min_score` threshold (default is 0.7)
- The system automatically lowers thresholds to 0.6, 0.5, 0.4, 0.3 if no results found
- Check if the search query is too specific

## Getting Help

If issues persist:

1. Run with trace logging: `RUST_LOG=trace ./target/release/project-rag`
2. Check the [README](../README.md) for configuration details
3. Open an issue on GitHub with logs and reproduction steps
