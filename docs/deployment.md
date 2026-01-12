# Deployment Guide

This guide covers deploying Project RAG for production use with Claude Code or Claude Desktop.

## Prerequisites

### Required

- **Rust**: 1.83+ with Rust 2024 edition support
- **protobuf-compiler**: Required for building
  ```bash
  # Ubuntu/Debian
  sudo apt-get install protobuf-compiler

  # macOS
  brew install protobuf
  ```

### Vector Database

**LanceDB (Default)**: No additional setup required. LanceDB is embedded and stores data in `./.lancedb` directory.

**Qdrant (Optional)**: If using the Qdrant backend, start a Qdrant server:
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_data:/qdrant/storage \
    qdrant/qdrant
```

## Installation

```bash
# Clone and navigate to the project
cd project-rag

# Build release binary (default LanceDB backend)
cargo build --release

# Or build with Qdrant backend
cargo build --release --no-default-features --features qdrant-backend

# Binary location: target/release/project-rag
```

## Configuration

### Claude Code

Add the MCP server using the CLI:

```bash
cd /path/to/project-rag
claude mcp add project --command "$(pwd)/target/release/project-rag"

# With logging enabled
claude mcp add project --command "$(pwd)/target/release/project-rag" --env RUST_LOG=info
```

Restart Claude Code to load the server.

### Claude Desktop

Add to your Claude Desktop config file:

| Platform | Location |
|----------|----------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

```json
{
  "mcpServers": {
    "project-rag": {
      "command": "/absolute/path/to/project-rag/target/release/project-rag",
      "env": {
        "RUST_LOG": "info"
      }
    }
  }
}
```

## First Run

On first run, FastEmbed downloads the embedding model (~50MB) to the cache directory (`~/.cache/fastembed`). Subsequent runs use the cached model.

## Performance

| Metric | Typical Value |
|--------|---------------|
| Indexing speed | ~1000 files/minute |
| Search latency | 20-30ms per query |
| Memory (base) | ~100MB |
| Memory (model) | ~50MB |
| Memory (per 10k chunks) | ~40MB |
| Storage (per chunk) | ~1.5KB |

## Security

- **Local-first**: All processing happens locally
- **No API calls**: Embeddings generated locally (no data leaves your machine)
- **No telemetry**: No usage tracking or analytics
- **Network**: Keep Qdrant on localhost only (if using Qdrant backend)

## Maintenance

### Regular Tasks

- Check disk usage periodically
- Update dependencies monthly: `cargo update && cargo build --release`
- Run tests after updates: `cargo test --lib`

### Clearing the Index

Use `/project:clear` or the `clear_index` tool to reset the database when needed.

## Next Steps

- See [slash-commands.md](slash-commands.md) for available commands
- See [troubleshooting.md](troubleshooting.md) for common issues
