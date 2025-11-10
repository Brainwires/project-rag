# MCP Slash Commands for Project RAG

## ✅ Available Now!

Project RAG now exposes **6 slash commands** via MCP Prompts that make it easy to use the RAG tools directly in Claude Code.

## Slash Commands

When you restart Claude Code, these commands will be available:

### `/mcp__project-rag__index`
Index a codebase directory to enable semantic search.

**Usage**:
```
/mcp__project-rag__index
```

**Optional Arguments**:
- `path`: Path to the codebase (default: current directory)

**What it does**: Triggers the `index_codebase` tool to create embeddings for all code files in the specified directory.

---

### `/mcp__project-rag__query`
Search the indexed codebase using semantic search.

**Usage**:
```
/mcp__project-rag__query
```

**Required Arguments**:
- `query`: What to search for (e.g., "authentication logic")

**What it does**: Triggers the `query_codebase` tool to perform semantic search across indexed code.

---

### `/mcp__project-rag__stats`
Get statistics about the indexed codebase.

**Usage**:
```
/mcp__project-rag__stats
```

**What it does**: Triggers the `get_statistics` tool to show index statistics including file counts, chunk counts, and language breakdown.

---

### `/mcp__project-rag__clear`
Clear all indexed data from the vector database.

**Usage**:
```
/mcp__project-rag__clear
```

**What it does**: Triggers the `clear_index` tool to delete all indexed data from Qdrant.

---

### `/mcp__project-rag__update`
Incrementally update the index with only changed files.

**Usage**:
```
/mcp__project-rag__update
```

**Optional Arguments**:
- `path`: Path to the codebase (default: current directory)

**What it does**: Triggers the `incremental_update` tool to detect and reindex only files that have changed since the last index.

---

### `/mcp__project-rag__search`
Advanced search with filters (file type, language, path).

**Usage**:
```
/mcp__project-rag__search
```

**Required Arguments**:
- `query`: What to search for

**What it does**: Triggers the `search_by_filters` tool for advanced semantic search with optional filtering by file extensions, languages, and path patterns.

---

## How Slash Commands Work

These slash commands are implemented using **MCP Prompts**. When you invoke a slash command:

1. Claude Code sends your command to the MCP server
2. The server returns a pre-formatted prompt
3. Claude receives the prompt and executes the corresponding tool
4. Results are displayed in Claude Code

This provides a convenient shortcut instead of manually typing:
```
Please use the index_codebase tool to index ~/dev/my-project
```

You can just type:
```
/mcp__project-rag__index
```

## Direct Tool Usage

You can also use the tools directly by asking Claude:

**Examples**:

1. **Index a codebase**:
   ```
   Please use the index_codebase tool to index ~/dev/my-project
   ```

2. **Query the codebase**:
   ```
   Please use the query_codebase tool to search for "authentication logic"
   ```

3. **Get statistics**:
   ```
   Please use the get_statistics tool to show me index stats
   ```

4. **Incremental update**:
   ```
   Please use the incremental_update tool to update the index for ~/dev/my-project
   ```

5. **Advanced search**:
   ```
   Please use the search_by_filters tool to find "database connection pool"
   in Rust files only
   ```

---

## Implementation Details

Slash commands are implemented using the rmcp SDK's prompts feature:
- Uses `#[prompt_router]` macro to define prompt routes
- Each prompt function is decorated with `#[prompt]` attribute
- Prompts return `PromptMessage` or `GetPromptResult` types
- Server capabilities include `enable_prompts()`

The implementation can be found in `src/mcp_server.rs`.

---

Last Updated: 2025-01-10
Status: **Production Ready** 🚀
