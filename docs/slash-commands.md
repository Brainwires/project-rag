# Slash Commands

Project RAG provides 9 slash commands via MCP Prompts for quick access in Claude Code.

## Quick Reference

| Command | Description |
|---------|-------------|
| `/project:index` | Index a codebase directory |
| `/project:query` | Search indexed code semantically |
| `/project:stats` | Get index statistics |
| `/project:clear` | Clear all indexed data |
| `/project:search` | Advanced search with filters |
| `/project:git-search` | Search git commit history |
| `/project:definition` | Find where a symbol is defined |
| `/project:references` | Find all references to a symbol |
| `/project:callgraph` | Get call graph for a function |

## Usage

### `/project:index`

Index a codebase directory to enable semantic search. Automatically performs full indexing for new codebases or incremental updates for previously indexed ones.

```
/project:index
```

Optional: Specify a path (defaults to current directory).

### `/project:query`

Search the indexed codebase using semantic search.

```
/project:query
```

Provide a search query like "authentication logic" or "database connection handling".

### `/project:stats`

Get statistics about the indexed codebase including file counts, chunk counts, and language breakdown.

```
/project:stats
```

### `/project:clear`

Clear all indexed data from the vector database.

```
/project:clear
```

### `/project:search`

Advanced search with filters for file type, language, or path patterns.

```
/project:search
```

Supports filtering by file extensions, programming languages, and path patterns.

### `/project:git-search`

Search git commit history using semantic search with on-demand indexing.

```
/project:git-search
```

Searches commit messages, diffs, and author information. Supports date range and author filtering.

### `/project:definition`

Find where a symbol (function, class, variable) is defined.

```
/project:definition
```

Specify file path, line number, and column position.

### `/project:references`

Find all references to a symbol across the codebase.

```
/project:references
```

Returns locations categorized by reference type (Call, Read, Write, Import, etc.).

### `/project:callgraph`

Get the call graph for a function showing callers and callees.

```
/project:callgraph
```

Useful for understanding code flow and impact analysis.

## How Slash Commands Work

Slash commands are implemented using MCP Prompts. When you invoke a command:

1. Claude Code sends the command to the MCP server
2. The server returns a pre-formatted prompt
3. Claude executes the corresponding tool
4. Results are displayed in Claude Code

This provides a convenient shortcut compared to manually requesting tool usage.
