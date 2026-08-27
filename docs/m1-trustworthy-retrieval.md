# M1 trustworthy retrieval architecture

## Data flow

Current-tree indexing now follows one identity path:

1. `ProjectPathResolver` canonicalizes the explicit project root.
2. Every discovered or user-supplied file is resolved against that root.
3. Filesystem canonicalization resolves symlinks and junctions before the
   project-boundary check.
4. The persistent and MCP-facing file key is the `/`-separated path relative
   to the project root.
5. Absolute paths remain internal aliases for filesystem access and index-root
   filtering; they are not serialized in search results.

`read_file`, `edit_file`, symbol tools, incremental indexing, and the file
walker use this resolver. A relative path is project-relative. If several
indexed projects contain the same relative path, the request must supply a
`project` id or an unambiguous absolute alias.

Current and history records share the vector-database implementation but are
logically separated by a required `origin` predicate and separate BM25
namespaces. Normal code retrieval always requests `origin = current`; the
dedicated Git tool requests `origin = history`.

## Persistence schema

M1 introduces identity schema version `2` and the LanceDB table
`code_embeddings_v2`. The v1 table is never opened by the v2 client, so old
absolute/history identities cannot mix with new records. An old hash cache is
reset with an index-health diagnostic, requiring a clean reindex.

The hash cache persists a stable `project_id` for every root. Supplying
`IndexRequest.project` treats that value as the configured project id. When it
is omitted, the first index creates and persists an opaque id. Each indexed
root must have a distinct id.

Compatibility consequences:

- Existing indexes require one clean reindex.
- Search result paths are canonical project-relative paths.
- Serialized search results no longer expose absolute `root_path` values.
- Every retrieval result exposes `origin`.
- Git records with invalid object ids or non-positive author/commit dates are
  excluded and recorded as diagnostics.

## Response budgets

Normal retrieval is snippet-first. It returns at most 100 results, 4,000
characters per result, and 20,000 characters total. A hit is centered around
the first query-term line with up to 15 context lines on either side. The
response reports observed/returned counts and truncation explicitly.

`read_file` defaults to `start_line = 1`, `line_count = 30`. Explicit larger
reads are allowed up to the 2,000-line hard response limit. The legacy
inclusive `end_line` request field remains supported but cannot be combined
with `line_count`.

EOF clamping and server truncation are separate:

- `range_clamped` means the requested range crossed file bounds.
- `content_truncated` means the hard server limit omitted valid requested
  content.
- `next_start_line` identifies the next readable line when content remains.

## Remaining M1 limitations

- Cursor continuation is not yet implemented; `next_cursor` is therefore
  `null` even when the bounded candidate probe reports truncation.
- Relocating a project preserves its relative file identities. Reusing an
  existing persisted index at a new root still requires explicit operational
  remapping or a clean reindex.
- Git diff text is retrieval evidence, not an authoritative current-tree
  dependency source. For a subdirectory project, `path_at_commit` is filtered
  and mapped to the project namespace, while diff snippets may still contain
  surrounding repository context.
