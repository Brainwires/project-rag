# M2: Trustworthy symbols and references

## Data flow

Indexing now performs one relations pass after file discovery:

1. Tree-sitter extracts declarations and definitions from every supported current-tree file.
2. Each declaration/definition receives a logical `symbol_id` and a separate source `location_id`.
3. A project-wide name index generates reference candidates.
4. Tree-sitter classifies occurrence context; textual name matching never establishes a target by itself.
5. Definition/declaration locations and reference occurrences are stored together in the v3 reference table.
6. `find_references`, relation statistics, callers, and callees read that persisted source of truth. Graph paths accept only `resolved` call edges.

M2 deliberately rebuilds the relations corpus on an incremental-index request. Dependency-aware invalidation and generation publication remain M5 work.

## Identity and evidence schema

A logical symbol digest includes:

- stable persisted `project_id`;
- language;
- qualified name;
- symbol kind;
- canonical signature;
- linkage;
- a project-relative scope discriminator only for internal, anonymous, file-local, or local symbols.

Absolute project location and declaration coordinates do not participate in external symbol identity. A `SourceLocation` separately stores stable project ID, canonical project-relative path, exact line/column span, and declaration/definition/reference role.

Reference rows expose:

- source and target symbol IDs where established;
- candidate IDs and reasons where not established;
- reference kind;
- independent `resolution_status` and `evidence_kind`;
- dispatch kind;
- canonical path and exact span;
- language and parser provenance.

Unqualified RepoMap name matches are `unresolved` for one candidate and `ambiguous` for multiple candidates, with `evidence_kind = heuristic`. An exact qualified syntactic match may be `resolved` with syntactic evidence. Comments, documentation, and strings are separately classified and excluded by default.

## Persistence and migration

- Overall index schema: v3.
- Definitions table: `relations_definitions_v3`.
- References table: `relations_references_v3`.
- A v2 cache migrates without changing its persisted `project_id` or current-file hashes.
- Run indexing once after migration to populate the v3 relation tables. The first incremental indexing pass rebuilds the complete relation corpus even when no source hashes changed.
- Older incompatible identity schemas still require a clean reindex.

## API compatibility

`find_references.total_count` remains as a compatibility alias for `total_matches`. New pagination metadata includes `returned_matches`, `results_truncated`, and `next_cursor`.

New optional filters are `language`, `path_filter`, `reference_kinds`, `resolution_statuses`, and `evidence_kinds`. `include_non_code` opts into documentation/comment/string matches when no explicit kind filter is supplied. Existing requests deserialize with code-only defaults.

Definition, symbol-list, reference, and call-graph results now expose stable identities and parser/evidence provenance. Consumers must not infer authority from the legacy `precision` label; `resolution_status` and `evidence_kind` are the edge-level contract.

## Known limitations

- RepoMap is a parser-backed classifier plus conservative name-candidate generator, not a full language name resolver.
- Canonical signatures normalize whitespace and callable qualification but do not yet model every language ABI qualifier.
- Indirect, virtual, callback, and dynamic dispatch remain `unknown` unless a future resolver proves more.
- Recursive graph depth, graph-form output, adjacency indexes, and graph budgets are M3.
- Full relation rebuilding on incremental indexing is correctness-first and will be replaced by M5 invalidation.

## Synthetic measurement

The ignored `benchmark_m2_relations_store` regression provides a repeatable local measurement. On the M2 development Windows host, a debug build writing 500 definitions plus 2,000 classified references took 247 ms, occupied 855,453 bytes, and an exact root/name query returning four rows took 49 ms. These are synthetic storage measurements, not production latency guarantees; parser time and full-project relation rebuilding depend on corpus size.
