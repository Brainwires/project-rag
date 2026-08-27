# M3: Correct dependency graphs

## Data flow

```text
get_call_graph position
-> canonical project file + logical root symbol
-> breadth-first stable-symbol frontier
-> indexed outgoing/incoming reference-table queries
-> explicit kind/status/language/path filters
-> unique nodes + provenance-bearing edges
-> budgets, totals, and continuation frontier
```

The reference table introduced in M2 remains the source of truth. M3 adds named
B-tree indexes on `source_symbol_id` (`relations_outgoing_v1`) and
`target_symbol_id` (`relations_incoming_v1`); it does not create a competing edge
store. Batch frontier queries replace per-edge definition and adjacency queries.

## Depth and graph semantics

- Depth 0 returns only the root node.
- Depth 1 returns the root and its direct selected-direction neighbors.
- Depth 2 additionally expands those neighbors once.
- A stable `symbol_id` visited set prevents cyclic or diamond-shaped graphs from
  recursively duplicating nodes.
- Every returned resolved edge has both endpoint nodes. If an extracted target has
  no stored definition location, the node is retained with
  `definition_available = false` and optional location fields.
- Ambiguous or unresolved outgoing observations can be included through
  `resolution_statuses`. They expose candidate sets but have no authoritative
  `target_symbol_id`, and traversal never follows their candidates.

## API and budgets

`get_call_graph` now returns graph-form `nodes` and `edges`. The old direct
`callers` and `callees` response arrays are removed. Direction remains controlled
by the compatible `include_callers` and `include_callees` request fields.

New request controls:

- `max_nodes` (default 200, hard maximum 5,000)
- `max_edges` (default 500, hard maximum 20,000)
- `edge_kinds` (default `call`, `constructor_call`)
- `resolution_statuses` (default `resolved`)
- `language_filters`
- `path_filters`

Caps set `graph_truncated`, preserve graph endpoint integrity, and return a
`continuation` object containing stable pending symbols, depth information, and the
cap reason. `estimated_or_known_total` is exact for a completed traversal and an
explicit lower bound (`exact = false`) when a cap stops further expansion.

Each edge exposes source and target identities, candidates where applicable,
reference kind, resolution, evidence, dispatch, canonical path, exact source
range, language, and parser provenance.

## Migration and compatibility

The M2 row schemas remain compatible. Existing v3 reference tables lazily build
the two adjacency indexes on the first graph query; a normal indexing pass builds
and warms them when references are published. No clean reindex is required.

The graph response shape is an intentional API compatibility change: consumers
must read `nodes` and `edges` instead of relationship-decorated `callers` and
`callees` arrays. Existing request payloads deserialize with M3 defaults.

## Remaining limitations

- Parser-backed RepoMap resolution remains conservative; ordinary graph requests
  default to resolved rows only.
- Ambiguous candidate membership is not represented as an incoming authoritative
  edge. It is visible when expanding its known source, which avoids inventing a
  target relationship.
- Build configurations, preprocessor state, generated wiring, and richer non-call
  dependencies are M4.
- Generation-safe graph caching and dependency-aware incremental invalidation are
  M5. M3 measures cold and warm storage/index latency without caching results.

## Regression and measurement coverage

Tests cover exact depth 0/1/2 behavior, cycles, diamonds, ambiguity, provenance,
filters, node/edge caps, continuation metadata, and physical adjacency-index
creation. The ignored `benchmark_m3_depth_two_cold_and_warm` test provides a
repeatable synthetic cold/warm depth-2 measurement; host-specific results are
recorded after verification rather than treated as a correctness gate.

On the M3 development Windows host, a debug build over a synthetic 500-symbol,
499-edge chain returned the depth-2 graph (3 nodes, 2 edges) in 154 ms cold and
141 ms warm. These figures isolate persisted adjacency traversal and definition
lookup; they are not production guarantees and do not include parsing/indexing.
