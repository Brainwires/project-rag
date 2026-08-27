# M4: Build-aware and non-call analysis

## Data flow

```text
project root + analysis configuration
-> compile_commands.json discovery and explicit configurations
-> stable semantic config_id catalog
-> per-file preprocessor evaluation
-> configuration-scoped persisted references
-> reference/graph config filters
-> conservative unused status + completeness + limitations
```

M4 reads compilation databases from explicitly configured paths or the project
root, `build/`, and `out/`. Entries with equivalent semantic flags are merged into
one configuration while retaining their project-relative source-file set. Absolute
project-root spelling is excluded from derived config IDs.

Explicit configurations support include paths, preprocessor definitions, language
standard, forced includes, generated-header paths, and optional source-file scope.
They may supplement or replace compilation database discovery.

## Conditional compilation

Each persisted reference now exposes `configuration_states` entries containing a
`config_id` and one of:

- `active`
- `inactive`
- `unknown_due_to_build_config`

The evaluator handles nested `#if`, `#ifdef`, `#ifndef`, `#elif`, `#else`, and
`#endif`, including simple `defined`, boolean, and numeric macro expressions.
Unsupported expressions remain unknown instead of being treated as absent.

Forced includes are persisted as explicit `include` observations with
`build-config/forced-include` provenance. Include resolution used by unused-import
analysis now searches discovered/configured include and generated-header paths.

`find_references` and `get_call_graph` accept `configurations` filters. A row
matches when the selected configuration marks it active or unknown; inactive rows
are excluded. Unknown rows remain visible because absence has not been proven.

## Conservative unused results

Unused findings and the response expose:

- `status`: `unused_in_analyzed_configurations` or `inconclusive`
- `analyzed_configurations`
- `analysis_completeness`: `complete` or `partial`
- `unresolved_dependency_kinds`
- `limitations`
- `safe_for_destructive_edit`

The public status model also includes `referenced`; referenced definitions are
currently rejected from the candidate list rather than emitted as findings.

Missing build configurations, unknown conditional state, parser omissions,
unverifiable includes, exhausted cross-index probes, dynamic registration,
reflection, generated paths, or forced includes prevent a universal unused claim.
A reference active only in one of several indexed configurations keeps its target
live across the combined analysis.

`safe_for_destructive_edit` is always false in M4. M5 removal validation is the
appropriate place to make a scoped SAFE/UNSAFE/INCONCLUSIVE decision.

## Non-call dependencies

Reference classification now distinguishes additional statically visible kinds:

- `member_read` and `member_write`
- `method_call`
- `object_construction` and `object_assignment`
- `owns`, `references`, and `points_to`
- `creates` and `destroys`
- existing `type_use` and `inheritance`

These are parser/syntax-backed observations, not full alias or lifetime analysis.
The implementation deliberately does not invent targets for indirect ownership or
dispatch relationships it cannot prove.

## Persistence and migration

Relation tables advance from v3 to v4 to store configuration state. Cache schema 4
preserves compatible retrieval hashes and stable project IDs; run indexing once to
publish v4 relation rows. Old and new relation identities live in separate tables,
so they cannot mix. Build discovery diagnostics are also surfaced through index
errors and cache/index-health diagnostics.

## Remaining limitations

- The preprocessor evaluator is intentionally conservative and is not a complete C
  preprocessor or compiler frontend.
- Configuration-driven metadata formats are reported as limitations through path
  and wiring detection; project-specific dynamic wiring tokens should be added to
  `analysis.dynamic_wiring_patterns`.
- Alias analysis, runtime registration resolution, generator execution, and full
  ownership/lifetime inference remain out of scope.
- Atomic patching, incremental invalidation, generation-safe caches, and removal
  validation are M5.

## Regression and measurement coverage

Tests cover compile database parsing (including malformed input fallback), stable
multi-config IDs, duplicate/unavailable selection, nested conditional
active/inactive/unknown states, active/unknown/inactive configuration-filter
boundaries, configuration-scoped persistence and graph filters, symbols referenced
only in another configuration, missing-config inconclusive status,
generated/dynamic wiring limitations, and non-call classification.

Ignored synthetic performance tests cover every M4 analysis path added here. On the
verification host in an unoptimised test build they measured:

- non-call extraction: 250 symbols / 313 observations in 12 ms
- configuration-filtered depth-two graph: 165 ms cold / 147 ms warm
- scoped relation persistence: 500 definitions / 2,000 references written in
  299 ms, exact-name query in 40 ms, database size 1,080,651 bytes
- conservative unused analysis: 200 files across four configurations in 31 ms

These host-specific measurements are regression baselines, not correctness or
production latency guarantees.
