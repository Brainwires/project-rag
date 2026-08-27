# M5: Runtime, editing, and removal validation

## Transaction data flow

```text
read_file raw-byte hash
-> apply_patch canonicalizes every path into one project root
-> validate all hashes/ranges/encodings/sizes/parents
-> optional dry-run result
-> stage sibling temporary files and fsync
-> commit with per-file backups and transaction rollback
-> one smart incremental index call
-> invalidate changed symbols and dependent reference files
-> publish one monotonic root generation
-> clear generation-scoped analysis caches
```

`apply_patch` is the primary editing API. It accepts up to 100 structured file
patches, including whole-file replacement/creation, line replacement/insertion,
and deletion. Existing files require the raw-byte SHA256 returned by `read_file`.
All aliases pass through the canonical project-relative path resolver, duplicate
canonical targets are rejected, and one transaction cannot span project roots.

The legacy `edit_file` API delegates to the same transaction engine with its
historical optional-hash behavior.

## Atomic filesystem behavior

The transaction validates every file before staging anything. UTF-8, UTF-8 BOM,
UTF-16LE, and UTF-16BE are decoded and re-encoded in their original form. Inserted
content adopts the existing LF or CRLF style. Unsupported/binary encodings are
rejected before writes.

Replacement data is written to sibling temporary files and flushed first. Commit
moves existing targets to transaction backups, publishes staged files, and rolls
back every already-committed target if any later operation fails. Backup cleanup
failure is logged as a warning because the requested content is already
committed. Crash-recovery journaling across process termination remains a future
hardening step; no portable filesystem offers a single atomic rename spanning
multiple files.

## Coherent index generations

Cache schema 5 adds a monotonic generation per indexed root. A generation advances
only after vector flushing and authoritative relation publication succeed. Dirty
roots block retrieval/analysis that could otherwise observe a partial generation.
Recovery after an interrupted publication performs a clean rebuild; approximate
row-count similarity is no longer treated as proof of coherence.

Graph cache keys include the canonical root/file, complete request filters and
build configurations, and the published generation. Semantic and filtered-query
cache entries include the complete request, canonical root filter, and a sorted
generation fingerprint for all indexed roots. An entry from generation N cannot
satisfy a generation N+1 lookup. Successful publication clears in-memory analysis
caches as an additional bound on obsolete entries.

## Incremental invalidation

Incremental indexing parses changed files, retains persisted definitions from
unchanged files, and derives changed symbol names from both old and new
definitions. Persisted references naming those symbols identify source files that
must be re-parsed and re-resolved. Only changed/removed definition rows and
changed/dependent reference rows are replaced. Build-input changes such as
`compile_commands.json` or CMake files conservatively invalidate every relation
source file.

Relation publication failures keep the root dirty, so graph and symbol readers do
not expose a partially updated table set.

## Removal validation

`validate_removal(symbol_id, configurations)` uses only current-tree authoritative
relations and returns `SAFE`, `UNSAFE`, or `INCONCLUSIVE` with:

- exact project/configuration/generation scope;
- resolved and uncertain reference evidence;
- blocking reference provenance;
- preprocessor and build coverage;
- generated/forced input detection;
- registration, reflection, plugin, and indirect-dispatch limitations.

Any resolved current-tree code dependency makes the result `UNSAFE`. Ambiguous or
unresolved candidate references, incomplete build coverage, unknown conditional
state, generated wiring, or dynamic wiring make it `INCONCLUSIVE`. `SAFE` is only
within the explicitly returned scope and supported dependency model. Git history
is never liveness evidence.

## Regression and performance coverage

Edge tests cover mid-commit rollback, UTF-16 BOM plus CRLF preservation, schema-4
generation migration, stale-generation cache rejection, independent definition
and reference invalidation, explicit verdict precedence, and uppercase removal
contracts. Ignored performance tests measure 50-file staging/commit, 100,000
generation-scoped graph and query-cache hits, dependency invalidation over 500
definitions and 2,000 references, and 100,000 removal verdict decisions.

One debug-profile run on the Windows verification host measured:

| Scenario | Workload | Elapsed |
|---|---:|---:|
| Atomic staging and commit | 50 files | 133 ms |
| Generation-scoped graph cache | 100,000 hits | 40,327 us |
| Generation-scoped query cache | 100,000 hits | 19,255 us |
| Relation invalidation | 500 definitions, 2,000 references, 10 changed symbols | 122 ms |
| Removal verdict decision | 100,000 evaluations | 1,156 us |

These host-specific microbenchmarks are regression baselines, not production
latency guarantees.

## Compatibility and limitations

- Cache schema 4 migrates safely to schema 5 with generation zero; the next
  successful index publishes generation one.
- Relation tables remain schema v4 because generation gates publication at the
  root/cache layer rather than changing reference identity.
- File commits are retained if reindexing fails. The response reports
  `reindex_failed`, the old generation remains published, and all analysis is
  blocked until `index_codebase` performs recovery.
- Removal validation is conservative static analysis, not compiler/linker or
  generator execution.
