//! find_unused: report import bindings and symbol definitions that nothing
//! references.
//!
//! The analysis is text-based on top of AST extraction, so it errs on the side
//! of "used": any word-boundary mention of a name -- call, type, comment,
//! import elsewhere -- counts as usage. What it CANNOT see is dynamic dispatch,
//! macro expansion, reflection and framework wiring, which is why every
//! candidate carries a confidence level and the tool never edits anything.
//!
//! Two checks:
//! - **imports**: an import binding is unused when its name never appears in
//!   the importing file outside import statements. File-local, needs no index.
//!   C/C++ `#include` is verified indirectly: the header is located among the
//!   scanned files or resolved on disk next to the including file or the scan
//!   root, and the question becomes "do any symbols it defines appear in this
//!   file?". An include that cannot be resolved is skipped -- counted in
//!   `unverifiable_import_details`, never flagged -- as are system `<...>`
//!   includes, C# namespace usings and Swift module imports.
//! - **symbols**: a definition is a dead-code candidate when its name appears
//!   nowhere else in the scanned corpus outside import lines and same-name
//!   definition spans. When only part of an indexed root is scanned, the
//!   BM25/hybrid index is probed for outside usage before flagging anything.

use super::RagClient;
use crate::build_config::{BuildConfigCatalog, ConfigurationState, PreprocessorState};
use crate::indexer::{FileInfo, FileWalker};
use crate::relations::repomap::language_name_for_extension;
use crate::relations::{Definition, RelationsProvider, SymbolKind, Visibility};
use crate::types::{
    AnalysisCompleteness, FindUnusedRequest, FindUnusedResponse, SymbolRejections, UnusedCandidate,
    UnusedStatus, UnverifiableImport,
};
use anyhow::{Context, Result};
use rayon::prelude::*;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Names that runtimes, traits and frameworks invoke implicitly; a zero-mention
/// count for these means nothing, so they are never flagged.
const IMPLICITLY_INVOKED: &[&str] = &[
    "main",
    "new",
    "default",
    "fmt",
    "drop",
    "clone",
    "eq",
    "ne",
    "cmp",
    "partial_cmp",
    "hash",
    "next",
    "from",
    "into",
    "try_from",
    "deref",
    "deref_mut",
    "index",
    "index_mut",
    "serialize",
    "deserialize",
    "to_string",
    "from_str",
    "__init__",
    "__str__",
    "__repr__",
    "__eq__",
    "__hash__",
    "__len__",
    "toString",
    "equals",
    "hashCode",
    "constructor",
    "render",
];

/// Cap on cross-index BM25 probes per call. Symbols past the budget are
/// conservatively treated as used and `probes_exhausted` is set.
const MAX_BM25_PROBES: usize = 100;
const PROBE_FILES_PER_NAME: usize = 10;

/// Cap on entries in `unverifiable_import_details`; the `unverifiable_imports`
/// count stays authoritative past it.
const MAX_UNVERIFIABLE_DETAILS: usize = 200;
/// Cap on header-symbol names spelled out in a candidate's probe string.
const MAX_PROBE_SYMBOLS_SHOWN: usize = 10;

/// Everything the pure, per-file part of the analysis produces.
struct Analyzed {
    files: Vec<FileInfo>,
    /// Definitions per file, parallel to `files`
    defs: Vec<Vec<Definition>>,
    /// identifier -> 1-based lines it appears on, per file
    idents: Vec<HashMap<String, Vec<usize>>>,
    /// (start_line, end_line) of every import statement, per file
    import_spans: Vec<Vec<(usize, usize)>>,
    /// name -> definition spans, per file; a mention inside a same-name
    /// definition (its own body, its impl block) is not usage
    def_spans_by_name: Vec<HashMap<String, Vec<(usize, usize)>>>,
    /// Definition nodes the extractor recognised but could not name; those
    /// symbols were never considered at all
    skipped_definitions: usize,
    /// Canonicalized paths of scanned files, for excluding them from probes
    scanned_paths: HashSet<PathBuf>,
    /// Per file, per 1-based line, states in every selected build configuration.
    configuration_states: Vec<Vec<Vec<ConfigurationState>>>,
}

/// Import candidates plus every binding that could not be verified, with the
/// reason it could not be.
#[derive(Default)]
struct ImportScan {
    candidates: Vec<UnusedCandidate>,
    unverifiable: Vec<UnverifiableImport>,
}

impl RagClient {
    /// Scan a file or directory for unused imports and unused symbols.
    pub async fn find_unused(&self, request: FindUnusedRequest) -> Result<FindUnusedResponse> {
        let start = Instant::now();
        request.validate().map_err(|e| anyhow::anyhow!(e))?;

        let normalized = Self::normalize_path(&request.path)?;

        // Symbol verification needs the index: without it a partial scan cannot
        // be told apart from a full one, and the cross-file probe has nothing to
        // search, so everything would look unused -- the dangerous direction.
        let indexed_root = self.find_indexed_root(&normalized).await;
        if request.check_symbols() {
            let Some(ref root) = indexed_root else {
                anyhow::bail!(
                    "'{}' is not inside an indexed root; run index_codebase first, \
                     or use check: \"imports\" for index-free import analysis",
                    request.path
                );
            };
            self.check_path_not_dirty(Some(root)).await?;
        }
        // Probes are needed only when the scan covers less than the indexed
        // root; scanning the whole root makes the in-memory corpus authoritative.
        let partial_scan = indexed_root.as_deref() != Some(normalized.as_str());

        let build_root = indexed_root
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let path = PathBuf::from(&normalized);
                if path.is_file() {
                    path.parent().unwrap_or(&path).to_path_buf()
                } else {
                    path
                }
            });
        let build_catalog = BuildConfigCatalog::discover(&build_root, &self.config.analysis)?
            .selected(&request.configurations);
        let analyzed_configurations = build_catalog.config_ids();
        let include_paths = build_catalog.include_paths(&build_root);

        // Gather and analyze files on a blocking thread (I/O + tree-sitter + regex).
        let provider = self.relations_provider.clone();
        let single_file = Path::new(&normalized).is_file();
        let files = if single_file {
            if indexed_root.is_some() {
                vec![
                    self.create_file_info(&normalized, request.project.clone())
                        .await?,
                ]
            } else {
                let path = Path::new(&normalized);
                let root = path.parent().unwrap_or(path);
                vec![RagClient::build_file_info_in_root(
                    path,
                    &root.to_string_lossy(),
                    request.project.clone(),
                )?]
            }
        } else {
            let walker = FileWalker::new(&normalized, request.max_file_size)
                .with_project(request.project.clone());
            tokio::task::spawn_blocking(move || walker.walk())
                .await
                .context("File walker task panicked")?
                .context("Failed to walk directory")?
        };

        // Include resolution reads and parses headers from disk, so the import
        // scan shares the analysis' blocking task.
        let scan_root = {
            let p = PathBuf::from(&normalized);
            if single_file {
                p.parent().map(|q| q.to_path_buf()).unwrap_or(p)
            } else {
                p
            }
        };
        let check_imports = request.check_imports();
        let resolver_provider = provider.clone();
        let resolver_project = request.project.clone();
        let analysis_catalog = build_catalog.clone();
        let (analyzed, import_scan) = tokio::task::spawn_blocking(move || {
            let analyzed = analyze(files, provider, &analysis_catalog);
            let import_scan = if check_imports {
                let mut resolver = IncludeResolver {
                    scan_root,
                    include_paths,
                    project: resolver_project,
                    provider: resolver_provider,
                    cache: HashMap::new(),
                };
                import_candidates(&analyzed, &mut resolver)
            } else {
                ImportScan::default()
            };
            (analyzed, import_scan)
        })
        .await
        .context("Analysis task panicked")?;

        let definitions_checked = analyzed.defs.iter().map(|d| d.len()).sum();

        let mut candidates = import_scan.candidates;
        let unverifiable_imports = import_scan.unverifiable.len();
        let mut unverifiable_import_details = import_scan.unverifiable;
        unverifiable_import_details.truncate(MAX_UNVERIFIABLE_DETAILS);

        let mut probes_exhausted = false;
        let mut symbol_rejections = SymbolRejections::default();
        if request.check_symbols() {
            let (pending, rejections) = symbol_candidates(&analyzed);
            symbol_rejections = rejections;
            if partial_scan {
                let (confirmed, exhausted, probe_errors) = self
                    .probe_unmentioned(pending, request.project.clone(), &analyzed.scanned_paths)
                    .await;
                candidates.extend(confirmed);
                probes_exhausted = exhausted;
                symbol_rejections.probe_errors = probe_errors;
            } else {
                // Full-root scan: the corpus already proved these unmentioned.
                candidates.extend(pending.into_iter().map(|(_, c)| c));
            }
        }

        // One entry per (file, name): a Rust struct and its impl block are two
        // definitions of the same name, and flagging both is noise.
        candidates.sort_by(|a, b| (&a.file_path, a.line).cmp(&(&b.file_path, b.line)));
        let mut seen = HashSet::new();
        candidates.retain(|c| seen.insert((c.file_path.clone(), c.name.clone())));

        let assessment = assess_analysis(
            &analyzed,
            &build_catalog,
            &self.config.analysis.dynamic_wiring_patterns,
            &self.config.analysis.generated_path_patterns,
            unverifiable_imports,
            probes_exhausted,
            &symbol_rejections,
        );
        finalize_candidates(&mut candidates, &analyzed, &assessment);

        let total_candidates = candidates.len();
        let truncated = total_candidates > request.limit;
        candidates.truncate(request.limit);

        Ok(FindUnusedResponse {
            scanned_root: normalized,
            files_scanned: analyzed.files.len(),
            definitions_checked,
            candidates,
            total_candidates,
            unverifiable_imports,
            unverifiable_import_details,
            symbol_rejections,
            skipped_definitions: analyzed.skipped_definitions,
            truncated,
            probes_exhausted,
            analyzed_configurations,
            analysis_completeness: assessment.completeness,
            unresolved_dependency_kinds: assessment.unresolved_dependency_kinds,
            limitations: assessment.limitations,
            safe_for_destructive_edit: false,
            precision: "medium".to_string(),
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// The indexed root that contains `normalized`, if any.
    async fn find_indexed_root(&self, normalized: &str) -> Option<String> {
        let cache = self.hash_cache.read().await;
        cache
            .roots
            .keys()
            .find(|root| {
                normalized == root.as_str()
                    || normalized
                        .strip_prefix(root.as_str())
                        .is_some_and(|rest| rest.starts_with(['/', '\\']))
            })
            .cloned()
    }

    /// Probe the index for usage of each pending name; return the candidates
    /// whose names are mentioned nowhere, whether the budget ran out, and how
    /// many probes errored (those names are conservatively treated as used).
    async fn probe_unmentioned(
        &self,
        pending: Vec<(String, UnusedCandidate)>,
        project: Option<String>,
        scanned_paths: &HashSet<PathBuf>,
    ) -> (Vec<UnusedCandidate>, bool, usize) {
        // Group candidates by name so each name is probed once.
        let mut by_name: HashMap<String, Vec<UnusedCandidate>> = HashMap::new();
        for (name, candidate) in pending {
            by_name.entry(name).or_default().push(candidate);
        }
        let mut names: Vec<String> = by_name.keys().cloned().collect();
        names.sort();

        let exhausted = names.len() > MAX_BM25_PROBES;
        let mut confirmed = Vec::new();
        let mut probe_errors = 0usize;

        for name in names.into_iter().take(MAX_BM25_PROBES) {
            let mut used = false;
            let probe_files = match self
                .files_mentioning(&name, project.clone(), PROBE_FILES_PER_NAME)
                .await
            {
                Ok(f) => f,
                Err(e) => {
                    // Cannot verify -- treat as used rather than flag blindly.
                    tracing::debug!("Probe failed for {}: {}", name, e);
                    probe_errors += 1;
                    continue;
                }
            };
            for file in probe_files {
                let canonical = tokio::fs::canonicalize(&file)
                    .await
                    .unwrap_or_else(|_| file.clone());
                if scanned_paths.contains(&canonical) {
                    continue; // already covered by the in-memory corpus check
                }
                match tokio::fs::read_to_string(&file).await {
                    Ok(content) if mentions_identifier(&content, &name) => {
                        used = true;
                        break;
                    }
                    _ => {}
                }
            }
            if !used {
                confirmed.extend(by_name.remove(&name).unwrap_or_default());
            }
        }
        (confirmed, exhausted, probe_errors)
    }
}

/// Pure per-file analysis: definitions, identifier occurrence maps, import and
/// definition spans.
fn analyze(
    files: Vec<FileInfo>,
    provider: Arc<crate::relations::HybridRelationsProvider>,
    build_catalog: &BuildConfigCatalog,
) -> Analyzed {
    let ident_re = Regex::new(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b").expect("static regex");

    let reported: Vec<(Vec<Definition>, usize)> = files
        .par_iter()
        .map(|file| {
            provider
                .extract_definitions_reporting(file)
                .map(|(defs, skipped)| (defs, skipped.len()))
                .unwrap_or_else(|e| {
                    tracing::debug!(
                        "Definition extraction failed for {}: {}",
                        file.relative_path,
                        e
                    );
                    (Vec::new(), 0)
                })
        })
        .collect();
    let skipped_definitions = reported.iter().map(|(_, skipped)| skipped).sum();
    let defs: Vec<Vec<Definition>> = reported.into_iter().map(|(defs, _)| defs).collect();

    let idents: Vec<HashMap<String, Vec<usize>>> = files
        .par_iter()
        .map(|file| identifier_lines(&file.content, &ident_re))
        .collect();

    let import_spans: Vec<Vec<(usize, usize)>> = defs
        .iter()
        .map(|file_defs| {
            file_defs
                .iter()
                .filter(|d| d.kind() == SymbolKind::Import)
                .map(|d| {
                    // tree-sitter ends a preproc_include past its newline, at
                    // column 0 of the NEXT row; unclamped, that span swallows
                    // the first code line after an include and hides every
                    // identifier on it from the usage check.
                    let end = if d.end_col == 0 && d.end_line > d.start_line() {
                        d.end_line - 1
                    } else {
                        d.end_line
                    };
                    (d.start_line(), end)
                })
                .collect()
        })
        .collect();

    let def_spans_by_name: Vec<HashMap<String, Vec<(usize, usize)>>> = defs
        .iter()
        .map(|file_defs| {
            let mut spans: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
            for d in file_defs {
                spans
                    .entry(d.name().to_string())
                    .or_default()
                    .push((d.start_line(), d.end_line));
            }
            spans
        })
        .collect();

    let scanned_paths = files
        .iter()
        .map(|f| std::fs::canonicalize(&f.path).unwrap_or_else(|_| f.path.clone()))
        .collect();
    let configuration_states = files
        .iter()
        .map(|file| build_catalog.states_for_file(&file.relative_path, &file.content))
        .collect();

    Analyzed {
        files,
        defs,
        idents,
        import_spans,
        def_spans_by_name,
        skipped_definitions,
        scanned_paths,
        configuration_states,
    }
}

/// Resolves quoted include paths on disk and enumerates the symbols the
/// resolved header defines. Parse results are cached per canonical path.
struct IncludeResolver {
    /// Base directory includes resolve against besides the including file's own
    scan_root: PathBuf,
    include_paths: Vec<PathBuf>,
    project: Option<String>,
    provider: Arc<crate::relations::HybridRelationsProvider>,
    cache: HashMap<PathBuf, Arc<Vec<String>>>,
}

impl IncludeResolver {
    /// `None` = not resolvable on disk (or unreadable / unparseable).
    /// `Some` with an empty list = resolved, but nothing extractable is
    /// defined in it.
    fn header_symbols(&mut self, including_file: &Path, needle: &str) -> Option<Arc<Vec<String>>> {
        let mut bases = vec![including_file.parent(), Some(self.scan_root.as_path())]
            .into_iter()
            .flatten()
            .map(Path::to_path_buf)
            .collect::<Vec<_>>();
        bases.extend(self.include_paths.clone());
        for base in bases {
            let candidate = base.join(needle);
            let Ok(canonical) = std::fs::canonicalize(&candidate) else {
                continue;
            };
            if !canonical.is_file() {
                continue;
            }
            if let Some(cached) = self.cache.get(&canonical) {
                return Some(cached.clone());
            }
            let Ok(info) = RagClient::build_file_info_in_root(
                &canonical,
                &self.scan_root.to_string_lossy(),
                self.project.clone(),
            ) else {
                return None;
            };
            let Ok((defs, _skipped)) = self.provider.extract_definitions_reporting(&info) else {
                return None;
            };
            let names: Vec<String> = defs
                .iter()
                .filter(|d| d.kind() != SymbolKind::Import && d.name().len() >= 2)
                .map(|d| d.name().to_string())
                .collect();
            let names = Arc::new(names);
            self.cache.insert(canonical, names.clone());
            return Some(names);
        }
        None
    }
}

/// Unused-import candidates plus the bindings that could not be verified.
fn import_candidates(analyzed: &Analyzed, resolver: &mut IncludeResolver) -> ImportScan {
    let mut scan = ImportScan::default();

    for (i, file_defs) in analyzed.defs.iter().enumerate() {
        // The extractor's taxonomy, from the extension -- NOT FileInfo.language.
        // The display taxonomy calls headers "C/C++ Header", which would route
        // every #include into the generic arm below and probe for a filename no
        // identifier token can ever match.
        let language = analyzed.files[i]
            .extension
            .as_deref()
            .and_then(language_name_for_extension);
        let file_path = &analyzed.files[i].relative_path;
        for def in file_defs.iter().filter(|d| d.kind() == SymbolKind::Import) {
            match language {
                Some("C") | Some("C++") => match check_include(analyzed, i, def, resolver) {
                    IncludeVerdict::Unused(candidate) => scan.candidates.push(*candidate),
                    IncludeVerdict::Used => {}
                    IncludeVerdict::Unverifiable(reason) => {
                        scan.unverifiable.push(UnverifiableImport {
                            file_path: file_path.clone(),
                            name: def.name().to_string(),
                            reason,
                        })
                    }
                },
                // Swift imports bind a module whose name need not appear in code.
                Some("Swift") => scan.unverifiable.push(UnverifiableImport {
                    file_path: file_path.clone(),
                    name: def.name().to_string(),
                    reason: "Swift module import; the module name need not appear in code"
                        .to_string(),
                }),
                // A plain C# `using Namespace;` makes members usable WITHOUT the
                // namespace name appearing; only alias usings are checkable.
                Some("C#") if !def.signature.contains('=') => {
                    scan.unverifiable.push(UnverifiableImport {
                        file_path: file_path.clone(),
                        name: def.name().to_string(),
                        reason: "C# namespace using; members are usable without the namespace name"
                            .to_string(),
                    })
                }
                // No language means no extraction should have produced imports;
                // never emit a candidate that no probe can support.
                None => scan.unverifiable.push(UnverifiableImport {
                    file_path: file_path.clone(),
                    name: def.name().to_string(),
                    reason: "file language unknown; import not checkable".to_string(),
                }),
                Some(_) => {
                    if !binding_used(analyzed, i, def.name()) {
                        let (confidence, note) = if language == Some("Rust") {
                            (
                                "medium",
                                "; note: Rust trait imports can be used implicitly via method calls",
                            )
                        } else {
                            ("high", "")
                        };
                        scan.candidates.push(make_candidate(
                            def,
                            file_path,
                            confidence,
                            format!(
                                "imported name '{}' is never referenced in this file{}",
                                def.name(),
                                note
                            ),
                            format!(
                                "whole-word search for '{}' in this file outside import lines",
                                def.name()
                            ),
                        ));
                    }
                }
            }
        }
    }
    scan
}

enum IncludeVerdict {
    Unused(Box<UnusedCandidate>),
    Used,
    /// Why the include could not be checked; travels into
    /// `unverifiable_import_details`.
    Unverifiable(String),
}

/// Verify a C/C++ `#include` indirectly: is any symbol defined by the included
/// header referenced in the including file? The header is located among the
/// scanned files first, then resolved on disk relative to the including file
/// and the scan root. An include that cannot be resolved is skipped
/// (unverifiable), never flagged.
fn check_include(
    analyzed: &Analyzed,
    i: usize,
    def: &Definition,
    resolver: &mut IncludeResolver,
) -> IncludeVerdict {
    // Only quoted local includes are resolvable; `<...>` system headers live on
    // include paths this tool does not know.
    if !def.signature.contains('"') {
        return IncludeVerdict::Unverifiable("system include (<...>); not resolvable".to_string());
    }

    let needle = def.name().trim_start_matches("./").replace('\\', "/");
    let target = (0..analyzed.files.len()).find(|&j| {
        j != i && {
            let p = analyzed.files[j].relative_path.replace('\\', "/");
            p == needle || p.ends_with(&format!("/{}", needle))
        }
    });

    let header_symbols: Vec<String> = match target {
        Some(j) => analyzed.defs[j]
            .iter()
            .filter(|d| d.kind() != SymbolKind::Import && d.name().len() >= 2)
            .map(|d| d.name().to_string())
            .collect(),
        None => match resolver.header_symbols(&analyzed.files[i].path, &needle) {
            Some(names) => names.as_ref().clone(),
            None => {
                return IncludeVerdict::Unverifiable(
                    "include not resolved in scan set or on disk".to_string(),
                );
            }
        },
    };
    if header_symbols.is_empty() {
        return IncludeVerdict::Unverifiable(
            "resolved header has no extractable definitions".to_string(),
        );
    }

    if header_symbols.iter().any(|n| binding_used(analyzed, i, n)) {
        IncludeVerdict::Used
    } else {
        let shown = header_symbols
            .iter()
            .take(MAX_PROBE_SYMBOLS_SHOWN)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(", ");
        let extra = header_symbols.len().saturating_sub(MAX_PROBE_SYMBOLS_SHOWN);
        let probe = if extra > 0 {
            format!(
                "searched this file for {} symbols defined by the header: {}, +{} more",
                header_symbols.len(),
                shown,
                extra
            )
        } else {
            format!(
                "searched this file for {} symbols defined by the header: {}",
                header_symbols.len(),
                shown
            )
        };
        IncludeVerdict::Unused(Box::new(make_candidate(
            def,
            &analyzed.files[i].relative_path,
            "medium",
            format!(
                "none of the {} symbols defined by '{}' are referenced in this file",
                header_symbols.len(),
                def.name()
            ),
            probe,
        )))
    }
}

/// True if `name` appears in file `i` on any line outside every import statement.
fn binding_used(analyzed: &Analyzed, i: usize, name: &str) -> bool {
    analyzed.idents[i].get(name).is_some_and(|lines| {
        lines.iter().any(|&line| {
            line_relevant(analyzed, i, line) && !line_in_spans(line, &analyzed.import_spans[i])
        })
    })
}

/// Symbol definitions with zero mentions anywhere in the scanned corpus, paired
/// with their name for probe grouping, plus counts of why the rest were
/// rejected.
fn symbol_candidates(analyzed: &Analyzed) -> (Vec<(String, UnusedCandidate)>, SymbolRejections) {
    let mut pending = Vec::new();
    let mut rejections = SymbolRejections::default();

    for (i, file_defs) in analyzed.defs.iter().enumerate() {
        for def in file_defs {
            if !symbol_eligible(def) {
                rejections.ineligible += 1;
                continue;
            }
            let name = def.name();
            if !line_relevant(analyzed, i, def.start_line()) {
                rejections.ineligible += 1;
                continue;
            }

            // A mention inside any same-name definition (its own body, its impl
            // block) is not usage.
            let own_spans = analyzed.def_spans_by_name[i]
                .get(name)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let used_here = analyzed.idents[i].get(name).is_some_and(|lines| {
                lines.iter().any(|&line| {
                    line_relevant(analyzed, i, line) && !line_in_spans(line, own_spans)
                })
            });
            if used_here {
                rejections.used_here += 1;
                continue;
            }

            // In other files, mentions on import lines and inside same-name
            // definition spans do not count either: importing a symbol is not
            // using it, and another definition of the same name is not a
            // reference to this one.
            let used_elsewhere = (0..analyzed.files.len()).any(|j| {
                if j == i {
                    return false;
                }
                let Some(lines) = analyzed.idents[j].get(name) else {
                    return false;
                };
                let spans = analyzed.def_spans_by_name[j]
                    .get(name)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                lines.iter().any(|&l| {
                    line_relevant(analyzed, j, l)
                        && !line_in_spans(l, &analyzed.import_spans[j])
                        && !line_in_spans(l, spans)
                })
            });
            if used_elsewhere {
                rejections.used_elsewhere += 1;
                continue;
            }

            let (confidence, reason) = if def.visibility == Visibility::Public {
                (
                    "low",
                    format!(
                        "no references to '{}' found; symbol is public and may be used externally",
                        name
                    ),
                )
            } else {
                (
                    "medium",
                    format!("no references to '{}' found in the scanned corpus", name),
                )
            };
            let probe = format!(
                "whole-word search for '{}' across {} scanned files, excluding import lines and same-name definition spans",
                name,
                analyzed.files.len()
            );
            pending.push((
                name.to_string(),
                make_candidate(
                    def,
                    &analyzed.files[i].relative_path,
                    confidence,
                    reason,
                    probe,
                ),
            ));
        }
    }
    (pending, rejections)
}

/// Kinds and names worth checking for dead code.
fn symbol_eligible(def: &Definition) -> bool {
    let kind_ok = matches!(
        def.kind(),
        SymbolKind::Function
            | SymbolKind::Method
            | SymbolKind::Class
            | SymbolKind::Struct
            | SymbolKind::Interface
            | SymbolKind::Trait
            | SymbolKind::Enum
            | SymbolKind::TypeAlias
            | SymbolKind::Constant
            | SymbolKind::Variable
            | SymbolKind::Module
    );
    let name = def.name();
    kind_ok
        && name.len() >= 3
        && !IMPLICITLY_INVOKED.contains(&name)
        && !name.starts_with("test_")
        && !name.starts_with("Test")
}

fn make_candidate(
    def: &Definition,
    file_path: &str,
    confidence: &str,
    reason: String,
    probe: String,
) -> UnusedCandidate {
    UnusedCandidate {
        file_path: file_path.to_string(),
        name: def.name().to_string(),
        kind: def.kind(),
        line: def.start_line(),
        confidence: confidence.to_string(),
        reason,
        signature: def.signature.clone(),
        probe,
        status: UnusedStatus::Inconclusive,
        analyzed_configurations: Vec::new(),
        analysis_completeness: AnalysisCompleteness::Partial,
        configuration_states: Vec::new(),
        unresolved_dependency_kinds: Vec::new(),
        limitations: Vec::new(),
        safe_for_destructive_edit: false,
    }
}

fn line_relevant(analyzed: &Analyzed, file_index: usize, line: usize) -> bool {
    let states = analyzed.configuration_states[file_index]
        .get(line)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    states.is_empty()
        || states
            .iter()
            .any(|state| state.state != PreprocessorState::Inactive)
}

struct AnalysisAssessment {
    completeness: AnalysisCompleteness,
    unresolved_dependency_kinds: Vec<String>,
    limitations: Vec<String>,
}

fn finalize_candidates(
    candidates: &mut [UnusedCandidate],
    analyzed: &Analyzed,
    assessment: &AnalysisAssessment,
) {
    for candidate in candidates {
        let states = analyzed
            .files
            .iter()
            .position(|file| file.relative_path == candidate.file_path)
            .and_then(|file_index| {
                analyzed.configuration_states[file_index]
                    .get(candidate.line)
                    .cloned()
            })
            .unwrap_or_default();
        candidate.configuration_states = states.clone();
        candidate.analyzed_configurations = states
            .iter()
            .filter(|state| state.state != PreprocessorState::Inactive)
            .map(|state| state.config_id.clone())
            .collect();
        candidate.analysis_completeness = assessment.completeness;
        candidate.unresolved_dependency_kinds = assessment.unresolved_dependency_kinds.clone();
        candidate.limitations = assessment.limitations.clone();
        candidate.status = if assessment.completeness == AnalysisCompleteness::Complete
            && !states
                .iter()
                .any(|state| state.state == PreprocessorState::UnknownDueToBuildConfig)
        {
            UnusedStatus::UnusedInAnalyzedConfigurations
        } else {
            UnusedStatus::Inconclusive
        };
        candidate.safe_for_destructive_edit = false;
    }
}

fn assess_analysis(
    analyzed: &Analyzed,
    catalog: &BuildConfigCatalog,
    configured_wiring_patterns: &[String],
    generated_path_patterns: &[String],
    unverifiable_imports: usize,
    probes_exhausted: bool,
    rejections: &SymbolRejections,
) -> AnalysisAssessment {
    let mut unresolved = Vec::<String>::new();
    let mut limitations = Vec::<String>::new();
    if catalog.configurations.is_empty() {
        unresolved.push("build_configuration".to_string());
        limitations.push("no build configuration was analyzed".to_string());
    }

    let is_compiled_language = |file: &FileInfo| {
        file.extension.as_deref().is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "c" | "cc" | "cpp" | "cxx" | "h" | "hh" | "hpp" | "hxx" | "m" | "mm"
            )
        })
    };
    let missing_file_configs = analyzed
        .files
        .iter()
        .filter(|file| {
            is_compiled_language(file)
                && catalog
                    .configurations_for_file(&file.relative_path)
                    .is_empty()
        })
        .count();
    if missing_file_configs > 0 {
        unresolved.push("conditional_compilation".to_string());
        limitations.push(format!(
            "{} compiled-language files have no matching build configuration",
            missing_file_configs
        ));
    }
    if analyzed.skipped_definitions > 0 {
        unresolved.push("parser_omissions".to_string());
        limitations.push(format!(
            "{} definition nodes could not be named",
            analyzed.skipped_definitions
        ));
    }
    if unverifiable_imports > 0 {
        unresolved.push("external_or_generated_include".to_string());
        limitations.push(format!(
            "{} imports/includes could not be verified",
            unverifiable_imports
        ));
    }
    if probes_exhausted || rejections.probe_errors > 0 {
        unresolved.push("cross_index_probe".to_string());
        limitations.push("cross-index usage verification was incomplete".to_string());
    }

    let mut wiring_patterns = vec![
        "register(".to_string(),
        "register_".to_string(),
        "reflection".to_string(),
        "Q_OBJECT".to_string(),
        "plugin".to_string(),
        "dynamic_cast".to_string(),
    ];
    wiring_patterns.extend(configured_wiring_patterns.iter().cloned());
    if analyzed.files.iter().any(|file| {
        wiring_patterns
            .iter()
            .any(|pattern| !pattern.is_empty() && file.content.contains(pattern))
    }) {
        unresolved.push("dynamic_or_registration_wiring".to_string());
        limitations.push(
            "dynamic registration/reflection wiring may create non-textual dependencies"
                .to_string(),
        );
    }
    if analyzed.files.iter().any(|file| {
        let path = file.relative_path.replace('\\', "/").to_ascii_lowercase();
        generated_path_patterns
            .iter()
            .any(|pattern| path.contains(&pattern.to_ascii_lowercase().replace('\\', "/")))
    }) || catalog.configurations.iter().any(|configuration| {
        !configuration.generated_header_paths.is_empty()
            || !configuration.forced_includes.is_empty()
    }) {
        unresolved.push("generated_or_forced_wiring".to_string());
        limitations.push(
            "generated headers or forced includes require build/generator validation".to_string(),
        );
    }

    unresolved.sort();
    unresolved.dedup();
    limitations.extend(
        catalog
            .diagnostics
            .iter()
            .filter(|diagnostic| {
                diagnostic.contains("Invalid") || diagnostic.contains("unavailable")
            })
            .cloned(),
    );
    limitations.sort();
    limitations.dedup();
    AnalysisAssessment {
        completeness: if unresolved.is_empty() {
            AnalysisCompleteness::Complete
        } else {
            AnalysisCompleteness::Partial
        },
        unresolved_dependency_kinds: unresolved,
        limitations,
    }
}

/// identifier -> 1-based lines it appears on.
fn identifier_lines(content: &str, re: &Regex) -> HashMap<String, Vec<usize>> {
    let mut map: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, line) in content.lines().enumerate() {
        for m in re.find_iter(line) {
            map.entry(m.as_str().to_string()).or_default().push(idx + 1);
        }
    }
    map
}

fn line_in_spans(line: usize, spans: &[(usize, usize)]) -> bool {
    spans.iter().any(|&(s, e)| line >= s && line <= e)
}

/// Word-boundary check: `name` appears in `text` as a whole identifier.
fn mentions_identifier(text: &str, name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let bytes = text.as_bytes();
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    for (pos, _) in text.match_indices(name) {
        let before_ok = pos == 0 || !is_ident(bytes[pos - 1]);
        let after = pos + name.len();
        let after_ok = after >= bytes.len() || !is_ident(bytes[after]);
        if before_ok && after_ok {
            return true;
        }
    }
    false
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FindUnusedRequest;
    use tempfile::TempDir;

    #[test]
    fn test_mentions_identifier_boundaries() {
        assert!(mentions_identifier("let x = foo();", "foo"));
        assert!(mentions_identifier("foo", "foo"));
        assert!(!mentions_identifier("food()", "foo"));
        assert!(!mentions_identifier("my_foo", "foo"));
        assert!(!mentions_identifier("foo1", "foo"));
        assert!(mentions_identifier("a.foo.b", "foo"));
        assert!(!mentions_identifier("", "foo"));
    }

    #[test]
    fn test_line_in_spans() {
        let spans = [(1, 1), (5, 8)];
        assert!(line_in_spans(1, &spans));
        assert!(line_in_spans(6, &spans));
        assert!(!line_in_spans(2, &spans));
        assert!(!line_in_spans(9, &spans));
    }

    #[test]
    fn test_identifier_lines() {
        let re = Regex::new(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b").unwrap();
        let map = identifier_lines("fn foo() {\n    bar();\n    foo();\n}\n", &re);
        assert_eq!(map.get("foo"), Some(&vec![1, 3]));
        assert_eq!(map.get("bar"), Some(&vec![2]));
        assert!(!map.contains_key("baz"));
    }

    async fn create_test_client() -> (RagClient, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("db").to_string_lossy().to_string();
        let cache_path = temp_dir.path().join("cache.json");
        let client = RagClient::new_with_db_path(&db_path, cache_path)
            .await
            .unwrap();
        (client, temp_dir)
    }

    fn make_request(path: &str, check: &str) -> FindUnusedRequest {
        FindUnusedRequest {
            path: path.to_string(),
            project: None,
            check: check.to_string(),
            limit: 100,
            max_file_size: 1_048_576,
            configurations: Vec::new(),
        }
    }

    #[tokio::test]
    async fn test_symbols_check_requires_indexed_root() {
        let (client, temp_dir) = create_test_client().await;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        std::fs::write(data_dir.join("a.rs"), "fn lonely_helper() {}\n").unwrap();

        let result = client
            .find_unused(make_request(&data_dir.to_string_lossy(), "symbols"))
            .await;
        assert!(result.is_err());
        assert!(format!("{:#}", result.unwrap_err()).contains("index_codebase"));
    }

    #[tokio::test]
    async fn test_imports_check_works_without_index() {
        let (client, temp_dir) = create_test_client().await;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        std::fs::write(
            data_dir.join("a.py"),
            "import os\nimport json\n\nprint(json.dumps({}))\n",
        )
        .unwrap();

        let response = client
            .find_unused(make_request(&data_dir.to_string_lossy(), "imports"))
            .await
            .unwrap();

        let names: Vec<&str> = response
            .candidates
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        assert!(
            names.contains(&"os"),
            "unused 'import os' should be flagged"
        );
        assert!(!names.contains(&"json"), "'json' is used");
        assert_eq!(response.candidates[0].confidence, "high");
    }

    #[tokio::test]
    async fn test_find_unused_end_to_end() {
        let (client, temp_dir) = create_test_client().await;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        std::fs::write(
            data_dir.join("lib.rs"),
            "use std::collections::HashMap;\n\
             pub fn used_helper() -> u32 { 41 }\n\
             fn orphan_helper() -> u32 { 42 }\n",
        )
        .unwrap();
        std::fs::write(
            data_dir.join("main.rs"),
            "fn main() { let _ = crate::used_helper(); }\n",
        )
        .unwrap();

        let index_req = crate::types::IndexRequest {
            path: data_dir.to_string_lossy().to_string(),
            project: None,
            include_patterns: vec![],
            exclude_patterns: vec![],
            max_file_size: 1_048_576,
        };
        client.index_codebase(index_req).await.unwrap();

        let response = client
            .find_unused(make_request(&data_dir.to_string_lossy(), "all"))
            .await
            .unwrap();

        assert_eq!(response.files_scanned, 2);
        let names: Vec<&str> = response
            .candidates
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        assert!(
            names.contains(&"HashMap"),
            "unused 'use HashMap' should be flagged, got: {:?}",
            names
        );
        assert!(
            names.contains(&"orphan_helper"),
            "unreferenced fn should be flagged, got: {:?}",
            names
        );
        assert!(!names.contains(&"used_helper"), "used_helper is referenced");
        assert!(!names.contains(&"main"), "entry points are never flagged");

        let orphan = response
            .candidates
            .iter()
            .find(|c| c.name == "orphan_helper")
            .unwrap();
        assert_eq!(orphan.confidence, "medium");
        let import = response
            .candidates
            .iter()
            .find(|c| c.name == "HashMap")
            .unwrap();
        assert_eq!(import.kind, SymbolKind::Import);
    }

    fn write(dir: &Path, name: &str, content: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, content).unwrap();
        p
    }

    #[tokio::test]
    async fn test_cpp_used_class_header_not_flagged() {
        let (client, temp_dir) = create_test_client().await;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        write(
            &data_dir,
            "Kiosk.Notify.h",
            "class KioskNotify {\npublic:\n  void fire();\n};\n",
        );
        write(
            &data_dir,
            "main.cpp",
            "#include \"Kiosk.Notify.h\"\nint main() { KioskNotify n; n.fire(); return 0; }\n",
        );

        let response = client
            .find_unused(make_request(&data_dir.to_string_lossy(), "imports"))
            .await
            .unwrap();

        assert!(
            response.candidates.is_empty(),
            "a used header must not be flagged, got: {:?}",
            response.candidates
        );
        assert_eq!(response.unverifiable_imports, 0);
        assert_eq!(response.symbol_rejections.used_elsewhere, 0);
    }

    #[tokio::test]
    async fn test_cpp_unresolvable_include_unverifiable() {
        let (client, temp_dir) = create_test_client().await;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        write(
            &data_dir,
            "main.cpp",
            "#include \"no_such.h\"\nint main() { return 0; }\n",
        );

        let response = client
            .find_unused(make_request(&data_dir.to_string_lossy(), "imports"))
            .await
            .unwrap();

        assert!(
            response.candidates.is_empty(),
            "an unresolvable include is skipped, never flagged, got: {:?}",
            response.candidates
        );
        assert_eq!(response.unverifiable_imports, 1);
        assert_eq!(response.unverifiable_import_details.len(), 1);
        assert_eq!(response.unverifiable_import_details[0].name, "no_such.h");
        assert!(
            response.unverifiable_import_details[0]
                .reason
                .contains("not resolved")
        );
    }

    #[tokio::test]
    async fn test_cpp_system_include_unverifiable() {
        let (client, temp_dir) = create_test_client().await;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        write(
            &data_dir,
            "main.cpp",
            "#include <memory>\nint main() { std::unique_ptr<int> p; return 0; }\n",
        );

        let response = client
            .find_unused(make_request(&data_dir.to_string_lossy(), "imports"))
            .await
            .unwrap();

        assert!(
            response.candidates.is_empty(),
            "got: {:?}",
            response.candidates
        );
        assert_eq!(response.unverifiable_imports, 1);
        assert!(
            response.unverifiable_import_details[0]
                .reason
                .contains("system include")
        );
    }

    #[tokio::test]
    async fn test_cpp_resolved_header_unused_flagged_medium() {
        let (client, temp_dir) = create_test_client().await;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        write(&data_dir, "unused.h", "class UnusedThing {};\n");
        write(
            &data_dir,
            "main.cpp",
            "#include \"unused.h\"\nint main() { return 0; }\n",
        );

        let response = client
            .find_unused(make_request(&data_dir.to_string_lossy(), "imports"))
            .await
            .unwrap();

        assert_eq!(
            response.candidates.len(),
            1,
            "got: {:?}",
            response.candidates
        );
        let candidate = &response.candidates[0];
        assert_eq!(candidate.name, "unused.h");
        assert_eq!(candidate.confidence, "medium");
        assert!(
            candidate.probe.contains("UnusedThing"),
            "the probe must disclose the symbols searched, got: {}",
            candidate.probe
        );
    }

    #[tokio::test]
    async fn test_cpp_single_file_scan_resolves_sibling_header() {
        let (client, temp_dir) = create_test_client().await;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        write(
            &data_dir,
            "Kiosk.Notify.h",
            "class KioskNotify {\npublic:\n  void fire();\n};\n",
        );
        let main_cpp = write(
            &data_dir,
            "main.cpp",
            "#include \"Kiosk.Notify.h\"\nint main() { KioskNotify n; n.fire(); return 0; }\n",
        );

        // Single-file scan: the header is outside the scan set and must be
        // resolved on disk next to the including file.
        let response = client
            .find_unused(make_request(&main_cpp.to_string_lossy(), "imports"))
            .await
            .unwrap();

        assert_eq!(response.files_scanned, 1);
        assert!(
            response.candidates.is_empty(),
            "got: {:?}",
            response.candidates
        );
        assert_eq!(response.unverifiable_imports, 0);
    }

    #[test]
    fn test_symbol_flagged_when_other_file_only_imports_it() {
        let make = |name: &str, content: &str| FileInfo {
            path: PathBuf::from(name),
            relative_path: name.to_string(),
            root_path: "/test".to_string(),
            project: None,
            extension: Some("rs".to_string()),
            language: None,
            content: content.to_string(),
            hash: "test_hash".to_string(),
        };
        let files = vec![
            make("a.rs", "fn ghost_fn() -> u32 { 7 }\n"),
            make("b.rs", "use crate::ghost_fn;\n"),
        ];
        let provider = Arc::new(crate::relations::HybridRelationsProvider::new(false).unwrap());
        let analyzed = analyze(files, provider, &BuildConfigCatalog::default());

        let (pending, rejections) = symbol_candidates(&analyzed);
        assert!(
            pending.iter().any(|(name, _)| name == "ghost_fn"),
            "an import elsewhere is not usage; got: {:?}",
            pending.iter().map(|(n, _)| n).collect::<Vec<_>>()
        );
        assert_eq!(rejections.used_elsewhere, 0);
    }

    fn analyzed_file(name: &str, extension: &str, content: &str) -> FileInfo {
        FileInfo {
            path: PathBuf::from(name),
            relative_path: name.to_string(),
            root_path: "/test".to_string(),
            project: Some("test".to_string()),
            extension: Some(extension.to_string()),
            language: None,
            content: content.to_string(),
            hash: "test_hash".to_string(),
        }
    }

    fn explicit_configuration(
        id: &str,
        definitions: &[&str],
    ) -> crate::build_config::BuildConfiguration {
        crate::build_config::BuildConfiguration {
            config_id: id.to_string(),
            source: crate::build_config::BuildConfigSource::Explicit,
            source_files: Vec::new(),
            include_paths: Vec::new(),
            preprocessor_definitions: definitions.iter().map(|value| value.to_string()).collect(),
            language_standard: Some("c++20".to_string()),
            forced_includes: Vec::new(),
            generated_header_paths: Vec::new(),
        }
    }

    #[test]
    fn missing_build_configuration_makes_unused_finding_inconclusive() {
        let provider = Arc::new(crate::relations::HybridRelationsProvider::new(false).unwrap());
        let catalog = BuildConfigCatalog::default();
        let analyzed = analyze(
            vec![analyzed_file(
                "src/a.cpp",
                "cpp",
                "static void orphaned() {}\n",
            )],
            provider,
            &catalog,
        );
        let (pending, rejections) = symbol_candidates(&analyzed);
        let mut candidates = pending
            .into_iter()
            .map(|(_, candidate)| candidate)
            .collect::<Vec<_>>();
        let assessment = assess_analysis(
            &analyzed,
            &catalog,
            &[],
            &["generated/".to_string()],
            0,
            false,
            &rejections,
        );
        finalize_candidates(&mut candidates, &analyzed, &assessment);
        let orphan = candidates
            .iter()
            .find(|candidate| candidate.name == "orphaned")
            .unwrap();
        assert_eq!(orphan.status, UnusedStatus::Inconclusive);
        assert!(
            orphan
                .unresolved_dependency_kinds
                .contains(&"build_configuration".to_string())
        );
        assert!(!orphan.safe_for_destructive_edit);
    }

    #[test]
    fn reference_active_only_in_another_indexed_configuration_counts_as_referenced() {
        let catalog = BuildConfigCatalog {
            configurations: vec![
                explicit_configuration("debug", &[]),
                explicit_configuration("release", &["RELEASE=1"]),
            ],
            diagnostics: Vec::new(),
        };
        let provider = Arc::new(crate::relations::HybridRelationsProvider::new(false).unwrap());
        let analyzed = analyze(
            vec![analyzed_file(
                "src/a.cpp",
                "cpp",
                "static void target_symbol() {}\n#ifdef RELEASE\nvoid use_target() { target_symbol(); }\n#endif\n",
            )],
            provider,
            &catalog,
        );
        let (pending, rejections) = symbol_candidates(&analyzed);
        assert!(
            !pending.iter().any(|(name, _)| name == "target_symbol"),
            "a reference active in release configuration must keep the symbol live"
        );
        assert!(rejections.used_here > 0);
    }

    #[test]
    fn dynamic_and_generated_wiring_prevents_destructive_safety() {
        let catalog = BuildConfigCatalog {
            configurations: vec![explicit_configuration("debug", &[])],
            diagnostics: Vec::new(),
        };
        let provider = Arc::new(crate::relations::HybridRelationsProvider::new(false).unwrap());
        let analyzed = analyze(
            vec![analyzed_file(
                "generated/a.cpp",
                "cpp",
                "static void orphaned() {}\nvoid wiring() { register(plugin); }\n",
            )],
            provider,
            &catalog,
        );
        let (pending, rejections) = symbol_candidates(&analyzed);
        let mut candidates = pending
            .into_iter()
            .map(|(_, candidate)| candidate)
            .collect::<Vec<_>>();
        let assessment = assess_analysis(
            &analyzed,
            &catalog,
            &[],
            &["generated/".to_string()],
            0,
            false,
            &rejections,
        );
        finalize_candidates(&mut candidates, &analyzed, &assessment);
        let orphan = candidates
            .iter()
            .find(|candidate| candidate.name == "orphaned")
            .unwrap();
        assert_eq!(orphan.status, UnusedStatus::Inconclusive);
        assert!(
            orphan
                .unresolved_dependency_kinds
                .contains(&"dynamic_or_registration_wiring".to_string())
        );
        assert!(
            orphan
                .unresolved_dependency_kinds
                .contains(&"generated_or_forced_wiring".to_string())
        );
        assert!(!orphan.safe_for_destructive_edit);
    }

    #[test]
    #[ignore = "manual synthetic M4 multi-configuration unused-analysis measurement"]
    fn benchmark_m4_multi_configuration_unused_analysis() {
        let files = (0..200)
            .map(|index| {
                analyzed_file(
                    &format!("src/file_{index}.cpp"),
                    "cpp",
                    &format!(
                        "static void orphan_{index}() {{}}\n#ifdef FEATURE\nstatic void live_{index}() {{}}\nvoid caller_{index}() {{ live_{index}(); }}\n#endif\n"
                    ),
                )
            })
            .collect::<Vec<_>>();
        let catalog = BuildConfigCatalog {
            configurations: vec![
                explicit_configuration("debug", &[]),
                explicit_configuration("feature", &["FEATURE=1"]),
                explicit_configuration("asan", &["ASAN=1"]),
                explicit_configuration("release", &["NDEBUG=1"]),
            ],
            diagnostics: Vec::new(),
        };
        let provider = Arc::new(crate::relations::HybridRelationsProvider::new(false).unwrap());

        let started = std::time::Instant::now();
        let analyzed = analyze(files, provider, &catalog);
        let (pending, rejections) = symbol_candidates(&analyzed);
        let mut candidates = pending
            .into_iter()
            .map(|(_, candidate)| candidate)
            .collect::<Vec<_>>();
        let assessment = assess_analysis(
            &analyzed,
            &catalog,
            &[],
            &["generated/".to_string()],
            0,
            false,
            &rejections,
        );
        finalize_candidates(&mut candidates, &analyzed, &assessment);
        let elapsed = started.elapsed();

        println!(
            "m4 unused synthetic: files={} configurations={} candidates={} elapsed_ms={}",
            analyzed.files.len(),
            catalog.configurations.len(),
            candidates.len(),
            elapsed.as_millis()
        );
        assert_eq!(analyzed.files.len(), 200);
        assert!(candidates.iter().all(|candidate| {
            candidate.status == UnusedStatus::UnusedInAnalyzedConfigurations
                && !candidate.safe_for_destructive_edit
        }));
    }
}
