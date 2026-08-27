//! Deterministic, bounded dependency-graph traversal.
//!
//! Traversal is breadth-first over the persisted reference table's incoming and
//! outgoing adjacency indexes. Only stable logical symbol IDs are visited; graph
//! observations that lack an authoritative target can be returned, but candidates
//! are never traversed as edges.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use anyhow::Result;

use crate::build_config::configuration_scope_matches;

use super::storage::RelationsStore;
use super::{
    CallGraphNode, Definition, GraphEdge, LocationRole, Reference, ReferenceKind, ResolutionStatus,
};
use crate::types::{GraphContinuation, GraphTotals};

#[derive(Debug, Clone)]
pub struct GraphTraversalOptions {
    pub depth: usize,
    pub include_incoming: bool,
    pub include_outgoing: bool,
    pub max_nodes: usize,
    pub max_edges: usize,
    pub edge_kinds: Vec<ReferenceKind>,
    pub resolution_statuses: Vec<ResolutionStatus>,
    pub language_filters: Vec<String>,
    pub path_filters: Vec<String>,
    pub configurations: Vec<String>,
}

#[derive(Debug)]
pub struct GraphTraversalResult {
    pub nodes: Vec<CallGraphNode>,
    pub edges: Vec<GraphEdge>,
    pub graph_truncated: bool,
    pub estimated_or_known_total: GraphTotals,
    pub continuation: Option<GraphContinuation>,
}

fn reference_matches(reference: &Reference, options: &GraphTraversalOptions) -> bool {
    options.edge_kinds.contains(&reference.reference_kind)
        && options
            .resolution_statuses
            .contains(&reference.resolution_status)
        && (options.language_filters.is_empty()
            || options
                .language_filters
                .iter()
                .any(|language| language.eq_ignore_ascii_case(&reference.language)))
        && (options.path_filters.is_empty()
            || options
                .path_filters
                .iter()
                .any(|filter| reference.file_path.contains(&filter.replace('\\', "/"))))
        && configuration_scope_matches(&reference.configuration_states, &options.configurations)
}

fn preferred_definitions(definitions: Vec<Definition>) -> HashMap<String, Definition> {
    let mut preferred = HashMap::<String, Definition>::new();
    for definition in definitions {
        let id = definition.to_storage_id();
        match preferred.get(&id) {
            Some(existing)
                if existing.location.role == LocationRole::Definition
                    || definition.location.role != LocationRole::Definition => {}
            _ => {
                preferred.insert(id, definition);
            }
        }
    }
    preferred
}

/// Traverse references up to an exact edge distance from `root_definition`.
pub async fn traverse_dependency_graph(
    store: &dyn RelationsStore,
    root_definition: &Definition,
    root_path: &str,
    options: &GraphTraversalOptions,
) -> Result<GraphTraversalResult> {
    let root_id = root_definition.to_storage_id();
    let mut nodes = BTreeMap::<String, CallGraphNode>::new();
    nodes.insert(
        root_id.clone(),
        CallGraphNode::from_definition(root_definition, 0),
    );

    let mut edges = Vec::<GraphEdge>::new();
    let mut returned_edge_ids = HashSet::<String>::new();
    let mut known_edge_ids = HashSet::<String>::new();
    let mut known_node_ids = HashSet::<String>::from([root_id.clone()]);
    let mut expanded = HashSet::<String>::new();
    let mut frontier = vec![root_id];
    let mut pending = BTreeSet::<String>::new();
    let mut truncated_reason = None::<String>;
    let mut continuation_depth = 0usize;

    for level in 0..options.depth {
        if frontier.is_empty() {
            break;
        }
        continuation_depth = level;
        for id in &frontier {
            expanded.insert(id.clone());
        }
        let frontier_set = frontier.iter().cloned().collect::<HashSet<_>>();

        let mut references = Vec::new();
        if options.include_outgoing {
            references.extend(
                store
                    .get_outgoing_references_in_root(&frontier, root_path)
                    .await?,
            );
        }
        if options.include_incoming {
            references.extend(
                store
                    .get_incoming_references_in_root(&frontier, root_path)
                    .await?,
            );
        }
        references.sort_by(|left, right| {
            (
                &left.file_path,
                left.start_line,
                left.start_col,
                &left.location_id,
            )
                .cmp(&(
                    &right.file_path,
                    right.start_line,
                    right.start_col,
                    &right.location_id,
                ))
        });
        references.dedup_by(|left, right| left.location_id == right.location_id);

        let mut level_rows = Vec::<(Reference, Vec<String>)>::new();
        let mut definition_ids = BTreeSet::<String>::new();
        for reference in references {
            if !reference_matches(&reference, options) {
                continue;
            }
            let Some(source_id) = reference.source_symbol_id.as_ref() else {
                continue;
            };
            let resolved_target = (reference.resolution_status == ResolutionStatus::Resolved
                && !reference.target_symbol_id.is_empty())
            .then_some(reference.target_symbol_id.as_str());
            let traversed_outgoing = options.include_outgoing && frontier_set.contains(source_id);
            let traversed_incoming = options.include_incoming
                && resolved_target.is_some_and(|target| frontier_set.contains(target));
            if !traversed_outgoing && !traversed_incoming {
                continue;
            }

            known_edge_ids.insert(reference.location_id.clone());
            let mut neighbors = Vec::new();
            if traversed_outgoing && let Some(target) = resolved_target {
                neighbors.push(target.to_string());
            }
            if traversed_incoming {
                neighbors.push(source_id.clone());
            }
            neighbors.sort();
            neighbors.dedup();
            for neighbor in &neighbors {
                known_node_ids.insert(neighbor.clone());
                if !nodes.contains_key(neighbor) {
                    definition_ids.insert(neighbor.clone());
                }
            }
            level_rows.push((reference, neighbors));
        }

        let definition_ids = definition_ids.into_iter().collect::<Vec<_>>();
        let definitions = preferred_definitions(
            store
                .find_definitions_by_symbol_ids_in_root(&definition_ids, root_path)
                .await?,
        );
        let mut next_frontier = BTreeSet::<String>::new();

        for (reference, neighbors) in level_rows {
            if returned_edge_ids.contains(&reference.location_id) {
                continue;
            }
            if edges.len() >= options.max_edges {
                pending.extend(neighbors);
                truncated_reason.get_or_insert_with(|| "max_edges reached".to_string());
                continue;
            }
            let mut endpoints_available = true;
            for neighbor in &neighbors {
                if nodes.contains_key(neighbor) {
                    continue;
                }
                if nodes.len() >= options.max_nodes {
                    endpoints_available = false;
                    pending.insert(neighbor.clone());
                    truncated_reason.get_or_insert_with(|| "max_nodes reached".to_string());
                    continue;
                }
                let node = definitions
                    .get(neighbor)
                    .map(|definition| CallGraphNode::from_definition(definition, level + 1))
                    .unwrap_or_else(|| {
                        CallGraphNode::without_definition(neighbor.clone(), level + 1)
                    });
                nodes.insert(neighbor.clone(), node);
                if level + 1 < options.depth && !expanded.contains(neighbor) {
                    next_frontier.insert(neighbor.clone());
                }
            }

            if !endpoints_available {
                continue;
            }
            if let Some(edge) = GraphEdge::from_reference(&reference) {
                returned_edge_ids.insert(reference.location_id.clone());
                edges.push(edge);
            }
        }

        if truncated_reason.is_some() {
            pending.extend(next_frontier);
            pending.extend(frontier);
            break;
        }
        frontier = next_frontier.into_iter().collect();
    }

    let graph_truncated = truncated_reason.is_some();
    let continuation = truncated_reason.map(|reason| GraphContinuation {
        pending_symbol_ids: pending.into_iter().collect(),
        next_depth: continuation_depth,
        remaining_depth: options.depth.saturating_sub(continuation_depth),
        reason,
    });

    let mut nodes = nodes.into_values().collect::<Vec<_>>();
    nodes.sort_by(|left, right| {
        (left.distance, &left.symbol_id).cmp(&(right.distance, &right.symbol_id))
    });
    edges.sort_by(|left, right| {
        (
            &left.path,
            left.start_line,
            left.start_column,
            &left.edge_id,
        )
            .cmp(&(
                &right.path,
                right.start_line,
                right.start_column,
                &right.edge_id,
            ))
    });

    Ok(GraphTraversalResult {
        nodes,
        edges,
        graph_truncated,
        estimated_or_known_total: GraphTotals {
            nodes: known_node_ids.len(),
            edges: known_edge_ids.len(),
            exact: !graph_truncated,
        },
        continuation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relations::storage::LanceRelationsStore;
    use crate::relations::{
        DispatchKind, EvidenceKind, ReferenceCandidate, SourceLocation, SymbolId, SymbolKind,
        Visibility,
    };
    use tempfile::TempDir;

    fn definition(name: &str, line: usize) -> Definition {
        let file = format!("src/{}.rs", name.to_lowercase());
        Definition {
            symbol_id: SymbolId::new(&file, name, SymbolKind::Function, line, 0),
            location: SourceLocation {
                project_id: "graph-project".to_string(),
                file_path: file,
                start_line: line,
                start_col: 0,
                end_line: line,
                end_col: name.len(),
                role: LocationRole::Definition,
            },
            root_path: Some("/graph".to_string()),
            project: Some("graph-project".to_string()),
            end_line: line + 3,
            end_col: 1,
            signature: format!("fn {name}()"),
            doc_comment: None,
            visibility: Visibility::Public,
            parent_id: None,
            parser: "tree-sitter/rust".to_string(),
            indexed_at: 1,
        }
    }

    fn call(
        source: &Definition,
        target: Option<&Definition>,
        line: usize,
        status: ResolutionStatus,
    ) -> Reference {
        let location = SourceLocation {
            project_id: "graph-project".to_string(),
            file_path: source.file_path().to_string(),
            start_line: line,
            start_col: 4,
            end_line: line,
            end_col: 9,
            role: LocationRole::Reference,
        };
        let target_id = target.map(Definition::to_storage_id).unwrap_or_default();
        Reference {
            file_path: source.file_path().to_string(),
            root_path: Some("/graph".to_string()),
            project: Some("graph-project".to_string()),
            start_line: line,
            end_line: line,
            start_col: 4,
            end_col: 9,
            location_id: location.to_storage_id(),
            source_symbol_id: Some(source.to_storage_id()),
            target_symbol_id: if status == ResolutionStatus::Resolved {
                target_id.clone()
            } else {
                String::new()
            },
            target_name: target
                .map(Definition::name)
                .unwrap_or("unknown")
                .to_string(),
            candidates: target
                .map(|definition| {
                    vec![ReferenceCandidate {
                        symbol_id: definition.to_storage_id(),
                        reason: "name and scope candidate".to_string(),
                    }]
                })
                .unwrap_or_default(),
            reference_kind: ReferenceKind::Call,
            resolution_status: status,
            evidence_kind: if status == ResolutionStatus::Resolved {
                EvidenceKind::Syntactic
            } else {
                EvidenceKind::Heuristic
            },
            dispatch_kind: if status == ResolutionStatus::Resolved {
                DispatchKind::Direct
            } else {
                DispatchKind::Unknown
            },
            configuration_states: Vec::new(),
            language: "Rust".to_string(),
            parser: "tree-sitter/rust".to_string(),
            indexed_at: 1,
        }
    }

    fn options(depth: usize) -> GraphTraversalOptions {
        GraphTraversalOptions {
            depth,
            include_incoming: false,
            include_outgoing: true,
            max_nodes: 100,
            max_edges: 100,
            edge_kinds: vec![ReferenceKind::Call],
            resolution_statuses: vec![ResolutionStatus::Resolved],
            language_filters: Vec::new(),
            path_filters: Vec::new(),
            configurations: Vec::new(),
        }
    }

    async fn stored_graph(
        definitions: &[Definition],
        references: Vec<Reference>,
    ) -> (TempDir, LanceRelationsStore) {
        let directory = TempDir::new().unwrap();
        let store = LanceRelationsStore::new(directory.path().to_path_buf())
            .await
            .unwrap();
        store
            .store_definitions(definitions.to_vec(), "/graph")
            .await
            .unwrap();
        store.store_references(references, "/graph").await.unwrap();
        (directory, store)
    }

    #[tokio::test]
    async fn depth_zero_one_and_two_have_exact_semantics() {
        let a = definition("A", 1);
        let b = definition("B", 10);
        let c = definition("C", 20);
        let references = vec![
            call(&a, Some(&b), 2, ResolutionStatus::Resolved),
            call(&b, Some(&c), 11, ResolutionStatus::Resolved),
        ];
        let (_directory, store) = stored_graph(&[a.clone(), b, c], references).await;

        let depth0 = traverse_dependency_graph(&store, &a, "/graph", &options(0))
            .await
            .unwrap();
        let depth1 = traverse_dependency_graph(&store, &a, "/graph", &options(1))
            .await
            .unwrap();
        let depth2 = traverse_dependency_graph(&store, &a, "/graph", &options(2))
            .await
            .unwrap();

        assert_eq!((depth0.nodes.len(), depth0.edges.len()), (1, 0));
        assert_eq!((depth1.nodes.len(), depth1.edges.len()), (2, 1));
        assert_eq!((depth2.nodes.len(), depth2.edges.len()), (3, 2));
        assert_eq!(depth2.nodes.iter().map(|node| node.distance).max(), Some(2));
    }

    #[tokio::test]
    async fn cycles_and_diamonds_return_unique_nodes_and_edges() {
        let a = definition("A", 1);
        let b = definition("B", 10);
        let c = definition("C", 20);
        let d = definition("D", 30);
        let references = vec![
            call(&a, Some(&b), 2, ResolutionStatus::Resolved),
            call(&a, Some(&c), 3, ResolutionStatus::Resolved),
            call(&b, Some(&d), 11, ResolutionStatus::Resolved),
            call(&c, Some(&d), 21, ResolutionStatus::Resolved),
            call(&d, Some(&a), 31, ResolutionStatus::Resolved),
        ];
        let (_directory, store) = stored_graph(&[a.clone(), b, c, d], references).await;

        let graph = traverse_dependency_graph(&store, &a, "/graph", &options(10))
            .await
            .unwrap();
        assert_eq!(graph.nodes.len(), 4);
        assert_eq!(graph.edges.len(), 5);
        assert_eq!(
            graph
                .nodes
                .iter()
                .map(|node| &node.symbol_id)
                .collect::<HashSet<_>>()
                .len(),
            4
        );
        assert!(!graph.graph_truncated);
        assert!(graph.estimated_or_known_total.exact);
    }

    #[tokio::test]
    async fn ambiguous_observations_are_visible_but_never_traversed() {
        let a = definition("A", 1);
        let b = definition("B", 10);
        let references = vec![call(&a, Some(&b), 2, ResolutionStatus::Ambiguous)];
        let (_directory, store) = stored_graph(&[a.clone(), b.clone()], references).await;
        let mut opts = options(2);
        opts.resolution_statuses = vec![ResolutionStatus::Ambiguous];

        let graph = traverse_dependency_graph(&store, &a, "/graph", &opts)
            .await
            .unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.edges[0].target_symbol_id, None);
        assert_eq!(graph.edges[0].candidates[0].symbol_id, b.to_storage_id());
        assert_eq!(graph.edges[0].evidence_kind, EvidenceKind::Heuristic);
        assert_eq!(graph.edges[0].path, "src/a.rs");
        assert_eq!(graph.edges[0].parser, "tree-sitter/rust");
    }

    #[tokio::test]
    async fn graph_budgets_report_truncation_and_continuation() {
        let a = definition("A", 1);
        let b = definition("B", 10);
        let c = definition("C", 20);
        let references = vec![
            call(&a, Some(&b), 2, ResolutionStatus::Resolved),
            call(&a, Some(&c), 3, ResolutionStatus::Resolved),
        ];
        let (_directory, store) = stored_graph(&[a.clone(), b, c], references).await;
        let mut opts = options(2);
        opts.max_nodes = 2;
        opts.max_edges = 1;

        let graph = traverse_dependency_graph(&store, &a, "/graph", &opts)
            .await
            .unwrap();
        assert!(graph.graph_truncated);
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
        assert!(graph.estimated_or_known_total.nodes >= 3);
        assert!(graph.estimated_or_known_total.edges >= 2);
        assert!(!graph.estimated_or_known_total.exact);
        assert!(graph.continuation.is_some());

        let mut edge_only_cap = options(2);
        edge_only_cap.max_edges = 1;
        let edge_capped = traverse_dependency_graph(&store, &a, "/graph", &edge_only_cap)
            .await
            .unwrap();
        assert_eq!(edge_capped.nodes.len(), 2);
        assert_eq!(edge_capped.edges.len(), 1);
        assert!(edge_capped.graph_truncated);
    }

    #[tokio::test]
    async fn graph_filters_apply_before_totals_and_traversal() {
        let a = definition("A", 1);
        let b = definition("B", 10);
        let c = definition("C", 20);
        let mut first = call(&a, Some(&b), 2, ResolutionStatus::Resolved);
        first.configuration_states = vec![crate::build_config::ConfigurationState {
            config_id: "debug".to_string(),
            state: crate::build_config::PreprocessorState::Active,
        }];
        let mut second = call(&a, Some(&c), 3, ResolutionStatus::Resolved);
        second.language = "C++".to_string();
        second.file_path = "generated/a.cpp".to_string();
        second.configuration_states = vec![crate::build_config::ConfigurationState {
            config_id: "release".to_string(),
            state: crate::build_config::PreprocessorState::Active,
        }];
        let references = vec![first, second];
        let (_directory, store) = stored_graph(&[a.clone(), b, c], references).await;
        let mut opts = options(1);
        opts.language_filters = vec!["rust".to_string()];
        opts.path_filters = vec!["src/".to_string()];
        opts.configurations = vec!["debug".to_string()];

        let graph = traverse_dependency_graph(&store, &a, "/graph", &opts)
            .await
            .unwrap();
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.estimated_or_known_total.edges, 1);
    }

    #[tokio::test]
    #[ignore = "synthetic M3 cold/warm latency measurement"]
    async fn benchmark_m3_depth_two_cold_and_warm() {
        let definitions = (0..500)
            .map(|index| definition(&format!("Node{index}"), index * 10 + 1))
            .collect::<Vec<_>>();
        let references = (0..499)
            .map(|index| {
                call(
                    &definitions[index],
                    Some(&definitions[index + 1]),
                    index * 10 + 2,
                    ResolutionStatus::Resolved,
                )
            })
            .collect::<Vec<_>>();
        let root = definitions[0].clone();
        let (_directory, store) = stored_graph(&definitions, references).await;
        let opts = options(2);

        let cold_start = std::time::Instant::now();
        let cold = traverse_dependency_graph(&store, &root, "/graph", &opts)
            .await
            .unwrap();
        let cold_elapsed = cold_start.elapsed();
        let warm_start = std::time::Instant::now();
        let warm = traverse_dependency_graph(&store, &root, "/graph", &opts)
            .await
            .unwrap();
        let warm_elapsed = warm_start.elapsed();

        println!(
            "m3_depth2 nodes={} edges={} cold_ms={} warm_ms={}",
            warm.nodes.len(),
            warm.edges.len(),
            cold_elapsed.as_millis(),
            warm_elapsed.as_millis()
        );
        assert_eq!(cold.nodes.len(), 3);
        assert_eq!(warm.edges.len(), 2);
    }

    #[tokio::test]
    #[ignore = "synthetic M4 configuration-filtered graph latency measurement"]
    async fn benchmark_m4_configuration_filtered_depth_two() {
        let definitions = (0..500)
            .map(|index| definition(&format!("Node{index}"), index * 10 + 1))
            .collect::<Vec<_>>();
        let references = (0..499)
            .map(|index| {
                let mut reference = call(
                    &definitions[index],
                    Some(&definitions[index + 1]),
                    index * 10 + 2,
                    ResolutionStatus::Resolved,
                );
                reference.configuration_states = vec![
                    crate::build_config::ConfigurationState {
                        config_id: "debug".to_string(),
                        state: crate::build_config::PreprocessorState::Active,
                    },
                    crate::build_config::ConfigurationState {
                        config_id: "release".to_string(),
                        state: if index % 2 == 0 {
                            crate::build_config::PreprocessorState::Active
                        } else {
                            crate::build_config::PreprocessorState::Inactive
                        },
                    },
                ];
                reference
            })
            .collect::<Vec<_>>();
        let root = definitions[0].clone();
        let (_directory, store) = stored_graph(&definitions, references).await;
        let mut opts = options(2);
        opts.configurations = vec!["debug".to_string()];

        let cold_started = std::time::Instant::now();
        let cold = traverse_dependency_graph(&store, &root, "/graph", &opts)
            .await
            .unwrap();
        let cold_elapsed = cold_started.elapsed();
        let warm_started = std::time::Instant::now();
        let warm = traverse_dependency_graph(&store, &root, "/graph", &opts)
            .await
            .unwrap();
        let warm_elapsed = warm_started.elapsed();

        println!(
            "m4 config-filtered graph: nodes={} edges={} cold_ms={} warm_ms={}",
            warm.nodes.len(),
            warm.edges.len(),
            cold_elapsed.as_millis(),
            warm_elapsed.as_millis()
        );
        assert_eq!((cold.nodes.len(), cold.edges.len()), (3, 2));
        assert_eq!((warm.nodes.len(), warm.edges.len()), (3, 2));
    }
}
