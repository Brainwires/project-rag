use super::*;
use crate::relations::types::{SymbolId, SymbolKind, Visibility};
use tempfile::TempDir;

fn make_def(name: &str, file: &str, start: usize, end: usize, kind: SymbolKind) -> Definition {
    Definition {
        symbol_id: SymbolId::new(file, name, kind, start, 0),
        location: crate::relations::SourceLocation {
            project_id: "proj".to_string(),
            file_path: file.to_string(),
            start_line: start,
            start_col: 0,
            end_line: start,
            end_col: name.len(),
            role: crate::relations::LocationRole::Definition,
        },
        root_path: Some("/test".to_string()),
        project: Some("proj".to_string()),
        end_line: end,
        end_col: 1,
        signature: format!("fn {}()", name),
        doc_comment: None,
        visibility: Visibility::Public,
        parent_id: None,
        parser: "tree-sitter/test".to_string(),
        indexed_at: 42,
    }
}

fn make_call_ref(target_id: &str, file: &str, line: usize) -> Reference {
    let location = crate::relations::SourceLocation {
        project_id: "proj".to_string(),
        file_path: file.to_string(),
        start_line: line,
        start_col: 4,
        end_line: line,
        end_col: 10,
        role: crate::relations::LocationRole::Reference,
    };
    Reference {
        file_path: file.to_string(),
        root_path: Some("/test".to_string()),
        project: Some("proj".to_string()),
        start_line: line,
        end_line: line,
        start_col: 4,
        end_col: 10,
        location_id: location.to_storage_id(),
        source_symbol_id: None,
        target_symbol_id: target_id.to_string(),
        target_name: crate::relations::Definition::name_from_storage_id(target_id)
            .unwrap_or("target")
            .to_string(),
        candidates: vec![crate::relations::ReferenceCandidate {
            symbol_id: target_id.to_string(),
            reason: "test".to_string(),
        }],
        reference_kind: ReferenceKind::Call,
        resolution_status: crate::relations::ResolutionStatus::Resolved,
        evidence_kind: crate::relations::EvidenceKind::Syntactic,
        dispatch_kind: crate::relations::DispatchKind::Direct,
        language: "Rust".to_string(),
        parser: "tree-sitter/test".to_string(),
        indexed_at: 42,
    }
}

async fn make_store() -> (TempDir, LanceRelationsStore) {
    let temp_dir = TempDir::new().unwrap();
    let store = LanceRelationsStore::new(temp_dir.path().to_path_buf())
        .await
        .unwrap();
    (temp_dir, store)
}

fn directory_size(path: &std::path::Path) -> u64 {
    std::fs::read_dir(path)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                directory_size(&path)
            } else {
                entry.metadata().map(|metadata| metadata.len()).unwrap_or(0)
            }
        })
        .sum()
}

#[tokio::test]
async fn test_store_creation() {
    let (_dir, store) = make_store().await;
    let stats = store.get_stats().await.unwrap();
    assert_eq!(stats.definition_count, 0);
    assert_eq!(stats.reference_count, 0);
}

#[tokio::test]
async fn test_store_empty_definitions() {
    let (_dir, store) = make_store().await;
    let count = store.store_definitions(Vec::new(), "/test").await.unwrap();
    assert_eq!(count, 0);
}

#[tokio::test]
async fn test_definitions_roundtrip() {
    let (_dir, store) = make_store().await;

    let defs = vec![
        make_def("greet", "src/lib.rs", 10, 20, SymbolKind::Function),
        make_def("Person", "src/lib.rs", 30, 50, SymbolKind::Struct),
    ];
    let count = store.store_definitions(defs, "/test").await.unwrap();
    assert_eq!(count, 2);

    let found = store.find_definitions_by_name("greet").await.unwrap();
    assert_eq!(found.len(), 1);
    let def = &found[0];
    assert_eq!(def.name(), "greet");
    assert_eq!(def.kind(), SymbolKind::Function);
    assert_eq!(def.file_path(), "src/lib.rs");
    assert_eq!(def.start_line(), 10);
    assert_eq!(def.end_line, 20);
    assert_eq!(def.visibility, Visibility::Public);
    assert_eq!(def.project.as_deref(), Some("proj"));
    assert_eq!(def.indexed_at, 42);
}

#[tokio::test]
async fn test_store_is_idempotent_per_file() {
    let (_dir, store) = make_store().await;

    let defs = vec![make_def(
        "greet",
        "src/lib.rs",
        10,
        20,
        SymbolKind::Function,
    )];
    store
        .store_definitions(defs.clone(), "/test")
        .await
        .unwrap();
    store.store_definitions(defs, "/test").await.unwrap();

    let stats = store.get_stats().await.unwrap();
    assert_eq!(
        stats.definition_count, 1,
        "re-storing the same file must not duplicate rows"
    );
}

#[tokio::test]
async fn test_store_and_delete_are_scoped_by_project_root() {
    let (_dir, store) = make_store().await;

    let mut first = make_def("greet", "src/lib.rs", 10, 20, SymbolKind::Function);
    first.root_path = Some("/root/a".to_string());
    first.project = Some("project-a".to_string());
    store
        .store_definitions(vec![first], "/root/a")
        .await
        .unwrap();

    let mut second = make_def("greet", "src/lib.rs", 10, 20, SymbolKind::Function);
    second.root_path = Some("/root/b".to_string());
    second.project = Some("project-b".to_string());
    store
        .store_definitions(vec![second.clone()], "/root/b")
        .await
        .unwrap();

    // Replacing root A must not remove root B's same relative path.
    let mut replacement = second.clone();
    replacement.root_path = Some("/root/a".to_string());
    replacement.project = Some("project-a".to_string());
    store
        .store_definitions(vec![replacement], "/root/a")
        .await
        .unwrap();
    assert_eq!(store.get_stats().await.unwrap().definition_count, 2);

    let removed = store
        .delete_by_file_in_root("src/lib.rs", "/root/a")
        .await
        .unwrap();
    assert_eq!(removed, 1);

    let remaining = store.find_definitions_by_name("greet").await.unwrap();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].root_path.as_deref(), Some("/root/b"));
    assert_eq!(remaining[0].project.as_deref(), Some("project-b"));
}

#[tokio::test]
async fn test_find_definition_at_innermost() {
    let (_dir, store) = make_store().await;

    // A method nested inside a class: line 12 is inside both.
    let defs = vec![
        make_def("MyClass", "src/lib.rs", 1, 100, SymbolKind::Class),
        make_def("helper", "src/lib.rs", 10, 15, SymbolKind::Method),
    ];
    store.store_definitions(defs, "/test").await.unwrap();

    let found = store
        .find_definition_at("src/lib.rs", 12, 0)
        .await
        .unwrap()
        .expect("should find a definition");
    assert_eq!(found.name(), "helper", "innermost definition must win");

    let outer = store
        .find_definition_at("src/lib.rs", 50, 0)
        .await
        .unwrap()
        .expect("should find the class");
    assert_eq!(outer.name(), "MyClass");

    let none = store
        .find_definition_at("src/lib.rs", 200, 0)
        .await
        .unwrap();
    assert!(none.is_none());
}

#[tokio::test]
async fn test_references_roundtrip_and_delete_by_file() {
    let (_dir, store) = make_store().await;

    let target = make_def("greet", "src/lib.rs", 10, 20, SymbolKind::Function);
    let target_id = target.to_storage_id();
    store
        .store_definitions(vec![target], "/test")
        .await
        .unwrap();
    store
        .store_references(
            vec![
                make_call_ref(&target_id, "src/main.rs", 5),
                make_call_ref(&target_id, "src/other.rs", 7),
            ],
            "/test",
        )
        .await
        .unwrap();

    let refs = store.find_references(&target_id).await.unwrap();
    assert_eq!(refs.len(), 2);
    assert!(refs.iter().all(|r| r.reference_kind == ReferenceKind::Call));

    // Deleting one file removes its references but not the other file's.
    let removed = store.delete_by_file("src/main.rs").await.unwrap();
    assert_eq!(removed, 1);
    let refs = store.find_references(&target_id).await.unwrap();
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].file_path, "src/other.rs");

    // Deleting the defining file removes the definition.
    let removed = store.delete_by_file("src/lib.rs").await.unwrap();
    assert_eq!(removed, 1);
    assert!(
        store
            .find_definitions_by_name("greet")
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn reference_name_query_and_statistics_share_persisted_rows() {
    let (_dir, store) = make_store().await;
    let target = make_def("Write", "src/api.rs", 1, 3, SymbolKind::Function);
    let target_id = target.to_storage_id();
    let mut call = make_call_ref(&target_id, "src/main.rs", 5);
    call.target_name = "Write".to_string();
    let mut comment = call.clone();
    comment.start_line = 6;
    comment.end_line = 6;
    comment.location_id = crate::relations::SourceLocation {
        project_id: "proj".to_string(),
        file_path: "src/main.rs".to_string(),
        start_line: 6,
        start_col: 4,
        end_line: 6,
        end_col: 9,
        role: crate::relations::LocationRole::Reference,
    }
    .to_storage_id();
    comment.reference_kind = ReferenceKind::Comment;
    comment.resolution_status = crate::relations::ResolutionStatus::Unresolved;
    comment.evidence_kind = crate::relations::EvidenceKind::Heuristic;
    comment.target_symbol_id.clear();

    store
        .store_references(vec![call, comment], "/test")
        .await
        .unwrap();
    let matches = store
        .find_references_by_name_in_root("Write", "/test")
        .await
        .unwrap();
    let at_call = store
        .find_reference_at_in_root("src/main.rs", "/test", 5, 5)
        .await
        .unwrap()
        .unwrap();
    let stats = store.get_stats().await.unwrap();
    assert_eq!(matches.len(), 2);
    assert_eq!(at_call.target_symbol_id, target_id);
    assert_eq!(stats.reference_count, 2);
    assert_eq!(stats.code_reference_count, 1);
}

#[tokio::test]
#[ignore = "manual synthetic M2 latency/index-size measurement"]
async fn benchmark_m2_relations_store() {
    let (dir, store) = make_store().await;
    let definitions: Vec<_> = (0..500)
        .map(|index| {
            make_def(
                &format!("symbol_{index}"),
                &format!("src/file_{}.rs", index / 10),
                index + 1,
                index + 2,
                SymbolKind::Function,
            )
        })
        .collect();
    let symbol_ids: Vec<_> = definitions.iter().map(Definition::to_storage_id).collect();
    let references: Vec<_> = (0..2_000)
        .map(|index| {
            let mut reference = make_call_ref(
                &symbol_ids[index % symbol_ids.len()],
                &format!("src/caller_{}.rs", index / 20),
                index + 1,
            );
            reference.target_name = format!("symbol_{}", index % symbol_ids.len());
            reference.location_id = format!("loc:v3:benchmark:{index}");
            reference
        })
        .collect();

    let write_started = std::time::Instant::now();
    store.store_definitions(definitions, "/test").await.unwrap();
    store.store_references(references, "/test").await.unwrap();
    let write_elapsed = write_started.elapsed();
    let query_started = std::time::Instant::now();
    let matches = store
        .find_references_by_name_in_root("symbol_42", "/test")
        .await
        .unwrap();
    let query_elapsed = query_started.elapsed();
    println!(
        "m2 synthetic: write_500_defs_2000_refs_ms={} query_matches={} query_ms={} db_bytes={}",
        write_elapsed.as_millis(),
        matches.len(),
        query_elapsed.as_millis(),
        directory_size(dir.path())
    );
}

#[tokio::test]
async fn test_callers_and_callees() {
    let (_dir, store) = make_store().await;

    // callee `greet` in lib.rs; caller `main` in main.rs calls it at line 5.
    let greet = make_def("greet", "src/lib.rs", 10, 20, SymbolKind::Function);
    let main_fn = make_def("main", "src/main.rs", 1, 30, SymbolKind::Function);
    let greet_id = greet.to_storage_id();
    let main_id = main_fn.to_storage_id();

    store
        .store_definitions(vec![greet, main_fn], "/test")
        .await
        .unwrap();
    let mut call = make_call_ref(&greet_id, "src/main.rs", 5);
    call.source_symbol_id = Some(main_id.clone());
    store.store_references(vec![call], "/test").await.unwrap();

    let callers = store.get_callers(&greet_id).await.unwrap();
    assert_eq!(callers.len(), 1);
    assert_eq!(callers[0].caller_id, main_id);
    assert_eq!(callers[0].callee_id, greet_id);
    assert_eq!(callers[0].call_site_file, "src/main.rs");
    assert_eq!(callers[0].call_site_line, 5);

    let callees = store.get_callees(&main_id).await.unwrap();
    assert_eq!(callees.len(), 1);
    assert_eq!(callees[0].caller_id, main_id);
    assert_eq!(callees[0].callee_id, greet_id);

    // A symbol with no calls has neither callers nor callees.
    let callees = store.get_callees(&greet_id).await.unwrap();
    assert!(callees.is_empty());
}

#[tokio::test]
async fn test_clear_and_stats() {
    let (_dir, store) = make_store().await;

    store
        .store_definitions(
            vec![
                make_def("a", "src/a.rs", 1, 5, SymbolKind::Function),
                make_def("b", "src/b.rs", 1, 5, SymbolKind::Function),
            ],
            "/test",
        )
        .await
        .unwrap();
    store
        .store_references(
            vec![make_call_ref("def:src/a.rs:a:1", "src/b.rs", 3)],
            "/test",
        )
        .await
        .unwrap();

    let stats = store.get_stats().await.unwrap();
    assert_eq!(stats.definition_count, 2);
    assert_eq!(stats.reference_count, 1);
    assert_eq!(stats.files_with_definitions, 2);

    store.clear().await.unwrap();

    let stats = store.get_stats().await.unwrap();
    assert_eq!(stats.definition_count, 0);
    assert_eq!(stats.reference_count, 0);
    assert_eq!(stats.files_with_definitions, 0);
}

#[tokio::test]
async fn test_sql_escaping_in_paths() {
    let (_dir, store) = make_store().await;

    // A path containing a single quote must not break the delete filter.
    let defs = vec![make_def("f", "src/it's.rs", 1, 5, SymbolKind::Function)];
    store.store_definitions(defs, "/test").await.unwrap();

    let removed = store.delete_by_file("src/it's.rs").await.unwrap();
    assert_eq!(removed, 1);
}
