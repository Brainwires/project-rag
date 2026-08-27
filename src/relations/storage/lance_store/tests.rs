use super::*;
use crate::relations::types::{SymbolId, Visibility};
use tempfile::TempDir;

fn make_def(name: &str, file: &str, start: usize, end: usize, kind: SymbolKind) -> Definition {
    Definition {
        symbol_id: SymbolId::new(file, name, kind, start, 0),
        root_path: Some("/test".to_string()),
        project: Some("proj".to_string()),
        end_line: end,
        end_col: 1,
        signature: format!("fn {}()", name),
        doc_comment: None,
        visibility: Visibility::Public,
        parent_id: None,
        indexed_at: 42,
    }
}

fn make_call_ref(target_id: &str, file: &str, line: usize) -> Reference {
    Reference {
        file_path: file.to_string(),
        root_path: Some("/test".to_string()),
        project: Some("proj".to_string()),
        start_line: line,
        end_line: line,
        start_col: 4,
        end_col: 10,
        target_symbol_id: target_id.to_string(),
        reference_kind: ReferenceKind::Call,
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
    store
        .store_references(vec![make_call_ref(&greet_id, "src/main.rs", 5)], "/test")
        .await
        .unwrap();

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
