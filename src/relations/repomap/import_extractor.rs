//! Import statement extraction.
//!
//! Turns import/use/include nodes into `SymbolKind::Import` definitions, one per
//! BOUND NAME rather than one per statement: `use a::{B, C as D};` yields `B` and
//! `D`, because those are the identifiers the rest of the file can actually
//! reference. That is the granularity an unused-import check needs.
//!
//! A statement that binds no checkable name (glob imports like `use x::*`,
//! side-effect imports like `import "./polyfill"`) yields nothing here; the caller
//! records it as a skipped definition so the omission is visible.

use chrono::Utc;
use tree_sitter::Node;

use crate::indexer::FileInfo;
use crate::relations::types::{Definition, SymbolId, SymbolKind, Visibility};

/// Check if a node kind represents an import statement for the given language.
pub fn is_import_node(kind: &str, language: &str) -> bool {
    match language {
        "Rust" => matches!(kind, "use_declaration" | "extern_crate_declaration"),
        "Python" => matches!(kind, "import_statement" | "import_from_statement"),
        "JavaScript" | "TypeScript" => kind == "import_statement",
        "Go" | "Java" | "Swift" => kind == "import_declaration",
        "C" | "C++" => kind == "preproc_include",
        "C#" => kind == "using_directive",
        "PHP" => kind == "namespace_use_declaration",
        _ => false,
    }
}

/// Extract one `Definition` per name the import binds into scope.
///
/// Returns an empty vector when no bound name could be extracted -- either because
/// the statement genuinely binds none (globs, side-effect imports) or because the
/// grammar shape was not recognised. The caller must surface that as a skipped
/// definition rather than dropping it silently.
pub fn extract_imports(
    node: Node,
    source: &str,
    language: &str,
    file_info: &FileInfo,
    parent_id: &Option<String>,
) -> Vec<Definition> {
    let mut names = binding_names(node, source, language);

    // A statement can mention the same name twice; one definition per name.
    let mut seen = std::collections::HashSet::new();
    names.retain(|n| !n.trim().is_empty() && seen.insert(n.clone()));

    let start_pos = node.start_position();
    let end_pos = node.end_position();
    let text = node_text(node, source);
    let signature = first_line(text);
    let visibility = Visibility::from_keywords(&signature);

    names
        .into_iter()
        .map(|name| Definition {
            symbol_id: SymbolId::new(
                &file_info.relative_path,
                name,
                SymbolKind::Import,
                start_pos.row + 1,
                start_pos.column,
            ),
            root_path: Some(file_info.root_path.clone()),
            project: file_info.project.clone(),
            end_line: end_pos.row + 1,
            end_col: end_pos.column,
            signature: signature.clone(),
            doc_comment: None,
            visibility,
            parent_id: parent_id.clone(),
            indexed_at: Utc::now().timestamp(),
        })
        .collect()
}

/// The names an import statement binds, per language.
fn binding_names(node: Node, source: &str, language: &str) -> Vec<String> {
    match language {
        "Rust" => rust_bindings(node, source),
        "Python" => python_bindings(node, source),
        "JavaScript" | "TypeScript" => js_bindings(node, source),
        "Go" => go_bindings(node, source),
        "Java" => java_bindings(node, source),
        "Swift" => swift_bindings(node, source),
        "C" | "C++" => c_include_bindings(node, source),
        "C#" => csharp_bindings(node, source),
        "PHP" => php_bindings(node, source),
        _ => Vec::new(),
    }
}

fn node_text<'a>(node: Node, source: &'a str) -> &'a str {
    source.get(node.start_byte()..node.end_byte()).unwrap_or("")
}

fn first_line(text: &str) -> String {
    text.lines().next().unwrap_or("").trim().to_string()
}

/// Rust: walk the use clause tree. `use_as_clause` binds its alias, a
/// `scoped_identifier` binds its final segment, `use_list` recurses, and
/// `use_wildcard` binds nothing checkable.
fn rust_bindings(node: Node, source: &str) -> Vec<String> {
    fn walk(node: Node, source: &str, out: &mut Vec<String>) {
        match node.kind() {
            "identifier" => out.push(node_text(node, source).to_string()),
            "scoped_identifier" => {
                if let Some(name) = node.child_by_field_name("name") {
                    out.push(node_text(name, source).to_string());
                }
            }
            "use_as_clause" => {
                if let Some(alias) = node.child_by_field_name("alias") {
                    out.push(node_text(alias, source).to_string());
                }
            }
            "use_list" => {
                let mut cursor = node.walk();
                for child in node.named_children(&mut cursor) {
                    walk(child, source, out);
                }
            }
            "scoped_use_list" => {
                if let Some(list) = node.child_by_field_name("list") {
                    walk(list, source, out);
                }
            }
            "use_wildcard" => {} // binds unknowable names
            _ => {}
        }
    }

    let mut out = Vec::new();
    if node.kind() == "extern_crate_declaration" {
        // `extern crate foo;` or `extern crate foo as bar;`
        let bound = node
            .child_by_field_name("alias")
            .or_else(|| node.child_by_field_name("name"));
        if let Some(n) = bound {
            out.push(node_text(n, source).to_string());
        }
        return out;
    }
    if let Some(argument) = node.child_by_field_name("argument") {
        walk(argument, source, &mut out);
    }
    out
}

/// Python: `import a.b` binds `a` (the top-level module); `from m import x`
/// binds `x`. Aliases win in both forms. Both grammars put the imported items
/// in the `name` field, and `from` puts the module in `module_name`.
fn python_bindings(node: Node, source: &str) -> Vec<String> {
    let from_import = node.kind() == "import_from_statement";
    let mut out = Vec::new();
    let mut cursor = node.walk();
    for item in node.children_by_field_name("name", &mut cursor) {
        match item.kind() {
            "aliased_import" => {
                if let Some(alias) = item.child_by_field_name("alias") {
                    out.push(node_text(alias, source).to_string());
                }
            }
            "dotted_name" => {
                let text = node_text(item, source);
                let segment = if from_import {
                    text.rsplit('.').next()
                } else {
                    text.split('.').next()
                };
                if let Some(s) = segment {
                    out.push(s.to_string());
                }
            }
            _ => {}
        }
    }
    out
}

/// JS/TS: default import, `* as ns`, and named specifiers (alias wins).
fn js_bindings(node: Node, source: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() != "import_clause" {
            continue;
        }
        let mut clause_cursor = child.walk();
        for part in child.named_children(&mut clause_cursor) {
            match part.kind() {
                "identifier" => out.push(node_text(part, source).to_string()),
                "namespace_import" => {
                    if let Some(id) = first_descendant_of_kind(part, "identifier") {
                        out.push(node_text(id, source).to_string());
                    }
                }
                "named_imports" => {
                    let mut spec_cursor = part.walk();
                    for spec in part.named_children(&mut spec_cursor) {
                        if spec.kind() != "import_specifier" {
                            continue;
                        }
                        let bound = spec
                            .child_by_field_name("alias")
                            .or_else(|| spec.child_by_field_name("name"));
                        if let Some(n) = bound {
                            out.push(node_text(n, source).trim_matches(['"', '\'']).to_string());
                        }
                    }
                }
                _ => {}
            }
        }
    }
    out
}

/// Go: each `import_spec` binds its explicit package name, or the final path
/// segment of the import string. `_` and `.` imports bind nothing checkable.
fn go_bindings(node: Node, source: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut specs = Vec::new();
    collect_descendants_of_kind(node, "import_spec", &mut specs);
    for spec in specs {
        if let Some(name) = spec.child_by_field_name("name") {
            let text = node_text(name, source);
            if text != "_" && text != "." {
                out.push(text.to_string());
            }
            continue;
        }
        if let Some(path) = spec.child_by_field_name("path") {
            let text = node_text(path, source).trim_matches(['"', '`']).to_string();
            if let Some(segment) = text.rsplit('/').next()
                && !segment.is_empty()
            {
                out.push(segment.to_string());
            }
        }
    }
    out
}

/// Java: `import java.util.List;` binds `List`. Wildcard imports bind
/// unknowable names and yield nothing.
fn java_bindings(node: Node, source: &str) -> Vec<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "asterisk" {
            return Vec::new();
        }
    }
    let mut out = Vec::new();
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "scoped_identifier" => {
                if let Some(name) = child.child_by_field_name("name") {
                    out.push(node_text(name, source).to_string());
                }
            }
            "identifier" => out.push(node_text(child, source).to_string()),
            _ => {}
        }
    }
    out
}

/// Swift: `import Foundation` binds the module name.
fn swift_bindings(node: Node, source: &str) -> Vec<String> {
    for kind in ["simple_identifier", "identifier"] {
        if let Some(id) = first_descendant_of_kind(node, kind) {
            return vec![node_text(id, source).to_string()];
        }
    }
    Vec::new()
}

/// C/C++: the "name" of an include is the header path itself, quotes and
/// angle brackets stripped: `#include <stdio.h>` binds `stdio.h`.
fn c_include_bindings(node: Node, source: &str) -> Vec<String> {
    let Some(path) = node.child_by_field_name("path") else {
        return Vec::new();
    };
    let text = node_text(path, source)
        .trim()
        .trim_matches(['"', '<', '>'])
        .to_string();
    if text.is_empty() {
        Vec::new()
    } else {
        vec![text]
    }
}

/// C#: an alias directive binds its alias; a plain `using System.Text;` is
/// recorded under its final segment. That final segment is a namespace, not a
/// usable identifier, so unused-detection for C# usings stays heuristic.
fn csharp_bindings(node: Node, source: &str) -> Vec<String> {
    let mut ids = Vec::new();
    collect_descendants_of_kind(node, "identifier", &mut ids);

    // An alias directive (`using Foo = System.Bar;`) is recognisable by the bare
    // `=` token; the alias is the identifier BEFORE it. Depending on grammar
    // version the alias may or may not be wrapped in a name_equals node, so key
    // off the token rather than the wrapper.
    let mut cursor = node.walk();
    let is_alias = node.children(&mut cursor).any(|c| c.kind() == "=")
        || first_descendant_of_kind(node, "name_equals").is_some();

    let bound = if is_alias { ids.first() } else { ids.last() };
    match bound {
        Some(id) => vec![node_text(*id, source).to_string()],
        None => Vec::new(),
    }
}

/// PHP: each use clause binds its alias or the final segment of the
/// qualified name.
fn php_bindings(node: Node, source: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut clauses = Vec::new();
    collect_descendants_of_kind(node, "namespace_use_clause", &mut clauses);
    for clause in clauses {
        if let Some(aliasing) = first_descendant_of_kind(clause, "namespace_aliasing_clause")
            && let Some(name) = first_descendant_of_kind(aliasing, "name")
        {
            out.push(node_text(name, source).to_string());
            continue;
        }
        let mut names = Vec::new();
        collect_descendants_of_kind(clause, "name", &mut names);
        if let Some(last) = names.last() {
            out.push(node_text(*last, source).to_string());
        }
    }
    out
}

fn first_descendant_of_kind<'a>(node: Node<'a>, kind: &str) -> Option<Node<'a>> {
    if node.kind() == kind {
        return Some(node);
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if let Some(found) = first_descendant_of_kind(child, kind) {
            return Some(found);
        }
    }
    None
}

fn collect_descendants_of_kind<'a>(node: Node<'a>, kind: &str, out: &mut Vec<Node<'a>>) {
    if node.kind() == kind {
        out.push(node);
        // A clause of some kind never nests inside itself in the grammars used here.
        return;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_descendants_of_kind(child, kind, out);
    }
}

#[cfg(test)]
mod tests {
    use super::super::symbol_extractor::SymbolExtractor;
    use crate::indexer::FileInfo;
    use crate::relations::types::{SymbolKind, Visibility};
    use std::path::PathBuf;

    fn make_file_info(content: &str, extension: &str) -> FileInfo {
        FileInfo {
            path: PathBuf::from(format!("test.{}", extension)),
            relative_path: format!("test.{}", extension),
            root_path: "/test".to_string(),
            project: None,
            extension: Some(extension.to_string()),
            language: None,
            content: content.to_string(),
            hash: "test_hash".to_string(),
        }
    }

    fn import_names(content: &str, extension: &str) -> Vec<String> {
        let file_info = make_file_info(content, extension);
        let extractor = SymbolExtractor::new();
        let definitions = extractor.extract_definitions(&file_info).unwrap();
        definitions
            .iter()
            .filter(|d| d.kind() == SymbolKind::Import)
            .map(|d| d.name().to_string())
            .collect()
    }

    #[test]
    fn test_rust_use_bindings() {
        let names = import_names(
            "use std::collections::HashMap;\nuse foo::bar as baz;\nuse a::{B, C as D};\n",
            "rs",
        );
        assert_eq!(names, vec!["HashMap", "baz", "B", "D"]);
    }

    #[test]
    fn test_rust_pub_use_is_public() {
        let file_info = make_file_info("pub use foo::Bar;\n", "rs");
        let defs = SymbolExtractor::new()
            .extract_definitions(&file_info)
            .unwrap();
        let import = defs
            .iter()
            .find(|d| d.kind() == SymbolKind::Import)
            .unwrap();
        assert_eq!(import.name(), "Bar");
        assert_eq!(import.visibility, Visibility::Public);
    }

    #[test]
    fn test_rust_glob_import_is_reported_skipped() {
        let file_info = make_file_info("use foo::*;\n", "rs");
        let (defs, skipped) = SymbolExtractor::new()
            .extract_definitions_reporting(&file_info)
            .unwrap();
        assert!(defs.iter().all(|d| d.kind() != SymbolKind::Import));
        assert_eq!(skipped.len(), 1);
        assert_eq!(skipped[0].kind, "use_declaration");
    }

    #[test]
    fn test_python_import_bindings() {
        let names = import_names(
            "import os\nimport numpy as np\nfrom collections import OrderedDict\nfrom x import a as b, c\nimport os.path\n",
            "py",
        );
        assert_eq!(names, vec!["os", "np", "OrderedDict", "b", "c", "os"]);
    }

    #[test]
    fn test_javascript_import_bindings() {
        let names = import_names(
            "import React from 'react';\nimport { useState, useEffect as ue } from 'react';\nimport * as path from 'path';\nimport './side-effect.css';\n",
            "js",
        );
        assert_eq!(names, vec!["React", "useState", "ue", "path"]);
    }

    #[test]
    fn test_typescript_import_bindings() {
        let names = import_names("import { Component } from '@angular/core';\n", "ts");
        assert_eq!(names, vec!["Component"]);
    }

    #[test]
    fn test_go_import_bindings() {
        let names = import_names(
            "package main\n\nimport (\n\tf \"fmt\"\n\t\"strings\"\n\t\"net/http\"\n\t_ \"embed\"\n)\n",
            "go",
        );
        assert_eq!(names, vec!["f", "strings", "http"]);
    }

    #[test]
    fn test_java_import_bindings() {
        let names = import_names(
            "import java.util.List;\nimport java.util.*;\n\nclass Foo {}\n",
            "java",
        );
        assert_eq!(names, vec!["List"]);
    }

    #[test]
    fn test_c_include_bindings() {
        let names = import_names("#include <stdio.h>\n#include \"myheader.h\"\n", "c");
        assert_eq!(names, vec!["stdio.h", "myheader.h"]);
    }

    #[test]
    fn test_cpp_include_bindings() {
        let names = import_names("#include <vector>\n", "cpp");
        assert_eq!(names, vec!["vector"]);
    }

    #[test]
    fn test_csharp_using_bindings() {
        let names = import_names(
            "using System.Text;\nusing Foo = System.Bar;\n\nclass C {}\n",
            "cs",
        );
        assert_eq!(names, vec!["Text", "Foo"]);
    }

    #[test]
    fn test_php_use_bindings() {
        let names = import_names(
            "<?php\nuse App\\Models\\User;\nuse Foo\\Bar as Baz;\n",
            "php",
        );
        assert_eq!(names, vec!["User", "Baz"]);
    }

    #[test]
    fn test_swift_import_bindings() {
        let names = import_names("import Foundation\n", "swift");
        assert_eq!(names, vec!["Foundation"]);
    }
}
