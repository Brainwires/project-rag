//! Build-configuration discovery and conservative preprocessor state tracking.

use anyhow::{Context, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use crate::project_path::ProjectPathResolver;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PreprocessorState {
    Active,
    Inactive,
    UnknownDueToBuildConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ConfigurationState {
    pub config_id: String,
    pub state: PreprocessorState,
}

/// Whether an observation is usable in the requested configuration scope.
/// Unknown observations remain visible because excluding them would turn an
/// incomplete build model into a false negative.
pub fn configuration_scope_matches(states: &[ConfigurationState], requested: &[String]) -> bool {
    requested.is_empty()
        || states.iter().any(|state| {
            requested.contains(&state.config_id) && state.state != PreprocessorState::Inactive
        })
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Explicit compile_commands.json locations, relative to the project root or absolute.
    #[serde(default)]
    pub compile_commands_paths: Vec<PathBuf>,
    /// Configurations used when no compilation database exists, or in addition to it.
    #[serde(default)]
    pub build_configurations: Vec<ExplicitBuildConfiguration>,
    /// Project-specific tokens that indicate reflection, registration, or generated wiring.
    #[serde(default)]
    pub dynamic_wiring_patterns: Vec<String>,
    /// Path fragments that identify generated source/header inputs.
    #[serde(default = "default_generated_path_patterns")]
    pub generated_path_patterns: Vec<String>,
}

fn default_generated_path_patterns() -> Vec<String> {
    vec![
        "generated/".to_string(),
        "autogen/".to_string(),
        "gen/".to_string(),
    ]
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExplicitBuildConfiguration {
    #[serde(default)]
    pub config_id: Option<String>,
    #[serde(default)]
    pub source_files: Vec<String>,
    #[serde(default)]
    pub include_paths: Vec<String>,
    #[serde(default)]
    pub preprocessor_definitions: Vec<String>,
    #[serde(default)]
    pub language_standard: Option<String>,
    #[serde(default)]
    pub forced_includes: Vec<String>,
    #[serde(default)]
    pub generated_header_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BuildConfigSource {
    CompileCommands,
    Explicit,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BuildConfiguration {
    pub config_id: String,
    pub source: BuildConfigSource,
    pub source_files: Vec<String>,
    pub include_paths: Vec<String>,
    pub preprocessor_definitions: Vec<String>,
    pub language_standard: Option<String>,
    pub forced_includes: Vec<String>,
    pub generated_header_paths: Vec<String>,
}

impl BuildConfiguration {
    fn macro_values(&self) -> HashMap<String, String> {
        self.preprocessor_definitions
            .iter()
            .filter_map(|definition| {
                let (name, value) = definition.split_once('=').unwrap_or((definition, "1"));
                (!name.is_empty()).then(|| (name.to_string(), value.to_string()))
            })
            .collect()
    }
}

#[derive(Debug, Clone, Default)]
pub struct BuildConfigCatalog {
    pub configurations: Vec<BuildConfiguration>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CompileCommandEntry {
    directory: String,
    file: String,
    #[serde(default)]
    arguments: Option<Vec<String>>,
    #[serde(default)]
    command: Option<String>,
}

impl BuildConfigCatalog {
    pub fn discover(project_root: &Path, analysis: &AnalysisConfig) -> Result<Self> {
        let resolver = ProjectPathResolver::new(project_root)?;
        let mut catalog = Self::default();
        let mut candidates = analysis.compile_commands_paths.clone();
        if candidates.is_empty() {
            candidates.extend([
                PathBuf::from("compile_commands.json"),
                PathBuf::from("build/compile_commands.json"),
                PathBuf::from("out/compile_commands.json"),
            ]);
        }

        let mut found_database = false;
        for candidate in candidates {
            let path = if candidate.is_absolute() {
                candidate
            } else {
                project_root.join(candidate)
            };
            if !path.is_file() {
                continue;
            }
            found_database = true;
            match parse_compile_commands(&path, project_root, &resolver) {
                Ok(configurations) => catalog.configurations.extend(configurations),
                Err(error) => catalog.diagnostics.push(format!(
                    "Invalid compilation database '{}': {:#}",
                    path.display(),
                    error
                )),
            }
        }
        if !found_database {
            catalog.diagnostics.push(
                "No compile_commands.json found; conditional C/C++ analysis requires explicit build configurations"
                    .to_string(),
            );
        }

        for explicit in &analysis.build_configurations {
            let mut configuration = BuildConfiguration {
                config_id: explicit.config_id.clone().unwrap_or_default(),
                source: BuildConfigSource::Explicit,
                source_files: normalize_values(&explicit.source_files),
                include_paths: normalize_values(&explicit.include_paths),
                preprocessor_definitions: normalize_values(&explicit.preprocessor_definitions),
                language_standard: explicit.language_standard.clone(),
                forced_includes: normalize_values(&explicit.forced_includes),
                generated_header_paths: normalize_values(&explicit.generated_header_paths),
            };
            if configuration.config_id.is_empty() {
                configuration.config_id = semantic_config_id(&configuration);
            }
            catalog.configurations.push(configuration);
        }

        catalog.configurations = merge_configurations(catalog.configurations);
        if catalog.configurations.is_empty() {
            catalog
                .diagnostics
                .push("No analyzed build configuration is available".to_string());
        }
        Ok(catalog)
    }

    pub fn selected(&self, requested: &[String]) -> Self {
        if requested.is_empty() {
            return self.clone();
        }
        let requested = requested.iter().collect::<BTreeSet<_>>();
        let configurations = self
            .configurations
            .iter()
            .filter(|configuration| requested.contains(&configuration.config_id))
            .cloned()
            .collect::<Vec<_>>();
        let mut diagnostics = self.diagnostics.clone();
        for missing in requested {
            if !configurations
                .iter()
                .any(|configuration| &configuration.config_id == missing)
            {
                diagnostics.push(format!(
                    "Requested build configuration '{}' is unavailable",
                    missing
                ));
            }
        }
        Self {
            configurations,
            diagnostics,
        }
    }

    pub fn config_ids(&self) -> Vec<String> {
        self.configurations
            .iter()
            .map(|configuration| configuration.config_id.clone())
            .collect()
    }

    pub fn configurations_for_file(&self, relative_path: &str) -> Vec<&BuildConfiguration> {
        let relative_path = relative_path.replace('\\', "/");
        let header = Path::new(&relative_path)
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| {
                matches!(
                    extension.to_ascii_lowercase().as_str(),
                    "h" | "hh" | "hpp" | "hxx" | "inc"
                )
            });
        self.configurations
            .iter()
            .filter(|configuration| {
                configuration.source_files.is_empty()
                    || header
                    || configuration
                        .source_files
                        .iter()
                        .any(|file| file == &relative_path)
            })
            .collect()
    }

    pub fn states_for_file(
        &self,
        relative_path: &str,
        content: &str,
    ) -> Vec<Vec<ConfigurationState>> {
        let line_count = content.lines().count().max(1);
        let mut states = vec![Vec::new(); line_count + 1];
        for configuration in self.configurations_for_file(relative_path) {
            let per_line = evaluate_preprocessor(content, configuration);
            for (line, line_states) in states.iter_mut().enumerate().take(line_count + 1).skip(1) {
                line_states.push(ConfigurationState {
                    config_id: configuration.config_id.clone(),
                    state: per_line
                        .get(line)
                        .copied()
                        .unwrap_or(PreprocessorState::UnknownDueToBuildConfig),
                });
            }
        }
        states
    }

    pub fn include_paths(&self, project_root: &Path) -> Vec<PathBuf> {
        let mut paths = BTreeSet::new();
        for configuration in &self.configurations {
            for path in configuration
                .include_paths
                .iter()
                .chain(&configuration.generated_header_paths)
            {
                let path = PathBuf::from(path);
                paths.insert(if path.is_absolute() {
                    path
                } else {
                    project_root.join(path)
                });
            }
        }
        paths.into_iter().collect()
    }
}

fn parse_compile_commands(
    path: &Path,
    project_root: &Path,
    resolver: &ProjectPathResolver,
) -> Result<Vec<BuildConfiguration>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let entries: Vec<CompileCommandEntry> = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse {}", path.display()))?;
    let mut configurations = Vec::new();
    for entry in entries {
        let working_directory = PathBuf::from(&entry.directory);
        let source = if Path::new(&entry.file).is_absolute() {
            PathBuf::from(&entry.file)
        } else {
            working_directory.join(&entry.file)
        };
        let Ok(source) = resolver.resolve_existing(&source.to_string_lossy()) else {
            continue;
        };
        let arguments = entry
            .arguments
            .unwrap_or_else(|| tokenize_command(entry.command.as_deref().unwrap_or_default()));
        let mut configuration = parse_arguments(&arguments, &working_directory, project_root);
        configuration.source = BuildConfigSource::CompileCommands;
        configuration.source_files = vec![source.relative];
        configuration.config_id = semantic_config_id(&configuration);
        configurations.push(configuration);
    }
    Ok(merge_configurations(configurations))
}

fn parse_arguments(
    arguments: &[String],
    directory: &Path,
    project_root: &Path,
) -> BuildConfiguration {
    let mut include_paths = Vec::new();
    let mut definitions = Vec::new();
    let mut forced_includes = Vec::new();
    let mut language_standard = None;
    let mut index = 0;
    while index < arguments.len() {
        let argument = &arguments[index];
        let next = || arguments.get(index + 1).cloned();
        if argument == "-I" || argument == "/I" || argument == "-isystem" {
            if let Some(value) = next() {
                include_paths.push(portable_path(&value, directory, project_root));
                index += 1;
            }
        } else if let Some(value) = argument
            .strip_prefix("-I")
            .filter(|value| !value.is_empty())
        {
            include_paths.push(portable_path(value, directory, project_root));
        } else if let Some(value) = argument
            .strip_prefix("/I")
            .filter(|value| !value.is_empty())
        {
            include_paths.push(portable_path(value, directory, project_root));
        } else if argument == "-D" || argument == "/D" {
            if let Some(value) = next() {
                definitions.push(value);
                index += 1;
            }
        } else if let Some(value) = argument
            .strip_prefix("-D")
            .filter(|value| !value.is_empty())
        {
            definitions.push(value.to_string());
        } else if let Some(value) = argument
            .strip_prefix("/D")
            .filter(|value| !value.is_empty())
        {
            definitions.push(value.to_string());
        } else if let Some(value) = argument.strip_prefix("-std=") {
            language_standard = Some(value.to_string());
        } else if let Some(value) = argument.strip_prefix("/std:") {
            language_standard = Some(value.to_string());
        } else if argument == "-include" || argument == "/FI" {
            if let Some(value) = next() {
                forced_includes.push(portable_path(&value, directory, project_root));
                index += 1;
            }
        } else if let Some(value) = argument
            .strip_prefix("/FI")
            .filter(|value| !value.is_empty())
        {
            forced_includes.push(portable_path(value, directory, project_root));
        }
        index += 1;
    }
    let generated_header_paths = include_paths
        .iter()
        .filter(|path| is_generated_path(path))
        .cloned()
        .collect();
    BuildConfiguration {
        config_id: String::new(),
        source: BuildConfigSource::CompileCommands,
        source_files: Vec::new(),
        include_paths: normalize_values(&include_paths),
        preprocessor_definitions: normalize_values(&definitions),
        language_standard,
        forced_includes: normalize_values(&forced_includes),
        generated_header_paths,
    }
}

fn portable_path(value: &str, directory: &Path, project_root: &Path) -> String {
    let value = value.trim_matches('"');
    let path = PathBuf::from(value);
    let absolute = if path.is_absolute() {
        path
    } else {
        directory.join(path)
    };
    absolute
        .strip_prefix(project_root)
        .map(|relative| relative.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| absolute.to_string_lossy().replace('\\', "/"))
}

fn is_generated_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    [
        "generated",
        "autogen",
        "/gen/",
        "\\gen\\",
        "/build/",
        "\\build\\",
    ]
    .iter()
    .any(|fragment| lower.contains(fragment))
}

fn normalize_values(values: &[String]) -> Vec<String> {
    values
        .iter()
        .map(|value| value.replace('\\', "/"))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn semantic_config_id(configuration: &BuildConfiguration) -> String {
    let payload = serde_json::json!({
        "include_paths": configuration.include_paths,
        "definitions": configuration.preprocessor_definitions,
        "standard": configuration.language_standard,
        "forced_includes": configuration.forced_includes,
        "generated_headers": configuration.generated_header_paths,
    });
    let digest = Sha256::digest(payload.to_string().as_bytes());
    format!("config:{:x}", digest)[..23].to_string()
}

fn merge_configurations(configurations: Vec<BuildConfiguration>) -> Vec<BuildConfiguration> {
    let mut merged = BTreeMap::<String, BuildConfiguration>::new();
    for mut configuration in configurations {
        merged
            .entry(configuration.config_id.clone())
            .and_modify(|existing| {
                existing
                    .source_files
                    .append(&mut configuration.source_files);
                existing.source_files = normalize_values(&existing.source_files);
            })
            .or_insert(configuration);
    }
    merged.into_values().collect()
}

fn tokenize_command(command: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    for character in command.chars() {
        match (quote, character) {
            (Some(active), c) if c == active => quote = None,
            (None, '"' | '\'') => quote = Some(character),
            (None, c) if c.is_whitespace() => {
                if !current.is_empty() {
                    values.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(character),
        }
    }
    if !current.is_empty() {
        values.push(current);
    }
    values
}

#[derive(Clone, Copy)]
struct ConditionalFrame {
    parent: PreprocessorState,
    branch_taken: Option<bool>,
}

fn combine(parent: PreprocessorState, condition: Option<bool>) -> PreprocessorState {
    match (parent, condition) {
        (PreprocessorState::Inactive, _) | (_, Some(false)) => PreprocessorState::Inactive,
        (PreprocessorState::Active, Some(true)) => PreprocessorState::Active,
        _ => PreprocessorState::UnknownDueToBuildConfig,
    }
}

pub fn evaluate_preprocessor(
    content: &str,
    configuration: &BuildConfiguration,
) -> Vec<PreprocessorState> {
    let macros = configuration.macro_values();
    let line_count = content.lines().count().max(1);
    let mut states = vec![PreprocessorState::Active; line_count + 1];
    let mut current = PreprocessorState::Active;
    let mut stack = Vec::<ConditionalFrame>::new();
    for (offset, line) in content.lines().enumerate() {
        let line_number = offset + 1;
        let trimmed = line.trim_start();
        states[line_number] = current;
        if let Some(expression) = trimmed.strip_prefix("#if ") {
            let condition = evaluate_expression(expression, &macros);
            stack.push(ConditionalFrame {
                parent: current,
                branch_taken: condition,
            });
            current = combine(current, condition);
        } else if let Some(name) = trimmed.strip_prefix("#ifdef ") {
            let condition = Some(macros.contains_key(name.trim()));
            stack.push(ConditionalFrame {
                parent: current,
                branch_taken: condition,
            });
            current = combine(current, condition);
        } else if let Some(name) = trimmed.strip_prefix("#ifndef ") {
            let condition = Some(!macros.contains_key(name.trim()));
            stack.push(ConditionalFrame {
                parent: current,
                branch_taken: condition,
            });
            current = combine(current, condition);
        } else if let Some(expression) = trimmed.strip_prefix("#elif ") {
            if let Some(frame) = stack.last_mut() {
                let condition = match frame.branch_taken {
                    Some(true) => Some(false),
                    Some(false) => evaluate_expression(expression, &macros),
                    None => None,
                };
                if condition == Some(true) {
                    frame.branch_taken = Some(true);
                }
                current = combine(frame.parent, condition);
            }
        } else if trimmed.starts_with("#else") {
            if let Some(frame) = stack.last() {
                current = combine(frame.parent, frame.branch_taken.map(|taken| !taken));
            }
        } else if trimmed.starts_with("#endif")
            && let Some(frame) = stack.pop()
        {
            current = frame.parent;
        }
    }
    states
}

fn evaluate_expression(expression: &str, macros: &HashMap<String, String>) -> Option<bool> {
    let expression = expression
        .trim()
        .trim_matches(|c| c == '(' || c == ')')
        .trim();
    if let Some((left, right)) = expression.split_once("||") {
        return match (
            evaluate_expression(left, macros),
            evaluate_expression(right, macros),
        ) {
            (Some(left), Some(right)) => Some(left || right),
            (Some(true), _) | (_, Some(true)) => Some(true),
            _ => None,
        };
    }
    if let Some((left, right)) = expression.split_once("&&") {
        return match (
            evaluate_expression(left, macros),
            evaluate_expression(right, macros),
        ) {
            (Some(left), Some(right)) => Some(left && right),
            (Some(false), _) | (_, Some(false)) => Some(false),
            _ => None,
        };
    }
    if let Some(inner) = expression.strip_prefix('!') {
        return evaluate_expression(inner, macros).map(|value| !value);
    }
    if let Some(name) = expression
        .strip_prefix("defined(")
        .and_then(|value| value.strip_suffix(')'))
    {
        return Some(macros.contains_key(name.trim()));
    }
    if let Some(name) = expression.strip_prefix("defined ") {
        return Some(macros.contains_key(name.trim()));
    }
    match expression {
        "0" => Some(false),
        "1" => Some(true),
        name if macros.contains_key(name) => macros
            .get(name)
            .and_then(|value| value.parse::<i64>().ok())
            .map(|value| value != 0)
            .or(Some(true)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn preprocessor_states_are_configuration_specific_and_unknown_is_explicit() {
        let debug = BuildConfiguration {
            config_id: "debug".to_string(),
            source: BuildConfigSource::Explicit,
            source_files: Vec::new(),
            include_paths: Vec::new(),
            preprocessor_definitions: vec!["DEBUG=1".to_string()],
            language_standard: Some("c++20".to_string()),
            forced_includes: Vec::new(),
            generated_header_paths: Vec::new(),
        };
        let states = evaluate_preprocessor(
            "#ifdef DEBUG\nint debug_only;\n#else\nint release_only;\n#endif\n#if MAYBE > 2\nint maybe;\n#endif\n",
            &debug,
        );
        assert_eq!(states[2], PreprocessorState::Active);
        assert_eq!(states[4], PreprocessorState::Inactive);
        assert_eq!(states[7], PreprocessorState::UnknownDueToBuildConfig);
    }

    #[test]
    fn compile_commands_extracts_required_build_inputs_and_merges_files() {
        let directory = TempDir::new().unwrap();
        std::fs::create_dir_all(directory.path().join("src")).unwrap();
        std::fs::create_dir_all(directory.path().join("generated")).unwrap();
        for file in ["a.cpp", "b.cpp"] {
            std::fs::write(directory.path().join("src").join(file), "int main() {}\n").unwrap();
        }
        let database = serde_json::json!([
            {
                "directory": directory.path(),
                "file": "src/a.cpp",
                "arguments": ["clang++", "-Igenerated", "-DDEBUG=1", "-std=c++20", "-include", "forced.h", "src/a.cpp"]
            },
            {
                "directory": directory.path(),
                "file": "src/b.cpp",
                "arguments": ["clang++", "-Igenerated", "-DDEBUG=1", "-std=c++20", "-include", "forced.h", "src/b.cpp"]
            }
        ]);
        std::fs::write(
            directory.path().join("compile_commands.json"),
            serde_json::to_vec(&database).unwrap(),
        )
        .unwrap();

        let catalog =
            BuildConfigCatalog::discover(directory.path(), &AnalysisConfig::default()).unwrap();
        assert_eq!(catalog.configurations.len(), 1);
        let configuration = &catalog.configurations[0];
        assert_eq!(configuration.source_files, ["src/a.cpp", "src/b.cpp"]);
        assert_eq!(configuration.preprocessor_definitions, ["DEBUG=1"]);
        assert_eq!(configuration.language_standard.as_deref(), Some("c++20"));
        assert_eq!(configuration.forced_includes, ["forced.h"]);
        assert_eq!(configuration.generated_header_paths, ["generated"]);
    }

    #[test]
    fn explicit_multiple_configurations_have_stable_distinct_ids() {
        let directory = TempDir::new().unwrap();
        let analysis = AnalysisConfig {
            build_configurations: vec![
                ExplicitBuildConfiguration {
                    preprocessor_definitions: vec!["DEBUG=1".to_string()],
                    ..Default::default()
                },
                ExplicitBuildConfiguration {
                    preprocessor_definitions: vec!["RELEASE=1".to_string()],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let first = BuildConfigCatalog::discover(directory.path(), &analysis).unwrap();
        let second = BuildConfigCatalog::discover(directory.path(), &analysis).unwrap();
        assert_eq!(first.config_ids(), second.config_ids());
        assert_eq!(first.configurations.len(), 2);
        assert_ne!(first.config_ids()[0], first.config_ids()[1]);
    }

    #[test]
    fn nested_elif_and_unknown_parent_preserve_conservative_states() {
        let configuration = BuildConfiguration {
            config_id: "debug".to_string(),
            source: BuildConfigSource::Explicit,
            source_files: Vec::new(),
            include_paths: Vec::new(),
            preprocessor_definitions: vec!["DEBUG=1".to_string()],
            language_standard: None,
            forced_includes: Vec::new(),
            generated_header_paths: Vec::new(),
        };
        let states = evaluate_preprocessor(
            "#if UNKNOWN_SWITCH\n#if DEBUG\nint uncertain_debug;\n#endif\n#elif DEBUG\nint known_debug;\n#else\nint fallback;\n#endif\n",
            &configuration,
        );

        assert_eq!(states[3], PreprocessorState::UnknownDueToBuildConfig);
        assert_eq!(states[6], PreprocessorState::UnknownDueToBuildConfig);
        assert_eq!(states[8], PreprocessorState::UnknownDueToBuildConfig);
    }

    #[test]
    fn malformed_database_is_diagnostic_and_explicit_config_still_works() {
        let directory = TempDir::new().unwrap();
        std::fs::write(directory.path().join("compile_commands.json"), "{not-json").unwrap();
        let analysis = AnalysisConfig {
            build_configurations: vec![ExplicitBuildConfiguration {
                config_id: Some("fallback".to_string()),
                preprocessor_definitions: vec!["FALLBACK=1".to_string()],
                ..Default::default()
            }],
            ..Default::default()
        };

        let catalog = BuildConfigCatalog::discover(directory.path(), &analysis).unwrap();
        assert_eq!(catalog.config_ids(), ["fallback"]);
        assert!(
            catalog
                .diagnostics
                .iter()
                .any(|message| message.contains("Invalid compilation database"))
        );
    }

    #[test]
    fn unavailable_selection_is_empty_and_reports_each_missing_id_once() {
        let catalog = BuildConfigCatalog {
            configurations: vec![BuildConfiguration {
                config_id: "debug".to_string(),
                source: BuildConfigSource::Explicit,
                source_files: Vec::new(),
                include_paths: Vec::new(),
                preprocessor_definitions: Vec::new(),
                language_standard: None,
                forced_includes: Vec::new(),
                generated_header_paths: Vec::new(),
            }],
            diagnostics: Vec::new(),
        };

        let selected = catalog.selected(&["missing".to_string(), "missing".to_string()]);
        assert!(selected.configurations.is_empty());
        assert_eq!(selected.diagnostics.len(), 1);
        assert!(selected.diagnostics[0].contains("'missing' is unavailable"));
    }

    #[test]
    fn configuration_scope_keeps_unknown_but_rejects_inactive_and_unscoped() {
        let active = ConfigurationState {
            config_id: "debug".to_string(),
            state: PreprocessorState::Active,
        };
        let inactive = ConfigurationState {
            config_id: "release".to_string(),
            state: PreprocessorState::Inactive,
        };
        let unknown = ConfigurationState {
            config_id: "asan".to_string(),
            state: PreprocessorState::UnknownDueToBuildConfig,
        };

        assert!(configuration_scope_matches(&[], &[]));
        assert!(configuration_scope_matches(
            &[active.clone(), inactive.clone(), unknown.clone()],
            &["debug".to_string()]
        ));
        assert!(configuration_scope_matches(
            &[active.clone(), inactive.clone(), unknown],
            &["asan".to_string()]
        ));
        assert!(!configuration_scope_matches(
            &[active, inactive],
            &["release".to_string()]
        ));
        assert!(!configuration_scope_matches(&[], &["debug".to_string()]));
    }
}
