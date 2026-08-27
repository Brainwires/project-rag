//! Conservative removal validation over the authoritative current-tree relation store.

use super::RagClient;
use crate::build_config::{BuildConfigCatalog, PreprocessorState, configuration_scope_matches};
use crate::relations::storage::RelationsStore;
use crate::relations::{ReferenceKind, ReferenceResult, ResolutionStatus};
use crate::types::{
    RemovalAnalysisScope, RemovalEvidence, RemovalVerdict, ValidateRemovalRequest,
    ValidateRemovalResponse,
};
use anyhow::Result;
use std::path::Path;
use std::time::Instant;

fn verdict_for_evidence(
    resolved: usize,
    uncertain: usize,
    limitations: &[String],
) -> RemovalVerdict {
    if resolved > 0 {
        RemovalVerdict::Unsafe
    } else if uncertain > 0 || !limitations.is_empty() {
        RemovalVerdict::Inconclusive
    } else {
        RemovalVerdict::Safe
    }
}

fn is_compiled_language(language: &str) -> bool {
    matches!(
        language.to_ascii_lowercase().as_str(),
        "c" | "c++" | "objective-c"
    )
}

impl RagClient {
    pub async fn validate_removal(
        &self,
        request: ValidateRemovalRequest,
    ) -> Result<ValidateRemovalResponse> {
        let started = Instant::now();
        request.validate().map_err(anyhow::Error::msg)?;

        let candidates = {
            let cache = self.hash_cache.read().await;
            cache
                .roots
                .keys()
                .filter(|root| {
                    request
                        .project
                        .as_deref()
                        .is_none_or(|project| cache.project_id(root) == Some(project))
                })
                .cloned()
                .collect::<Vec<_>>()
        };
        for root in &candidates {
            self.check_path_not_dirty(Some(root)).await?;
        }
        let mut matches = Vec::new();
        for root in candidates {
            let definitions = self
                .relations_store
                .find_definitions_by_symbol_id_in_root(&request.symbol_id, &root)
                .await?;
            if !definitions.is_empty() {
                matches.push((root, definitions));
            }
        }
        if matches.len() > 1 {
            anyhow::bail!("symbol_id matches multiple project roots; specify project");
        }
        let Some((root, definitions)) = matches.pop() else {
            anyhow::bail!("symbol_id was not found in the current authoritative index");
        };
        let generation = self.hash_cache.read().await.generation(&root);
        let project_id = self
            .hash_cache
            .read()
            .await
            .project_id(&root)
            .unwrap_or_default()
            .to_string();
        let symbol_name = definitions
            .first()
            .map(|definition| definition.name().to_string());
        let language = definitions
            .first()
            .map(|definition| definition.symbol_id.language.clone())
            .unwrap_or_default();

        let catalog = BuildConfigCatalog::discover(Path::new(&root), &self.config.analysis)?
            .selected(&request.configurations);
        let analyzed_configurations = catalog.config_ids();
        let reference_scope = if request.configurations.is_empty() {
            analyzed_configurations.clone()
        } else {
            request.configurations.clone()
        };
        let mut limitations = catalog
            .diagnostics
            .iter()
            .filter(|message| message.contains("Invalid") || message.contains("unavailable"))
            .cloned()
            .collect::<Vec<_>>();
        if is_compiled_language(&language) && catalog.configurations.is_empty() {
            limitations
                .push("no matching compiled-language build configuration was analyzed".to_string());
        }
        if is_compiled_language(&language)
            && definitions.iter().any(|definition| {
                catalog
                    .configurations_for_file(definition.file_path())
                    .is_empty()
            })
        {
            limitations.push(
                "at least one definition file has no matching build configuration".to_string(),
            );
        }

        for definition in &definitions {
            if let Ok(content) =
                std::fs::read_to_string(Path::new(&root).join(definition.file_path()))
            {
                let states = catalog.states_for_file(definition.file_path(), &content);
                if states.get(definition.start_line()).is_some_and(|line| {
                    line.iter()
                        .any(|state| state.state == PreprocessorState::UnknownDueToBuildConfig)
                }) {
                    limitations.push(format!(
                        "definition in '{}' has unknown preprocessor state",
                        definition.file_path()
                    ));
                }
            }
        }

        let mut references = if let Some(name) = &symbol_name {
            self.relations_store
                .find_references_by_name_in_root(name, &root)
                .await?
        } else {
            Vec::new()
        };
        references.retain(|reference| {
            reference.reference_kind.is_code()
                && !matches!(
                    reference.reference_kind,
                    ReferenceKind::Definition | ReferenceKind::Declaration
                )
                && (reference.target_symbol_id == request.symbol_id
                    || reference
                        .candidates
                        .iter()
                        .any(|candidate| candidate.symbol_id == request.symbol_id))
                && configuration_scope_matches(&reference.configuration_states, &reference_scope)
        });
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
        let resolved = references
            .iter()
            .filter(|reference| reference.resolution_status == ResolutionStatus::Resolved)
            .count();
        let uncertain = references.len().saturating_sub(resolved);

        let mut generated_or_forced = catalog.configurations.iter().any(|configuration| {
            !configuration.forced_includes.is_empty()
                || !configuration.generated_header_paths.is_empty()
        });
        let mut dynamic_wiring = false;
        let mut wiring_patterns = vec![
            "register(".to_string(),
            "register_".to_string(),
            "reflection".to_string(),
            "Q_OBJECT".to_string(),
            "plugin".to_string(),
            "dynamic_cast".to_string(),
        ];
        wiring_patterns.extend(self.config.analysis.dynamic_wiring_patterns.clone());
        let (indexed_paths, generated_patterns) = {
            let cache = self.hash_cache.read().await;
            (
                cache
                    .get_root(&root)
                    .map(|files| files.keys().cloned().collect::<Vec<_>>())
                    .unwrap_or_default(),
                self.config.analysis.generated_path_patterns.clone(),
            )
        };
        for relative in indexed_paths {
            let lower = relative.replace('\\', "/").to_ascii_lowercase();
            generated_or_forced |= generated_patterns
                .iter()
                .any(|pattern| lower.contains(&pattern.replace('\\', "/").to_ascii_lowercase()));
            if !dynamic_wiring
                && let Ok(content) = std::fs::read_to_string(Path::new(&root).join(&relative))
            {
                dynamic_wiring = wiring_patterns
                    .iter()
                    .any(|pattern| !pattern.is_empty() && content.contains(pattern));
            }
        }
        if generated_or_forced {
            limitations.push(
                "generated code, generated headers, or forced includes require generator/build validation"
                    .to_string(),
            );
        }
        if dynamic_wiring {
            limitations.push(
                "dynamic registration, reflection, or plugin wiring may create non-textual references"
                    .to_string(),
            );
        }
        if references.iter().any(|reference| {
            matches!(
                reference.dispatch_kind,
                crate::relations::DispatchKind::Indirect
                    | crate::relations::DispatchKind::Callback
                    | crate::relations::DispatchKind::Dynamic
                    | crate::relations::DispatchKind::Unknown
            ) && reference.resolution_status != ResolutionStatus::Resolved
        }) {
            limitations.push("indirect or dynamic dispatch target set is incomplete".to_string());
        }
        limitations.sort();
        limitations.dedup();

        let verdict = verdict_for_evidence(resolved, uncertain, &limitations);
        let mut evidence = vec![
            RemovalEvidence {
                kind: "resolved_current_tree_references".to_string(),
                summary:
                    "Resolved current-tree code dependencies in the selected configuration scope"
                        .to_string(),
                count: resolved,
            },
            RemovalEvidence {
                kind: "ambiguous_or_unresolved_references".to_string(),
                summary:
                    "Candidate-bearing dependencies that cannot establish one authoritative target"
                        .to_string(),
                count: uncertain,
            },
        ];
        if generated_or_forced || dynamic_wiring {
            evidence.push(RemovalEvidence {
                kind: "non_source_wiring".to_string(),
                summary: "Generated/configuration/dynamic wiring detected in current project"
                    .to_string(),
                count: usize::from(generated_or_forced) + usize::from(dynamic_wiring),
            });
        }

        Ok(ValidateRemovalResponse {
            verdict,
            symbol_id: request.symbol_id,
            symbol_name,
            analysis_scope: RemovalAnalysisScope {
                project_id,
                origin: "current".to_string(),
                configurations: analyzed_configurations,
                index_generation: generation,
                dependency_model: vec![
                    "resolved/ambiguous/unresolved references".to_string(),
                    "preprocessor and build configurations".to_string(),
                    "generated/forced includes".to_string(),
                    "dynamic registration and indirect dispatch".to_string(),
                ],
            },
            evidence,
            limitations,
            blocking_references: references.iter().map(ReferenceResult::from).collect(),
            duration_ms: started.elapsed().as_millis() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_precedence_is_unsafe_then_inconclusive_then_safe() {
        assert_eq!(
            verdict_for_evidence(1, 2, &["gap".to_string()]),
            RemovalVerdict::Unsafe
        );
        assert_eq!(
            verdict_for_evidence(0, 1, &[]),
            RemovalVerdict::Inconclusive
        );
        assert_eq!(
            verdict_for_evidence(0, 0, &["gap".to_string()]),
            RemovalVerdict::Inconclusive
        );
        assert_eq!(verdict_for_evidence(0, 0, &[]), RemovalVerdict::Safe);
    }

    #[test]
    #[ignore = "manual M5 removal-decision throughput measurement"]
    fn benchmark_m5_removal_decision() {
        let started = std::time::Instant::now();
        let mut safe = 0;
        let gap = ["gap".to_string()];
        for index in 0..100_000 {
            let limitations = if index % 10 == 0 { gap.as_slice() } else { &[] };
            if verdict_for_evidence(0, 0, limitations) == RemovalVerdict::Safe {
                safe += 1;
            }
        }
        println!(
            "m5 removal decision: evaluations=100000 safe={} elapsed_us={}",
            safe,
            started.elapsed().as_micros()
        );
        assert_eq!(safe, 90_000);
    }
}
