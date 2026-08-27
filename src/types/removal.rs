//! Conservative current-tree removal validation contracts.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RemovalVerdict {
    Safe,
    Unsafe,
    Inconclusive,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ValidateRemovalRequest {
    pub symbol_id: String,
    #[serde(default)]
    pub configurations: Vec<String>,
    #[serde(default)]
    pub project: Option<String>,
}

impl ValidateRemovalRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.symbol_id.trim().is_empty() {
            return Err("symbol_id cannot be empty".to_string());
        }
        if self.configurations.iter().any(|id| id.trim().is_empty()) {
            return Err("configurations cannot contain an empty config_id".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RemovalAnalysisScope {
    pub project_id: String,
    pub origin: String,
    pub configurations: Vec<String>,
    pub index_generation: u64,
    pub dependency_model: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RemovalEvidence {
    pub kind: String,
    pub summary: String,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ValidateRemovalResponse {
    pub verdict: RemovalVerdict,
    pub symbol_id: String,
    pub symbol_name: Option<String>,
    pub analysis_scope: RemovalAnalysisScope,
    pub evidence: Vec<RemovalEvidence>,
    pub limitations: Vec<String>,
    pub blocking_references: Vec<crate::relations::ReferenceResult>,
    pub duration_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_serializes_as_explicit_uppercase_contract() {
        assert_eq!(
            serde_json::to_string(&RemovalVerdict::Safe).unwrap(),
            "\"SAFE\""
        );
        assert_eq!(
            serde_json::to_string(&RemovalVerdict::Unsafe).unwrap(),
            "\"UNSAFE\""
        );
        assert_eq!(
            serde_json::to_string(&RemovalVerdict::Inconclusive).unwrap(),
            "\"INCONCLUSIVE\""
        );
    }
}
