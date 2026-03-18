use burn::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::kimi::attn_res_model::KimiAttnResModel;
use crate::kimi::config::KimiArtifactConfig;
use crate::kimi::import::{KimiArtifactUnderstanding, KimiImportSelection};
use crate::kimi::payload::KimiBaselinePayloadError;

/// Explicit warm-start policy for loading baseline Kimi tensors into an
/// AttnRes-Kimi model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KimiAttnResBootstrapPolicy {
    /// Import baseline checkpoint tensors for the matching baseline modules
    /// while leaving every AttnRes operator freshly initialized.
    ///
    /// This is a structural bootstrap only. It is not a numerical-parity mode
    /// and it does not justify quality claims until additional training or
    /// continued pretraining succeeds.
    BaselineImportWithFreshAttnRes,
}

impl Default for KimiAttnResBootstrapPolicy {
    fn default() -> Self {
        Self::BaselineImportWithFreshAttnRes
    }
}

/// Honest status label for AttnRes bootstrap loads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KimiAttnResBootstrapParityStatus {
    StructuralBootstrapOnly,
}

/// Executed summary of one AttnRes bootstrap load.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiAttnResBootstrapReport {
    pub policy: KimiAttnResBootstrapPolicy,
    pub baseline_selection: KimiImportSelection,
    pub imported_baseline_tensor_count: usize,
    pub imported_baseline_module_count: usize,
    pub attn_res_operator_count: usize,
    pub parity_status: KimiAttnResBootstrapParityStatus,
    pub requires_training_before_quality_eval: bool,
    pub notes: Vec<String>,
}

/// Loaded AttnRes model plus the explicit bootstrap report that explains what
/// was and was not initialized from the baseline checkpoint.
#[derive(Debug)]
pub struct KimiAttnResBootstrapLoadResult<B: Backend> {
    pub model: KimiAttnResModel<B>,
    pub report: KimiAttnResBootstrapReport,
}

impl KimiArtifactConfig {
    pub fn default_attn_res_bootstrap_policy(&self) -> KimiAttnResBootstrapPolicy {
        KimiAttnResBootstrapPolicy::default()
    }
}

impl KimiArtifactUnderstanding {
    pub fn try_bootstrap_attn_res_model_from_dir<B: Backend, P: AsRef<Path>>(
        &self,
        dir: P,
        num_blocks: usize,
        selection: KimiImportSelection,
        policy: KimiAttnResBootstrapPolicy,
        device: &B::Device,
    ) -> Result<KimiAttnResBootstrapLoadResult<B>, KimiBaselinePayloadError> {
        let mut model = self.config.try_init_attn_res_model(num_blocks, device)?;
        model.try_load_baseline_payloads_from_dir(self, dir, selection.clone())?;
        let plan = self.try_slice_plan(selection.clone())?;

        Ok(KimiAttnResBootstrapLoadResult {
            model,
            report: KimiAttnResBootstrapReport {
                policy,
                baseline_selection: selection,
                imported_baseline_tensor_count: plan.coverage.mapped_tensors.len(),
                imported_baseline_module_count: plan.coverage.module_coverage.len(),
                attn_res_operator_count: self.config.num_hidden_layers * 2,
                parity_status: KimiAttnResBootstrapParityStatus::StructuralBootstrapOnly,
                requires_training_before_quality_eval: true,
                notes: vec![
                    "Baseline checkpoint tensors were imported only for baseline-compatible modules.".to_string(),
                    "All AttnRes operators remain freshly initialized, including zero pseudo-query vectors.".to_string(),
                    "This load path is a structural bootstrap and must not be treated as baseline parity.".to_string(),
                    "Quality or benchmark claims remain blocked until additional AttnRes training or continued pretraining succeeds.".to_string(),
                ],
            },
        })
    }
}
