use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};
use std::path::Path;

use crate::kimi::config::{KimiArtifactConfig, KimiArtifactConfigError};
use crate::kimi::index::{KimiShardIndex, KimiShardIndexError};
use crate::kimi::phase::{KimiMilestonePhase, KIMI_IMPLEMENTED_PHASE};
use crate::kimi::schedule::KimiLayerSchedule;

pub const KIMI_CONFIG_FILENAME: &str = "config.json";
pub const KIMI_SHARD_INDEX_FILENAME: &str = "model.safetensors.index.json";

/// Import modes defined by RFC 0003.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KimiImportMode {
    Inspect,
    Slice,
    Full,
}

/// Future-facing selection surface for slice import.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct KimiImportSelection {
    pub layer_indices: Vec<usize>,
    pub include_embeddings: bool,
    pub include_final_norm: bool,
    pub include_lm_head: bool,
}

/// Planning result for an import request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiImportPlan {
    pub mode: KimiImportMode,
    pub phase: KimiMilestonePhase,
    pub selection: KimiImportSelection,
    pub required_shards: Vec<String>,
}

/// Machine-readable report for the current repository-supported import surface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiImportReport {
    pub phase: KimiMilestonePhase,
    pub ready_modes: Vec<KimiImportMode>,
    pub deferred_modes: Vec<KimiImportMode>,
    pub total_parameters: u64,
    pub total_size_bytes: u64,
    pub tensor_count: usize,
    pub shard_count: usize,
}

/// Validated Phase A artifact-understanding bundle.
#[derive(Debug, Clone, PartialEq)]
pub struct KimiArtifactUnderstanding {
    pub config: KimiArtifactConfig,
    pub layer_schedule: KimiLayerSchedule,
    pub shard_index: KimiShardIndex,
    pub report: KimiImportReport,
}

/// Typed failures for Phase A artifact understanding and import planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KimiImportError {
    Config(KimiArtifactConfigError),
    ShardIndex(KimiShardIndexError),
    ReadFailed {
        path: String,
        detail: String,
    },
    DuplicateSelectedLayer {
        layer_idx: usize,
    },
    SelectedLayerOutOfRange {
        layer_idx: usize,
        num_hidden_layers: usize,
    },
    ModeNotYetImplemented {
        mode: KimiImportMode,
        implemented_phase: KimiMilestonePhase,
        required_phase: KimiMilestonePhase,
        detail: &'static str,
    },
}

impl Display for KimiImportError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(err) => write!(f, "{err}"),
            Self::ShardIndex(err) => write!(f, "{err}"),
            Self::ReadFailed { path, detail } => {
                write!(f, "failed to read Kimi artifacts from '{path}': {detail}")
            }
            Self::DuplicateSelectedLayer { layer_idx } => {
                write!(f, "layer selection contains duplicate layer_idx {layer_idx}")
            }
            Self::SelectedLayerOutOfRange {
                layer_idx,
                num_hidden_layers,
            } => write!(
                f,
                "selected layer_idx ({layer_idx}) must be < num_hidden_layers ({num_hidden_layers})"
            ),
            Self::ModeNotYetImplemented {
                mode,
                implemented_phase,
                required_phase,
                detail,
            } => write!(
                f,
                "{mode} import is not implemented yet: repository is at {implemented_phase}, requested work starts in {required_phase}; {detail}"
            ),
        }
    }
}

impl std::error::Error for KimiImportError {}

impl Display for KimiImportMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Inspect => write!(f, "inspect"),
            Self::Slice => write!(f, "slice"),
            Self::Full => write!(f, "full"),
        }
    }
}

impl From<KimiArtifactConfigError> for KimiImportError {
    fn from(err: KimiArtifactConfigError) -> Self {
        Self::Config(err)
    }
}

impl From<KimiShardIndexError> for KimiImportError {
    fn from(err: KimiShardIndexError) -> Self {
        Self::ShardIndex(err)
    }
}

impl KimiImportSelection {
    pub fn try_validate(&self, num_hidden_layers: usize) -> Result<(), KimiImportError> {
        let mut seen = BTreeSet::new();

        for &layer_idx in &self.layer_indices {
            if layer_idx >= num_hidden_layers {
                return Err(KimiImportError::SelectedLayerOutOfRange {
                    layer_idx,
                    num_hidden_layers,
                });
            }
            if !seen.insert(layer_idx) {
                return Err(KimiImportError::DuplicateSelectedLayer { layer_idx });
            }
        }

        Ok(())
    }
}

impl KimiArtifactUnderstanding {
    pub fn try_from_parts(
        config: KimiArtifactConfig,
        shard_index: KimiShardIndex,
    ) -> Result<Self, KimiImportError> {
        config.try_validate()?;
        shard_index.try_validate()?;

        let layer_schedule = config.try_layer_schedule()?;
        let report = KimiImportReport {
            phase: KIMI_IMPLEMENTED_PHASE,
            ready_modes: vec![KimiImportMode::Inspect],
            deferred_modes: vec![KimiImportMode::Slice, KimiImportMode::Full],
            total_parameters: shard_index.metadata.total_parameter_count,
            total_size_bytes: shard_index.metadata.total_size_bytes,
            tensor_count: shard_index.tensor_count(),
            shard_count: shard_index.shard_count(),
        };

        Ok(Self {
            config,
            layer_schedule,
            shard_index,
            report,
        })
    }

    pub fn load_from_dir<P: AsRef<Path>>(dir: P) -> Result<Self, KimiImportError> {
        let dir = dir.as_ref();
        ensure_dir_exists(dir)?;

        let config = KimiArtifactConfig::load(dir.join(KIMI_CONFIG_FILENAME))?;
        let shard_index = KimiShardIndex::load(dir.join(KIMI_SHARD_INDEX_FILENAME))?;
        Self::try_from_parts(config, shard_index)
    }

    pub fn phase(&self) -> KimiMilestonePhase {
        KIMI_IMPLEMENTED_PHASE
    }

    pub fn inspect_plan(&self) -> KimiImportPlan {
        KimiImportPlan {
            mode: KimiImportMode::Inspect,
            phase: self.phase(),
            selection: KimiImportSelection::default(),
            required_shards: Vec::new(),
        }
    }

    pub fn try_slice_plan(
        &self,
        selection: KimiImportSelection,
    ) -> Result<KimiImportPlan, KimiImportError> {
        selection.try_validate(self.config.num_hidden_layers)?;
        Err(KimiImportError::ModeNotYetImplemented {
            mode: KimiImportMode::Slice,
            implemented_phase: self.phase(),
            required_phase: KimiMilestonePhase::BaselineImplementation,
            detail: "tensor-to-module mapping and shard slicing are deferred to RFC 0002/0003",
        })
    }

    pub fn try_full_plan(&self) -> Result<KimiImportPlan, KimiImportError> {
        Err(KimiImportError::ModeNotYetImplemented {
            mode: KimiImportMode::Full,
            implemented_phase: self.phase(),
            required_phase: KimiMilestonePhase::BaselineImplementation,
            detail: "full checkpoint loading is deferred to RFC 0002/0003",
        })
    }
}

fn ensure_dir_exists(dir: &Path) -> Result<(), KimiImportError> {
    let metadata = std::fs::metadata(dir).map_err(|err| KimiImportError::ReadFailed {
        path: dir.display().to_string(),
        detail: err.to_string(),
    })?;

    if metadata.is_dir() {
        Ok(())
    } else {
        Err(KimiImportError::ReadFailed {
            path: dir.display().to_string(),
            detail: "path is not a directory".to_string(),
        })
    }
}
