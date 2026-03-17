use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

use crate::kimi::config::{KimiArtifactConfig, KimiArtifactConfigError};
use crate::kimi::index::{
    KimiShardIndex, KimiShardIndexError, KimiTensorLocator, KimiTensorLocatorError,
};
use crate::kimi::phase::{KimiMilestonePhase, KIMI_ARTIFACT_UNDERSTANDING_PHASE};
use crate::kimi::schedule::{KimiAttentionLayerKind, KimiFeedForwardLayerKind, KimiLayerSchedule};

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

/// Runtime dtype for the local RFC 0002/RFC 0003 checkpoint path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KimiImportRuntimeDtype {
    Float32,
    BFloat16,
}

/// Explicit artifact-to-runtime dtype action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KimiImportDtypeAction {
    Preserve,
    PromoteToF32,
}

/// Public dtype policy for Kimi checkpoint planning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiImportDtypePolicy {
    pub artifact_dtype: String,
    pub runtime_dtype: KimiImportRuntimeDtype,
    pub action: KimiImportDtypeAction,
}

/// Addressable submodule surface for the current local Kimi baseline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KimiModuleRef {
    Embeddings,
    FinalNorm,
    LmHead,
    DecoderLayer {
        layer_idx: usize,
        component: KimiLayerModuleRef,
    },
}

/// Layer-local module surface for the current baseline Kimi scaffold.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KimiLayerModuleRef {
    InputNorm,
    Attention { kind: KimiAttentionLayerKind },
    PostAttentionNorm,
    FeedForward { kind: KimiFeedForwardLayerKind },
}

/// Typed reasons for unsupported public tensors on the current local baseline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KimiUnsupportedTensorReason {
    UnsupportedTensorNamePattern,
    TensorDoesNotMatchAttentionKind {
        expected: KimiAttentionLayerKind,
    },
    TensorDoesNotMatchFeedForwardKind {
        expected: KimiFeedForwardLayerKind,
    },
    MlaKvLatentProjectionContainsUnsupportedRopeRows,
    MlaKvLayerNormUnsupported,
    MlaKvUpProjectionUnsupported,
    MlaLowRankQueryImportDeferred,
    KdaAuxiliaryTensorUnsupported,
    MoeRouterBiasCorrectionUnsupported,
    SharedExpertsPackingUnsupported {
        num_shared_experts: usize,
    },
    ExpertIndexOutOfRange {
        expert_idx: usize,
        num_experts: usize,
    },
}

/// Support classification for one tensor name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KimiTensorMappingStatus {
    SupportedDirect,
    Unsupported { reason: KimiUnsupportedTensorReason },
}

/// Explicit tensor-to-module mapping for the current baseline Kimi scaffold.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiTensorMapping {
    pub tensor_name: String,
    pub shard_path: String,
    pub module: Option<KimiModuleRef>,
    pub local_parameter_paths: Vec<String>,
    pub status: KimiTensorMappingStatus,
}

/// Missing supported tensor for a selected module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiMissingTensor {
    pub tensor_name: String,
    pub module: KimiModuleRef,
    pub local_parameter_paths: Vec<String>,
}

/// Duplicate supported tensor ownership inside the local mapping table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiDuplicateTensor {
    pub tensor_name: String,
    pub modules: Vec<KimiModuleRef>,
}

/// Coverage summary for one selected module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiModuleCoverage {
    pub module: KimiModuleRef,
    pub required_tensors: Vec<String>,
    pub mapped_tensors: Vec<String>,
    pub missing_tensors: Vec<String>,
    pub unsupported_tensors: Vec<String>,
}

/// Machine-readable coverage result for a plan or inspection pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiImportCoverageReport {
    pub mode: KimiImportMode,
    pub selection: KimiImportSelection,
    pub selected_modules: Vec<KimiModuleRef>,
    pub required_shards: Vec<String>,
    pub dtype_policy: KimiImportDtypePolicy,
    pub mapped_tensors: Vec<KimiTensorMapping>,
    pub unsupported_tensors: Vec<KimiTensorMapping>,
    pub missing_tensors: Vec<KimiMissingTensor>,
    pub duplicate_tensors: Vec<KimiDuplicateTensor>,
    pub module_coverage: Vec<KimiModuleCoverage>,
}

/// Typed failure when a plan/report is not fully loadable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KimiImportCoverageError {
    DuplicateRequiredTensor {
        tensor_name: String,
        modules: Vec<KimiModuleRef>,
    },
    MissingRequiredTensor {
        tensor_name: String,
        module: KimiModuleRef,
    },
    UnsupportedTensor {
        tensor_name: String,
        module: Option<KimiModuleRef>,
        reason: KimiUnsupportedTensorReason,
    },
}

impl Display for KimiImportCoverageError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateRequiredTensor {
                tensor_name,
                modules,
            } => write!(
                f,
                "tensor '{tensor_name}' is claimed by multiple local modules: {:?}",
                modules
            ),
            Self::MissingRequiredTensor { tensor_name, module } => write!(
                f,
                "required tensor '{tensor_name}' for module {module:?} is missing from the shard index"
            ),
            Self::UnsupportedTensor {
                tensor_name,
                module,
                reason,
            } => write!(
                f,
                "tensor '{tensor_name}' is not loadable for module {module:?}: {reason:?}"
            ),
        }
    }
}

impl std::error::Error for KimiImportCoverageError {}

/// Planning result for an import request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiImportPlan {
    pub mode: KimiImportMode,
    pub phase: KimiMilestonePhase,
    pub selection: KimiImportSelection,
    pub selected_modules: Vec<KimiModuleRef>,
    pub required_shards: Vec<String>,
    pub dtype_policy: KimiImportDtypePolicy,
    pub coverage: KimiImportCoverageReport,
}

/// Machine-readable report for the repository-backed RFC 0003 planning surface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiImportReport {
    pub phase: KimiMilestonePhase,
    pub ready_modes: Vec<KimiImportMode>,
    pub deferred_modes: Vec<KimiImportMode>,
    pub payload_loading_implemented: bool,
    pub total_parameters: u64,
    pub total_size_bytes: u64,
    pub tensor_count: usize,
    pub shard_count: usize,
    pub dtype_policy: KimiImportDtypePolicy,
    pub coverage: KimiImportCoverageReport,
}

/// Resolved shard path for a planned import.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiResolvedShard {
    pub shard_path: String,
    pub resolved_path: PathBuf,
}

/// Filesystem resolver for shard plans. This resolves paths only; it does not load tensors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiShardResolver {
    root_dir: PathBuf,
}

/// Typed failures for shard-path resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KimiShardResolverError {
    MissingShardFile {
        shard_path: String,
        resolved_path: String,
    },
}

impl Display for KimiShardResolverError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingShardFile {
                shard_path,
                resolved_path,
            } => write!(
                f,
                "required shard '{shard_path}' does not exist at '{resolved_path}'"
            ),
        }
    }
}

impl std::error::Error for KimiShardResolverError {}

/// Validated Phase A artifact-understanding bundle plus RFC 0003 planning/reporting.
#[derive(Debug, Clone, PartialEq)]
pub struct KimiArtifactUnderstanding {
    pub config: KimiArtifactConfig,
    pub layer_schedule: KimiLayerSchedule,
    pub shard_index: KimiShardIndex,
    pub report: KimiImportReport,
}

/// Typed failures for artifact understanding and RFC 0003 planning.
#[derive(Debug, Clone, PartialEq)]
pub enum KimiImportError {
    Config(KimiArtifactConfigError),
    ShardIndex(KimiShardIndexError),
    TensorLocator(KimiTensorLocatorError),
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
    UnsupportedArtifactDtype {
        dtype: String,
    },
}

impl Display for KimiImportError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(err) => write!(f, "{err}"),
            Self::ShardIndex(err) => write!(f, "{err}"),
            Self::TensorLocator(err) => write!(f, "{err}"),
            Self::ReadFailed { path, detail } => {
                write!(f, "failed to read Kimi artifacts from '{path}': {detail}")
            }
            Self::DuplicateSelectedLayer { layer_idx } => {
                write!(
                    f,
                    "layer selection contains duplicate layer_idx {layer_idx}"
                )
            }
            Self::SelectedLayerOutOfRange {
                layer_idx,
                num_hidden_layers,
            } => write!(
                f,
                "selected layer_idx ({layer_idx}) must be < num_hidden_layers ({num_hidden_layers})"
            ),
            Self::UnsupportedArtifactDtype { dtype } => write!(
                f,
                "artifact dtype '{dtype}' is not supported by the RFC 0003 import scaffolding"
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

impl From<KimiTensorLocatorError> for KimiImportError {
    fn from(err: KimiTensorLocatorError) -> Self {
        Self::TensorLocator(err)
    }
}

impl KimiImportSelection {
    pub fn full(num_hidden_layers: usize) -> Self {
        Self {
            layer_indices: (0..num_hidden_layers).collect(),
            include_embeddings: true,
            include_final_norm: true,
            include_lm_head: true,
        }
    }

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

impl KimiImportDtypePolicy {
    pub fn for_artifact_dtype(dtype: &str) -> Result<Self, KimiImportError> {
        match dtype {
            "bfloat16" => Ok(Self {
                artifact_dtype: dtype.to_string(),
                runtime_dtype: KimiImportRuntimeDtype::Float32,
                action: KimiImportDtypeAction::PromoteToF32,
            }),
            "float32" => Ok(Self {
                artifact_dtype: dtype.to_string(),
                runtime_dtype: KimiImportRuntimeDtype::Float32,
                action: KimiImportDtypeAction::Preserve,
            }),
            other => Err(KimiImportError::UnsupportedArtifactDtype {
                dtype: other.to_string(),
            }),
        }
    }
}

impl KimiImportCoverageReport {
    pub fn is_fully_loadable(&self) -> bool {
        self.duplicate_tensors.is_empty()
            && self.missing_tensors.is_empty()
            && self.unsupported_tensors.is_empty()
    }

    pub fn try_require_loadable(&self) -> Result<(), KimiImportCoverageError> {
        if let Some(duplicate) = self.duplicate_tensors.first() {
            return Err(KimiImportCoverageError::DuplicateRequiredTensor {
                tensor_name: duplicate.tensor_name.clone(),
                modules: duplicate.modules.clone(),
            });
        }

        if let Some(missing) = self.missing_tensors.first() {
            return Err(KimiImportCoverageError::MissingRequiredTensor {
                tensor_name: missing.tensor_name.clone(),
                module: missing.module.clone(),
            });
        }

        if let Some(unsupported) = self.unsupported_tensors.first() {
            let KimiTensorMappingStatus::Unsupported { reason } = &unsupported.status else {
                unreachable!("unsupported_tensors must only contain unsupported entries");
            };
            return Err(KimiImportCoverageError::UnsupportedTensor {
                tensor_name: unsupported.tensor_name.clone(),
                module: unsupported.module.clone(),
                reason: reason.clone(),
            });
        }

        Ok(())
    }
}

impl KimiImportPlan {
    pub fn is_fully_loadable(&self) -> bool {
        self.coverage.is_fully_loadable()
    }

    pub fn try_require_loadable(&self) -> Result<(), KimiImportCoverageError> {
        self.coverage.try_require_loadable()
    }
}

impl KimiShardResolver {
    pub fn new<P: AsRef<Path>>(root_dir: P) -> Self {
        Self {
            root_dir: root_dir.as_ref().to_path_buf(),
        }
    }

    pub fn resolve_shard(&self, shard_path: &str) -> KimiResolvedShard {
        KimiResolvedShard {
            shard_path: shard_path.to_string(),
            resolved_path: self.root_dir.join(shard_path),
        }
    }

    pub fn try_resolve_plan(
        &self,
        plan: &KimiImportPlan,
    ) -> Result<Vec<KimiResolvedShard>, KimiShardResolverError> {
        plan.required_shards
            .iter()
            .map(|shard_path| {
                let resolved = self.resolve_shard(shard_path);
                if resolved.resolved_path.is_file() {
                    Ok(resolved)
                } else {
                    Err(KimiShardResolverError::MissingShardFile {
                        shard_path: shard_path.clone(),
                        resolved_path: resolved.resolved_path.display().to_string(),
                    })
                }
            })
            .collect()
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
        let full_selection = KimiImportSelection::full(config.num_hidden_layers);
        let dtype_policy = KimiImportDtypePolicy::for_artifact_dtype(&config.dtype)?;
        let coverage = build_coverage_report(
            &config,
            &layer_schedule,
            &shard_index,
            KimiImportMode::Inspect,
            full_selection.clone(),
            false,
            &dtype_policy,
        );
        let report = KimiImportReport {
            phase: KIMI_ARTIFACT_UNDERSTANDING_PHASE,
            ready_modes: vec![
                KimiImportMode::Inspect,
                KimiImportMode::Slice,
                KimiImportMode::Full,
            ],
            deferred_modes: Vec::new(),
            payload_loading_implemented: false,
            total_parameters: shard_index.metadata.total_parameter_count,
            total_size_bytes: shard_index.metadata.total_size_bytes,
            tensor_count: shard_index.tensor_count(),
            shard_count: shard_index.shard_count(),
            dtype_policy: dtype_policy.clone(),
            coverage,
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
        KIMI_ARTIFACT_UNDERSTANDING_PHASE
    }

    pub fn inspect_plan(&self) -> KimiImportPlan {
        let selection = KimiImportSelection::full(self.config.num_hidden_layers);
        KimiImportPlan {
            mode: KimiImportMode::Inspect,
            phase: self.phase(),
            selection,
            selected_modules: self.report.coverage.selected_modules.clone(),
            required_shards: Vec::new(),
            dtype_policy: self.report.dtype_policy.clone(),
            coverage: self.report.coverage.clone(),
        }
    }

    pub fn try_slice_plan(
        &self,
        selection: KimiImportSelection,
    ) -> Result<KimiImportPlan, KimiImportError> {
        selection.try_validate(self.config.num_hidden_layers)?;
        self.build_plan(KimiImportMode::Slice, selection, true)
    }

    pub fn try_full_plan(&self) -> Result<KimiImportPlan, KimiImportError> {
        let selection = KimiImportSelection::full(self.config.num_hidden_layers);
        self.build_plan(KimiImportMode::Full, selection, true)
    }

    fn build_plan(
        &self,
        mode: KimiImportMode,
        selection: KimiImportSelection,
        include_required_shards: bool,
    ) -> Result<KimiImportPlan, KimiImportError> {
        let dtype_policy = KimiImportDtypePolicy::for_artifact_dtype(&self.config.dtype)?;
        let coverage = build_coverage_report(
            &self.config,
            &self.layer_schedule,
            &self.shard_index,
            mode,
            selection.clone(),
            include_required_shards,
            &dtype_policy,
        );

        Ok(KimiImportPlan {
            mode,
            phase: self.phase(),
            selection,
            selected_modules: coverage.selected_modules.clone(),
            required_shards: coverage.required_shards.clone(),
            dtype_policy,
            coverage,
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

#[derive(Debug, Clone)]
struct KimiSelectionScope {
    selected_layers: BTreeSet<usize>,
    include_embeddings: bool,
    include_final_norm: bool,
    include_lm_head: bool,
}

impl KimiSelectionScope {
    fn from_selection(selection: &KimiImportSelection) -> Self {
        Self {
            selected_layers: selection.layer_indices.iter().copied().collect(),
            include_embeddings: selection.include_embeddings,
            include_final_norm: selection.include_final_norm,
            include_lm_head: selection.include_lm_head,
        }
    }

    fn includes_layer(&self, layer_idx: usize) -> bool {
        self.selected_layers.contains(&layer_idx)
    }
}

#[derive(Debug, Clone)]
struct KimiPlannedTensor {
    tensor_name: String,
    module: KimiModuleRef,
    local_parameter_paths: Vec<String>,
}

fn build_coverage_report(
    config: &KimiArtifactConfig,
    layer_schedule: &KimiLayerSchedule,
    shard_index: &KimiShardIndex,
    mode: KimiImportMode,
    selection: KimiImportSelection,
    include_required_shards: bool,
    dtype_policy: &KimiImportDtypePolicy,
) -> KimiImportCoverageReport {
    let locator = KimiTensorLocator::from_index(shard_index);
    let selected_modules = selected_modules_for_selection(&selection, layer_schedule);
    let planned_tensors = selected_modules
        .iter()
        .flat_map(|module| planned_tensors_for_module(config, module))
        .collect::<Vec<_>>();

    let mut tensor_to_modules: BTreeMap<String, Vec<KimiModuleRef>> = BTreeMap::new();
    for planned in &planned_tensors {
        tensor_to_modules
            .entry(planned.tensor_name.clone())
            .or_default()
            .push(planned.module.clone());
    }

    let duplicate_tensors = tensor_to_modules
        .iter()
        .filter(|(_, modules)| modules.len() > 1)
        .map(|(tensor_name, modules)| KimiDuplicateTensor {
            tensor_name: tensor_name.clone(),
            modules: modules.clone(),
        })
        .collect::<Vec<_>>();

    let mut mapped_tensors = Vec::new();
    let mut missing_tensors = Vec::new();
    let mut required_tensors_by_module: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut mapped_tensors_by_module: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut missing_tensors_by_module: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for planned in &planned_tensors {
        let module_key = format!("{:?}", planned.module);
        required_tensors_by_module
            .entry(module_key.clone())
            .or_default()
            .push(planned.tensor_name.clone());

        match locator.shard_for_tensor(&planned.tensor_name) {
            Some(shard_path) => {
                mapped_tensors.push(KimiTensorMapping {
                    tensor_name: planned.tensor_name.clone(),
                    shard_path: shard_path.to_string(),
                    module: Some(planned.module.clone()),
                    local_parameter_paths: planned.local_parameter_paths.clone(),
                    status: KimiTensorMappingStatus::SupportedDirect,
                });
                mapped_tensors_by_module
                    .entry(module_key)
                    .or_default()
                    .push(planned.tensor_name.clone());
            }
            None => {
                missing_tensors.push(KimiMissingTensor {
                    tensor_name: planned.tensor_name.clone(),
                    module: planned.module.clone(),
                    local_parameter_paths: planned.local_parameter_paths.clone(),
                });
                missing_tensors_by_module
                    .entry(module_key)
                    .or_default()
                    .push(planned.tensor_name.clone());
            }
        }
    }

    let supported_tensor_names = planned_tensors
        .iter()
        .map(|planned| planned.tensor_name.as_str())
        .collect::<BTreeSet<_>>();
    let selection_scope = KimiSelectionScope::from_selection(&selection);
    let mut unsupported_tensors = Vec::new();
    let mut unsupported_tensors_by_module: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for (tensor_name, shard_path) in locator.weight_map() {
        if supported_tensor_names.contains(tensor_name.as_str()) {
            continue;
        }

        if let Some(mapping) = classify_selected_tensor(
            config,
            layer_schedule,
            &selection_scope,
            tensor_name,
            shard_path,
        ) {
            if let Some(module) = &mapping.module {
                unsupported_tensors_by_module
                    .entry(format!("{module:?}"))
                    .or_default()
                    .push(mapping.tensor_name.clone());
            }
            unsupported_tensors.push(mapping);
        }
    }

    let required_shards = if include_required_shards {
        mapped_tensors
            .iter()
            .map(|mapping| mapping.shard_path.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    } else {
        Vec::new()
    };

    let module_coverage = selected_modules
        .iter()
        .map(|module| {
            let key = format!("{module:?}");
            KimiModuleCoverage {
                module: module.clone(),
                required_tensors: required_tensors_by_module.remove(&key).unwrap_or_default(),
                mapped_tensors: mapped_tensors_by_module.remove(&key).unwrap_or_default(),
                missing_tensors: missing_tensors_by_module.remove(&key).unwrap_or_default(),
                unsupported_tensors: unsupported_tensors_by_module
                    .remove(&key)
                    .unwrap_or_default(),
            }
        })
        .collect();

    KimiImportCoverageReport {
        mode,
        selection,
        selected_modules,
        required_shards,
        dtype_policy: dtype_policy.clone(),
        mapped_tensors,
        unsupported_tensors,
        missing_tensors,
        duplicate_tensors,
        module_coverage,
    }
}

fn selected_modules_for_selection(
    selection: &KimiImportSelection,
    layer_schedule: &KimiLayerSchedule,
) -> Vec<KimiModuleRef> {
    let mut modules = Vec::new();

    if selection.include_embeddings {
        modules.push(KimiModuleRef::Embeddings);
    }

    for &layer_idx in &selection.layer_indices {
        let scheduled = layer_schedule
            .try_layer(layer_idx)
            .expect("validated selections must point to real layers");
        modules.push(KimiModuleRef::DecoderLayer {
            layer_idx,
            component: KimiLayerModuleRef::InputNorm,
        });
        modules.push(KimiModuleRef::DecoderLayer {
            layer_idx,
            component: KimiLayerModuleRef::Attention {
                kind: scheduled.attention_kind,
            },
        });
        modules.push(KimiModuleRef::DecoderLayer {
            layer_idx,
            component: KimiLayerModuleRef::PostAttentionNorm,
        });
        modules.push(KimiModuleRef::DecoderLayer {
            layer_idx,
            component: KimiLayerModuleRef::FeedForward {
                kind: scheduled.feed_forward_kind,
            },
        });
    }

    if selection.include_final_norm {
        modules.push(KimiModuleRef::FinalNorm);
    }

    if selection.include_lm_head {
        modules.push(KimiModuleRef::LmHead);
    }

    modules
}

fn planned_tensors_for_module(
    config: &KimiArtifactConfig,
    module: &KimiModuleRef,
) -> Vec<KimiPlannedTensor> {
    match module {
        KimiModuleRef::Embeddings => vec![planned_tensor(
            "model.embed_tokens.weight",
            module,
            vec!["embedding.weight".to_string()],
        )],
        KimiModuleRef::FinalNorm => vec![planned_tensor(
            "model.norm.weight",
            module,
            vec!["final_norm.gamma".to_string()],
        )],
        KimiModuleRef::LmHead => vec![planned_tensor(
            "lm_head.weight",
            module,
            vec!["lm_head.weight".to_string()],
        )],
        KimiModuleRef::DecoderLayer {
            layer_idx,
            component,
        } => match component {
            KimiLayerModuleRef::InputNorm => vec![planned_tensor(
                format!("model.layers.{layer_idx}.input_layernorm.weight"),
                module,
                vec![format!("layers[{layer_idx}].input_norm.gamma")],
            )],
            KimiLayerModuleRef::Attention { kind } => match kind {
                KimiAttentionLayerKind::FullAttention => vec![
                    planned_tensor(
                        format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                        module,
                        vec![format!("layers[{layer_idx}].attention.q_proj.weight")],
                    ),
                    planned_tensor(
                        format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                        module,
                        vec![format!("layers[{layer_idx}].attention.out_proj.weight")],
                    ),
                ],
                KimiAttentionLayerKind::LinearAttentionKda => vec![
                    planned_tensor(
                        format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                        module,
                        vec![format!("layers[{layer_idx}].attention.q_proj.weight")],
                    ),
                    planned_tensor(
                        format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                        module,
                        vec![format!("layers[{layer_idx}].attention.k_proj.weight")],
                    ),
                    planned_tensor(
                        format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                        module,
                        vec![format!("layers[{layer_idx}].attention.v_proj.weight")],
                    ),
                    planned_tensor(
                        format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                        module,
                        vec![format!("layers[{layer_idx}].attention.out_proj.weight")],
                    ),
                ],
            },
            KimiLayerModuleRef::PostAttentionNorm => vec![planned_tensor(
                format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                module,
                vec![format!("layers[{layer_idx}].post_attention_norm.gamma")],
            )],
            KimiLayerModuleRef::FeedForward { kind } => match kind {
                KimiFeedForwardLayerKind::DenseMlp => vec![
                    planned_tensor(
                        format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                        module,
                        vec![format!(
                            "layers[{layer_idx}].feed_forward.inner.gate_proj.weight"
                        )],
                    ),
                    planned_tensor(
                        format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                        module,
                        vec![format!(
                            "layers[{layer_idx}].feed_forward.inner.up_proj.weight"
                        )],
                    ),
                    planned_tensor(
                        format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                        module,
                        vec![format!(
                            "layers[{layer_idx}].feed_forward.inner.down_proj.weight"
                        )],
                    ),
                ],
                KimiFeedForwardLayerKind::SparseMoe => {
                    let mut planned = vec![planned_tensor(
                        format!("model.layers.{layer_idx}.block_sparse_moe.gate.weight"),
                        module,
                        vec![format!("layers[{layer_idx}].feed_forward.router.weight")],
                    )];

                    if config.num_shared_experts == 1 {
                        planned.extend([
                            planned_tensor(
                                format!(
                                    "model.layers.{layer_idx}.block_sparse_moe.shared_experts.gate_proj.weight"
                                ),
                                module,
                                vec![format!(
                                    "layers[{layer_idx}].feed_forward.shared_experts[0].gate_proj.weight"
                                )],
                            ),
                            planned_tensor(
                                format!(
                                    "model.layers.{layer_idx}.block_sparse_moe.shared_experts.up_proj.weight"
                                ),
                                module,
                                vec![format!(
                                    "layers[{layer_idx}].feed_forward.shared_experts[0].up_proj.weight"
                                )],
                            ),
                            planned_tensor(
                                format!(
                                    "model.layers.{layer_idx}.block_sparse_moe.shared_experts.down_proj.weight"
                                ),
                                module,
                                vec![format!(
                                    "layers[{layer_idx}].feed_forward.shared_experts[0].down_proj.weight"
                                )],
                            ),
                        ]);
                    }

                    for expert_idx in 0..config.num_experts {
                        planned.extend([
                            planned_tensor(
                                format!(
                                    "model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight"
                                ),
                                module,
                                vec![format!(
                                    "layers[{layer_idx}].feed_forward.experts[{expert_idx}].gate_proj.weight"
                                )],
                            ),
                            planned_tensor(
                                format!(
                                    "model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight"
                                ),
                                module,
                                vec![format!(
                                    "layers[{layer_idx}].feed_forward.experts[{expert_idx}].down_proj.weight"
                                )],
                            ),
                            planned_tensor(
                                format!(
                                    "model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight"
                                ),
                                module,
                                vec![format!(
                                    "layers[{layer_idx}].feed_forward.experts[{expert_idx}].up_proj.weight"
                                )],
                            ),
                        ]);
                    }

                    planned
                }
            },
        },
    }
}

fn planned_tensor(
    tensor_name: impl Into<String>,
    module: &KimiModuleRef,
    local_parameter_paths: Vec<String>,
) -> KimiPlannedTensor {
    KimiPlannedTensor {
        tensor_name: tensor_name.into(),
        module: module.clone(),
        local_parameter_paths,
    }
}

fn classify_selected_tensor(
    config: &KimiArtifactConfig,
    layer_schedule: &KimiLayerSchedule,
    selection_scope: &KimiSelectionScope,
    tensor_name: &str,
    shard_path: &str,
) -> Option<KimiTensorMapping> {
    match tensor_name {
        "model.embed_tokens.weight" => {
            if selection_scope.include_embeddings {
                return Some(unsupported_mapping(
                    tensor_name,
                    shard_path,
                    Some(KimiModuleRef::Embeddings),
                    KimiUnsupportedTensorReason::UnsupportedTensorNamePattern,
                ));
            }
            return None;
        }
        "model.norm.weight" => {
            if selection_scope.include_final_norm {
                return Some(unsupported_mapping(
                    tensor_name,
                    shard_path,
                    Some(KimiModuleRef::FinalNorm),
                    KimiUnsupportedTensorReason::UnsupportedTensorNamePattern,
                ));
            }
            return None;
        }
        "lm_head.weight" => {
            if selection_scope.include_lm_head {
                return Some(unsupported_mapping(
                    tensor_name,
                    shard_path,
                    Some(KimiModuleRef::LmHead),
                    KimiUnsupportedTensorReason::UnsupportedTensorNamePattern,
                ));
            }
            return None;
        }
        _ => {}
    }

    let layer_prefix = "model.layers.";
    if !tensor_name.starts_with(layer_prefix) {
        return None;
    }

    let remainder = &tensor_name[layer_prefix.len()..];
    let (layer_idx_str, remainder) = remainder.split_once('.')?;
    let layer_idx = layer_idx_str.parse::<usize>().ok()?;
    if !selection_scope.includes_layer(layer_idx) {
        return None;
    }

    if remainder == "input_layernorm.weight" {
        return Some(unsupported_mapping(
            tensor_name,
            shard_path,
            Some(KimiModuleRef::DecoderLayer {
                layer_idx,
                component: KimiLayerModuleRef::InputNorm,
            }),
            KimiUnsupportedTensorReason::UnsupportedTensorNamePattern,
        ));
    }

    if remainder == "post_attention_layernorm.weight" {
        return Some(unsupported_mapping(
            tensor_name,
            shard_path,
            Some(KimiModuleRef::DecoderLayer {
                layer_idx,
                component: KimiLayerModuleRef::PostAttentionNorm,
            }),
            KimiUnsupportedTensorReason::UnsupportedTensorNamePattern,
        ));
    }

    if let Some(leaf) = remainder.strip_prefix("self_attn.") {
        let kind = layer_schedule
            .try_attention_kind(layer_idx)
            .expect("validated layer_idx must exist");
        let module = KimiModuleRef::DecoderLayer {
            layer_idx,
            component: KimiLayerModuleRef::Attention { kind },
        };
        return attention_tensor_mapping(config, tensor_name, shard_path, module, kind, leaf);
    }

    if remainder.starts_with("mlp.") || remainder.starts_with("block_sparse_moe.") {
        let kind = layer_schedule
            .try_feed_forward_kind(layer_idx)
            .expect("validated layer_idx must exist");
        let module = KimiModuleRef::DecoderLayer {
            layer_idx,
            component: KimiLayerModuleRef::FeedForward { kind },
        };
        return feed_forward_tensor_mapping(
            config,
            tensor_name,
            shard_path,
            module,
            kind,
            remainder,
        );
    }

    Some(unsupported_mapping(
        tensor_name,
        shard_path,
        None,
        KimiUnsupportedTensorReason::UnsupportedTensorNamePattern,
    ))
}

fn attention_tensor_mapping(
    config: &KimiArtifactConfig,
    tensor_name: &str,
    shard_path: &str,
    module: KimiModuleRef,
    kind: KimiAttentionLayerKind,
    leaf: &str,
) -> Option<KimiTensorMapping> {
    let status = match kind {
        KimiAttentionLayerKind::FullAttention => match leaf {
            "kv_a_proj_with_mqa.weight" => {
                KimiUnsupportedTensorReason::MlaKvLatentProjectionContainsUnsupportedRopeRows
            }
            "kv_a_layernorm.weight" => KimiUnsupportedTensorReason::MlaKvLayerNormUnsupported,
            "kv_b_proj.weight" => KimiUnsupportedTensorReason::MlaKvUpProjectionUnsupported,
            "q_a_proj.weight" | "q_a_layernorm.weight" | "q_b_proj.weight"
                if config.q_lora_rank.is_some() =>
            {
                KimiUnsupportedTensorReason::MlaLowRankQueryImportDeferred
            }
            "k_proj.weight" | "v_proj.weight" | "q_conv1d.weight" | "k_conv1d.weight"
            | "v_conv1d.weight" | "A_log" | "dt_bias" | "b_proj.weight" | "f_a_proj.weight"
            | "f_b_proj.weight" | "g_a_proj.weight" | "g_b_proj.weight" | "o_norm.weight" => {
                KimiUnsupportedTensorReason::TensorDoesNotMatchAttentionKind { expected: kind }
            }
            _ => KimiUnsupportedTensorReason::UnsupportedTensorNamePattern,
        },
        KimiAttentionLayerKind::LinearAttentionKda => match leaf {
            "A_log" | "dt_bias" | "b_proj.weight" | "f_a_proj.weight" | "f_b_proj.weight"
            | "g_a_proj.weight" | "g_b_proj.weight" | "q_conv1d.weight" | "k_conv1d.weight"
            | "v_conv1d.weight" | "o_norm.weight" => {
                KimiUnsupportedTensorReason::KdaAuxiliaryTensorUnsupported
            }
            "kv_a_proj_with_mqa.weight"
            | "kv_a_layernorm.weight"
            | "kv_b_proj.weight"
            | "q_a_proj.weight"
            | "q_a_layernorm.weight"
            | "q_b_proj.weight" => {
                KimiUnsupportedTensorReason::TensorDoesNotMatchAttentionKind { expected: kind }
            }
            _ => KimiUnsupportedTensorReason::UnsupportedTensorNamePattern,
        },
    };

    Some(unsupported_mapping(
        tensor_name,
        shard_path,
        Some(module),
        status,
    ))
}

fn feed_forward_tensor_mapping(
    config: &KimiArtifactConfig,
    tensor_name: &str,
    shard_path: &str,
    module: KimiModuleRef,
    kind: KimiFeedForwardLayerKind,
    remainder: &str,
) -> Option<KimiTensorMapping> {
    let status = match kind {
        KimiFeedForwardLayerKind::DenseMlp => {
            if remainder.starts_with("block_sparse_moe.") {
                KimiUnsupportedTensorReason::TensorDoesNotMatchFeedForwardKind { expected: kind }
            } else {
                KimiUnsupportedTensorReason::UnsupportedTensorNamePattern
            }
        }
        KimiFeedForwardLayerKind::SparseMoe => {
            if remainder == "block_sparse_moe.gate.e_score_correction_bias" {
                KimiUnsupportedTensorReason::MoeRouterBiasCorrectionUnsupported
            } else if remainder.starts_with("mlp.") {
                KimiUnsupportedTensorReason::TensorDoesNotMatchFeedForwardKind { expected: kind }
            } else if remainder.starts_with("block_sparse_moe.shared_experts.")
                && config.num_shared_experts != 1
            {
                KimiUnsupportedTensorReason::SharedExpertsPackingUnsupported {
                    num_shared_experts: config.num_shared_experts,
                }
            } else if let Some((expert_idx, _leaf)) = parse_sparse_expert_leaf(remainder) {
                if expert_idx >= config.num_experts {
                    KimiUnsupportedTensorReason::ExpertIndexOutOfRange {
                        expert_idx,
                        num_experts: config.num_experts,
                    }
                } else {
                    KimiUnsupportedTensorReason::UnsupportedTensorNamePattern
                }
            } else {
                KimiUnsupportedTensorReason::UnsupportedTensorNamePattern
            }
        }
    };

    Some(unsupported_mapping(
        tensor_name,
        shard_path,
        Some(module),
        status,
    ))
}

fn parse_sparse_expert_leaf(remainder: &str) -> Option<(usize, &str)> {
    let prefix = "block_sparse_moe.experts.";
    let suffix = remainder.strip_prefix(prefix)?;
    let (expert_idx, leaf) = suffix.split_once('.')?;
    Some((expert_idx.parse().ok()?, leaf))
}

fn unsupported_mapping(
    tensor_name: &str,
    shard_path: &str,
    module: Option<KimiModuleRef>,
    reason: KimiUnsupportedTensorReason,
) -> KimiTensorMapping {
    KimiTensorMapping {
        tensor_name: tensor_name.to_string(),
        shard_path: shard_path.to_string(),
        module,
        local_parameter_paths: Vec::new(),
        status: KimiTensorMappingStatus::Unsupported { reason },
    }
}
