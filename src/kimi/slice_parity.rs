use burn::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::path::Path;

use crate::kimi::config::KimiArtifactConfig;
use crate::kimi::import::{
    KimiArtifactUnderstanding, KimiImportCoverageError, KimiImportError, KimiImportSelection,
    KimiModuleRef,
};
use crate::kimi::model::KimiLinearModel;
use crate::kimi::payload::KimiBaselinePayloadError;

pub const KIMI_BASELINE_SLICE_REQUEST_FILENAME: &str = "baseline-slice-request.json";
pub const KIMI_BASELINE_SLICE_REQUEST_KIND: &str = "attnres.kimi.baseline_slice_request";
pub const KIMI_BASELINE_SLICE_REQUEST_VERSION: u32 = 1;
pub const KIMI_BASELINE_SLICE_PARITY_FILENAME: &str = "baseline-slice-parity.json";
pub const KIMI_BASELINE_SLICE_PARITY_KIND: &str = "attnres.kimi.baseline_slice_parity";
pub const KIMI_BASELINE_SLICE_PARITY_VERSION: u32 = 1;

const KIMI_BASELINE_SLICE_TOLERANCE_METRIC: &str = "max_abs_diff";
const KIMI_BASELINE_SLICE_RUNTIME_DTYPE: &str = "float32";

/// Machine-readable tensor payload for baseline-only slice parity fixtures.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiBaselineSliceParityTensor {
    pub dims: Vec<usize>,
    pub values: Vec<f32>,
}

/// Selected post-layer hidden-state capture for one prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiBaselineSliceParityHiddenState {
    pub layer_idx: usize,
    pub tensor: KimiBaselineSliceParityTensor,
}

/// One prompt specification for baseline-only slice parity execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiBaselineSliceParityPromptSpec {
    pub name: String,
    pub input_ids: Vec<usize>,
}

/// Caller-provided slice request before module/tensor expansion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiBaselineSliceRequestSpec {
    pub seed: u64,
    pub import_selection: KimiImportSelection,
    pub selected_hidden_layers: Vec<usize>,
    pub prompts: Vec<KimiBaselineSliceParityPromptSpec>,
    pub tolerances: KimiBaselineSliceParityToleranceSpec,
}

/// One prompt result produced by an external baseline reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiBaselineSliceParityPromptResult {
    pub prompt_name: String,
    pub input_ids: Vec<usize>,
    pub logits: KimiBaselineSliceParityTensor,
    pub hidden_states: Vec<KimiBaselineSliceParityHiddenState>,
}

/// Baseline artifact/config fingerprint expected by an external slice fixture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiBaselineSliceParityArtifactSpec {
    pub model_type: String,
    pub dtype: String,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

/// Explicit tolerance metadata for fixture consumption.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiBaselineSliceParityToleranceSpec {
    pub metric: String,
    pub runtime_dtype: String,
    pub logits_max_abs_diff: f32,
    pub hidden_state_max_abs_diff: f32,
}

/// Exact supported baseline slice requested by the external reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiBaselineSliceParitySliceSpec {
    pub import_selection: KimiImportSelection,
    pub selected_hidden_layers: Vec<usize>,
    pub requested_modules: Vec<KimiModuleRef>,
    pub required_tensors: Vec<String>,
    pub tolerances: KimiBaselineSliceParityToleranceSpec,
}

/// Machine-readable request manifest emitted for an external baseline reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiBaselineSliceRequestManifest {
    pub kind: String,
    pub version: u32,
    pub seed: u64,
    pub artifact: KimiBaselineSliceParityArtifactSpec,
    pub slice: KimiBaselineSliceParitySliceSpec,
    pub prompts: Vec<KimiBaselineSliceParityPromptSpec>,
}

/// Baseline-only external-reference fixture consumed by the local parity harness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiBaselineSliceParityFixture {
    pub kind: String,
    pub version: u32,
    pub seed: u64,
    pub artifact: KimiBaselineSliceParityArtifactSpec,
    pub slice: KimiBaselineSliceParitySliceSpec,
    pub prompts: Vec<KimiBaselineSliceParityPromptSpec>,
    pub prompt_results: Vec<KimiBaselineSliceParityPromptResult>,
}

/// Typed failures for baseline-only Gate 2 slice parity fixture loading/comparison.
#[derive(Debug, Clone, PartialEq)]
pub enum KimiBaselineSliceParityError {
    ManifestReadFailed {
        path: String,
        detail: String,
    },
    ManifestWriteFailed {
        path: String,
        detail: String,
    },
    ManifestParseFailed {
        path: String,
        detail: String,
    },
    UnexpectedManifestKind {
        kind: String,
    },
    UnsupportedManifestVersion {
        version: u32,
    },
    ReadFailed {
        path: String,
        detail: String,
    },
    ParseFailed {
        path: String,
        detail: String,
    },
    UnexpectedFixtureKind {
        kind: String,
    },
    UnsupportedFixtureVersion {
        version: u32,
    },
    Import(KimiImportError),
    Coverage(KimiImportCoverageError),
    Payload(KimiBaselinePayloadError),
    ArtifactFieldMismatch {
        field: String,
        expected: String,
        actual: String,
    },
    SelectedHiddenLayerNotInImportSelection {
        layer_idx: usize,
        import_selection_layers: Vec<usize>,
    },
    UnsupportedModuleRequest {
        module: KimiModuleRef,
    },
    MissingRequiredModule {
        module: KimiModuleRef,
    },
    UnsupportedTensorRequest {
        tensor_name: String,
    },
    MissingRequiredTensor {
        tensor_name: String,
    },
    ManifestFieldMismatch {
        field: String,
        expected: String,
        actual: String,
    },
    FixtureManifestMismatch {
        field: String,
        expected: String,
        actual: String,
    },
    UnsupportedToleranceMetadata {
        field: String,
        expected: String,
        actual: String,
    },
    InvalidToleranceValue {
        field: String,
        value: f32,
    },
    PromptCountMismatch {
        expected: usize,
        actual: usize,
    },
    PromptNameMismatch {
        index: usize,
        expected: String,
        actual: String,
    },
    PromptTokenMismatch {
        prompt_name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    HiddenStateCountMismatch {
        prompt_name: String,
        expected: usize,
        actual: usize,
    },
    HiddenStateLayerMismatch {
        prompt_name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    TensorShapeMismatch {
        label: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    LogitsToleranceExceeded {
        prompt_name: String,
        max_abs_diff: f32,
        tolerance: f32,
    },
    HiddenStateToleranceExceeded {
        prompt_name: String,
        layer_idx: usize,
        max_abs_diff: f32,
        tolerance: f32,
    },
}

impl Display for KimiBaselineSliceParityError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ManifestReadFailed { path, detail } => {
                write!(f, "failed to read baseline slice request manifest '{path}': {detail}")
            }
            Self::ManifestWriteFailed { path, detail } => write!(
                f,
                "failed to write baseline slice request manifest '{path}': {detail}"
            ),
            Self::ManifestParseFailed { path, detail } => write!(
                f,
                "failed to parse baseline slice request manifest '{path}': {detail}"
            ),
            Self::UnexpectedManifestKind { kind } => write!(
                f,
                "expected baseline slice request manifest kind '{KIMI_BASELINE_SLICE_REQUEST_KIND}', got '{kind}'"
            ),
            Self::UnsupportedManifestVersion { version } => write!(
                f,
                "expected baseline slice request manifest version {KIMI_BASELINE_SLICE_REQUEST_VERSION}, got {version}"
            ),
            Self::ReadFailed { path, detail } => {
                write!(f, "failed to read baseline slice parity fixture '{path}': {detail}")
            }
            Self::ParseFailed { path, detail } => write!(
                f,
                "failed to parse baseline slice parity fixture '{path}': {detail}"
            ),
            Self::UnexpectedFixtureKind { kind } => write!(
                f,
                "expected baseline slice parity fixture kind '{KIMI_BASELINE_SLICE_PARITY_KIND}', got '{kind}'"
            ),
            Self::UnsupportedFixtureVersion { version } => write!(
                f,
                "expected baseline slice parity fixture version {KIMI_BASELINE_SLICE_PARITY_VERSION}, got {version}"
            ),
            Self::Import(err) => write!(f, "{err}"),
            Self::Coverage(err) => write!(f, "{err}"),
            Self::Payload(err) => write!(f, "{err}"),
            Self::ArtifactFieldMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "fixture artifact field '{field}' expected '{expected}', got '{actual}' from the loaded artifact"
            ),
            Self::SelectedHiddenLayerNotInImportSelection {
                layer_idx,
                import_selection_layers,
            } => write!(
                f,
                "selected hidden layer {layer_idx} is not present in the imported slice {:?}",
                import_selection_layers
            ),
            Self::UnsupportedModuleRequest { module } => write!(
                f,
                "fixture requested unsupported or mismatched baseline module {module:?}"
            ),
            Self::MissingRequiredModule { module } => write!(
                f,
                "fixture omitted required baseline module {module:?} for the selected slice"
            ),
            Self::UnsupportedTensorRequest { tensor_name } => write!(
                f,
                "fixture requested unsupported or mismatched baseline tensor '{tensor_name}'"
            ),
            Self::MissingRequiredTensor { tensor_name } => write!(
                f,
                "fixture omitted required baseline tensor '{tensor_name}' for the selected slice"
            ),
            Self::ManifestFieldMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "baseline slice request manifest field '{field}' expected '{expected}', got '{actual}'"
            ),
            Self::FixtureManifestMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "baseline slice parity fixture field '{field}' expected '{expected}' from the request manifest, got '{actual}'"
            ),
            Self::UnsupportedToleranceMetadata {
                field,
                expected,
                actual,
            } => write!(
                f,
                "fixture tolerance field '{field}' expected '{expected}', got '{actual}'"
            ),
            Self::InvalidToleranceValue { field, value } => write!(
                f,
                "fixture tolerance field '{field}' must be finite and >= 0, got {value}"
            ),
            Self::PromptCountMismatch { expected, actual } => write!(
                f,
                "fixture prompt count mismatch: expected {expected} prompt results, got {actual}"
            ),
            Self::PromptNameMismatch {
                index,
                expected,
                actual,
            } => write!(
                f,
                "fixture prompt[{index}] name mismatch: expected '{expected}', got '{actual}'"
            ),
            Self::PromptTokenMismatch {
                prompt_name,
                expected,
                actual,
            } => write!(
                f,
                "fixture prompt '{prompt_name}' token mismatch: expected {:?}, got {:?}",
                expected, actual
            ),
            Self::HiddenStateCountMismatch {
                prompt_name,
                expected,
                actual,
            } => write!(
                f,
                "fixture prompt '{prompt_name}' hidden-state count mismatch: expected {expected}, got {actual}"
            ),
            Self::HiddenStateLayerMismatch {
                prompt_name,
                expected,
                actual,
            } => write!(
                f,
                "fixture prompt '{prompt_name}' hidden-state layer mismatch: expected {:?}, got {:?}",
                expected, actual
            ),
            Self::TensorShapeMismatch {
                label,
                expected,
                actual,
            } => write!(
                f,
                "{label} tensor shape mismatch: expected {:?}, got {:?}",
                expected, actual
            ),
            Self::LogitsToleranceExceeded {
                prompt_name,
                max_abs_diff,
                tolerance,
            } => write!(
                f,
                "prompt '{prompt_name}' logits max_abs_diff {max_abs_diff} exceeded tolerance {tolerance}"
            ),
            Self::HiddenStateToleranceExceeded {
                prompt_name,
                layer_idx,
                max_abs_diff,
                tolerance,
            } => write!(
                f,
                "prompt '{prompt_name}' hidden state for layer {layer_idx} max_abs_diff {max_abs_diff} exceeded tolerance {tolerance}"
            ),
        }
    }
}

impl std::error::Error for KimiBaselineSliceParityError {}

impl From<KimiImportError> for KimiBaselineSliceParityError {
    fn from(err: KimiImportError) -> Self {
        Self::Import(err)
    }
}

impl From<KimiImportCoverageError> for KimiBaselineSliceParityError {
    fn from(err: KimiImportCoverageError) -> Self {
        Self::Coverage(err)
    }
}

impl From<KimiBaselinePayloadError> for KimiBaselineSliceParityError {
    fn from(err: KimiBaselinePayloadError) -> Self {
        Self::Payload(err)
    }
}

impl KimiBaselineSliceParityToleranceSpec {
    pub fn try_validate(&self) -> Result<(), KimiBaselineSliceParityError> {
        if self.metric != KIMI_BASELINE_SLICE_TOLERANCE_METRIC {
            return Err(KimiBaselineSliceParityError::UnsupportedToleranceMetadata {
                field: "metric".to_string(),
                expected: KIMI_BASELINE_SLICE_TOLERANCE_METRIC.to_string(),
                actual: self.metric.clone(),
            });
        }
        if self.runtime_dtype != KIMI_BASELINE_SLICE_RUNTIME_DTYPE {
            return Err(KimiBaselineSliceParityError::UnsupportedToleranceMetadata {
                field: "runtime_dtype".to_string(),
                expected: KIMI_BASELINE_SLICE_RUNTIME_DTYPE.to_string(),
                actual: self.runtime_dtype.clone(),
            });
        }
        validate_tolerance_value("logits_max_abs_diff", self.logits_max_abs_diff)?;
        validate_tolerance_value("hidden_state_max_abs_diff", self.hidden_state_max_abs_diff)?;
        Ok(())
    }
}

impl KimiBaselineSliceRequestSpec {
    pub fn try_validate(
        &self,
        num_hidden_layers: usize,
    ) -> Result<(), KimiBaselineSliceParityError> {
        self.tolerances.try_validate()?;
        self.import_selection.try_validate(num_hidden_layers)?;
        validate_selected_hidden_layers(
            num_hidden_layers,
            &self.import_selection,
            &self.selected_hidden_layers,
        )?;
        Ok(())
    }
}

impl KimiArtifactUnderstanding {
    pub fn try_build_baseline_slice_request_manifest(
        &self,
        request: KimiBaselineSliceRequestSpec,
    ) -> Result<KimiBaselineSliceRequestManifest, KimiBaselineSliceParityError> {
        request.try_validate(self.config.num_hidden_layers)?;

        let plan = self.try_slice_plan(request.import_selection.clone())?;
        plan.try_require_loadable()?;
        let requested_modules = plan.selected_modules.clone();
        let required_tensors = required_tensor_names(&plan);

        let manifest = KimiBaselineSliceRequestManifest {
            kind: KIMI_BASELINE_SLICE_REQUEST_KIND.to_string(),
            version: KIMI_BASELINE_SLICE_REQUEST_VERSION,
            seed: request.seed,
            artifact: baseline_slice_artifact_spec(&self.config),
            slice: KimiBaselineSliceParitySliceSpec {
                import_selection: request.import_selection,
                selected_hidden_layers: request.selected_hidden_layers,
                requested_modules,
                required_tensors,
                tolerances: request.tolerances,
            },
            prompts: request.prompts,
        };
        manifest.try_validate_against_understanding(self)?;
        Ok(manifest)
    }
}

impl KimiBaselineSliceRequestManifest {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, KimiBaselineSliceParityError> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path).map_err(|err| {
            KimiBaselineSliceParityError::ManifestReadFailed {
                path: path.display().to_string(),
                detail: err.to_string(),
            }
        })?;
        let manifest: Self = serde_json::from_str(&json).map_err(|err| {
            KimiBaselineSliceParityError::ManifestParseFailed {
                path: path.display().to_string(),
                detail: err.to_string(),
            }
        })?;
        manifest.try_validate_static()?;
        Ok(manifest)
    }

    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<(), KimiBaselineSliceParityError> {
        let path = path.as_ref();
        let json = serde_json::to_string_pretty(self).expect("manifest serialization must succeed");
        std::fs::write(path, json).map_err(|err| {
            KimiBaselineSliceParityError::ManifestWriteFailed {
                path: path.display().to_string(),
                detail: err.to_string(),
            }
        })
    }

    pub fn try_validate_static(&self) -> Result<(), KimiBaselineSliceParityError> {
        if self.kind != KIMI_BASELINE_SLICE_REQUEST_KIND {
            return Err(KimiBaselineSliceParityError::UnexpectedManifestKind {
                kind: self.kind.clone(),
            });
        }
        if self.version != KIMI_BASELINE_SLICE_REQUEST_VERSION {
            return Err(KimiBaselineSliceParityError::UnsupportedManifestVersion {
                version: self.version,
            });
        }

        KimiBaselineSliceRequestSpec {
            seed: self.seed,
            import_selection: self.slice.import_selection.clone(),
            selected_hidden_layers: self.slice.selected_hidden_layers.clone(),
            prompts: self.prompts.clone(),
            tolerances: self.slice.tolerances.clone(),
        }
        .try_validate(self.artifact.num_hidden_layers)
    }

    pub fn try_into_fixture(
        self,
        prompt_results: Vec<KimiBaselineSliceParityPromptResult>,
    ) -> Result<KimiBaselineSliceParityFixture, KimiBaselineSliceParityError> {
        let fixture = KimiBaselineSliceParityFixture {
            kind: KIMI_BASELINE_SLICE_PARITY_KIND.to_string(),
            version: KIMI_BASELINE_SLICE_PARITY_VERSION,
            seed: self.seed,
            artifact: self.artifact,
            slice: self.slice,
            prompts: self.prompts,
            prompt_results,
        };
        fixture.try_validate_static()?;
        Ok(fixture)
    }

    fn try_validate_against_understanding(
        &self,
        understanding: &KimiArtifactUnderstanding,
    ) -> Result<(), KimiBaselineSliceParityError> {
        let expected_artifact = baseline_slice_artifact_spec(&understanding.config);
        compare_manifest_field(
            "artifact.model_type",
            &self.artifact.model_type,
            &expected_artifact.model_type,
        )?;
        compare_manifest_field(
            "artifact.dtype",
            &self.artifact.dtype,
            &expected_artifact.dtype,
        )?;
        compare_manifest_field(
            "artifact.num_hidden_layers",
            &self.artifact.num_hidden_layers,
            &expected_artifact.num_hidden_layers,
        )?;
        compare_manifest_field(
            "artifact.hidden_size",
            &self.artifact.hidden_size,
            &expected_artifact.hidden_size,
        )?;
        compare_manifest_field(
            "artifact.vocab_size",
            &self.artifact.vocab_size,
            &expected_artifact.vocab_size,
        )?;

        let plan = understanding.try_slice_plan(self.slice.import_selection.clone())?;
        plan.try_require_loadable()?;

        compare_manifest_field(
            "slice.requested_modules",
            &self.slice.requested_modules,
            &plan.selected_modules,
        )?;
        compare_manifest_field(
            "slice.required_tensors",
            &self.slice.required_tensors,
            &required_tensor_names(&plan),
        )?;

        Ok(())
    }
}

impl KimiBaselineSliceParityFixture {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, KimiBaselineSliceParityError> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path).map_err(|err| {
            KimiBaselineSliceParityError::ReadFailed {
                path: path.display().to_string(),
                detail: err.to_string(),
            }
        })?;
        let fixture: Self = serde_json::from_str(&json).map_err(|err| {
            KimiBaselineSliceParityError::ParseFailed {
                path: path.display().to_string(),
                detail: err.to_string(),
            }
        })?;
        fixture.try_validate_static()?;
        Ok(fixture)
    }

    pub fn try_validate_static(&self) -> Result<(), KimiBaselineSliceParityError> {
        if self.kind != KIMI_BASELINE_SLICE_PARITY_KIND {
            return Err(KimiBaselineSliceParityError::UnexpectedFixtureKind {
                kind: self.kind.clone(),
            });
        }
        if self.version != KIMI_BASELINE_SLICE_PARITY_VERSION {
            return Err(KimiBaselineSliceParityError::UnsupportedFixtureVersion {
                version: self.version,
            });
        }

        self.slice.tolerances.try_validate()?;
        self.slice
            .import_selection
            .try_validate(self.artifact.num_hidden_layers)?;
        validate_selected_hidden_layers(
            self.artifact.num_hidden_layers,
            &self.slice.import_selection,
            &self.slice.selected_hidden_layers,
        )?;
        self.try_validate_prompt_metadata()?;
        Ok(())
    }

    pub fn try_validate_matches_manifest(
        &self,
        manifest: &KimiBaselineSliceRequestManifest,
    ) -> Result<(), KimiBaselineSliceParityError> {
        self.try_validate_static()?;
        manifest.try_validate_static()?;

        compare_fixture_manifest_field("seed", &self.seed, &manifest.seed)?;
        compare_fixture_manifest_field(
            "artifact.model_type",
            &self.artifact.model_type,
            &manifest.artifact.model_type,
        )?;
        compare_fixture_manifest_field(
            "artifact.dtype",
            &self.artifact.dtype,
            &manifest.artifact.dtype,
        )?;
        compare_fixture_manifest_field(
            "artifact.num_hidden_layers",
            &self.artifact.num_hidden_layers,
            &manifest.artifact.num_hidden_layers,
        )?;
        compare_fixture_manifest_field(
            "artifact.hidden_size",
            &self.artifact.hidden_size,
            &manifest.artifact.hidden_size,
        )?;
        compare_fixture_manifest_field(
            "artifact.vocab_size",
            &self.artifact.vocab_size,
            &manifest.artifact.vocab_size,
        )?;
        compare_fixture_manifest_field(
            "slice.import_selection",
            &self.slice.import_selection,
            &manifest.slice.import_selection,
        )?;
        compare_fixture_manifest_field(
            "slice.selected_hidden_layers",
            &self.slice.selected_hidden_layers,
            &manifest.slice.selected_hidden_layers,
        )?;
        compare_fixture_manifest_field(
            "slice.requested_modules",
            &self.slice.requested_modules,
            &manifest.slice.requested_modules,
        )?;
        compare_fixture_manifest_field(
            "slice.required_tensors",
            &self.slice.required_tensors,
            &manifest.slice.required_tensors,
        )?;
        compare_fixture_manifest_field(
            "slice.tolerances.metric",
            &self.slice.tolerances.metric,
            &manifest.slice.tolerances.metric,
        )?;
        compare_fixture_manifest_field(
            "slice.tolerances.runtime_dtype",
            &self.slice.tolerances.runtime_dtype,
            &manifest.slice.tolerances.runtime_dtype,
        )?;
        compare_fixture_manifest_field(
            "slice.tolerances.logits_max_abs_diff",
            &self.slice.tolerances.logits_max_abs_diff,
            &manifest.slice.tolerances.logits_max_abs_diff,
        )?;
        compare_fixture_manifest_field(
            "slice.tolerances.hidden_state_max_abs_diff",
            &self.slice.tolerances.hidden_state_max_abs_diff,
            &manifest.slice.tolerances.hidden_state_max_abs_diff,
        )?;
        compare_fixture_manifest_field(
            "prompts.len",
            &self.prompts.len(),
            &manifest.prompts.len(),
        )?;
        for (index, (prompt, expected)) in
            self.prompts.iter().zip(manifest.prompts.iter()).enumerate()
        {
            compare_fixture_manifest_field(
                &format!("prompts[{index}].name"),
                &prompt.name,
                &expected.name,
            )?;
            compare_fixture_manifest_field(
                &format!("prompts[{index}].input_ids"),
                &prompt.input_ids,
                &expected.input_ids,
            )?;
        }

        Ok(())
    }

    fn try_validate_prompt_metadata(&self) -> Result<(), KimiBaselineSliceParityError> {
        if self.prompts.len() != self.prompt_results.len() {
            return Err(KimiBaselineSliceParityError::PromptCountMismatch {
                expected: self.prompts.len(),
                actual: self.prompt_results.len(),
            });
        }

        for (index, (prompt, result)) in self
            .prompts
            .iter()
            .zip(self.prompt_results.iter())
            .enumerate()
        {
            if prompt.name != result.prompt_name {
                return Err(KimiBaselineSliceParityError::PromptNameMismatch {
                    index,
                    expected: prompt.name.clone(),
                    actual: result.prompt_name.clone(),
                });
            }
            if prompt.input_ids != result.input_ids {
                return Err(KimiBaselineSliceParityError::PromptTokenMismatch {
                    prompt_name: prompt.name.clone(),
                    expected: prompt.input_ids.clone(),
                    actual: result.input_ids.clone(),
                });
            }
            if result.hidden_states.len() != self.slice.selected_hidden_layers.len() {
                return Err(KimiBaselineSliceParityError::HiddenStateCountMismatch {
                    prompt_name: prompt.name.clone(),
                    expected: self.slice.selected_hidden_layers.len(),
                    actual: result.hidden_states.len(),
                });
            }

            let actual_layers = result
                .hidden_states
                .iter()
                .map(|state| state.layer_idx)
                .collect::<Vec<_>>();
            if actual_layers != self.slice.selected_hidden_layers {
                return Err(KimiBaselineSliceParityError::HiddenStateLayerMismatch {
                    prompt_name: prompt.name.clone(),
                    expected: self.slice.selected_hidden_layers.clone(),
                    actual: actual_layers,
                });
            }
        }

        Ok(())
    }

    fn try_validate_against_understanding(
        &self,
        understanding: &KimiArtifactUnderstanding,
    ) -> Result<(), KimiBaselineSliceParityError> {
        let expected_artifact = baseline_slice_artifact_spec(&understanding.config);
        compare_artifact_field(
            "model_type",
            &self.artifact.model_type,
            &expected_artifact.model_type,
        )?;
        compare_artifact_field("dtype", &self.artifact.dtype, &expected_artifact.dtype)?;
        compare_artifact_field(
            "num_hidden_layers",
            &self.artifact.num_hidden_layers.to_string(),
            &expected_artifact.num_hidden_layers.to_string(),
        )?;
        compare_artifact_field(
            "hidden_size",
            &self.artifact.hidden_size.to_string(),
            &expected_artifact.hidden_size.to_string(),
        )?;
        compare_artifact_field(
            "vocab_size",
            &self.artifact.vocab_size.to_string(),
            &expected_artifact.vocab_size.to_string(),
        )?;

        let plan = understanding.try_slice_plan(self.slice.import_selection.clone())?;
        plan.try_require_loadable()?;

        for module in &self.slice.requested_modules {
            if !plan.selected_modules.contains(module) {
                return Err(KimiBaselineSliceParityError::UnsupportedModuleRequest {
                    module: module.clone(),
                });
            }
        }
        for module in &plan.selected_modules {
            if !self.slice.requested_modules.contains(module) {
                return Err(KimiBaselineSliceParityError::MissingRequiredModule {
                    module: module.clone(),
                });
            }
        }

        let expected_tensors = plan
            .coverage
            .mapped_tensors
            .iter()
            .map(|mapping| mapping.tensor_name.as_str())
            .collect::<Vec<_>>();
        for tensor_name in &self.slice.required_tensors {
            if !expected_tensors.contains(&tensor_name.as_str()) {
                return Err(KimiBaselineSliceParityError::UnsupportedTensorRequest {
                    tensor_name: tensor_name.clone(),
                });
            }
        }
        for tensor_name in expected_tensors {
            if !self
                .slice
                .required_tensors
                .iter()
                .any(|requested| requested == tensor_name)
            {
                return Err(KimiBaselineSliceParityError::MissingRequiredTensor {
                    tensor_name: tensor_name.to_string(),
                });
            }
        }

        Ok(())
    }
}

/// Consume an external baseline-only slice parity fixture from disk and compare
/// it against a locally loadable sharded artifact using `KimiLinearModel`.
pub fn compare_baseline_slice_parity_fixture_from_dir<
    B: Backend,
    P: AsRef<Path>,
    Q: AsRef<Path>,
>(
    artifact_dir: P,
    fixture_path: Q,
    device: &B::Device,
) -> Result<(), KimiBaselineSliceParityError> {
    let fixture = KimiBaselineSliceParityFixture::load(fixture_path)?;
    compare_baseline_slice_parity_fixture::<B, P>(artifact_dir, &fixture, device)
}

/// Compare an already-loaded baseline-only slice parity fixture against a
/// locally loadable sharded artifact using `KimiLinearModel`.
pub fn compare_baseline_slice_parity_fixture<B: Backend, P: AsRef<Path>>(
    artifact_dir: P,
    fixture: &KimiBaselineSliceParityFixture,
    device: &B::Device,
) -> Result<(), KimiBaselineSliceParityError> {
    let artifact_dir = artifact_dir.as_ref();
    fixture.try_validate_static()?;
    let understanding = KimiArtifactUnderstanding::load_from_dir(artifact_dir)?;
    compare_baseline_slice_parity_fixture_with_understanding::<B>(
        artifact_dir,
        &understanding,
        fixture,
        device,
    )
}

/// Consume an external baseline-only slice request manifest plus a
/// `baseline-slice-parity.json` fixture and compare the fixture against the
/// locally loadable `KimiLinearModel` slice requested by that manifest.
pub fn compare_baseline_slice_parity_fixture_with_manifest_from_dir<
    B: Backend,
    P: AsRef<Path>,
    Q: AsRef<Path>,
    R: AsRef<Path>,
>(
    artifact_dir: P,
    manifest_path: Q,
    fixture_path: R,
    device: &B::Device,
) -> Result<(), KimiBaselineSliceParityError> {
    let manifest = KimiBaselineSliceRequestManifest::load(manifest_path)?;
    let fixture = KimiBaselineSliceParityFixture::load(fixture_path)?;
    compare_baseline_slice_parity_fixture_with_manifest::<B, P>(
        artifact_dir,
        &manifest,
        &fixture,
        device,
    )
}

/// Validate a baseline-only slice request manifest plus matching fixture
/// against a locally loadable sharded artifact using `KimiLinearModel`.
pub fn compare_baseline_slice_parity_fixture_with_manifest<B: Backend, P: AsRef<Path>>(
    artifact_dir: P,
    manifest: &KimiBaselineSliceRequestManifest,
    fixture: &KimiBaselineSliceParityFixture,
    device: &B::Device,
) -> Result<(), KimiBaselineSliceParityError> {
    let artifact_dir = artifact_dir.as_ref();
    manifest.try_validate_static()?;
    let understanding = KimiArtifactUnderstanding::load_from_dir(artifact_dir)?;
    manifest.try_validate_against_understanding(&understanding)?;
    fixture.try_validate_matches_manifest(manifest)?;
    compare_baseline_slice_parity_fixture_with_understanding::<B>(
        artifact_dir,
        &understanding,
        fixture,
        device,
    )
}

fn validate_tolerance_value(field: &str, value: f32) -> Result<(), KimiBaselineSliceParityError> {
    if !value.is_finite() || value < 0.0 {
        return Err(KimiBaselineSliceParityError::InvalidToleranceValue {
            field: field.to_string(),
            value,
        });
    }
    Ok(())
}

fn validate_selected_hidden_layers(
    num_hidden_layers: usize,
    import_selection: &KimiImportSelection,
    selected_hidden_layers: &[usize],
) -> Result<(), KimiBaselineSliceParityError> {
    KimiImportSelection {
        layer_indices: selected_hidden_layers.to_vec(),
        include_embeddings: false,
        include_final_norm: false,
        include_lm_head: false,
    }
    .try_validate(num_hidden_layers)?;

    for &layer_idx in selected_hidden_layers {
        if !import_selection.layer_indices.contains(&layer_idx) {
            return Err(
                KimiBaselineSliceParityError::SelectedHiddenLayerNotInImportSelection {
                    layer_idx,
                    import_selection_layers: import_selection.layer_indices.clone(),
                },
            );
        }
    }

    Ok(())
}

fn compare_artifact_field(
    field: &str,
    actual: &str,
    expected: &str,
) -> Result<(), KimiBaselineSliceParityError> {
    if actual != expected {
        return Err(KimiBaselineSliceParityError::ArtifactFieldMismatch {
            field: field.to_string(),
            expected: expected.to_string(),
            actual: actual.to_string(),
        });
    }
    Ok(())
}

fn compare_manifest_field<T: Serialize + PartialEq>(
    field: &str,
    actual: &T,
    expected: &T,
) -> Result<(), KimiBaselineSliceParityError> {
    if actual != expected {
        return Err(KimiBaselineSliceParityError::ManifestFieldMismatch {
            field: field.to_string(),
            expected: json_field_value(expected),
            actual: json_field_value(actual),
        });
    }
    Ok(())
}

fn compare_fixture_manifest_field<T: Serialize + PartialEq>(
    field: &str,
    actual: &T,
    expected: &T,
) -> Result<(), KimiBaselineSliceParityError> {
    if actual != expected {
        return Err(KimiBaselineSliceParityError::FixtureManifestMismatch {
            field: field.to_string(),
            expected: json_field_value(expected),
            actual: json_field_value(actual),
        });
    }
    Ok(())
}

fn json_field_value<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("slice parity values should serialize to json")
}

fn baseline_slice_artifact_spec(
    config: &KimiArtifactConfig,
) -> KimiBaselineSliceParityArtifactSpec {
    KimiBaselineSliceParityArtifactSpec {
        model_type: config.model_type.clone(),
        dtype: config.dtype.clone(),
        num_hidden_layers: config.num_hidden_layers,
        hidden_size: config.hidden_size,
        vocab_size: config.vocab_size,
    }
}

fn required_tensor_names(plan: &crate::kimi::import::KimiImportPlan) -> Vec<String> {
    plan.coverage
        .mapped_tensors
        .iter()
        .map(|mapping| mapping.tensor_name.clone())
        .collect()
}

fn compare_baseline_slice_parity_fixture_with_understanding<B: Backend>(
    artifact_dir: &Path,
    understanding: &KimiArtifactUnderstanding,
    fixture: &KimiBaselineSliceParityFixture,
    device: &B::Device,
) -> Result<(), KimiBaselineSliceParityError> {
    fixture.try_validate_static()?;
    fixture.try_validate_against_understanding(understanding)?;

    B::seed(device, fixture.seed);
    let model: KimiLinearModel<B> = understanding.try_init_baseline_model_from_dir(
        artifact_dir,
        fixture.slice.import_selection.clone(),
        device,
    )?;

    compare_model_to_fixture(&model, fixture, device)
}

fn compare_model_to_fixture<B: Backend>(
    model: &KimiLinearModel<B>,
    fixture: &KimiBaselineSliceParityFixture,
    device: &B::Device,
) -> Result<(), KimiBaselineSliceParityError> {
    for (prompt, expected) in fixture.prompts.iter().zip(fixture.prompt_results.iter()) {
        let input_ids = baseline_slice_input_ids::<B>(&prompt.input_ids, device);
        let observed_logits = tensor_to_parity(model.forward(input_ids.clone()));
        assert_tensor_shape(
            &format!("prompt '{}' logits", prompt.name),
            &expected.logits,
            &observed_logits,
        )?;
        let logits_diff = max_abs_diff(&expected.logits, &observed_logits);
        if logits_diff > fixture.slice.tolerances.logits_max_abs_diff {
            return Err(KimiBaselineSliceParityError::LogitsToleranceExceeded {
                prompt_name: prompt.name.clone(),
                max_abs_diff: logits_diff,
                tolerance: fixture.slice.tolerances.logits_max_abs_diff,
            });
        }

        let observed_hidden_states =
            trace_hidden_states(model, input_ids, &fixture.slice.selected_hidden_layers);
        for (expected_state, observed_state) in expected
            .hidden_states
            .iter()
            .zip(observed_hidden_states.iter())
        {
            assert_tensor_shape(
                &format!(
                    "prompt '{}' hidden state layer {}",
                    prompt.name, expected_state.layer_idx
                ),
                &expected_state.tensor,
                &observed_state.tensor,
            )?;
            let diff = max_abs_diff(&expected_state.tensor, &observed_state.tensor);
            if diff > fixture.slice.tolerances.hidden_state_max_abs_diff {
                return Err(KimiBaselineSliceParityError::HiddenStateToleranceExceeded {
                    prompt_name: prompt.name.clone(),
                    layer_idx: expected_state.layer_idx,
                    max_abs_diff: diff,
                    tolerance: fixture.slice.tolerances.hidden_state_max_abs_diff,
                });
            }
        }
    }

    Ok(())
}

fn assert_tensor_shape(
    label: &str,
    expected: &KimiBaselineSliceParityTensor,
    observed: &KimiBaselineSliceParityTensor,
) -> Result<(), KimiBaselineSliceParityError> {
    if expected.dims != observed.dims {
        return Err(KimiBaselineSliceParityError::TensorShapeMismatch {
            label: label.to_string(),
            expected: expected.dims.clone(),
            actual: observed.dims.clone(),
        });
    }
    Ok(())
}

fn max_abs_diff(
    expected: &KimiBaselineSliceParityTensor,
    observed: &KimiBaselineSliceParityTensor,
) -> f32 {
    expected
        .values
        .iter()
        .zip(observed.values.iter())
        .map(|(expected, observed)| (expected - observed).abs())
        .fold(0.0f32, f32::max)
}

fn trace_hidden_states<B: Backend>(
    model: &KimiLinearModel<B>,
    input_ids: Tensor<B, 2, Int>,
    selected_hidden_layers: &[usize],
) -> Vec<KimiBaselineSliceParityHiddenState> {
    let mut hidden = model.embed_tokens(input_ids);
    let mut hidden_states = Vec::new();

    for (layer_idx, layer) in model.layers().iter().enumerate() {
        hidden = layer.forward(hidden);
        if selected_hidden_layers.contains(&layer_idx) {
            hidden_states.push(KimiBaselineSliceParityHiddenState {
                layer_idx,
                tensor: tensor_to_parity(hidden.clone()),
            });
        }
    }

    hidden_states
}

fn baseline_slice_input_ids<B: Backend>(tokens: &[usize], device: &B::Device) -> Tensor<B, 2, Int> {
    let ints = tokens.iter().map(|&token| token as i64).collect::<Vec<_>>();
    Tensor::<B, 1, Int>::from_ints(ints.as_slice(), device).reshape([1, tokens.len()])
}

fn tensor_to_parity<const D: usize, B: Backend>(
    tensor: Tensor<B, D>,
) -> KimiBaselineSliceParityTensor {
    let dims = tensor.dims().into_iter().collect::<Vec<_>>();
    let numel = dims.iter().product::<usize>();
    let values: Vec<f32> = tensor.reshape([numel]).into_data().to_vec().unwrap();
    KimiBaselineSliceParityTensor { dims, values }
}
