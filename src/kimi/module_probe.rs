use burn::nn::LinearConfig;
use burn::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::path::Path;

use crate::kimi::cache::{KimiKdaCache, KimiMlaCache};
use crate::kimi::config::{KimiArtifactConfig, KimiArtifactConfigError};
use crate::kimi::import::{KimiArtifactUnderstanding, KimiImportError};
use crate::kimi::index::KimiTensorLocator;
use crate::kimi::payload::{
    load_named_tensor_payloads, load_param_tensor, KimiBaselinePayloadError, KimiDecodedTensor,
};
use crate::kimi::schedule::KimiAttentionLayerKind;
use crate::kimi::slice_parity::{
    KimiBaselineSliceParityArtifactSpec, KimiBaselineSliceParityTensor,
};
use crate::rms_norm::RmsNormConfig;

pub const KIMI_MODULE_PROBE_REQUEST_KIND: &str = "attnres.kimi.module_probe_request";
pub const KIMI_MODULE_PROBE_FIXTURE_KIND: &str = "attnres.kimi.module_probe_fixture";
pub const KIMI_MODULE_PROBE_VERSION: u32 = 1;
pub const KIMI_MODULE_PROBE_RUNTIME_DTYPE: &str = "float32";
pub const KIMI_MODULE_PROBE_DEFAULT_SEED: u64 = 20260318;
pub const KIMI_MODULE_PROBE_DEFAULT_SEQUENCE_LEN: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KimiModuleProbeTarget {
    KdaAttention { layer_idx: usize },
    MlaAttention { layer_idx: usize },
    FinalNorm,
    LmHead,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiModuleProbeToleranceSpec {
    pub runtime_dtype: String,
    pub output_max_abs_diff: f32,
    pub cache_max_abs_diff: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiModuleProbeRequestCase {
    pub name: String,
    pub target: KimiModuleProbeTarget,
    pub input: KimiBaselineSliceParityTensor,
    #[serde(default)]
    pub compare_decode: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiModuleProbeRequest {
    pub kind: String,
    pub version: u32,
    pub seed: u64,
    pub artifact: KimiBaselineSliceParityArtifactSpec,
    pub tolerances: KimiModuleProbeToleranceSpec,
    pub probes: Vec<KimiModuleProbeRequestCase>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiModuleProbeFingerprint {
    pub tensor_names: Vec<String>,
    pub shard_paths: Vec<String>,
    pub tensor_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KimiModuleProbeCache {
    Mla {
        processed_tokens: usize,
        keys: KimiBaselineSliceParityTensor,
        values: KimiBaselineSliceParityTensor,
    },
    Kda {
        processed_tokens: usize,
        q_conv_state: KimiBaselineSliceParityTensor,
        k_conv_state: KimiBaselineSliceParityTensor,
        v_conv_state: KimiBaselineSliceParityTensor,
        recurrent_state: KimiBaselineSliceParityTensor,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiModuleProbeDecodeStep {
    pub token_index: usize,
    pub output: KimiBaselineSliceParityTensor,
    pub cache: KimiModuleProbeCache,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiModuleProbeFixtureCase {
    pub name: String,
    pub target: KimiModuleProbeTarget,
    pub input: KimiBaselineSliceParityTensor,
    pub output: KimiBaselineSliceParityTensor,
    pub compare_decode: bool,
    pub decode_steps: Vec<KimiModuleProbeDecodeStep>,
    pub fingerprint: KimiModuleProbeFingerprint,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiModuleProbeFixture {
    pub kind: String,
    pub version: u32,
    pub seed: u64,
    pub artifact: KimiBaselineSliceParityArtifactSpec,
    pub tolerances: KimiModuleProbeToleranceSpec,
    pub probes: Vec<KimiModuleProbeFixtureCase>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum KimiModuleProbeError {
    Import(KimiImportError),
    Payload(KimiBaselinePayloadError),
    UnexpectedRequestKind { kind: String },
    UnexpectedFixtureKind { kind: String },
    UnsupportedVersion { version: u32 },
    UnsupportedRuntimeDtype { runtime_dtype: String },
    ArtifactFieldMismatch { field: String, expected: String, actual: String },
    EmptyProbeList,
    ProbeCountMismatch { expected: usize, actual: usize },
    ProbeMetadataMismatch {
        probe_name: String,
        field: String,
        expected: String,
        actual: String,
    },
    ProbeLayerOutOfRange { layer_idx: usize, num_hidden_layers: usize },
    ProbeAttentionKindMismatch {
        layer_idx: usize,
        expected: KimiAttentionLayerKind,
        actual: KimiAttentionLayerKind,
    },
    DecodeOnlySupportedForAttention { probe_name: String },
    InputShapeMismatch {
        probe_name: String,
        expected_hidden_size: usize,
        actual_dims: Vec<usize>,
    },
    TensorShapeMismatch {
        probe_name: String,
        tensor_label: String,
        expected_dims: Vec<usize>,
        actual_dims: Vec<usize>,
    },
    DecodeStepCountMismatch {
        probe_name: String,
        expected: usize,
        actual: usize,
    },
    ProcessedTokenMismatch {
        probe_name: String,
        cache_label: String,
        expected: usize,
        actual: usize,
    },
    FingerprintMismatch {
        probe_name: String,
        expected: KimiModuleProbeFingerprint,
        actual: KimiModuleProbeFingerprint,
    },
    OutputToleranceExceeded {
        probe_name: String,
        max_abs_diff: f32,
        tolerance: f32,
    },
    CacheToleranceExceeded {
        probe_name: String,
        cache_label: String,
        max_abs_diff: f32,
        tolerance: f32,
    },
}

impl Display for KimiModuleProbeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Import(err) => write!(f, "{err}"),
            Self::Payload(err) => write!(f, "{err}"),
            Self::UnexpectedRequestKind { kind } => write!(
                f,
                "expected module probe request kind '{KIMI_MODULE_PROBE_REQUEST_KIND}', got '{kind}'"
            ),
            Self::UnexpectedFixtureKind { kind } => write!(
                f,
                "expected module probe fixture kind '{KIMI_MODULE_PROBE_FIXTURE_KIND}', got '{kind}'"
            ),
            Self::UnsupportedVersion { version } => write!(
                f,
                "expected module probe version {KIMI_MODULE_PROBE_VERSION}, got {version}"
            ),
            Self::UnsupportedRuntimeDtype { runtime_dtype } => write!(
                f,
                "expected module probe runtime dtype '{KIMI_MODULE_PROBE_RUNTIME_DTYPE}', got '{runtime_dtype}'"
            ),
            Self::ArtifactFieldMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "module probe artifact field '{field}' expected '{expected}', got '{actual}'"
            ),
            Self::EmptyProbeList => write!(f, "module probe request must contain at least one probe"),
            Self::ProbeCountMismatch { expected, actual } => write!(
                f,
                "module probe fixture expected {expected} probes, got {actual}"
            ),
            Self::ProbeMetadataMismatch {
                probe_name,
                field,
                expected,
                actual,
            } => write!(
                f,
                "module probe '{probe_name}' field '{field}' expected '{expected}', got '{actual}'"
            ),
            Self::ProbeLayerOutOfRange {
                layer_idx,
                num_hidden_layers,
            } => write!(
                f,
                "module probe layer {layer_idx} is out of range for num_hidden_layers={num_hidden_layers}"
            ),
            Self::ProbeAttentionKindMismatch {
                layer_idx,
                expected,
                actual,
            } => write!(
                f,
                "module probe layer {layer_idx} expected attention kind {expected:?}, got {actual:?}"
            ),
            Self::DecodeOnlySupportedForAttention { probe_name } => write!(
                f,
                "module probe '{probe_name}' requested decode comparison for a non-attention target"
            ),
            Self::InputShapeMismatch {
                probe_name,
                expected_hidden_size,
                actual_dims,
            } => write!(
                f,
                "module probe '{probe_name}' expected input dims [batch, seq, {expected_hidden_size}], got {:?}",
                actual_dims
            ),
            Self::TensorShapeMismatch {
                probe_name,
                tensor_label,
                expected_dims,
                actual_dims,
            } => write!(
                f,
                "module probe '{probe_name}' tensor '{tensor_label}' expected dims {:?}, got {:?}",
                expected_dims, actual_dims
            ),
            Self::DecodeStepCountMismatch {
                probe_name,
                expected,
                actual,
            } => write!(
                f,
                "module probe '{probe_name}' expected {expected} decode steps, got {actual}"
            ),
            Self::ProcessedTokenMismatch {
                probe_name,
                cache_label,
                expected,
                actual,
            } => write!(
                f,
                "module probe '{probe_name}' cache '{cache_label}' expected processed_tokens={expected}, got {actual}"
            ),
            Self::FingerprintMismatch {
                probe_name,
                expected,
                actual,
            } => write!(
                f,
                "module probe '{probe_name}' tensor fingerprint mismatch: expected {:?}, got {:?}",
                expected, actual
            ),
            Self::OutputToleranceExceeded {
                probe_name,
                max_abs_diff,
                tolerance,
            } => write!(
                f,
                "module probe '{probe_name}' output max_abs_diff {max_abs_diff} exceeded tolerance {tolerance}"
            ),
            Self::CacheToleranceExceeded {
                probe_name,
                cache_label,
                max_abs_diff,
                tolerance,
            } => write!(
                f,
                "module probe '{probe_name}' cache '{cache_label}' max_abs_diff {max_abs_diff} exceeded tolerance {tolerance}"
            ),
        }
    }
}

impl std::error::Error for KimiModuleProbeError {}

impl From<KimiImportError> for KimiModuleProbeError {
    fn from(err: KimiImportError) -> Self {
        Self::Import(err)
    }
}

impl From<KimiArtifactConfigError> for KimiModuleProbeError {
    fn from(err: KimiArtifactConfigError) -> Self {
        Self::Import(KimiImportError::Config(err))
    }
}

impl From<KimiBaselinePayloadError> for KimiModuleProbeError {
    fn from(err: KimiBaselinePayloadError) -> Self {
        Self::Payload(err)
    }
}

pub fn build_default_module_probe_request(
    config: &KimiArtifactConfig,
) -> Result<KimiModuleProbeRequest, KimiModuleProbeError> {
    let artifact = KimiBaselineSliceParityArtifactSpec {
        model_type: config.model_type.clone(),
        dtype: config.dtype.clone(),
        num_hidden_layers: config.num_hidden_layers,
        hidden_size: config.hidden_size,
        vocab_size: config.vocab_size,
    };

    let first_kda = config
        .linear_attn_config
        .kda_layers
        .first()
        .copied()
        .map(|layer_idx| layer_idx - 1);
    let first_mla = config
        .linear_attn_config
        .full_attn_layers
        .first()
        .copied()
        .map(|layer_idx| layer_idx - 1);

    let mut probes = Vec::new();
    if let Some(layer_idx) = first_kda {
        probes.push(KimiModuleProbeRequestCase {
            name: format!("kda_layer_{layer_idx}_seq{KIMI_MODULE_PROBE_DEFAULT_SEQUENCE_LEN}"),
            target: KimiModuleProbeTarget::KdaAttention { layer_idx },
            input: default_probe_input(
                config.hidden_size,
                KIMI_MODULE_PROBE_DEFAULT_SEQUENCE_LEN,
                0.0,
            ),
            compare_decode: true,
        });
    }
    if let Some(layer_idx) = first_mla {
        probes.push(KimiModuleProbeRequestCase {
            name: format!("mla_layer_{layer_idx}_seq{KIMI_MODULE_PROBE_DEFAULT_SEQUENCE_LEN}"),
            target: KimiModuleProbeTarget::MlaAttention { layer_idx },
            input: default_probe_input(
                config.hidden_size,
                KIMI_MODULE_PROBE_DEFAULT_SEQUENCE_LEN,
                1000.0,
            ),
            compare_decode: true,
        });
    }
    probes.push(KimiModuleProbeRequestCase {
        name: "final_norm_seq4".to_string(),
        target: KimiModuleProbeTarget::FinalNorm,
        input: default_probe_input(
            config.hidden_size,
            KIMI_MODULE_PROBE_DEFAULT_SEQUENCE_LEN,
            2000.0,
        ),
        compare_decode: false,
    });
    probes.push(KimiModuleProbeRequestCase {
        name: "lm_head_seq4".to_string(),
        target: KimiModuleProbeTarget::LmHead,
        input: default_probe_input(
            config.hidden_size,
            KIMI_MODULE_PROBE_DEFAULT_SEQUENCE_LEN,
            3000.0,
        ),
        compare_decode: false,
    });

    Ok(KimiModuleProbeRequest {
        kind: KIMI_MODULE_PROBE_REQUEST_KIND.to_string(),
        version: KIMI_MODULE_PROBE_VERSION,
        seed: KIMI_MODULE_PROBE_DEFAULT_SEED,
        artifact,
        tolerances: KimiModuleProbeToleranceSpec {
            runtime_dtype: KIMI_MODULE_PROBE_RUNTIME_DTYPE.to_string(),
            output_max_abs_diff: 1e-4,
            cache_max_abs_diff: 1e-4,
        },
        probes,
    })
}

pub fn generate_module_probe_fixture_from_dir<B: Backend, P: AsRef<Path>>(
    artifact_dir: P,
    request: &KimiModuleProbeRequest,
    device: &B::Device,
) -> Result<KimiModuleProbeFixture, KimiModuleProbeError> {
    let artifact_dir = artifact_dir.as_ref();
    let understanding = KimiArtifactUnderstanding::load_from_dir(artifact_dir)?;
    request.try_validate_against_understanding(&understanding)?;

    B::seed(device, request.seed);
    let probes = request
        .probes
        .iter()
        .map(|probe| generate_probe_case::<B>(&understanding, artifact_dir, probe, device))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(KimiModuleProbeFixture {
        kind: KIMI_MODULE_PROBE_FIXTURE_KIND.to_string(),
        version: KIMI_MODULE_PROBE_VERSION,
        seed: request.seed,
        artifact: request.artifact.clone(),
        tolerances: request.tolerances.clone(),
        probes,
    })
}

pub fn compare_module_probe_fixture_from_dir<B: Backend, P: AsRef<Path>>(
    artifact_dir: P,
    request: &KimiModuleProbeRequest,
    fixture: &KimiModuleProbeFixture,
    device: &B::Device,
) -> Result<(), KimiModuleProbeError> {
    fixture.try_validate_static()?;
    request.try_validate_fixture_metadata(fixture)?;
    let observed = generate_module_probe_fixture_from_dir::<B, _>(artifact_dir, request, device)?;
    if fixture.probes.len() != observed.probes.len() {
        return Err(KimiModuleProbeError::ProbeCountMismatch {
            expected: fixture.probes.len(),
            actual: observed.probes.len(),
        });
    }

    for (expected, actual) in fixture.probes.iter().zip(observed.probes.iter()) {
        compare_probe_metadata(expected, actual)?;
        if expected.fingerprint != actual.fingerprint {
            return Err(KimiModuleProbeError::FingerprintMismatch {
                probe_name: expected.name.clone(),
                expected: expected.fingerprint.clone(),
                actual: actual.fingerprint.clone(),
            });
        }

        ensure_same_tensor_dims(&expected.name, "output", &expected.output, &actual.output)?;
        let output_diff = max_abs_diff(&expected.output, &actual.output);
        if output_diff > fixture.tolerances.output_max_abs_diff {
            return Err(KimiModuleProbeError::OutputToleranceExceeded {
                probe_name: expected.name.clone(),
                max_abs_diff: output_diff,
                tolerance: fixture.tolerances.output_max_abs_diff,
            });
        }

        if expected.decode_steps.len() != actual.decode_steps.len() {
            return Err(KimiModuleProbeError::DecodeStepCountMismatch {
                probe_name: expected.name.clone(),
                expected: expected.decode_steps.len(),
                actual: actual.decode_steps.len(),
            });
        }
        for (expected_step, actual_step) in expected.decode_steps.iter().zip(actual.decode_steps.iter()) {
            if expected_step.token_index != actual_step.token_index {
                return Err(KimiModuleProbeError::ProbeMetadataMismatch {
                    probe_name: expected.name.clone(),
                    field: "decode_steps.token_index".to_string(),
                    expected: expected_step.token_index.to_string(),
                    actual: actual_step.token_index.to_string(),
                });
            }
            ensure_same_tensor_dims(
                &expected.name,
                &format!("decode_steps[{}].output", expected_step.token_index),
                &expected_step.output,
                &actual_step.output,
            )?;
            let diff = max_abs_diff(&expected_step.output, &actual_step.output);
            if diff > fixture.tolerances.output_max_abs_diff {
                return Err(KimiModuleProbeError::OutputToleranceExceeded {
                    probe_name: format!("{} decode step {}", expected.name, expected_step.token_index),
                    max_abs_diff: diff,
                    tolerance: fixture.tolerances.output_max_abs_diff,
                });
            }
            compare_cache(
                &expected.name,
                &expected_step.cache,
                &actual_step.cache,
                fixture.tolerances.cache_max_abs_diff,
            )?;
        }
    }

    Ok(())
}

impl KimiModuleProbeRequest {
    pub fn try_validate_against_understanding(
        &self,
        understanding: &KimiArtifactUnderstanding,
    ) -> Result<(), KimiModuleProbeError> {
        self.try_validate_static()?;
        compare_artifact(&self.artifact, &understanding.config)?;
        for probe in &self.probes {
            validate_probe_request(probe, understanding)?;
        }
        Ok(())
    }

    fn try_validate_static(&self) -> Result<(), KimiModuleProbeError> {
        if self.kind != KIMI_MODULE_PROBE_REQUEST_KIND {
            return Err(KimiModuleProbeError::UnexpectedRequestKind {
                kind: self.kind.clone(),
            });
        }
        if self.version != KIMI_MODULE_PROBE_VERSION {
            return Err(KimiModuleProbeError::UnsupportedVersion {
                version: self.version,
            });
        }
        if self.tolerances.runtime_dtype != KIMI_MODULE_PROBE_RUNTIME_DTYPE {
            return Err(KimiModuleProbeError::UnsupportedRuntimeDtype {
                runtime_dtype: self.tolerances.runtime_dtype.clone(),
            });
        }
        if self.probes.is_empty() {
            return Err(KimiModuleProbeError::EmptyProbeList);
        }
        Ok(())
    }

    fn try_validate_fixture_metadata(
        &self,
        fixture: &KimiModuleProbeFixture,
    ) -> Result<(), KimiModuleProbeError> {
        fixture.try_validate_static()?;
        if fixture.seed != self.seed {
            return Err(KimiModuleProbeError::ArtifactFieldMismatch {
                field: "seed".to_string(),
                expected: self.seed.to_string(),
                actual: fixture.seed.to_string(),
            });
        }
        if fixture.artifact != self.artifact {
            return Err(KimiModuleProbeError::ArtifactFieldMismatch {
                field: "artifact".to_string(),
                expected: serde_json::to_string(&self.artifact).unwrap(),
                actual: serde_json::to_string(&fixture.artifact).unwrap(),
            });
        }
        if fixture.probes.len() != self.probes.len() {
            return Err(KimiModuleProbeError::ProbeCountMismatch {
                expected: self.probes.len(),
                actual: fixture.probes.len(),
            });
        }
        for (request_probe, fixture_probe) in self.probes.iter().zip(fixture.probes.iter()) {
            compare_probe_request_to_fixture(request_probe, fixture_probe)?;
        }
        Ok(())
    }
}

impl KimiModuleProbeFixture {
    pub fn try_validate_static(&self) -> Result<(), KimiModuleProbeError> {
        if self.kind != KIMI_MODULE_PROBE_FIXTURE_KIND {
            return Err(KimiModuleProbeError::UnexpectedFixtureKind {
                kind: self.kind.clone(),
            });
        }
        if self.version != KIMI_MODULE_PROBE_VERSION {
            return Err(KimiModuleProbeError::UnsupportedVersion {
                version: self.version,
            });
        }
        if self.tolerances.runtime_dtype != KIMI_MODULE_PROBE_RUNTIME_DTYPE {
            return Err(KimiModuleProbeError::UnsupportedRuntimeDtype {
                runtime_dtype: self.tolerances.runtime_dtype.clone(),
            });
        }
        Ok(())
    }
}

fn validate_probe_request(
    probe: &KimiModuleProbeRequestCase,
    understanding: &KimiArtifactUnderstanding,
) -> Result<(), KimiModuleProbeError> {
    let dims = &probe.input.dims;
    if dims.len() != 3 || *dims.last().unwrap() != understanding.config.hidden_size {
        return Err(KimiModuleProbeError::InputShapeMismatch {
            probe_name: probe.name.clone(),
            expected_hidden_size: understanding.config.hidden_size,
            actual_dims: dims.clone(),
        });
    }

    match probe.target {
        KimiModuleProbeTarget::KdaAttention { layer_idx } => {
            validate_attention_probe_layer(
                understanding,
                layer_idx,
                KimiAttentionLayerKind::LinearAttentionKda,
            )?;
        }
        KimiModuleProbeTarget::MlaAttention { layer_idx } => {
            validate_attention_probe_layer(
                understanding,
                layer_idx,
                KimiAttentionLayerKind::FullAttention,
            )?;
        }
        KimiModuleProbeTarget::FinalNorm | KimiModuleProbeTarget::LmHead => {
            if probe.compare_decode {
                return Err(KimiModuleProbeError::DecodeOnlySupportedForAttention {
                    probe_name: probe.name.clone(),
                });
            }
        }
    }

    Ok(())
}

fn validate_attention_probe_layer(
    understanding: &KimiArtifactUnderstanding,
    layer_idx: usize,
    expected_kind: KimiAttentionLayerKind,
) -> Result<(), KimiModuleProbeError> {
    if layer_idx >= understanding.config.num_hidden_layers {
        return Err(KimiModuleProbeError::ProbeLayerOutOfRange {
            layer_idx,
            num_hidden_layers: understanding.config.num_hidden_layers,
        });
    }
    let actual_kind = understanding
        .layer_schedule
        .try_attention_kind(layer_idx)
        .map_err(KimiArtifactConfigError::from)?;
    if actual_kind != expected_kind {
        return Err(KimiModuleProbeError::ProbeAttentionKindMismatch {
            layer_idx,
            expected: expected_kind,
            actual: actual_kind,
        });
    }
    Ok(())
}

fn generate_probe_case<B: Backend>(
    understanding: &KimiArtifactUnderstanding,
    artifact_dir: &Path,
    probe: &KimiModuleProbeRequestCase,
    device: &B::Device,
) -> Result<KimiModuleProbeFixtureCase, KimiModuleProbeError> {
    let tensor_names = tensor_names_for_probe(&probe.target, &understanding.config)?;
    let locator = KimiTensorLocator::from_index(&understanding.shard_index);
    let mut shard_paths = tensor_names
        .iter()
        .map(|tensor_name| {
            locator
                .shard_for_tensor(tensor_name)
                .ok_or_else(|| KimiImportError::TensorLocator(
                    crate::kimi::index::KimiTensorLocatorError::MissingTensor {
                        tensor_name: tensor_name.clone(),
                    },
                ))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    shard_paths.sort();
    shard_paths.dedup();
    let payloads =
        load_named_tensor_payloads(&locator, &tensor_names, artifact_dir, &understanding.config.dtype)?;
    let fingerprint = KimiModuleProbeFingerprint {
        tensor_names: tensor_names.clone(),
        shard_paths,
        tensor_fingerprint: fnv64_tensor_payloads(&tensor_names, &payloads),
    };
    let input = parity_tensor_to_backend::<B>(&probe.input, device);

    let (output, decode_steps) = match probe.target {
        KimiModuleProbeTarget::KdaAttention { layer_idx } => {
            let baseline = understanding.config.try_baseline_config()?;
            let mut attention = baseline.attention.init_kda(device);
            apply_attention_payloads_kda(&mut attention, layer_idx, &tensor_names, &payloads)?;
            let (output, _) = attention.forward(input.clone(), None);
            let decode_steps = if probe.compare_decode {
                generate_kda_decode_steps(&attention, input)
            } else {
                Vec::new()
            };
            (output, decode_steps)
        }
        KimiModuleProbeTarget::MlaAttention { layer_idx } => {
            let baseline = understanding.config.try_baseline_config()?;
            let mut attention = baseline.attention.init_mla(device);
            apply_attention_payloads_mla(&mut attention, layer_idx, &tensor_names, &payloads)?;
            let (output, _) = attention.forward(input.clone(), None);
            let decode_steps = if probe.compare_decode {
                generate_mla_decode_steps(&attention, input)
            } else {
                Vec::new()
            };
            (output, decode_steps)
        }
        KimiModuleProbeTarget::FinalNorm => {
            let mut final_norm = RmsNormConfig::new(understanding.config.hidden_size)
                .with_eps(understanding.config.rms_norm_eps)
                .init(device);
            let payload = payloads
                .get("model.norm.weight")
                .expect("module probe final norm payload should exist");
            load_param_tensor(final_norm.gamma_param_mut(), "model.norm.weight", payload)?;
            (final_norm.forward(input), Vec::new())
        }
        KimiModuleProbeTarget::LmHead => {
            let mut lm_head = LinearConfig::new(
                understanding.config.hidden_size,
                understanding.config.vocab_size,
            )
            .with_bias(false)
            .init(device);
            let payload = payloads
                .get("lm_head.weight")
                .expect("module probe lm_head payload should exist");
            load_param_tensor(&mut lm_head.weight, "lm_head.weight", payload)?;
            (lm_head.forward(input), Vec::new())
        }
    };

    Ok(KimiModuleProbeFixtureCase {
        name: probe.name.clone(),
        target: probe.target.clone(),
        input: probe.input.clone(),
        output: tensor_to_parity(output),
        compare_decode: probe.compare_decode,
        decode_steps,
        fingerprint,
    })
}

fn tensor_names_for_probe(
    target: &KimiModuleProbeTarget,
    _config: &KimiArtifactConfig,
) -> Result<Vec<String>, KimiModuleProbeError> {
    Ok(match target {
        KimiModuleProbeTarget::KdaAttention { layer_idx } => vec![
            format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
            format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
            format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
            format!("model.layers.{layer_idx}.self_attn.q_conv1d.weight"),
            format!("model.layers.{layer_idx}.self_attn.k_conv1d.weight"),
            format!("model.layers.{layer_idx}.self_attn.v_conv1d.weight"),
            format!("model.layers.{layer_idx}.self_attn.A_log"),
            format!("model.layers.{layer_idx}.self_attn.f_a_proj.weight"),
            format!("model.layers.{layer_idx}.self_attn.f_b_proj.weight"),
            format!("model.layers.{layer_idx}.self_attn.dt_bias"),
            format!("model.layers.{layer_idx}.self_attn.b_proj.weight"),
            format!("model.layers.{layer_idx}.self_attn.g_a_proj.weight"),
            format!("model.layers.{layer_idx}.self_attn.g_b_proj.weight"),
            format!("model.layers.{layer_idx}.self_attn.o_norm.weight"),
            format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
        ],
        KimiModuleProbeTarget::MlaAttention { layer_idx } => vec![
            format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
            format!("model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight"),
            format!("model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight"),
            format!("model.layers.{layer_idx}.self_attn.kv_b_proj.weight"),
            format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
        ],
        KimiModuleProbeTarget::FinalNorm => vec!["model.norm.weight".to_string()],
        KimiModuleProbeTarget::LmHead => {
            vec!["lm_head.weight".to_string()]
        }
    })
}

fn apply_attention_payloads_kda<B: Backend>(
    attention: &mut crate::kimi::attention::KimiKdaAttention<B>,
    layer_idx: usize,
    tensor_names: &[String],
    payloads: &std::collections::BTreeMap<String, KimiDecodedTensor>,
) -> Result<(), KimiModuleProbeError> {
    let prefix = format!("model.layers.{layer_idx}.self_attn.");
    for tensor_name in tensor_names {
        let leaf = tensor_name
            .strip_prefix(&prefix)
            .expect("module probe KDA tensor should match layer prefix");
        let payload = payloads.get(tensor_name).expect("module probe payload must exist");
        attention.try_apply_tensor_payload(tensor_name, leaf, payload)?;
    }
    Ok(())
}

fn apply_attention_payloads_mla<B: Backend>(
    attention: &mut crate::kimi::attention::KimiMlaAttention<B>,
    layer_idx: usize,
    tensor_names: &[String],
    payloads: &std::collections::BTreeMap<String, KimiDecodedTensor>,
) -> Result<(), KimiModuleProbeError> {
    let prefix = format!("model.layers.{layer_idx}.self_attn.");
    for tensor_name in tensor_names {
        let leaf = tensor_name
            .strip_prefix(&prefix)
            .expect("module probe MLA tensor should match layer prefix");
        let payload = payloads.get(tensor_name).expect("module probe payload must exist");
        attention.try_apply_tensor_payload(tensor_name, leaf, payload)?;
    }
    Ok(())
}

fn generate_kda_decode_steps<B: Backend>(
    attention: &crate::kimi::attention::KimiKdaAttention<B>,
    input: Tensor<B, 3>,
) -> Vec<KimiModuleProbeDecodeStep> {
    let [batch, seq_len, hidden] = input.dims();
    let mut cache: Option<KimiKdaCache<B>> = None;
    let mut steps = Vec::with_capacity(seq_len);
    for token_index in 0..seq_len {
        let token = input.clone().slice([0..batch, token_index..token_index + 1, 0..hidden]);
        let (output, next_cache) = attention.forward(token, cache.as_ref());
        steps.push(KimiModuleProbeDecodeStep {
            token_index,
            output: tensor_to_parity(output),
            cache: KimiModuleProbeCache::Kda {
                processed_tokens: next_cache.processed_tokens(),
                q_conv_state: tensor_to_parity(next_cache.q_conv_state().clone()),
                k_conv_state: tensor_to_parity(next_cache.k_conv_state().clone()),
                v_conv_state: tensor_to_parity(next_cache.v_conv_state().clone()),
                recurrent_state: tensor_to_parity(next_cache.recurrent_state().clone()),
            },
        });
        cache = Some(next_cache);
    }
    steps
}

fn generate_mla_decode_steps<B: Backend>(
    attention: &crate::kimi::attention::KimiMlaAttention<B>,
    input: Tensor<B, 3>,
) -> Vec<KimiModuleProbeDecodeStep> {
    let [batch, seq_len, hidden] = input.dims();
    let mut cache: Option<KimiMlaCache<B>> = None;
    let mut steps = Vec::with_capacity(seq_len);
    for token_index in 0..seq_len {
        let token = input.clone().slice([0..batch, token_index..token_index + 1, 0..hidden]);
        let (output, next_cache) = attention.forward(token, cache.as_ref());
        steps.push(KimiModuleProbeDecodeStep {
            token_index,
            output: tensor_to_parity(output),
            cache: KimiModuleProbeCache::Mla {
                processed_tokens: next_cache.processed_tokens(),
                keys: tensor_to_parity(next_cache.keys().clone()),
                values: tensor_to_parity(next_cache.values().clone()),
            },
        });
        cache = Some(next_cache);
    }
    steps
}

fn compare_cache(
    probe_name: &str,
    expected: &KimiModuleProbeCache,
    actual: &KimiModuleProbeCache,
    tolerance: f32,
) -> Result<(), KimiModuleProbeError> {
    match (expected, actual) {
        (
            KimiModuleProbeCache::Mla {
                processed_tokens: expected_processed_tokens,
                keys: expected_keys,
                values: expected_values,
            },
            KimiModuleProbeCache::Mla {
                processed_tokens: actual_processed_tokens,
                keys: actual_keys,
                values: actual_values,
            },
        ) => {
            compare_processed_tokens(
                probe_name,
                "mla",
                *expected_processed_tokens,
                *actual_processed_tokens,
            )?;
            ensure_same_tensor_dims(probe_name, "mla.keys", expected_keys, actual_keys)?;
            ensure_same_tensor_dims(probe_name, "mla.values", expected_values, actual_values)?;
            let keys_diff = max_abs_diff(expected_keys, actual_keys);
            if keys_diff > tolerance {
                return Err(KimiModuleProbeError::CacheToleranceExceeded {
                    probe_name: probe_name.to_string(),
                    cache_label: "mla.keys".to_string(),
                    max_abs_diff: keys_diff,
                    tolerance,
                });
            }
            let values_diff = max_abs_diff(expected_values, actual_values);
            if values_diff > tolerance {
                return Err(KimiModuleProbeError::CacheToleranceExceeded {
                    probe_name: probe_name.to_string(),
                    cache_label: "mla.values".to_string(),
                    max_abs_diff: values_diff,
                    tolerance,
                });
            }
        }
        (
            KimiModuleProbeCache::Kda {
                processed_tokens: expected_processed_tokens,
                q_conv_state: expected_q,
                k_conv_state: expected_k,
                v_conv_state: expected_v,
                recurrent_state: expected_recurrent,
            },
            KimiModuleProbeCache::Kda {
                processed_tokens: actual_processed_tokens,
                q_conv_state: actual_q,
                k_conv_state: actual_k,
                v_conv_state: actual_v,
                recurrent_state: actual_recurrent,
            },
        ) => {
            compare_processed_tokens(
                probe_name,
                "kda",
                *expected_processed_tokens,
                *actual_processed_tokens,
            )?;
            for (label, expected_tensor, actual_tensor) in [
                ("kda.q_conv_state", expected_q, actual_q),
                ("kda.k_conv_state", expected_k, actual_k),
                ("kda.v_conv_state", expected_v, actual_v),
                ("kda.recurrent_state", expected_recurrent, actual_recurrent),
            ] {
                ensure_same_tensor_dims(probe_name, label, expected_tensor, actual_tensor)?;
                let diff = max_abs_diff(expected_tensor, actual_tensor);
                if diff > tolerance {
                    return Err(KimiModuleProbeError::CacheToleranceExceeded {
                        probe_name: probe_name.to_string(),
                        cache_label: label.to_string(),
                        max_abs_diff: diff,
                        tolerance,
                    });
                }
            }
        }
        _ => {
            return Err(KimiModuleProbeError::CacheToleranceExceeded {
                probe_name: probe_name.to_string(),
                cache_label: "cache_kind".to_string(),
                max_abs_diff: f32::INFINITY,
                tolerance,
            });
        }
    }

    Ok(())
}

fn compare_probe_request_to_fixture(
    request_probe: &KimiModuleProbeRequestCase,
    fixture_probe: &KimiModuleProbeFixtureCase,
) -> Result<(), KimiModuleProbeError> {
    if fixture_probe.name != request_probe.name {
        return Err(KimiModuleProbeError::ProbeMetadataMismatch {
            probe_name: request_probe.name.clone(),
            field: "name".to_string(),
            expected: request_probe.name.clone(),
            actual: fixture_probe.name.clone(),
        });
    }
    if fixture_probe.target != request_probe.target {
        return Err(KimiModuleProbeError::ProbeMetadataMismatch {
            probe_name: request_probe.name.clone(),
            field: "target".to_string(),
            expected: serde_json::to_string(&request_probe.target).unwrap(),
            actual: serde_json::to_string(&fixture_probe.target).unwrap(),
        });
    }
    if fixture_probe.input != request_probe.input {
        return Err(KimiModuleProbeError::ProbeMetadataMismatch {
            probe_name: request_probe.name.clone(),
            field: "input".to_string(),
            expected: serde_json::to_string(&request_probe.input).unwrap(),
            actual: serde_json::to_string(&fixture_probe.input).unwrap(),
        });
    }
    if fixture_probe.compare_decode != request_probe.compare_decode {
        return Err(KimiModuleProbeError::ProbeMetadataMismatch {
            probe_name: request_probe.name.clone(),
            field: "compare_decode".to_string(),
            expected: request_probe.compare_decode.to_string(),
            actual: fixture_probe.compare_decode.to_string(),
        });
    }
    Ok(())
}

fn compare_probe_metadata(
    expected: &KimiModuleProbeFixtureCase,
    actual: &KimiModuleProbeFixtureCase,
) -> Result<(), KimiModuleProbeError> {
    for (field, expected_value, actual_value) in [
        ("name", expected.name.clone(), actual.name.clone()),
        (
            "target",
            serde_json::to_string(&expected.target).unwrap(),
            serde_json::to_string(&actual.target).unwrap(),
        ),
        (
            "input",
            serde_json::to_string(&expected.input).unwrap(),
            serde_json::to_string(&actual.input).unwrap(),
        ),
        (
            "compare_decode",
            expected.compare_decode.to_string(),
            actual.compare_decode.to_string(),
        ),
    ] {
        if expected_value != actual_value {
            return Err(KimiModuleProbeError::ProbeMetadataMismatch {
                probe_name: expected.name.clone(),
                field: field.to_string(),
                expected: expected_value,
                actual: actual_value,
            });
        }
    }
    Ok(())
}

fn ensure_same_tensor_dims(
    probe_name: &str,
    tensor_label: &str,
    expected: &KimiBaselineSliceParityTensor,
    actual: &KimiBaselineSliceParityTensor,
) -> Result<(), KimiModuleProbeError> {
    if expected.dims != actual.dims {
        return Err(KimiModuleProbeError::TensorShapeMismatch {
            probe_name: probe_name.to_string(),
            tensor_label: tensor_label.to_string(),
            expected_dims: expected.dims.clone(),
            actual_dims: actual.dims.clone(),
        });
    }
    Ok(())
}

fn compare_processed_tokens(
    probe_name: &str,
    cache_label: &str,
    expected: usize,
    actual: usize,
) -> Result<(), KimiModuleProbeError> {
    if expected != actual {
        return Err(KimiModuleProbeError::ProcessedTokenMismatch {
            probe_name: probe_name.to_string(),
            cache_label: cache_label.to_string(),
            expected,
            actual,
        });
    }
    Ok(())
}

fn compare_artifact(
    artifact: &KimiBaselineSliceParityArtifactSpec,
    config: &KimiArtifactConfig,
) -> Result<(), KimiModuleProbeError> {
    for (field, expected, actual) in [
        ("model_type", artifact.model_type.clone(), config.model_type.clone()),
        ("dtype", artifact.dtype.clone(), config.dtype.clone()),
        (
            "num_hidden_layers",
            artifact.num_hidden_layers.to_string(),
            config.num_hidden_layers.to_string(),
        ),
        (
            "hidden_size",
            artifact.hidden_size.to_string(),
            config.hidden_size.to_string(),
        ),
        (
            "vocab_size",
            artifact.vocab_size.to_string(),
            config.vocab_size.to_string(),
        ),
    ] {
        if expected != actual {
            return Err(KimiModuleProbeError::ArtifactFieldMismatch {
                field: field.to_string(),
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn default_probe_input(
    hidden_size: usize,
    sequence_len: usize,
    offset: f32,
) -> KimiBaselineSliceParityTensor {
    let values = (0..(hidden_size * sequence_len))
        .map(|idx| {
            let x = offset + idx as f32 + 1.0;
            (x * 0.013).sin() + (x * 0.007).cos()
        })
        .collect::<Vec<_>>();
    KimiBaselineSliceParityTensor {
        dims: vec![1, sequence_len, hidden_size],
        values,
    }
}

fn parity_tensor_to_backend<B: Backend>(
    tensor: &KimiBaselineSliceParityTensor,
    device: &B::Device,
) -> Tensor<B, 3> {
    let dims = tensor.dims.clone();
    let [batch, seq_len, hidden] = [dims[0], dims[1], dims[2]];
    Tensor::<B, 3>::from_data(
        burn::tensor::TensorData::new(tensor.values.clone(), [batch, seq_len, hidden]),
        device,
    )
}

fn tensor_to_parity<const D: usize, B: Backend>(
    tensor: Tensor<B, D>,
) -> KimiBaselineSliceParityTensor {
    let dims = tensor.dims().into_iter().collect::<Vec<_>>();
    let numel = dims.iter().product::<usize>();
    let values = tensor.reshape([numel]).into_data().to_vec().unwrap();
    KimiBaselineSliceParityTensor { dims, values }
}

fn max_abs_diff(
    expected: &KimiBaselineSliceParityTensor,
    actual: &KimiBaselineSliceParityTensor,
) -> f32 {
    expected
        .values
        .iter()
        .zip(actual.values.iter())
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0f32, f32::max)
}

fn fnv64_tensor_payloads(
    tensor_names: &[String],
    payloads: &std::collections::BTreeMap<String, KimiDecodedTensor>,
) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for tensor_name in tensor_names {
        hash = fnv64_bytes(hash, tensor_name.as_bytes());
        let payload = payloads
            .get(tensor_name)
            .expect("probe fingerprint payload should exist");
        for dim in &payload.shape {
            hash = fnv64_bytes(hash, &(*dim as u64).to_le_bytes());
        }
        for value in &payload.values {
            hash = fnv64_bytes(hash, &value.to_le_bytes());
        }
    }
    format!("{hash:016x}")
}

fn fnv64_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
