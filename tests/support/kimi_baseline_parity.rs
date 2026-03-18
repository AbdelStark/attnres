#![allow(dead_code)]

use attnres::kimi::{
    KimiArtifactConfig, KimiArtifactUnderstanding, KimiAttentionLayerKind, KimiImportCoverageError,
    KimiImportError, KimiImportSelection, KimiKdaCache, KimiLinearModel, KimiMlaCache,
};
use burn::backend::NdArray;
use burn::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

pub type TestBackend = NdArray;
pub type TestDevice = <TestBackend as Backend>::Device;

pub const KIMI_BASELINE_PARITY_FILENAME: &str = "baseline-parity.json";
pub const KIMI_BASELINE_PARITY_KIND: &str = "attnres.kimi.baseline_parity";
pub const KIMI_BASELINE_PARITY_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiBaselineParityTensor {
    pub dims: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiBaselineParityPromptSpec {
    pub name: String,
    pub input_ids: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiBaselineParityHiddenState {
    pub layer_idx: usize,
    pub tensor: KimiBaselineParityTensor,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KimiBaselineParityCacheLayer {
    Mla {
        layer_idx: usize,
        processed_tokens: usize,
        keys: KimiBaselineParityTensor,
        values: KimiBaselineParityTensor,
    },
    Kda {
        layer_idx: usize,
        processed_tokens: usize,
        q_conv_state: KimiBaselineParityTensor,
        k_conv_state: KimiBaselineParityTensor,
        v_conv_state: KimiBaselineParityTensor,
        recurrent_state: KimiBaselineParityTensor,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiBaselineParityCacheStep {
    pub token_index: usize,
    pub token_id: usize,
    pub layers: Vec<KimiBaselineParityCacheLayer>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiBaselineParityPrompt {
    pub name: String,
    pub input_ids: Vec<usize>,
    pub logits: KimiBaselineParityTensor,
    pub hidden_states: Vec<KimiBaselineParityHiddenState>,
    pub cache_steps: Vec<KimiBaselineParityCacheStep>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiBaselineParityFixture {
    pub kind: String,
    pub version: u32,
    pub seed: u64,
    pub selected_hidden_layers: Vec<usize>,
    pub prompts: Vec<KimiBaselineParityPrompt>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KimiBaselineParityBundle {
    pub fixture_path: PathBuf,
    pub fixture: KimiBaselineParityFixture,
    pub understanding: KimiArtifactUnderstanding,
}

#[derive(Debug, Clone, PartialEq)]
pub enum KimiBaselineParityHarnessError {
    ReadFailed { path: String, detail: String },
    ParseFailed { path: String, detail: String },
    UnexpectedFixtureKind { kind: String },
    UnsupportedFixtureVersion { version: u32 },
    Import(KimiImportError),
    Coverage(KimiImportCoverageError),
}

impl Display for KimiBaselineParityHarnessError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadFailed { path, detail } => {
                write!(f, "failed to read baseline parity fixture '{path}': {detail}")
            }
            Self::ParseFailed { path, detail } => write!(
                f,
                "failed to parse baseline parity fixture '{path}': {detail}"
            ),
            Self::UnexpectedFixtureKind { kind } => write!(
                f,
                "expected baseline parity fixture kind '{KIMI_BASELINE_PARITY_KIND}', got '{kind}'"
            ),
            Self::UnsupportedFixtureVersion { version } => write!(
                f,
                "expected baseline parity fixture version {KIMI_BASELINE_PARITY_VERSION}, got {version}"
            ),
            Self::Import(err) => write!(f, "{err}"),
            Self::Coverage(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for KimiBaselineParityHarnessError {}

impl From<KimiImportError> for KimiBaselineParityHarnessError {
    fn from(err: KimiImportError) -> Self {
        Self::Import(err)
    }
}

impl From<KimiImportCoverageError> for KimiBaselineParityHarnessError {
    fn from(err: KimiImportCoverageError) -> Self {
        Self::Coverage(err)
    }
}

impl KimiBaselineParityFixture {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, KimiBaselineParityHarnessError> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path).map_err(|err| {
            KimiBaselineParityHarnessError::ReadFailed {
                path: path.display().to_string(),
                detail: err.to_string(),
            }
        })?;
        let fixture: Self = serde_json::from_str(&json).map_err(|err| {
            KimiBaselineParityHarnessError::ParseFailed {
                path: path.display().to_string(),
                detail: err.to_string(),
            }
        })?;
        fixture.try_validate()?;
        Ok(fixture)
    }

    pub fn try_validate(&self) -> Result<(), KimiBaselineParityHarnessError> {
        if self.kind != KIMI_BASELINE_PARITY_KIND {
            return Err(KimiBaselineParityHarnessError::UnexpectedFixtureKind {
                kind: self.kind.clone(),
            });
        }
        if self.version != KIMI_BASELINE_PARITY_VERSION {
            return Err(KimiBaselineParityHarnessError::UnsupportedFixtureVersion {
                version: self.version,
            });
        }
        Ok(())
    }

    pub fn prompt_specs(&self) -> Vec<KimiBaselineParityPromptSpec> {
        self.prompts
            .iter()
            .map(|prompt| KimiBaselineParityPromptSpec {
                name: prompt.name.clone(),
                input_ids: prompt.input_ids.clone(),
            })
            .collect()
    }
}

pub fn load_bundle_from_dir<P: AsRef<Path>>(
    dir: P,
) -> Result<KimiBaselineParityBundle, KimiBaselineParityHarnessError> {
    let dir = dir.as_ref();
    let understanding = KimiArtifactUnderstanding::load_from_dir(dir)?;
    let fixture_path = dir.join(KIMI_BASELINE_PARITY_FILENAME);
    let fixture = KimiBaselineParityFixture::load(&fixture_path)?;
    let selection = KimiImportSelection {
        layer_indices: fixture.selected_hidden_layers.clone(),
        include_embeddings: true,
        include_final_norm: true,
        include_lm_head: true,
    };
    let plan = understanding.try_slice_plan(selection)?;
    plan.try_require_loadable()?;

    Ok(KimiBaselineParityBundle {
        fixture_path,
        fixture,
        understanding,
    })
}

pub fn generate_fixture(
    config: &KimiArtifactConfig,
    seed: u64,
    selected_hidden_layers: &[usize],
    prompt_specs: &[KimiBaselineParityPromptSpec],
) -> KimiBaselineParityFixture {
    let device = Default::default();
    seed_backend(&device, seed);
    let model: KimiLinearModel<TestBackend> = config.try_init_model(&device).unwrap();

    let prompts = prompt_specs
        .iter()
        .map(|prompt| generate_prompt(&model, prompt, selected_hidden_layers, &device))
        .collect();

    KimiBaselineParityFixture {
        kind: KIMI_BASELINE_PARITY_KIND.to_string(),
        version: KIMI_BASELINE_PARITY_VERSION,
        seed,
        selected_hidden_layers: selected_hidden_layers.to_vec(),
        prompts,
    }
}

pub fn assert_bundle_matches(bundle: &KimiBaselineParityBundle, tolerance: f32) {
    let observed = generate_fixture(
        &bundle.understanding.config,
        bundle.fixture.seed,
        &bundle.fixture.selected_hidden_layers,
        &bundle.fixture.prompt_specs(),
    );
    assert_fixture_matches(&bundle.fixture, &observed, tolerance);
}

pub fn assert_fixture_matches(
    expected: &KimiBaselineParityFixture,
    observed: &KimiBaselineParityFixture,
    tolerance: f32,
) {
    assert_eq!(observed.kind, expected.kind);
    assert_eq!(observed.version, expected.version);
    assert_eq!(observed.seed, expected.seed);
    assert_eq!(
        observed.selected_hidden_layers,
        expected.selected_hidden_layers
    );
    assert_eq!(observed.prompts.len(), expected.prompts.len());

    for (prompt_idx, (expected_prompt, observed_prompt)) in expected
        .prompts
        .iter()
        .zip(observed.prompts.iter())
        .enumerate()
    {
        let prompt_label = format!("prompt[{prompt_idx}] {}", expected_prompt.name);
        assert_eq!(
            observed_prompt.name, expected_prompt.name,
            "{prompt_label} name"
        );
        assert_eq!(
            observed_prompt.input_ids, expected_prompt.input_ids,
            "{prompt_label} input ids"
        );
        assert_tensor_close(
            &expected_prompt.logits,
            &observed_prompt.logits,
            tolerance,
            &format!("{prompt_label} logits"),
        );
        assert_eq!(
            observed_prompt.hidden_states.len(),
            expected_prompt.hidden_states.len(),
            "{prompt_label} hidden state count",
        );
        for (state_idx, (expected_state, observed_state)) in expected_prompt
            .hidden_states
            .iter()
            .zip(observed_prompt.hidden_states.iter())
            .enumerate()
        {
            assert_eq!(
                observed_state.layer_idx, expected_state.layer_idx,
                "{prompt_label} hidden[{state_idx}] layer_idx"
            );
            assert_tensor_close(
                &expected_state.tensor,
                &observed_state.tensor,
                tolerance,
                &format!(
                    "{prompt_label} hidden[{state_idx}] layer {}",
                    expected_state.layer_idx
                ),
            );
        }
        assert_eq!(
            observed_prompt.cache_steps.len(),
            expected_prompt.cache_steps.len(),
            "{prompt_label} cache step count",
        );
        for (step_idx, (expected_step, observed_step)) in expected_prompt
            .cache_steps
            .iter()
            .zip(observed_prompt.cache_steps.iter())
            .enumerate()
        {
            let step_label = format!("{prompt_label} cache_step[{step_idx}]");
            assert_eq!(
                observed_step.token_index, expected_step.token_index,
                "{step_label} token_index"
            );
            assert_eq!(
                observed_step.token_id, expected_step.token_id,
                "{step_label} token_id"
            );
            assert_eq!(
                observed_step.layers.len(),
                expected_step.layers.len(),
                "{step_label} layer count",
            );
            for (layer_idx, (expected_layer, observed_layer)) in expected_step
                .layers
                .iter()
                .zip(observed_step.layers.iter())
                .enumerate()
            {
                assert_cache_layer_close(
                    expected_layer,
                    observed_layer,
                    tolerance,
                    &format!("{step_label} layer[{layer_idx}]"),
                );
            }
        }
    }
}

fn assert_cache_layer_close(
    expected: &KimiBaselineParityCacheLayer,
    observed: &KimiBaselineParityCacheLayer,
    tolerance: f32,
    label: &str,
) {
    match (expected, observed) {
        (
            KimiBaselineParityCacheLayer::Mla {
                layer_idx: expected_layer_idx,
                processed_tokens: expected_processed_tokens,
                keys: expected_keys,
                values: expected_values,
            },
            KimiBaselineParityCacheLayer::Mla {
                layer_idx: observed_layer_idx,
                processed_tokens: observed_processed_tokens,
                keys: observed_keys,
                values: observed_values,
            },
        ) => {
            assert_eq!(observed_layer_idx, expected_layer_idx, "{label} layer_idx");
            assert_eq!(
                observed_processed_tokens, expected_processed_tokens,
                "{label} processed_tokens"
            );
            assert_tensor_close(
                expected_keys,
                observed_keys,
                tolerance,
                &format!("{label} keys"),
            );
            assert_tensor_close(
                expected_values,
                observed_values,
                tolerance,
                &format!("{label} values"),
            );
        }
        (
            KimiBaselineParityCacheLayer::Kda {
                layer_idx: expected_layer_idx,
                processed_tokens: expected_processed_tokens,
                q_conv_state: expected_q_conv_state,
                k_conv_state: expected_k_conv_state,
                v_conv_state: expected_v_conv_state,
                recurrent_state: expected_recurrent_state,
            },
            KimiBaselineParityCacheLayer::Kda {
                layer_idx: observed_layer_idx,
                processed_tokens: observed_processed_tokens,
                q_conv_state: observed_q_conv_state,
                k_conv_state: observed_k_conv_state,
                v_conv_state: observed_v_conv_state,
                recurrent_state: observed_recurrent_state,
            },
        ) => {
            assert_eq!(observed_layer_idx, expected_layer_idx, "{label} layer_idx");
            assert_eq!(
                observed_processed_tokens, expected_processed_tokens,
                "{label} processed_tokens"
            );
            assert_tensor_close(
                expected_q_conv_state,
                observed_q_conv_state,
                tolerance,
                &format!("{label} q_conv_state"),
            );
            assert_tensor_close(
                expected_k_conv_state,
                observed_k_conv_state,
                tolerance,
                &format!("{label} k_conv_state"),
            );
            assert_tensor_close(
                expected_v_conv_state,
                observed_v_conv_state,
                tolerance,
                &format!("{label} v_conv_state"),
            );
            assert_tensor_close(
                expected_recurrent_state,
                observed_recurrent_state,
                tolerance,
                &format!("{label} recurrent_state"),
            );
        }
        _ => panic!("{label} cache kind mismatch"),
    }
}

fn assert_tensor_close(
    expected: &KimiBaselineParityTensor,
    observed: &KimiBaselineParityTensor,
    tolerance: f32,
    label: &str,
) {
    assert_eq!(observed.dims, expected.dims, "{label} dims");
    assert_eq!(observed.values.len(), expected.values.len(), "{label} len");

    let max_diff = expected
        .values
        .iter()
        .zip(observed.values.iter())
        .map(|(expected, observed)| (expected - observed).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff <= tolerance,
        "{label} max_abs_diff {max_diff} exceeded tolerance {tolerance}"
    );
}

fn generate_prompt(
    model: &KimiLinearModel<TestBackend>,
    prompt: &KimiBaselineParityPromptSpec,
    selected_hidden_layers: &[usize],
    device: &TestDevice,
) -> KimiBaselineParityPrompt {
    let input_ids = input_ids(&prompt.input_ids, device);
    let logits = tensor_to_parity(model.forward(input_ids.clone()));
    let hidden_states = trace_hidden_states(model, input_ids, selected_hidden_layers);
    let cache_steps = trace_cache_steps(model, &prompt.input_ids, device);

    KimiBaselineParityPrompt {
        name: prompt.name.clone(),
        input_ids: prompt.input_ids.clone(),
        logits,
        hidden_states,
        cache_steps,
    }
}

fn trace_hidden_states(
    model: &KimiLinearModel<TestBackend>,
    input_ids: Tensor<TestBackend, 2, Int>,
    selected_hidden_layers: &[usize],
) -> Vec<KimiBaselineParityHiddenState> {
    let mut hidden = model.embed_tokens(input_ids);
    let selected = selected_hidden_layers
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let mut hidden_states = Vec::new();

    for (layer_idx, layer) in model.layers().iter().enumerate() {
        hidden = layer.forward(hidden);
        if selected.contains(&layer_idx) {
            hidden_states.push(KimiBaselineParityHiddenState {
                layer_idx,
                tensor: tensor_to_parity(hidden.clone()),
            });
        }
    }

    hidden_states
}

fn trace_cache_steps(
    model: &KimiLinearModel<TestBackend>,
    prompt_tokens: &[usize],
    device: &TestDevice,
) -> Vec<KimiBaselineParityCacheStep> {
    let mut cache = model.new_cache();
    let mut steps = Vec::with_capacity(prompt_tokens.len());

    for (token_index, &token_id) in prompt_tokens.iter().enumerate() {
        let token = input_ids(&[token_id], device);
        model.try_forward_hidden_cached(token, &mut cache).unwrap();

        let layers = model
            .layers()
            .iter()
            .enumerate()
            .map(|(layer_idx, layer)| match layer.attention_kind() {
                KimiAttentionLayerKind::FullAttention => {
                    let state = cache.try_mla(layer_idx).unwrap().unwrap();
                    parity_mla_layer(layer_idx, state)
                }
                KimiAttentionLayerKind::LinearAttentionKda => {
                    let state = cache.try_kda(layer_idx).unwrap().unwrap();
                    parity_kda_layer(layer_idx, state)
                }
            })
            .collect();

        steps.push(KimiBaselineParityCacheStep {
            token_index,
            token_id,
            layers,
        });
    }

    steps
}

fn parity_mla_layer(
    layer_idx: usize,
    state: &KimiMlaCache<TestBackend>,
) -> KimiBaselineParityCacheLayer {
    KimiBaselineParityCacheLayer::Mla {
        layer_idx,
        processed_tokens: state.processed_tokens(),
        keys: tensor_to_parity(state.keys().clone()),
        values: tensor_to_parity(state.values().clone()),
    }
}

fn parity_kda_layer(
    layer_idx: usize,
    state: &KimiKdaCache<TestBackend>,
) -> KimiBaselineParityCacheLayer {
    KimiBaselineParityCacheLayer::Kda {
        layer_idx,
        processed_tokens: state.processed_tokens(),
        q_conv_state: tensor_to_parity(state.q_conv_state().clone()),
        k_conv_state: tensor_to_parity(state.k_conv_state().clone()),
        v_conv_state: tensor_to_parity(state.v_conv_state().clone()),
        recurrent_state: tensor_to_parity(state.recurrent_state().clone()),
    }
}

fn input_ids(tokens: &[usize], device: &TestDevice) -> Tensor<TestBackend, 2, Int> {
    let ints = tokens.iter().map(|&token| token as i64).collect::<Vec<_>>();
    Tensor::<TestBackend, 1, Int>::from_ints(ints.as_slice(), device).reshape([1, tokens.len()])
}

fn tensor_to_parity<const D: usize>(tensor: Tensor<TestBackend, D>) -> KimiBaselineParityTensor {
    let dims = tensor.dims().into_iter().collect::<Vec<_>>();
    let numel = dims.iter().product::<usize>();
    let values = tensor.reshape([numel]).into_data().to_vec().unwrap();
    KimiBaselineParityTensor { dims, values }
}

fn seed_backend(device: &TestDevice, seed: u64) {
    TestBackend::seed(device, seed);
}
