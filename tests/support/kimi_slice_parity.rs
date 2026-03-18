#![allow(dead_code)]

use attnres::kimi::{
    KimiArtifactUnderstanding, KimiBaselineSliceParityFixture, KimiBaselineSliceParityHiddenState,
    KimiBaselineSliceParityPromptResult, KimiBaselineSliceParityPromptSpec,
    KimiBaselineSliceParityTensor, KimiBaselineSliceParityToleranceSpec,
    KimiBaselineSliceRequestManifest, KimiBaselineSliceRequestSpec, KimiImportSelection,
    KimiLinearModel, KIMI_BASELINE_SLICE_PARITY_FILENAME, KIMI_BASELINE_SLICE_REQUEST_FILENAME,
};
use burn::prelude::*;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use super::kimi_local_artifact::{input_ids, TestBackend};

#[derive(Debug)]
pub struct LocalSliceParityFixtureFile {
    root_dir: PathBuf,
    path: PathBuf,
}

#[derive(Debug)]
pub struct LocalSliceRequestManifestFile {
    root_dir: PathBuf,
    path: PathBuf,
}

static LOCAL_SLICE_PARITY_COUNTER: AtomicU64 = AtomicU64::new(0);
const LOCAL_SLICE_PARITY_SEED: u64 = 20260318;

impl LocalSliceParityFixtureFile {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl LocalSliceRequestManifestFile {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for LocalSliceParityFixtureFile {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root_dir);
    }
}

impl Drop for LocalSliceRequestManifestFile {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root_dir);
    }
}

pub fn write_valid_fixture(
    artifact_dir: &Path,
    selection: KimiImportSelection,
    selected_hidden_layers: &[usize],
) -> LocalSliceParityFixtureFile {
    let fixture = build_valid_fixture(artifact_dir, selection, selected_hidden_layers);
    write_fixture(&fixture)
}

pub fn build_valid_fixture(
    artifact_dir: &Path,
    selection: KimiImportSelection,
    selected_hidden_layers: &[usize],
) -> KimiBaselineSliceParityFixture {
    let manifest = build_valid_manifest(artifact_dir, selection, selected_hidden_layers);
    build_valid_fixture_from_manifest(artifact_dir, &manifest)
}

pub fn write_fixture(fixture: &KimiBaselineSliceParityFixture) -> LocalSliceParityFixtureFile {
    write_fixture_json(&serde_json::to_string_pretty(fixture).unwrap())
}

pub fn write_fixture_value(value: &Value) -> LocalSliceParityFixtureFile {
    write_fixture_json(&serde_json::to_string_pretty(value).unwrap())
}

pub fn write_valid_manifest(
    artifact_dir: &Path,
    selection: KimiImportSelection,
    selected_hidden_layers: &[usize],
) -> LocalSliceRequestManifestFile {
    let manifest = build_valid_manifest(artifact_dir, selection, selected_hidden_layers);
    write_manifest(&manifest)
}

pub fn build_valid_manifest(
    artifact_dir: &Path,
    selection: KimiImportSelection,
    selected_hidden_layers: &[usize],
) -> KimiBaselineSliceRequestManifest {
    let understanding = KimiArtifactUnderstanding::load_from_dir(artifact_dir).unwrap();
    understanding
        .try_build_baseline_slice_request_manifest(valid_request_spec(
            selection,
            selected_hidden_layers,
        ))
        .unwrap()
}

pub fn build_valid_fixture_from_manifest(
    artifact_dir: &Path,
    manifest: &KimiBaselineSliceRequestManifest,
) -> KimiBaselineSliceParityFixture {
    let understanding = KimiArtifactUnderstanding::load_from_dir(artifact_dir).unwrap();
    let device = Default::default();
    TestBackend::seed(&device, manifest.seed);
    let model: KimiLinearModel<TestBackend> = understanding
        .try_init_baseline_model_from_dir(
            artifact_dir,
            manifest.slice.import_selection.clone(),
            &device,
        )
        .unwrap();
    let prompt_results = manifest
        .prompts
        .iter()
        .map(|prompt| {
            build_prompt_result(
                &model,
                prompt,
                &manifest.slice.selected_hidden_layers,
                &device,
            )
        })
        .collect::<Vec<_>>();

    manifest.clone().try_into_fixture(prompt_results).unwrap()
}

pub fn write_manifest(
    manifest: &KimiBaselineSliceRequestManifest,
) -> LocalSliceRequestManifestFile {
    let root_dir = unique_temp_dir();
    fs::create_dir_all(&root_dir).unwrap();
    let path = root_dir.join(KIMI_BASELINE_SLICE_REQUEST_FILENAME);
    manifest.write(&path).unwrap();
    LocalSliceRequestManifestFile { root_dir, path }
}

pub fn write_manifest_value(value: &Value) -> LocalSliceRequestManifestFile {
    write_manifest_json(&serde_json::to_string_pretty(value).unwrap())
}

fn write_fixture_json(json: &str) -> LocalSliceParityFixtureFile {
    let root_dir = unique_temp_dir();
    fs::create_dir_all(&root_dir).unwrap();
    let path = root_dir.join(KIMI_BASELINE_SLICE_PARITY_FILENAME);
    fs::write(&path, json).unwrap();
    LocalSliceParityFixtureFile { root_dir, path }
}

fn write_manifest_json(json: &str) -> LocalSliceRequestManifestFile {
    let root_dir = unique_temp_dir();
    fs::create_dir_all(&root_dir).unwrap();
    let path = root_dir.join(KIMI_BASELINE_SLICE_REQUEST_FILENAME);
    fs::write(&path, json).unwrap();
    LocalSliceRequestManifestFile { root_dir, path }
}

fn build_prompt_result(
    model: &KimiLinearModel<TestBackend>,
    prompt: &KimiBaselineSliceParityPromptSpec,
    selected_hidden_layers: &[usize],
    device: &<TestBackend as Backend>::Device,
) -> KimiBaselineSliceParityPromptResult {
    let input_ids = input_ids(&prompt.input_ids, device);
    let logits = tensor_to_fixture(model.forward(input_ids.clone()));
    let hidden_states = trace_hidden_states(model, input_ids, selected_hidden_layers);

    KimiBaselineSliceParityPromptResult {
        prompt_name: prompt.name.clone(),
        input_ids: prompt.input_ids.clone(),
        logits,
        hidden_states,
    }
}

fn trace_hidden_states(
    model: &KimiLinearModel<TestBackend>,
    input_ids: Tensor<TestBackend, 2, Int>,
    selected_hidden_layers: &[usize],
) -> Vec<KimiBaselineSliceParityHiddenState> {
    let mut hidden = model.embed_tokens(input_ids);
    let mut hidden_states = Vec::new();

    for (layer_idx, layer) in model.layers().iter().enumerate() {
        hidden = layer.forward(hidden);
        if selected_hidden_layers.contains(&layer_idx) {
            hidden_states.push(KimiBaselineSliceParityHiddenState {
                layer_idx,
                tensor: tensor_to_fixture(hidden.clone()),
            });
        }
    }

    hidden_states
}

fn tensor_to_fixture<const D: usize>(
    tensor: Tensor<TestBackend, D>,
) -> KimiBaselineSliceParityTensor {
    let dims = tensor.dims().into_iter().collect::<Vec<_>>();
    let numel = dims.iter().product::<usize>();
    let values = tensor.reshape([numel]).into_data().to_vec().unwrap();
    KimiBaselineSliceParityTensor { dims, values }
}

fn valid_request_spec(
    selection: KimiImportSelection,
    selected_hidden_layers: &[usize],
) -> KimiBaselineSliceRequestSpec {
    KimiBaselineSliceRequestSpec {
        seed: LOCAL_SLICE_PARITY_SEED,
        import_selection: selection,
        selected_hidden_layers: selected_hidden_layers.to_vec(),
        prompts: vec![
            KimiBaselineSliceParityPromptSpec {
                name: "ascending_len4".to_string(),
                input_ids: vec![0, 1, 2, 3],
            },
            KimiBaselineSliceParityPromptSpec {
                name: "repeat_len3".to_string(),
                input_ids: vec![5, 5, 1],
            },
        ],
        tolerances: KimiBaselineSliceParityToleranceSpec {
            metric: "max_abs_diff".to_string(),
            runtime_dtype: "float32".to_string(),
            logits_max_abs_diff: 1e-6,
            hidden_state_max_abs_diff: 1e-6,
        },
    }
}

fn unique_temp_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should move forward")
        .as_nanos();
    let counter = LOCAL_SLICE_PARITY_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "attnres-kimi-slice-parity-{}-{nanos}-{counter}",
        std::process::id(),
    ))
}
