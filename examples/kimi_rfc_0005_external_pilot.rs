use attnres::kimi::{
    compare_baseline_slice_parity_fixture_with_manifest_from_dir, KimiArtifactConfig,
    KimiArtifactUnderstanding, KimiBaselineSliceParityFixture, KimiBaselineSliceParityHiddenState,
    KimiBaselineSliceParityPromptResult, KimiBaselineSliceParityPromptSpec,
    KimiBaselineSliceParityTensor, KimiBaselineSliceParityToleranceSpec,
    KimiBaselineSliceRequestManifest, KimiBaselineSliceRequestSpec, KimiImportSelection,
    KimiLinearModel, KIMI_BASELINE_SLICE_REQUEST_FILENAME,
};
use burn::backend::NdArray;
use burn::prelude::*;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

type BackendImpl = NdArray;
type Device = <BackendImpl as Backend>::Device;

const PILOT_SEED: u64 = 20260318;
const LOCAL_INIT_CONTRACT_FILENAME: &str = "local-init-contract.json";
const LOCAL_INIT_CONTRACT_KIND: &str = "attnres.kimi.local_init_contract";
const LOCAL_INIT_CONTRACT_VERSION: u32 = 1;
const LOCAL_INIT_CONTRACT_STRATEGY: &str = "burn.ndarray.lazy_linear_kaiming_uniform.v1";

#[derive(Debug, Clone)]
struct LocalSafetensorTensor {
    dtype: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl LocalSafetensorTensor {
    fn from_f32_values(shape: Vec<usize>, values: Vec<f32>) -> Self {
        Self {
            dtype: "F32".to_string(),
            shape,
            bytes: values
                .into_iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }
    }

    fn filled_f32(shape: Vec<usize>, base: f32) -> Self {
        let numel = shape.iter().product::<usize>();
        let values = (0..numel)
            .map(|idx| base + idx as f32 * 0.001)
            .collect::<Vec<_>>();
        Self::from_f32_values(shape, values)
    }
}

#[derive(Debug, Clone)]
struct TinyBaselinePayloadArtifactBuilder {
    config_json: Value,
    index_json: Value,
    shards: BTreeMap<String, BTreeMap<String, LocalSafetensorTensor>>,
}

#[derive(Debug, Serialize)]
struct LocalInitContract {
    kind: String,
    version: u32,
    strategy: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    match args.as_slice() {
        [_, command, output_dir] if command == "write-artifact" => {
            write_artifact(Path::new(output_dir))?;
        }
        [_, command, artifact_dir, output_dir] if command == "emit-request-bundle" => {
            emit_request_bundle(Path::new(artifact_dir), Path::new(output_dir))?;
        }
        [_, command, artifact_dir, manifest_path, output_path]
            if command == "generate-rust-fixture" =>
        {
            generate_rust_fixture(
                Path::new(artifact_dir),
                Path::new(manifest_path),
                Path::new(output_path),
            )?;
        }
        [_, command, artifact_dir, manifest_path, fixture_path]
            if command == "validate-fixture" =>
        {
            validate_fixture(
                Path::new(artifact_dir),
                Path::new(manifest_path),
                Path::new(fixture_path),
            )?;
        }
        _ => {
            eprintln!(
                "usage:\n  cargo run --example kimi_rfc_0005_external_pilot -- write-artifact <output-dir>\n  cargo run --example kimi_rfc_0005_external_pilot -- emit-request-bundle <artifact-dir> <output-dir>\n  cargo run --example kimi_rfc_0005_external_pilot -- generate-rust-fixture <artifact-dir> <manifest-path> <output-path>\n  cargo run --example kimi_rfc_0005_external_pilot -- validate-fixture <artifact-dir> <manifest-path> <fixture-path>"
            );
            std::process::exit(1);
        }
    }

    Ok(())
}

fn write_artifact(output_dir: &Path) -> Result<(), Box<dyn Error>> {
    if output_dir.exists() {
        fs::remove_dir_all(output_dir)?;
    }
    fs::create_dir_all(output_dir)?;

    let builder = TinyBaselinePayloadArtifactBuilder::new()?;
    fs::write(
        output_dir.join("config.json"),
        serde_json::to_string_pretty(&builder.config_json)?,
    )?;
    fs::write(
        output_dir.join("model.safetensors.index.json"),
        serde_json::to_string_pretty(&builder.index_json)?,
    )?;
    for (shard_name, tensors) in &builder.shards {
        write_safetensor_shard(&output_dir.join(shard_name), tensors)?;
    }

    Ok(())
}

fn emit_request_bundle(artifact_dir: &Path, output_dir: &Path) -> Result<(), Box<dyn Error>> {
    if output_dir.exists() {
        fs::remove_dir_all(output_dir)?;
    }
    fs::create_dir_all(output_dir)?;

    let understanding = KimiArtifactUnderstanding::load_from_dir(artifact_dir)?;
    let manifest = understanding.try_build_baseline_slice_request_manifest(fixed_request_spec())?;
    manifest.write(output_dir.join(KIMI_BASELINE_SLICE_REQUEST_FILENAME))?;

    let init_contract = local_init_contract();
    fs::write(
        output_dir.join(LOCAL_INIT_CONTRACT_FILENAME),
        serde_json::to_string_pretty(&init_contract)?,
    )?;

    Ok(())
}

fn validate_fixture(
    artifact_dir: &Path,
    manifest_path: &Path,
    fixture_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let device = Default::default();
    compare_baseline_slice_parity_fixture_with_manifest_from_dir::<BackendImpl, _, _, _>(
        artifact_dir,
        manifest_path,
        fixture_path,
        &device,
    )?;
    Ok(())
}

fn generate_rust_fixture(
    artifact_dir: &Path,
    manifest_path: &Path,
    output_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let manifest = KimiBaselineSliceRequestManifest::load(manifest_path)?;
    let understanding = KimiArtifactUnderstanding::load_from_dir(artifact_dir)?;
    let device = Default::default();
    BackendImpl::seed(&device, manifest.seed);
    let model: KimiLinearModel<BackendImpl> = understanding.try_init_baseline_model_from_dir(
        artifact_dir,
        manifest.slice.import_selection.clone(),
        &device,
    )?;
    let prompt_results = manifest
        .prompts
        .iter()
        .map(|prompt| build_prompt_result(&model, prompt, &manifest.slice.selected_hidden_layers))
        .collect::<Vec<_>>();
    let fixture: KimiBaselineSliceParityFixture = manifest.try_into_fixture(prompt_results)?;
    fs::write(output_path, serde_json::to_string_pretty(&fixture)?)?;
    Ok(())
}

fn fixed_request_spec() -> KimiBaselineSliceRequestSpec {
    KimiBaselineSliceRequestSpec {
        seed: PILOT_SEED,
        import_selection: KimiImportSelection::full(2),
        selected_hidden_layers: vec![0],
        prompts: vec![
            KimiBaselineSliceParityPromptSpec {
                name: "single_token_0".to_string(),
                input_ids: vec![0],
            },
            KimiBaselineSliceParityPromptSpec {
                name: "single_token_5".to_string(),
                input_ids: vec![5],
            },
        ],
        tolerances: KimiBaselineSliceParityToleranceSpec {
            metric: "max_abs_diff".to_string(),
            runtime_dtype: "float32".to_string(),
            logits_max_abs_diff: 0.5,
            hidden_state_max_abs_diff: 1.0,
        },
    }
}

fn local_init_contract() -> LocalInitContract {
    LocalInitContract {
        kind: LOCAL_INIT_CONTRACT_KIND.to_string(),
        version: LOCAL_INIT_CONTRACT_VERSION,
        strategy: LOCAL_INIT_CONTRACT_STRATEGY.to_string(),
    }
}

fn tensor_to_fixture<const D: usize>(
    tensor: Tensor<BackendImpl, D>,
) -> KimiBaselineSliceParityTensor {
    let dims = tensor.dims().into_iter().collect::<Vec<_>>();
    let numel = dims.iter().product::<usize>();
    let values = tensor.reshape([numel]).into_data().to_vec().unwrap();
    KimiBaselineSliceParityTensor { dims, values }
}

fn build_prompt_result(
    model: &KimiLinearModel<BackendImpl>,
    prompt: &KimiBaselineSliceParityPromptSpec,
    selected_hidden_layers: &[usize],
) -> KimiBaselineSliceParityPromptResult {
    let device = Default::default();
    let input_ids = input_ids(&prompt.input_ids, &device);
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
    model: &KimiLinearModel<BackendImpl>,
    input_ids: Tensor<BackendImpl, 2, Int>,
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

fn input_ids(tokens: &[usize], device: &Device) -> Tensor<BackendImpl, 2, Int> {
    let ints = tokens.iter().map(|&token| token as i64).collect::<Vec<_>>();
    Tensor::<BackendImpl, 1, Int>::from_ints(ints.as_slice(), device).reshape([1, tokens.len()])
}

impl TinyBaselinePayloadArtifactBuilder {
    fn new() -> Result<Self, Box<dyn Error>> {
        let config_json: Value = serde_json::from_str(&read_fixture_text("config.json")?)?;
        let index_json: Value =
            serde_json::from_str(&read_fixture_text("model.safetensors.index.json")?)?;
        let config: KimiArtifactConfig = serde_json::from_value(config_json.clone())?;
        let shards = build_default_shards(&config, &index_json)?;

        Ok(Self {
            config_json,
            index_json,
            shards,
        })
    }
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("kimi")
        .join("tiny-random-baseline")
}

fn read_fixture_text(file_name: &str) -> Result<String, Box<dyn Error>> {
    Ok(fs::read_to_string(fixture_dir().join(file_name))?)
}

fn build_default_shards(
    config: &KimiArtifactConfig,
    index_json: &Value,
) -> Result<BTreeMap<String, BTreeMap<String, LocalSafetensorTensor>>, Box<dyn Error>> {
    let mut shards = BTreeMap::<String, BTreeMap<String, LocalSafetensorTensor>>::new();

    let weight_map = index_json["weight_map"]
        .as_object()
        .ok_or("weight_map must be an object")?;
    for (idx, (tensor_name, shard_path)) in weight_map.iter().enumerate() {
        let base = 0.01 * (idx as f32 + 1.0);
        let tensor = default_tensor_for_name(config, tensor_name, base)?;
        let shard_path = shard_path
            .as_str()
            .ok_or("shard path must be a string")?
            .to_string();
        shards
            .entry(shard_path)
            .or_default()
            .insert(tensor_name.clone(), tensor);
    }

    Ok(shards)
}

fn default_tensor_for_name(
    config: &KimiArtifactConfig,
    tensor_name: &str,
    base: f32,
) -> Result<LocalSafetensorTensor, Box<dyn Error>> {
    let hidden = config.hidden_size;
    let dense_intermediate = config.intermediate_size;
    let moe_intermediate = config.moe_intermediate_size;
    let qk_dim = config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim);
    let mla_out_dim = config.num_attention_heads * config.v_head_dim;
    let kda_qk_dim = config.linear_attn_config.num_heads * config.linear_attn_config.head_dim;
    let kda_v_dim = config.linear_attn_config.num_heads * config.v_head_dim;

    let shape = match tensor_name {
        "model.embed_tokens.weight" => vec![config.vocab_size, hidden],
        "model.layers.0.input_layernorm.weight" => vec![hidden],
        "model.layers.0.self_attn.q_proj.weight" => vec![hidden, kda_qk_dim],
        "model.layers.0.self_attn.k_proj.weight" => vec![hidden, kda_qk_dim],
        "model.layers.0.self_attn.v_proj.weight" => vec![hidden, kda_v_dim],
        "model.layers.0.self_attn.o_proj.weight" => vec![kda_v_dim, hidden],
        "model.layers.0.post_attention_layernorm.weight" => vec![hidden],
        "model.layers.0.mlp.gate_proj.weight" => vec![hidden, dense_intermediate],
        "model.layers.0.mlp.up_proj.weight" => vec![hidden, dense_intermediate],
        "model.layers.0.mlp.down_proj.weight" => vec![dense_intermediate, hidden],
        "model.layers.1.input_layernorm.weight" => vec![hidden],
        "model.layers.1.self_attn.q_proj.weight" => vec![hidden, qk_dim],
        "model.layers.1.self_attn.o_proj.weight" => vec![mla_out_dim, hidden],
        "model.layers.1.post_attention_layernorm.weight" => vec![hidden],
        "model.layers.1.block_sparse_moe.gate.weight" => vec![hidden, config.num_experts],
        "model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight" => {
            vec![hidden, moe_intermediate]
        }
        "model.layers.1.block_sparse_moe.shared_experts.up_proj.weight" => {
            vec![hidden, moe_intermediate]
        }
        "model.layers.1.block_sparse_moe.shared_experts.down_proj.weight" => {
            vec![moe_intermediate, hidden]
        }
        "model.layers.1.block_sparse_moe.experts.0.w1.weight"
        | "model.layers.1.block_sparse_moe.experts.0.w3.weight"
        | "model.layers.1.block_sparse_moe.experts.1.w1.weight"
        | "model.layers.1.block_sparse_moe.experts.1.w3.weight" => {
            vec![hidden, moe_intermediate]
        }
        "model.layers.1.block_sparse_moe.experts.0.w2.weight"
        | "model.layers.1.block_sparse_moe.experts.1.w2.weight" => {
            vec![moe_intermediate, hidden]
        }
        "model.norm.weight" => vec![hidden],
        "lm_head.weight" => vec![hidden, config.vocab_size],
        other => {
            return Err(format!("missing default tensor shape for {other}").into());
        }
    };

    Ok(LocalSafetensorTensor::filled_f32(shape, base))
}

fn write_safetensor_shard(
    path: &Path,
    tensors: &BTreeMap<String, LocalSafetensorTensor>,
) -> Result<(), Box<dyn Error>> {
    let mut offset = 0usize;
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();

    for (tensor_name, tensor) in tensors {
        let expected_bytes = tensor
            .shape
            .iter()
            .product::<usize>()
            .checked_mul(4)
            .ok_or("tensor byte length overflowed usize")?;
        if tensor.bytes.len() != expected_bytes {
            return Err(format!(
                "tensor {tensor_name} byte length {} does not match shape {:?}",
                tensor.bytes.len(),
                tensor.shape
            )
            .into());
        }
        let next_offset = offset + tensor.bytes.len();
        header.insert(
            tensor_name.clone(),
            serde_json::json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [offset, next_offset],
            }),
        );
        offset = next_offset;
        data.extend_from_slice(&tensor.bytes);
    }

    let mut header_bytes = serde_json::to_vec(&Value::Object(header))?;
    let aligned_len = header_bytes.len().next_multiple_of(8);
    header_bytes.resize(aligned_len, b' ');

    let mut file = Vec::with_capacity(8 + header_bytes.len() + data.len());
    file.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file.extend_from_slice(&header_bytes);
    file.extend_from_slice(&data);
    fs::write(path, file)?;

    Ok(())
}
