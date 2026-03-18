#![allow(dead_code)]

use attnres::kimi::{KimiArtifactConfig, KimiImportSelection};
use burn::backend::NdArray;
use burn::prelude::*;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub type TestBackend = NdArray;
pub type TestDevice = <TestBackend as Backend>::Device;

#[derive(Debug, Clone)]
pub struct LocalSafetensorTensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl LocalSafetensorTensor {
    pub fn from_f32_values(shape: Vec<usize>, values: Vec<f32>) -> Self {
        Self {
            dtype: "F32".to_string(),
            shape,
            bytes: values
                .into_iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }
    }

    pub fn filled_f32(shape: Vec<usize>, base: f32) -> Self {
        let numel = shape.iter().product::<usize>();
        let values = (0..numel)
            .map(|idx| base + idx as f32 * 0.001)
            .collect::<Vec<_>>();
        Self::from_f32_values(shape, values)
    }

    pub fn raw(dtype: &str, shape: Vec<usize>, bytes: Vec<u8>) -> Self {
        Self {
            dtype: dtype.to_string(),
            shape,
            bytes,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TinyBaselinePayloadArtifactBuilder {
    config_json: Value,
    index_json: Value,
    shards: BTreeMap<String, BTreeMap<String, LocalSafetensorTensor>>,
}

#[derive(Debug)]
pub struct LocalArtifactDir {
    path: PathBuf,
}

static LOCAL_ARTIFACT_COUNTER: AtomicU64 = AtomicU64::new(0);

impl LocalArtifactDir {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for LocalArtifactDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

impl TinyBaselinePayloadArtifactBuilder {
    pub fn new() -> Self {
        let config_json: Value =
            serde_json::from_str(&read_fixture_text("config.json")).expect("fixture config");
        let index_json: Value =
            serde_json::from_str(&read_fixture_text("model.safetensors.index.json"))
                .expect("fixture shard index");
        let config: KimiArtifactConfig =
            serde_json::from_value(config_json.clone()).expect("typed fixture config");
        let shards = build_default_shards(&config, &index_json);

        Self {
            config_json,
            index_json,
            shards,
        }
    }

    pub fn single_layer_dense_kda() -> Self {
        let config_json = serde_json::json!({
            "model_type": "kimi_linear",
            "dtype": "float32",
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 12,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "kv_lora_rank": 4,
            "q_lora_rank": null,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 4,
            "mla_use_nope": true,
            "hidden_act": "silu",
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "num_experts": 2,
            "num_experts_per_token": 1,
            "num_shared_experts": 1,
            "moe_renormalize": true,
            "moe_router_activation_func": "sigmoid",
            "routed_scaling_factor": 1.0,
            "use_grouped_topk": true,
            "num_expert_group": 1,
            "topk_group": 1,
            "tie_word_embeddings": false,
            "use_cache": true,
            "rms_norm_eps": 1e-5,
            "linear_attn_config": {
                "full_attn_layers": [],
                "kda_layers": [1],
                "num_heads": 2,
                "head_dim": 4,
                "short_conv_kernel_size": 3
            }
        });
        let index_json = serde_json::json!({
            "metadata": {
                "total_parameters": 12,
                "total_size": 48
            },
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.v_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.q_conv1d.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.k_conv1d.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.v_conv1d.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.A_log": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.f_a_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.f_b_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.dt_bias": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.b_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.g_a_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.g_b_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.o_norm.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.o_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.post_attention_layernorm.weight": "model-00002-of-00002.safetensors",
                "model.layers.0.mlp.gate_proj.weight": "model-00002-of-00002.safetensors",
                "model.layers.0.mlp.up_proj.weight": "model-00002-of-00002.safetensors",
                "model.layers.0.mlp.down_proj.weight": "model-00002-of-00002.safetensors",
                "model.norm.weight": "model-00002-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors"
            }
        });
        let config: KimiArtifactConfig =
            serde_json::from_value(config_json.clone()).expect("typed single-layer config");
        let shards = build_default_shards(&config, &index_json);

        Self {
            config_json,
            index_json,
            shards,
        }
    }

    pub fn config(&self) -> KimiArtifactConfig {
        serde_json::from_value(self.config_json.clone()).expect("typed config")
    }

    pub fn full_selection(&self) -> KimiImportSelection {
        KimiImportSelection::full(self.config().num_hidden_layers)
    }

    pub fn insert_weight_map_tensor(&mut self, tensor_name: &str, shard_path: &str) {
        self.index_json["weight_map"]
            .as_object_mut()
            .expect("weight_map object")
            .insert(
                tensor_name.to_string(),
                Value::String(shard_path.to_string()),
            );
    }

    pub fn replace_tensor_payload(&mut self, tensor_name: &str, tensor: LocalSafetensorTensor) {
        let shard_path = self.shard_path_for_tensor(tensor_name);
        self.shards
            .entry(shard_path)
            .or_default()
            .insert(tensor_name.to_string(), tensor);
    }

    pub fn remove_tensor_payload(&mut self, tensor_name: &str) {
        let shard_path = self.shard_path_for_tensor(tensor_name);
        self.shards
            .get_mut(&shard_path)
            .expect("tensor shard should exist")
            .remove(tensor_name);
    }

    pub fn write(self) -> LocalArtifactDir {
        let dir = LocalArtifactDir {
            path: unique_temp_dir(),
        };
        fs::create_dir_all(dir.path()).unwrap();
        fs::write(
            dir.path().join("config.json"),
            serde_json::to_string_pretty(&self.config_json).unwrap(),
        )
        .unwrap();
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_string_pretty(&self.index_json).unwrap(),
        )
        .unwrap();
        for (shard_name, tensors) in &self.shards {
            write_safetensor_shard(&dir.path().join(shard_name), tensors);
        }
        dir
    }

    fn shard_path_for_tensor(&self, tensor_name: &str) -> String {
        self.index_json["weight_map"][tensor_name]
            .as_str()
            .unwrap_or_else(|| panic!("missing weight_map entry for {tensor_name}"))
            .to_string()
    }
}

pub fn input_ids(tokens: &[usize], device: &TestDevice) -> Tensor<TestBackend, 2, Int> {
    let ints = tokens.iter().map(|&token| token as i64).collect::<Vec<_>>();
    Tensor::<TestBackend, 1, Int>::from_ints(ints.as_slice(), device).reshape([1, tokens.len()])
}

pub fn max_abs_diff<const D: usize>(
    lhs: Tensor<TestBackend, D>,
    rhs: Tensor<TestBackend, D>,
) -> f32 {
    (lhs - rhs).abs().max().into_scalar()
}

pub fn seed_backend(device: &TestDevice, seed: u64) {
    TestBackend::seed(device, seed);
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("kimi")
        .join("tiny-random-baseline")
}

fn read_fixture_text(file_name: &str) -> String {
    fs::read_to_string(fixture_dir().join(file_name)).unwrap()
}

fn unique_temp_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should move forward")
        .as_nanos();
    let counter = LOCAL_ARTIFACT_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "attnres-kimi-local-payload-{}-{nanos}-{counter}",
        std::process::id(),
    ))
}

fn build_default_shards(
    config: &KimiArtifactConfig,
    index_json: &Value,
) -> BTreeMap<String, BTreeMap<String, LocalSafetensorTensor>> {
    let mut shards = BTreeMap::<String, BTreeMap<String, LocalSafetensorTensor>>::new();

    for (idx, (tensor_name, shard_path)) in index_json["weight_map"]
        .as_object()
        .expect("fixture weight_map object")
        .iter()
        .enumerate()
    {
        let base = 0.01 * (idx as f32 + 1.0);
        let tensor = default_tensor_for_name(config, tensor_name, base);
        shards
            .entry(shard_path.as_str().expect("shard path string").to_string())
            .or_default()
            .insert(tensor_name.clone(), tensor);
    }

    shards
}

fn default_tensor_for_name(
    config: &KimiArtifactConfig,
    tensor_name: &str,
    base: f32,
) -> LocalSafetensorTensor {
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
        "model.layers.0.self_attn.q_conv1d.weight" | "model.layers.0.self_attn.k_conv1d.weight" => {
            vec![
                kda_qk_dim,
                1,
                config.linear_attn_config.short_conv_kernel_size,
            ]
        }
        "model.layers.0.self_attn.v_conv1d.weight" => {
            vec![
                kda_v_dim,
                1,
                config.linear_attn_config.short_conv_kernel_size,
            ]
        }
        "model.layers.0.self_attn.A_log" => vec![1, 1, config.linear_attn_config.num_heads, 1],
        "model.layers.0.self_attn.f_a_proj.weight" | "model.layers.0.self_attn.g_a_proj.weight" => {
            vec![hidden, config.linear_attn_config.head_dim]
        }
        "model.layers.0.self_attn.f_b_proj.weight" | "model.layers.0.self_attn.g_b_proj.weight" => {
            vec![config.linear_attn_config.head_dim, kda_v_dim]
        }
        "model.layers.0.self_attn.dt_bias" => vec![kda_v_dim],
        "model.layers.0.self_attn.b_proj.weight" => {
            vec![hidden, config.linear_attn_config.num_heads]
        }
        "model.layers.0.self_attn.o_norm.weight" => vec![config.linear_attn_config.head_dim],
        "model.layers.0.self_attn.o_proj.weight" => vec![kda_v_dim, hidden],
        "model.layers.0.post_attention_layernorm.weight" => vec![hidden],
        "model.layers.0.mlp.gate_proj.weight" => vec![hidden, dense_intermediate],
        "model.layers.0.mlp.up_proj.weight" => vec![hidden, dense_intermediate],
        "model.layers.0.mlp.down_proj.weight" => vec![dense_intermediate, hidden],
        "model.layers.1.input_layernorm.weight" => vec![hidden],
        "model.layers.1.self_attn.q_proj.weight" => vec![hidden, qk_dim],
        "model.layers.1.self_attn.kv_a_proj_with_mqa.weight" => {
            vec![hidden, config.kv_lora_rank + config.qk_rope_head_dim]
        }
        "model.layers.1.self_attn.kv_a_layernorm.weight" => vec![config.kv_lora_rank],
        "model.layers.1.self_attn.kv_b_proj.weight" => {
            vec![
                config.kv_lora_rank,
                config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
            ]
        }
        "model.layers.1.self_attn.o_proj.weight" => vec![mla_out_dim, hidden],
        "model.layers.1.post_attention_layernorm.weight" => vec![hidden],
        "model.layers.1.block_sparse_moe.gate.weight" => vec![hidden, config.num_experts],
        "model.layers.1.block_sparse_moe.gate.e_score_correction_bias" => {
            vec![config.num_experts]
        }
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
        other => panic!("missing default tensor shape for {other}"),
    };

    LocalSafetensorTensor::filled_f32(shape, base)
}

fn write_safetensor_shard(path: &Path, tensors: &BTreeMap<String, LocalSafetensorTensor>) {
    let mut offset = 0usize;
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();

    for (tensor_name, tensor) in tensors {
        let expected_bytes = tensor
            .shape
            .iter()
            .product::<usize>()
            .checked_mul(dtype_size(&tensor.dtype))
            .expect("tensor byte length should fit in usize");
        assert_eq!(
            tensor.bytes.len(),
            expected_bytes,
            "tensor {tensor_name} byte length does not match shape {:?} and dtype {}",
            tensor.shape,
            tensor.dtype
        );
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

    let mut header_bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
    let aligned_len = header_bytes.len().next_multiple_of(8);
    header_bytes.resize(aligned_len, b' ');

    let mut file = Vec::with_capacity(8 + header_bytes.len() + data.len());
    file.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file.extend_from_slice(&header_bytes);
    file.extend_from_slice(&data);
    fs::write(path, file).unwrap();
}

fn dtype_size(dtype: &str) -> usize {
    match dtype {
        "F32" => 4,
        "BF16" | "F16" => 2,
        other => panic!("unsupported test dtype {other}"),
    }
}
