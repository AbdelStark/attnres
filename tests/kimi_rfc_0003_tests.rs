use attnres::kimi::{
    KimiArtifactConfig, KimiArtifactUnderstanding, KimiImportCoverageError, KimiImportMode,
    KimiImportSelection, KimiModuleRef, KimiShardIndex, KimiShardResolver, KimiTensorLocator,
    KimiTensorLocatorError,
};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn reduced_config_json() -> String {
    r#"{
        "model_type": "kimi_linear",
        "dtype": "bfloat16",
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "moe_intermediate_size": 24,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "kv_lora_rank": 8,
        "q_lora_rank": null,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 8,
        "mla_use_nope": true,
        "hidden_act": "silu",
        "first_k_dense_replace": 2,
        "moe_layer_freq": 2,
        "num_experts": 2,
        "num_experts_per_token": 1,
        "num_shared_experts": 1,
        "tie_word_embeddings": false,
        "use_cache": true,
        "rms_norm_eps": 1e-5,
        "linear_attn_config": {
            "full_attn_layers": [2, 4],
            "kda_layers": [1, 3],
            "num_heads": 4,
            "head_dim": 8,
            "short_conv_kernel_size": 3
        }
    }"#
    .to_string()
}

fn supported_slice_index_json() -> String {
    r#"{
        "metadata": {
            "total_parameters": 512,
            "total_size": 2048
        },
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
            "model.layers.0.input_layernorm.weight": "model-00001-of-00003.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00003.safetensors",
            "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00003.safetensors",
            "model.layers.0.self_attn.v_proj.weight": "model-00001-of-00003.safetensors",
            "model.layers.0.self_attn.o_proj.weight": "model-00002-of-00003.safetensors",
            "model.layers.0.post_attention_layernorm.weight": "model-00002-of-00003.safetensors",
            "model.layers.0.mlp.gate_proj.weight": "model-00002-of-00003.safetensors",
            "model.layers.0.mlp.up_proj.weight": "model-00002-of-00003.safetensors",
            "model.layers.0.mlp.down_proj.weight": "model-00002-of-00003.safetensors",
            "model.norm.weight": "model-00003-of-00003.safetensors",
            "lm_head.weight": "model-00003-of-00003.safetensors"
        }
    }"#
    .to_string()
}

fn rich_index_json() -> String {
    r#"{
        "metadata": {
            "total_parameters": 2048,
            "total_size": 8192
        },
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
            "model.layers.0.input_layernorm.weight": "model-00001-of-00004.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
            "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00004.safetensors",
            "model.layers.0.self_attn.v_proj.weight": "model-00001-of-00004.safetensors",
            "model.layers.0.self_attn.o_proj.weight": "model-00001-of-00004.safetensors",
            "model.layers.0.self_attn.q_conv1d.weight": "model-00001-of-00004.safetensors",
            "model.layers.0.post_attention_layernorm.weight": "model-00001-of-00004.safetensors",
            "model.layers.0.mlp.gate_proj.weight": "model-00002-of-00004.safetensors",
            "model.layers.0.mlp.up_proj.weight": "model-00002-of-00004.safetensors",
            "model.layers.0.mlp.down_proj.weight": "model-00002-of-00004.safetensors",

            "model.layers.1.input_layernorm.weight": "model-00002-of-00004.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
            "model.layers.1.self_attn.kv_a_layernorm.weight": "model-00002-of-00004.safetensors",
            "model.layers.1.self_attn.kv_a_proj_with_mqa.weight": "model-00002-of-00004.safetensors",
            "model.layers.1.self_attn.kv_b_proj.weight": "model-00002-of-00004.safetensors",
            "model.layers.1.self_attn.o_proj.weight": "model-00002-of-00004.safetensors",
            "model.layers.1.post_attention_layernorm.weight": "model-00002-of-00004.safetensors",
            "model.layers.1.mlp.gate_proj.weight": "model-00002-of-00004.safetensors",
            "model.layers.1.mlp.up_proj.weight": "model-00002-of-00004.safetensors",
            "model.layers.1.mlp.down_proj.weight": "model-00002-of-00004.safetensors",

            "model.layers.2.input_layernorm.weight": "model-00003-of-00004.safetensors",
            "model.layers.2.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
            "model.layers.2.self_attn.k_proj.weight": "model-00003-of-00004.safetensors",
            "model.layers.2.self_attn.v_proj.weight": "model-00003-of-00004.safetensors",
            "model.layers.2.self_attn.o_proj.weight": "model-00003-of-00004.safetensors",
            "model.layers.2.self_attn.o_norm.weight": "model-00003-of-00004.safetensors",
            "model.layers.2.post_attention_layernorm.weight": "model-00003-of-00004.safetensors",
            "model.layers.2.block_sparse_moe.gate.weight": "model-00003-of-00004.safetensors",
            "model.layers.2.block_sparse_moe.gate.e_score_correction_bias": "model-00003-of-00004.safetensors",
            "model.layers.2.block_sparse_moe.shared_experts.gate_proj.weight": "model-00003-of-00004.safetensors",
            "model.layers.2.block_sparse_moe.shared_experts.up_proj.weight": "model-00003-of-00004.safetensors",
            "model.layers.2.block_sparse_moe.shared_experts.down_proj.weight": "model-00003-of-00004.safetensors",
            "model.layers.2.block_sparse_moe.experts.0.w1.weight": "model-00004-of-00004.safetensors",
            "model.layers.2.block_sparse_moe.experts.0.w2.weight": "model-00004-of-00004.safetensors",
            "model.layers.2.block_sparse_moe.experts.0.w3.weight": "model-00004-of-00004.safetensors",
            "model.layers.2.block_sparse_moe.experts.1.w1.weight": "model-00004-of-00004.safetensors",
            "model.layers.2.block_sparse_moe.experts.1.w2.weight": "model-00004-of-00004.safetensors",
            "model.layers.2.block_sparse_moe.experts.1.w3.weight": "model-00004-of-00004.safetensors",

            "model.norm.weight": "model-00004-of-00004.safetensors",
            "lm_head.weight": "model-00004-of-00004.safetensors"
        }
    }"#
    .to_string()
}

fn unique_temp_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should move forward")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "attnres-kimi-rfc-0003-{}-{nanos}",
        std::process::id()
    ))
}

fn understanding_from(config_json: &str, index_json: &str) -> KimiArtifactUnderstanding {
    KimiArtifactUnderstanding::try_from_parts(
        KimiArtifactConfig::from_json_str(config_json).unwrap(),
        KimiShardIndex::from_json_str(index_json).unwrap(),
    )
    .unwrap()
}

#[test]
fn kimi_rfc_0003_tensor_locator_deduplicates_required_shards() {
    let index = KimiShardIndex::from_json_str(&supported_slice_index_json()).unwrap();
    let locator = KimiTensorLocator::from_index(&index);

    let embed = locator.locate("model.embed_tokens.weight").unwrap();
    assert_eq!(embed.shard_path, "model-00001-of-00003.safetensors");
    assert_eq!(
        locator
            .required_shards_for_tensors([
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.o_proj.weight"
            ])
            .unwrap(),
        vec![
            "model-00001-of-00003.safetensors".to_string(),
            "model-00002-of-00003.safetensors".to_string()
        ]
    );
    assert_eq!(
        locator.locate("missing.tensor"),
        Err(KimiTensorLocatorError::MissingTensor {
            tensor_name: "missing.tensor".to_string()
        })
    );
}

#[test]
fn kimi_rfc_0003_slice_plan_computes_required_shards_for_supported_subset() {
    let understanding = understanding_from(&reduced_config_json(), &supported_slice_index_json());

    let plan = understanding
        .try_slice_plan(KimiImportSelection {
            layer_indices: vec![0],
            include_embeddings: true,
            include_final_norm: true,
            include_lm_head: true,
        })
        .unwrap();

    assert_eq!(plan.mode, KimiImportMode::Slice);
    assert_eq!(plan.required_shards.len(), 3);
    assert!(plan.is_fully_loadable());
    plan.try_require_loadable().unwrap();
}

#[test]
fn kimi_rfc_0003_reports_unsupported_and_missing_tensors_honestly() {
    let understanding = understanding_from(&reduced_config_json(), &rich_index_json());

    let plan = understanding
        .try_slice_plan(KimiImportSelection {
            layer_indices: vec![0, 1, 2],
            include_embeddings: true,
            include_final_norm: true,
            include_lm_head: true,
        })
        .unwrap();

    assert!(!plan.is_fully_loadable());
    assert!(plan
        .coverage
        .unsupported_tensors
        .iter()
        .any(|tensor| tensor.tensor_name == "model.layers.0.self_attn.q_conv1d.weight"));
    assert!(plan
        .coverage
        .unsupported_tensors
        .iter()
        .any(|tensor| tensor.tensor_name == "model.layers.1.self_attn.kv_a_proj_with_mqa.weight"));
    assert!(plan
        .coverage
        .unsupported_tensors
        .iter()
        .any(|tensor| tensor.tensor_name
            == "model.layers.2.block_sparse_moe.gate.e_score_correction_bias"));
    assert!(
        plan.coverage
            .missing_tensors
            .iter()
            .any(|missing| missing.tensor_name == "model.layers.2.self_attn.q_proj.weight")
            == false
    );
    assert!(matches!(
        plan.try_require_loadable(),
        Err(KimiImportCoverageError::UnsupportedTensor { .. })
    ));
}

#[test]
fn kimi_rfc_0003_module_coverage_tracks_supported_sparse_moe_tensors() {
    let understanding = understanding_from(&reduced_config_json(), &rich_index_json());

    let full_plan = understanding.try_full_plan().unwrap();
    let sparse_module = full_plan
        .coverage
        .module_coverage
        .iter()
        .find(|coverage| {
            matches!(
                coverage.module,
                KimiModuleRef::DecoderLayer { layer_idx: 2, .. }
            ) && coverage
                .required_tensors
                .iter()
                .any(|name| name.contains("block_sparse_moe"))
        })
        .unwrap();

    assert!(sparse_module
        .mapped_tensors
        .iter()
        .any(|tensor| tensor.ends_with("block_sparse_moe.experts.1.w3.weight")));
    assert!(sparse_module
        .unsupported_tensors
        .iter()
        .any(|tensor| tensor.ends_with("block_sparse_moe.gate.e_score_correction_bias")));
}

#[test]
fn kimi_rfc_0003_duplicate_and_missing_coverage_errors_are_typed() {
    let understanding = understanding_from(&reduced_config_json(), &supported_slice_index_json());
    let plan = understanding.try_full_plan().unwrap();

    let mut duplicate_report = plan.coverage.clone();
    duplicate_report
        .duplicate_tensors
        .push(attnres::kimi::KimiDuplicateTensor {
            tensor_name: "lm_head.weight".to_string(),
            modules: vec![KimiModuleRef::LmHead, KimiModuleRef::LmHead],
        });
    assert!(matches!(
        duplicate_report.try_require_loadable(),
        Err(KimiImportCoverageError::DuplicateRequiredTensor { .. })
    ));

    let mut missing_report = plan.coverage.clone();
    missing_report
        .missing_tensors
        .push(attnres::kimi::KimiMissingTensor {
            tensor_name: "model.layers.3.self_attn.q_proj.weight".to_string(),
            module: KimiModuleRef::DecoderLayer {
                layer_idx: 3,
                component: attnres::kimi::KimiLayerModuleRef::Attention {
                    kind: attnres::kimi::KimiAttentionLayerKind::FullAttention,
                },
            },
            local_parameter_paths: vec!["layers[3].attention.q_proj.weight".to_string()],
        });
    assert!(matches!(
        missing_report.try_require_loadable(),
        Err(KimiImportCoverageError::MissingRequiredTensor { .. })
    ));
}

#[test]
fn kimi_rfc_0003_smoke_resolves_slice_shards_without_fake_loading() {
    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).unwrap();
    fs::write(temp_dir.join("config.json"), reduced_config_json()).unwrap();
    fs::write(
        temp_dir.join("model.safetensors.index.json"),
        supported_slice_index_json(),
    )
    .unwrap();
    fs::write(temp_dir.join("model-00001-of-00003.safetensors"), b"").unwrap();
    fs::write(temp_dir.join("model-00002-of-00003.safetensors"), b"").unwrap();
    fs::write(temp_dir.join("model-00003-of-00003.safetensors"), b"").unwrap();

    let understanding = KimiArtifactUnderstanding::load_from_dir(&temp_dir).unwrap();
    let plan = understanding
        .try_slice_plan(KimiImportSelection {
            layer_indices: vec![0],
            include_embeddings: true,
            include_final_norm: true,
            include_lm_head: true,
        })
        .unwrap();
    let resolver = KimiShardResolver::new(&temp_dir);
    let resolved = resolver.try_resolve_plan(&plan).unwrap();

    assert_eq!(resolved.len(), 3);
    assert!(resolved
        .iter()
        .all(|shard| shard.resolved_path.exists() && shard.resolved_path.is_file()));

    fs::remove_dir_all(temp_dir).unwrap();
}
