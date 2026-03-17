use attnres::kimi::{
    KimiArtifactConfig, KimiArtifactConfigError, KimiArtifactUnderstanding, KimiAttentionLayerKind,
    KimiFeedForwardLayerKind, KimiImportError, KimiImportMode, KimiImportSelection,
    KimiLayerSchedule, KimiLayerScheduleError, KimiMilestonePhase, KimiShardIndex,
    KimiShardIndexError,
};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn sample_config_json() -> String {
    r#"{
        "model_type": "kimi_linear",
        "dtype": "bfloat16",
        "hidden_size": 2304,
        "intermediate_size": 9216,
        "moe_intermediate_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 72,
        "kv_lora_rank": 512,
        "q_lora_rank": null,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mla_use_nope": true,
        "hidden_act": "silu",
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "num_experts": 256,
        "num_experts_per_token": 8,
        "num_shared_experts": 1,
        "tie_word_embeddings": false,
        "use_cache": true,
        "linear_attn_config": {
            "full_attn_layers": [4],
            "kda_layers": [1, 2, 3],
            "num_heads": 32,
            "head_dim": 128,
            "short_conv_kernel_size": 4
        }
    }"#
    .to_string()
}

fn sample_shard_index_json() -> String {
    r#"{
        "metadata": {
            "total_parameters": 123456,
            "total_size": 654321
        },
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors"
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
        "attnres-kimi-phase-a-{}-{nanos}",
        std::process::id()
    ))
}

#[test]
fn kimi_artifact_config_builds_typed_layer_schedule() {
    let config = KimiArtifactConfig::from_json_str(&sample_config_json()).unwrap();
    let schedule = config.try_layer_schedule().unwrap();

    assert_eq!(schedule.num_hidden_layers(), 4);
    assert_eq!(schedule.full_attention_layers_zero_based(), &[3]);
    assert_eq!(schedule.kda_layers_zero_based(), &[0, 1, 2]);
    assert_eq!(
        schedule.try_attention_kind(0).unwrap(),
        KimiAttentionLayerKind::LinearAttentionKda
    );
    assert_eq!(
        schedule.try_attention_kind(3).unwrap(),
        KimiAttentionLayerKind::FullAttention
    );
    assert_eq!(
        schedule.try_feed_forward_kind(0).unwrap(),
        KimiFeedForwardLayerKind::DenseMlp
    );
    assert_eq!(
        schedule.try_feed_forward_kind(1).unwrap(),
        KimiFeedForwardLayerKind::SparseMoe
    );
}

#[test]
fn kimi_artifact_config_rejects_wrong_model_type() {
    let json = sample_config_json().replace("\"kimi_linear\"", "\"kimi_k2\"");
    let err = KimiArtifactConfig::from_json_str(&json).unwrap_err();

    assert_eq!(
        err,
        KimiArtifactConfigError::UnsupportedModelType {
            model_type: "kimi_k2".to_string()
        }
    );
}

#[test]
fn kimi_layer_schedule_rejects_zero_in_one_based_lists() {
    let err =
        KimiLayerSchedule::try_from_one_based_lists(4, &[0], &[1, 2, 3, 4], 1, 1).unwrap_err();

    assert_eq!(
        err,
        KimiLayerScheduleError::OneBasedLayerIndexMustBePositive {
            schedule_name: "full_attn_layers",
            position: 0
        }
    );
}

#[test]
fn kimi_artifact_config_rejects_missing_attention_assignment() {
    let json = sample_config_json().replace("\"kda_layers\": [1, 2, 3]", "\"kda_layers\": [1, 2]");
    let err = KimiArtifactConfig::from_json_str(&json).unwrap_err();

    assert_eq!(
        err,
        KimiArtifactConfigError::LayerSchedule(KimiLayerScheduleError::MissingLayerAssignment {
            one_based_layer_idx: 3
        })
    );
}

#[test]
fn kimi_shard_index_parses_and_reports_unique_shards() {
    let index = KimiShardIndex::from_json_str(&sample_shard_index_json()).unwrap();

    assert_eq!(index.tensor_count(), 3);
    assert_eq!(index.shard_count(), 2);
    assert_eq!(
        index.shard_for_tensor("lm_head.weight"),
        Some("model-00002-of-00002.safetensors")
    );
}

#[test]
fn kimi_shard_index_rejects_empty_weight_map() {
    let err = KimiShardIndex::from_json_str(
        r#"{
            "metadata": {
                "total_parameters": 1,
                "total_size": 1
            },
            "weight_map": {}
        }"#,
    )
    .unwrap_err();

    assert_eq!(err, KimiShardIndexError::WeightMapMustNotBeEmpty);
}

#[test]
fn phase_a_artifact_understanding_loads_from_dir_and_reports_inspect_mode() {
    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).unwrap();
    fs::write(temp_dir.join("config.json"), sample_config_json()).unwrap();
    fs::write(
        temp_dir.join("model.safetensors.index.json"),
        sample_shard_index_json(),
    )
    .unwrap();

    let understanding = KimiArtifactUnderstanding::load_from_dir(&temp_dir).unwrap();
    let inspect_plan = understanding.inspect_plan();

    assert_eq!(
        understanding.phase(),
        KimiMilestonePhase::ArtifactUnderstanding
    );
    assert_eq!(
        understanding.report.ready_modes,
        vec![KimiImportMode::Inspect]
    );
    assert_eq!(
        understanding.report.deferred_modes,
        vec![KimiImportMode::Slice, KimiImportMode::Full]
    );
    assert_eq!(inspect_plan.mode, KimiImportMode::Inspect);
    assert!(inspect_plan.required_shards.is_empty());

    fs::remove_dir_all(temp_dir).unwrap();
}

#[test]
fn slice_plan_is_explicitly_deferred_after_selection_validation() {
    let understanding = KimiArtifactUnderstanding::try_from_parts(
        KimiArtifactConfig::from_json_str(&sample_config_json()).unwrap(),
        KimiShardIndex::from_json_str(&sample_shard_index_json()).unwrap(),
    )
    .unwrap();

    let err = understanding
        .try_slice_plan(KimiImportSelection {
            layer_indices: vec![0, 3],
            include_embeddings: true,
            include_final_norm: false,
            include_lm_head: false,
        })
        .unwrap_err();

    assert_eq!(
        err,
        KimiImportError::ModeNotYetImplemented {
            mode: KimiImportMode::Slice,
            implemented_phase: KimiMilestonePhase::ArtifactUnderstanding,
            required_phase: KimiMilestonePhase::BaselineImplementation,
            detail: "tensor-to-module mapping and shard slicing are deferred to RFC 0002/0003"
        }
    );
}

#[test]
fn slice_plan_rejects_duplicate_layer_selection_before_deferred_mode_error() {
    let understanding = KimiArtifactUnderstanding::try_from_parts(
        KimiArtifactConfig::from_json_str(&sample_config_json()).unwrap(),
        KimiShardIndex::from_json_str(&sample_shard_index_json()).unwrap(),
    )
    .unwrap();

    let err = understanding
        .try_slice_plan(KimiImportSelection {
            layer_indices: vec![1, 1],
            include_embeddings: false,
            include_final_norm: false,
            include_lm_head: false,
        })
        .unwrap_err();

    assert_eq!(
        err,
        KimiImportError::DuplicateSelectedLayer { layer_idx: 1 }
    );
}
