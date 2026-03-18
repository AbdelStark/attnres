use attnres::kimi::{
    KimiArtifactConfig, KimiAttnResConfig, KimiAttnResModel, KimiLayerSchedule,
    KimiLayerScheduleError,
};
use burn::backend::NdArray;
use burn::prelude::*;

type TestBackend = NdArray;

fn max_abs_diff<const D: usize>(lhs: Tensor<TestBackend, D>, rhs: Tensor<TestBackend, D>) -> f32 {
    (lhs - rhs).abs().max().into_scalar()
}

fn seed_backend(device: &<TestBackend as Backend>::Device, seed: u64) {
    TestBackend::seed(device, seed);
}

fn mixed_reduced_config() -> KimiArtifactConfig {
    KimiArtifactConfig::from_json_str(
        r#"{
            "model_type": "kimi_linear",
            "dtype": "float32",
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
            "num_experts": 4,
            "num_experts_per_token": 2,
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
        }"#,
    )
    .unwrap()
}

fn full_attn_res_three_layer_config() -> KimiArtifactConfig {
    KimiArtifactConfig::from_json_str(
        r#"{
            "model_type": "kimi_linear",
            "dtype": "float32",
            "vocab_size": 32,
            "hidden_size": 12,
            "intermediate_size": 24,
            "moe_intermediate_size": 20,
            "num_hidden_layers": 3,
            "num_attention_heads": 3,
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
            "moe_layer_freq": 2,
            "num_experts": 4,
            "num_experts_per_token": 2,
            "num_shared_experts": 1,
            "tie_word_embeddings": false,
            "use_cache": true,
            "rms_norm_eps": 1e-5,
            "linear_attn_config": {
                "full_attn_layers": [2],
                "kda_layers": [1, 3],
                "num_heads": 3,
                "head_dim": 4,
                "short_conv_kernel_size": 3
            }
        }"#,
    )
    .unwrap()
}

fn mla_only_config() -> KimiArtifactConfig {
    KimiArtifactConfig::from_json_str(
        r#"{
            "model_type": "kimi_linear",
            "dtype": "float32",
            "vocab_size": 32,
            "hidden_size": 12,
            "intermediate_size": 24,
            "moe_intermediate_size": 20,
            "num_hidden_layers": 2,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "kv_lora_rank": 4,
            "q_lora_rank": null,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 4,
            "mla_use_nope": true,
            "hidden_act": "silu",
            "first_k_dense_replace": 2,
            "moe_layer_freq": 2,
            "num_experts": 4,
            "num_experts_per_token": 2,
            "num_shared_experts": 1,
            "tie_word_embeddings": false,
            "use_cache": true,
            "rms_norm_eps": 1e-5,
            "linear_attn_config": {
                "full_attn_layers": [1, 2],
                "kda_layers": [],
                "num_heads": 3,
                "head_dim": 4,
                "short_conv_kernel_size": 3
            }
        }"#,
    )
    .unwrap()
}

fn input_ids_for_seq(
    seq_len: usize,
    device: &<TestBackend as Backend>::Device,
) -> Tensor<TestBackend, 2, Int> {
    Tensor::<TestBackend, 2, Int>::zeros([1, seq_len], device)
}

fn init_attn_res_model(
    config: KimiArtifactConfig,
    num_blocks: usize,
    device: &<TestBackend as Backend>::Device,
) -> KimiAttnResModel<TestBackend> {
    config.try_init_attn_res_model(num_blocks, device).unwrap()
}

#[test]
fn kimi_rfc_0005_gate4_every_decoder_layer_has_two_distinct_attn_res_ops() {
    let device = Default::default();
    seed_backend(&device, 7);
    let model = init_attn_res_model(mixed_reduced_config(), 4, &device);

    for (layer_idx, layer) in model.layers().iter().enumerate() {
        let (attn_op, mlp_op) = layer.attn_res_ops();
        assert!(
            !std::ptr::eq(attn_op, mlp_op),
            "layer {layer_idx} should keep separate AttnRes operators for attention and MLP",
        );
    }
}

#[test]
fn kimi_rfc_0005_gate4_block_state_tracks_mixed_mla_kda_schedule_and_keeps_embedding_first() {
    let device = Default::default();
    seed_backend(&device, 11);
    let model = init_attn_res_model(mixed_reduced_config(), 4, &device);
    let input_ids = input_ids_for_seq(4, &device);
    let embeddings = model.embed_tokens(input_ids);
    let mut state = attnres::kimi::KimiAttnResBlockState::new(embeddings.clone());

    for (layer_idx, expected_blocks) in [1, 2, 3, 4].into_iter().enumerate() {
        state = model.layers()[layer_idx].forward(state);

        assert_eq!(
            state.num_blocks(),
            expected_blocks,
            "layer {layer_idx} should preserve sublayer-space block boundaries on the mixed MLA/KDA schedule",
        );
        assert!(state.partial_block().is_some());
        assert!(
            max_abs_diff(state.blocks()[0].clone(), embeddings.clone()) < 1e-6,
            "embedding block must stay at blocks[0] after layer {layer_idx}",
        );
    }
}

#[test]
fn kimi_rfc_0005_gate5_standard_and_two_phase_agree_on_supported_reduced_configs() {
    let device = Default::default();
    let cases: [(&str, KimiAttnResConfig, usize, u64); 2] = [
        (
            "mixed_four_layer_blocks4",
            mixed_reduced_config().try_attn_res_config(4).unwrap(),
            6,
            31,
        ),
        (
            "three_layer_full_attnres_blocks6",
            full_attn_res_three_layer_config()
                .try_attn_res_config(6)
                .unwrap(),
            5,
            47,
        ),
    ];

    for (case_name, config, seq_len, seed) in cases {
        seed_backend(&device, seed);
        let model = config.try_init_model::<TestBackend>(&device).unwrap();
        let input_ids = input_ids_for_seq(seq_len, &device);

        let standard_hidden = model.forward_hidden(input_ids.clone());
        let two_phase_hidden = model.forward_hidden_two_phase(input_ids.clone());
        let hidden_diff = max_abs_diff(standard_hidden, two_phase_hidden);
        assert!(
            hidden_diff < 1e-3,
            "{case_name} hidden states diverged between standard and two-phase paths, diff={hidden_diff}",
        );

        let standard_logits = model.forward(input_ids.clone());
        let two_phase_logits = model.forward_two_phase(input_ids);
        let logits_diff = max_abs_diff(standard_logits, two_phase_logits);
        assert!(
            logits_diff < 1e-3,
            "{case_name} logits diverged between standard and two-phase paths, diff={logits_diff}",
        );
    }
}

#[test]
fn kimi_rfc_0005_rejects_overlapping_attention_schedule_lists() {
    let err =
        KimiLayerSchedule::try_from_one_based_lists(4, &[1, 2], &[2, 3, 4], 1, 1).unwrap_err();

    assert_eq!(
        err,
        KimiLayerScheduleError::LayerAssignedTwice {
            one_based_layer_idx: 2
        }
    );
}

#[test]
fn kimi_rfc_0005_rejects_duplicate_attention_indices_within_a_schedule_list() {
    let err =
        KimiLayerSchedule::try_from_one_based_lists(4, &[2, 2], &[1, 3, 4], 1, 1).unwrap_err();

    assert_eq!(
        err,
        KimiLayerScheduleError::DuplicateLayerIndex {
            schedule_name: "full_attn_layers",
            one_based_layer_idx: 2
        }
    );
}

#[test]
fn kimi_rfc_0005_guards_against_one_based_to_zero_based_off_by_one_regressions() {
    let schedule = KimiLayerSchedule::try_from_one_based_lists(4, &[1, 4], &[2, 3], 1, 1).unwrap();

    assert_eq!(schedule.full_attention_layers_zero_based(), &[0, 3]);
    assert_eq!(schedule.kda_layers_zero_based(), &[1, 2]);
}

#[test]
fn kimi_rfc_0005_cached_forward_requires_cache_reset_when_batch_size_changes() {
    let device = Default::default();
    seed_backend(&device, 59);
    let model = init_attn_res_model(mla_only_config(), 2, &device);
    let mut cache = model.new_cache();

    model
        .try_forward_hidden_cached(Tensor::zeros([2, 1], &device), &mut cache)
        .unwrap();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = model.try_forward_hidden_cached(Tensor::zeros([1, 1], &device), &mut cache);
    }))
    .expect_err("cache reuse across batch-reset boundaries should fail loudly");
    let panic_text = if let Some(text) = panic.downcast_ref::<String>() {
        text.clone()
    } else if let Some(text) = panic.downcast_ref::<&str>() {
        text.to_string()
    } else {
        String::from("non-string panic payload")
    };
    assert!(
        panic_text.contains("MlaCacheBatchMismatch"),
        "unexpected panic for cache reuse across batch-reset boundaries: {panic_text}",
    );

    cache.clear();
    let recovered = model
        .try_forward_hidden_cached(Tensor::zeros([1, 1], &device), &mut cache)
        .unwrap();
    assert_eq!(recovered.dims(), [1, 1, 12]);
}
