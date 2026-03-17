use attnres::config::ConfigError;
use attnres::kimi::{
    KimiArtifactConfig, KimiAttentionLayerKind, KimiAttnResBlockState, KimiAttnResConfigError,
    KimiAttnResModel, KimiAttnResStateError, KimiFeedForwardLayerKind,
};
use burn::backend::NdArray;
use burn::prelude::*;

type TestBackend = NdArray;

fn reduced_baseline_config_json() -> &'static str {
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
    }"#
}

fn three_layer_config_json() -> &'static str {
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
    }"#
}

fn reduced_config() -> KimiArtifactConfig {
    KimiArtifactConfig::from_json_str(reduced_baseline_config_json()).unwrap()
}

fn three_layer_config() -> KimiArtifactConfig {
    KimiArtifactConfig::from_json_str(three_layer_config_json()).unwrap()
}

#[test]
fn kimi_rfc_0004_layer_typing_stays_schedule_driven_under_attn_res() {
    let device = Default::default();
    let model: KimiAttnResModel<TestBackend> = reduced_config()
        .try_attn_res_config(4)
        .unwrap()
        .try_init_model(&device)
        .unwrap();

    assert_eq!(
        model.layers()[0].attention_kind(),
        KimiAttentionLayerKind::LinearAttentionKda
    );
    assert_eq!(
        model.layers()[1].attention_kind(),
        KimiAttentionLayerKind::FullAttention
    );
    assert_eq!(
        model.layers()[2].attention_kind(),
        KimiAttentionLayerKind::LinearAttentionKda
    );
    assert_eq!(
        model.layers()[3].attention_kind(),
        KimiAttentionLayerKind::FullAttention
    );
}

#[test]
fn kimi_rfc_0004_dense_vs_sparse_moe_selection_remains_distinct() {
    let device = Default::default();
    let model: KimiAttnResModel<TestBackend> = reduced_config()
        .try_attn_res_config(4)
        .unwrap()
        .try_init_model(&device)
        .unwrap();

    assert_eq!(
        model.layers()[0].feed_forward_kind(),
        KimiFeedForwardLayerKind::DenseMlp
    );
    assert_eq!(
        model.layers()[1].feed_forward_kind(),
        KimiFeedForwardLayerKind::DenseMlp
    );
    assert_eq!(
        model.layers()[2].feed_forward_kind(),
        KimiFeedForwardLayerKind::SparseMoe
    );
    assert_eq!(
        model.layers()[3].feed_forward_kind(),
        KimiFeedForwardLayerKind::DenseMlp
    );
    assert!(!model.layers()[1].uses_moe());
    assert!(model.layers()[2].uses_moe());
}

#[test]
fn kimi_rfc_0004_block_boundaries_stay_in_sublayer_space() {
    let device = Default::default();
    let config = three_layer_config().try_attn_res_config(6).unwrap();
    let layer0 = config.try_init_layer::<TestBackend>(0, &device).unwrap();
    let state = KimiAttnResBlockState::new(Tensor::<TestBackend, 3>::zeros([1, 3, 12], &device));

    assert_eq!(config.block_size(), 1);
    let state = layer0.forward(state);
    assert_eq!(
        state.num_blocks(),
        2,
        "layer 0 should close a block between attention and MLP when block_size == 1"
    );
    assert_eq!(state.partial_block().unwrap().dims(), [1, 3, 12]);
}

#[test]
fn kimi_rfc_0004_forward_paths_smoke_test_shapes_and_cache() {
    let device = Default::default();
    let model: KimiAttnResModel<TestBackend> = reduced_config()
        .try_init_attn_res_model(4, &device)
        .unwrap();
    let input_ids = Tensor::<TestBackend, 2, Int>::zeros([2, 5], &device);

    let hidden = model.forward_hidden(input_ids.clone());
    let logits = model.forward(input_ids);

    assert_eq!(hidden.dims(), [2, 5, 16]);
    assert_eq!(logits.dims(), [2, 5, 64]);
    assert!(model.supports_cache());

    let mut cache = model.new_cache();
    let cached_hidden = model
        .try_forward_hidden_cached(
            Tensor::<TestBackend, 2, Int>::zeros([1, 1], &device),
            &mut cache,
        )
        .unwrap();
    let cached_logits = model
        .try_forward_cached(
            Tensor::<TestBackend, 2, Int>::zeros([1, 1], &device),
            &mut cache,
        )
        .unwrap();

    assert_eq!(cached_hidden.dims(), [1, 1, 16]);
    assert_eq!(cached_logits.dims(), [1, 1, 64]);
    assert_eq!(cache.try_kda(0).unwrap().unwrap().processed_tokens(), 2);
    assert_eq!(cache.try_mla(1).unwrap().unwrap().processed_tokens(), 2);
}

#[test]
fn kimi_rfc_0004_two_phase_matches_standard_on_supported_reduced_config() {
    let device = Default::default();
    let model: KimiAttnResModel<TestBackend> = reduced_config()
        .try_init_attn_res_model(4, &device)
        .unwrap();
    let input_ids = Tensor::<TestBackend, 2, Int>::zeros([1, 4], &device);

    let standard_hidden = model.forward_hidden(input_ids.clone());
    let two_phase_hidden = model.forward_hidden_two_phase(input_ids.clone());
    let hidden_diff: f32 = (standard_hidden - two_phase_hidden)
        .abs()
        .max()
        .into_scalar();
    assert!(
        hidden_diff < 1e-3,
        "two-phase hidden forward should match standard forward, diff={hidden_diff}"
    );

    let standard_logits = model.forward(input_ids.clone());
    let two_phase_logits = model.forward_two_phase(input_ids);
    let logits_diff: f32 = (standard_logits - two_phase_logits)
        .abs()
        .max()
        .into_scalar();
    assert!(
        logits_diff < 1e-3,
        "two-phase logits should match standard forward, diff={logits_diff}"
    );
}

#[test]
fn kimi_rfc_0004_validation_and_state_failures_are_explicit() {
    assert_eq!(
        KimiAttnResBlockState::<TestBackend>::try_from_parts(Vec::new(), None).unwrap_err(),
        KimiAttnResStateError::CompletedBlocksMustNotBeEmpty
    );

    let invalid_config = reduced_config().try_attn_res_config(3);
    assert_eq!(
        invalid_config.unwrap_err(),
        KimiAttnResConfigError::AttnRes(ConfigError::NumLayersMustBeDivisibleByNumBlocks {
            num_layers: 8,
            num_blocks: 3,
        })
    );
}
