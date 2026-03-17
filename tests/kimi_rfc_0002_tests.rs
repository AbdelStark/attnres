use attnres::kimi::{
    KimiArtifactConfig, KimiAttentionLayerKind, KimiCacheError, KimiFeedForwardLayerKind,
    KimiKdaCache, KimiLinearModel, KimiMilestonePhase, KimiMlaCache, KIMI_IMPLEMENTED_PHASE,
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

fn reduced_config() -> KimiArtifactConfig {
    KimiArtifactConfig::from_json_str(reduced_baseline_config_json()).unwrap()
}

#[test]
fn kimi_rfc_0002_layer_typing_drives_module_selection() {
    let device = Default::default();
    let config = reduced_config();
    let baseline = config.try_baseline_config().unwrap();
    let model = baseline.try_init_model::<TestBackend>(&device).unwrap();

    assert_eq!(
        KIMI_IMPLEMENTED_PHASE,
        KimiMilestonePhase::BaselineImplementation
    );
    assert_eq!(baseline.attention.kv_repeat_factor(), 2);
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
fn kimi_rfc_0002_moe_placement_follows_schedule_rules() {
    let device = Default::default();
    let model: KimiLinearModel<TestBackend> = reduced_config().try_init_model(&device).unwrap();

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
fn kimi_rfc_0002_cache_types_stay_separate_and_validate_shapes() {
    let device = Default::default();
    let model: KimiLinearModel<TestBackend> = reduced_config().try_init_model(&device).unwrap();
    let mut cache = model.new_cache();

    assert!(matches!(
        cache.try_mla(0),
        Err(KimiCacheError::ExpectedMlaLayer {
            layer_idx: 0,
            found: KimiAttentionLayerKind::LinearAttentionKda,
        })
    ));
    assert!(matches!(
        cache.try_kda(1),
        Err(KimiCacheError::ExpectedKdaLayer {
            layer_idx: 1,
            found: KimiAttentionLayerKind::FullAttention,
        })
    ));

    let mla_cache = KimiMlaCache::try_new(
        Tensor::<TestBackend, 4>::zeros([1, 2, 2, 8], &device),
        Tensor::<TestBackend, 4>::zeros([1, 2, 2, 8], &device),
        8,
        8,
    )
    .unwrap();
    cache.update_mla(1, mla_cache).unwrap();
    assert_eq!(cache.try_mla(1).unwrap().unwrap().processed_tokens(), 2);
    cache.clear_layer(1).unwrap();
    assert!(cache.try_mla(1).unwrap().is_none());

    let mla_err = KimiMlaCache::try_new(
        Tensor::<TestBackend, 4>::zeros([1, 2, 2, 8], &device),
        Tensor::<TestBackend, 4>::zeros([1, 2, 1, 8], &device),
        8,
        8,
    )
    .unwrap_err();
    assert_eq!(
        mla_err,
        KimiCacheError::MlaCacheSequenceLengthMismatch {
            keys_seq_len: 2,
            values_seq_len: 1,
        }
    );

    let kda_cache = KimiKdaCache::try_new(
        Tensor::<TestBackend, 4>::zeros([1, 4, 2, 8], &device),
        Tensor::<TestBackend, 4>::zeros([1, 4, 8, 8], &device),
        Tensor::<TestBackend, 3>::zeros([1, 4, 8], &device),
        2,
        2,
        8,
        8,
    )
    .unwrap();
    cache.update_kda(0, kda_cache).unwrap();
    assert_eq!(cache.try_kda(0).unwrap().unwrap().processed_tokens(), 2);

    let kda_err = KimiKdaCache::try_new(
        Tensor::<TestBackend, 4>::zeros([1, 4, 3, 8], &device),
        Tensor::<TestBackend, 4>::zeros([1, 4, 8, 8], &device),
        Tensor::<TestBackend, 3>::zeros([1, 4, 8], &device),
        2,
        2,
        8,
        8,
    )
    .unwrap_err();
    assert_eq!(
        kda_err,
        KimiCacheError::KdaCacheConvStateTooLong {
            history_len: 3,
            max_history_len: 2,
        }
    );
}

#[test]
fn kimi_rfc_0002_forward_paths_produce_expected_shapes() {
    let device = Default::default();
    let model: KimiLinearModel<TestBackend> = reduced_config().try_init_model(&device).unwrap();

    let input_ids = Tensor::<TestBackend, 2, Int>::zeros([2, 5], &device);
    let hidden = model.forward_hidden(input_ids.clone());
    let logits = model.forward(input_ids);

    assert_eq!(hidden.dims(), [2, 5, 16]);
    assert_eq!(logits.dims(), [2, 5, 64]);

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
