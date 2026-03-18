mod support;

use attnres::kimi::{
    KimiAttentionLayerKind, KimiAttnResModel, KimiBaselinePayloadError, KimiImportCoverageError,
    KimiImportSelection, KimiLayerModuleRef, KimiModuleRef, KimiShardResolverError,
};
use support::kimi_local_artifact::{
    input_ids, max_abs_diff, seed_backend, TestBackend, TinyBaselinePayloadArtifactBuilder,
};

#[test]
fn kimi_rfc_0005_gate2_local_payload_harness_changes_attn_res_logits_deterministically() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let config = builder.config();
    let num_blocks = config.num_hidden_layers * 2;
    let artifact = builder.write();
    let device = Default::default();
    let prompt = input_ids(&[0, 1, 2, 3], &device);

    seed_backend(&device, 20260318);
    let baseline = config
        .try_init_attn_res_model::<TestBackend>(num_blocks, &device)
        .unwrap();
    let baseline_logits = baseline.forward(prompt.clone());

    seed_backend(&device, 20260318);
    let loaded_once = KimiAttnResModel::<TestBackend>::try_from_artifact_dir(
        artifact.path(),
        num_blocks,
        selection.clone(),
        &device,
    )
    .unwrap();
    let loaded_once_hidden = loaded_once.forward_hidden(prompt.clone());
    let loaded_once_hidden_two_phase = loaded_once.forward_hidden_two_phase(prompt.clone());
    let loaded_once_logits = loaded_once.forward(prompt.clone());
    let loaded_once_logits_two_phase = loaded_once.forward_two_phase(prompt.clone());

    seed_backend(&device, 20260318);
    let loaded_twice = KimiAttnResModel::<TestBackend>::try_from_artifact_dir(
        artifact.path(),
        num_blocks,
        selection,
        &device,
    )
    .unwrap();
    let loaded_twice_logits = loaded_twice.forward(prompt.clone());

    let baseline_diff = max_abs_diff(baseline_logits, loaded_once_logits.clone());
    assert!(
        baseline_diff > 1e-4,
        "loaded AttnRes-Kimi logits should differ from the randomly initialized model, diff={baseline_diff}",
    );

    let hidden_diff = max_abs_diff(loaded_once_hidden, loaded_once_hidden_two_phase);
    assert!(
        hidden_diff < 1e-3,
        "loaded AttnRes-Kimi hidden states should agree between standard and two-phase paths, diff={hidden_diff}",
    );

    let logits_diff = max_abs_diff(loaded_once_logits.clone(), loaded_once_logits_two_phase);
    assert!(
        logits_diff < 1e-3,
        "loaded AttnRes-Kimi logits should agree between standard and two-phase paths, diff={logits_diff}",
    );

    let repeat_diff = max_abs_diff(loaded_once_logits, loaded_twice_logits);
    assert!(
        repeat_diff < 1e-6,
        "loading the same artifact into AttnRes-Kimi twice with the same seed should be deterministic, diff={repeat_diff}",
    );
}

#[test]
fn kimi_rfc_0005_gate2_loaded_attn_res_model_keeps_cached_decode_available() {
    let builder = TinyBaselinePayloadArtifactBuilder::single_layer_dense_kda();
    let selection = builder.full_selection();
    let config = builder.config();
    let artifact = builder.write();
    let device = Default::default();

    seed_backend(&device, 20260318);
    let model = KimiAttnResModel::<TestBackend>::try_from_artifact_dir(
        artifact.path(),
        config.num_hidden_layers * 2,
        selection,
        &device,
    )
    .unwrap();
    let mut cache = model.new_cache();
    let hidden = model
        .try_forward_hidden_cached(input_ids(&[0, 1], &device), &mut cache)
        .unwrap();

    assert_eq!(hidden.dims(), [1, 2, 8]);
    assert_eq!(cache.try_kda(0).unwrap().unwrap().processed_tokens(), 2);
}

#[test]
fn kimi_rfc_0005_gate2_attn_res_rejects_unsupported_selected_tensor_before_loading_payloads() {
    let mut builder = TinyBaselinePayloadArtifactBuilder::new();
    builder.insert_weight_map_tensor(
        "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
        "model-00002-of-00002.safetensors",
    );
    let selection = builder.full_selection();
    let config = builder.config();
    let artifact = builder.write();
    let device = Default::default();

    let err = KimiAttnResModel::<TestBackend>::try_from_artifact_dir(
        artifact.path(),
        config.num_hidden_layers * 2,
        selection,
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiBaselinePayloadError::Coverage(KimiImportCoverageError::UnsupportedTensor {
            tensor_name,
            ..
        }) if tensor_name == "model.layers.1.self_attn.kv_a_proj_with_mqa.weight"
    ));
}

#[test]
fn kimi_rfc_0005_gate2_attn_res_rejects_missing_required_shard_files() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let config = builder.config();
    let artifact = builder.write();
    let device = Default::default();

    std::fs::remove_file(artifact.path().join("model-00002-of-00002.safetensors")).unwrap();
    let err = KimiAttnResModel::<TestBackend>::try_from_artifact_dir(
        artifact.path(),
        config.num_hidden_layers * 2,
        selection,
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiBaselinePayloadError::ShardResolver(KimiShardResolverError::MissingShardFile {
            shard_path,
            ..
        }) if shard_path == "model-00002-of-00002.safetensors"
    ));
}

#[test]
fn kimi_rfc_0005_gate2_attn_res_rejects_incomplete_selected_layer_payload_coverage() {
    let mut builder = TinyBaselinePayloadArtifactBuilder::new();
    builder.remove_tensor_payload("model.layers.0.self_attn.k_proj.weight");
    let config = builder.config();
    let selection = KimiImportSelection {
        layer_indices: vec![0],
        include_embeddings: true,
        include_final_norm: true,
        include_lm_head: true,
    };
    let artifact = builder.write();
    let device = Default::default();

    let err = KimiAttnResModel::<TestBackend>::try_from_artifact_dir(
        artifact.path(),
        config.num_hidden_layers * 2,
        selection,
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiBaselinePayloadError::IncompleteModulePayload {
            module:
                KimiModuleRef::DecoderLayer {
                    layer_idx: 0,
                    component:
                        KimiLayerModuleRef::Attention {
                            kind: KimiAttentionLayerKind::LinearAttentionKda,
                        },
                },
            missing_tensors,
        } if missing_tensors == vec!["model.layers.0.self_attn.k_proj.weight".to_string()]
    ));
}
