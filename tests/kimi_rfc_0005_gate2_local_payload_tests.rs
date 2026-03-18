mod support;

use attnres::kimi::{
    KimiAttentionLayerKind, KimiBaselinePayloadError, KimiImportCoverageError, KimiImportSelection,
    KimiLayerModuleRef, KimiLinearModel, KimiModuleRef, KimiShardResolverError,
};
use support::kimi_local_artifact::{
    input_ids, max_abs_diff, seed_backend, LocalSafetensorTensor, TestBackend,
    TinyBaselinePayloadArtifactBuilder,
};

#[test]
fn kimi_rfc_0005_gate2_local_payload_harness_changes_baseline_logits_deterministically() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let config = builder.config();
    let artifact = builder.write();
    let device = Default::default();
    let prompt = input_ids(&[0, 1, 2, 3], &device);

    seed_backend(&device, 20260318);
    let baseline = config.try_init_model::<TestBackend>(&device).unwrap();
    let baseline_logits = baseline.forward(prompt.clone());

    seed_backend(&device, 20260318);
    let loaded_once = KimiLinearModel::<TestBackend>::try_from_artifact_dir(
        artifact.path(),
        selection.clone(),
        &device,
    )
    .unwrap();
    let loaded_once_logits = loaded_once.forward(prompt.clone());

    seed_backend(&device, 20260318);
    let loaded_twice =
        KimiLinearModel::<TestBackend>::try_from_artifact_dir(artifact.path(), selection, &device)
            .unwrap();
    let loaded_twice_logits = loaded_twice.forward(prompt);

    let baseline_diff = max_abs_diff(baseline_logits, loaded_once_logits.clone());
    assert!(
        baseline_diff > 1e-4,
        "loaded baseline logits should differ from the randomly initialized baseline, diff={baseline_diff}",
    );

    let repeat_diff = max_abs_diff(loaded_once_logits, loaded_twice_logits);
    assert!(
        repeat_diff < 1e-6,
        "loading the same sharded artifact twice with the same seed should be deterministic, diff={repeat_diff}",
    );
}

#[test]
fn kimi_rfc_0005_gate2_rejects_unsupported_selected_tensor_before_loading_payloads() {
    let mut builder = TinyBaselinePayloadArtifactBuilder::new();
    builder.insert_weight_map_tensor(
        "model.layers.1.self_attn.q_a_proj.weight",
        "model-00002-of-00002.safetensors",
    );
    let selection = builder.full_selection();
    let artifact = builder.write();
    let device = Default::default();

    let err =
        KimiLinearModel::<TestBackend>::try_from_artifact_dir(artifact.path(), selection, &device)
            .unwrap_err();

    assert!(matches!(
        err,
        KimiBaselinePayloadError::Coverage(KimiImportCoverageError::UnsupportedTensor {
            tensor_name,
            ..
        }) if tensor_name == "model.layers.1.self_attn.q_a_proj.weight"
    ));
}

#[test]
fn kimi_rfc_0005_gate2_rejects_missing_required_shard_files() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let device = Default::default();

    std::fs::remove_file(artifact.path().join("model-00002-of-00002.safetensors")).unwrap();
    let err =
        KimiLinearModel::<TestBackend>::try_from_artifact_dir(artifact.path(), selection, &device)
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
fn kimi_rfc_0005_gate2_rejects_unsupported_payload_dtypes() {
    let mut builder = TinyBaselinePayloadArtifactBuilder::new();
    builder.replace_tensor_payload(
        "model.embed_tokens.weight",
        LocalSafetensorTensor::raw("F16", vec![16, 8], vec![0; 16 * 8 * 2]),
    );
    let selection = builder.full_selection();
    let artifact = builder.write();
    let device = Default::default();

    let err =
        KimiLinearModel::<TestBackend>::try_from_artifact_dir(artifact.path(), selection, &device)
            .unwrap_err();

    assert!(matches!(
        err,
        KimiBaselinePayloadError::UnsupportedTensorDtype { tensor_name, dtype }
            if tensor_name == "model.embed_tokens.weight" && dtype == "F16"
    ));
}

#[test]
fn kimi_rfc_0005_gate2_rejects_tensor_shape_mismatches() {
    let mut builder = TinyBaselinePayloadArtifactBuilder::new();
    builder.replace_tensor_payload(
        "model.embed_tokens.weight",
        LocalSafetensorTensor::filled_f32(vec![15, 8], 0.25),
    );
    let selection = builder.full_selection();
    let artifact = builder.write();
    let device = Default::default();

    let err =
        KimiLinearModel::<TestBackend>::try_from_artifact_dir(artifact.path(), selection, &device)
            .unwrap_err();

    match err {
        KimiBaselinePayloadError::TensorShapeMismatch {
            tensor_name,
            expected,
            actual,
        } => {
            assert_eq!(tensor_name, "model.embed_tokens.weight");
            assert_eq!(expected, vec![16, 8]);
            assert_eq!(actual, vec![15, 8]);
        }
        other => panic!("expected tensor shape mismatch, got {other:?}"),
    }
}

#[test]
fn kimi_rfc_0005_gate2_rejects_incomplete_selected_layer_payload_coverage() {
    let mut builder = TinyBaselinePayloadArtifactBuilder::new();
    builder.remove_tensor_payload("model.layers.0.self_attn.k_proj.weight");
    let selection = KimiImportSelection {
        layer_indices: vec![0],
        include_embeddings: true,
        include_final_norm: true,
        include_lm_head: true,
    };
    let artifact = builder.write();
    let device = Default::default();

    let err =
        KimiLinearModel::<TestBackend>::try_from_artifact_dir(artifact.path(), selection, &device)
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
