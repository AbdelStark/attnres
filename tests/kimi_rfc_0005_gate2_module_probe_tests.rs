mod support;

use attnres::kimi::{
    build_default_module_probe_request, compare_module_probe_fixture_from_dir,
    generate_module_probe_fixture_from_dir, KimiAttentionLayerKind, KimiModuleProbeCache,
    KimiModuleProbeError, KimiModuleProbeTarget, KIMI_MODULE_PROBE_DEFAULT_SEQUENCE_LEN,
};
use support::kimi_local_artifact::{seed_backend, TestBackend, TinyBaselinePayloadArtifactBuilder};

#[test]
fn kimi_rfc_0005_gate2_module_probe_roundtrip_covers_kda_mla_norm_and_lm_head() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let config = builder.config();
    let artifact = builder.write();
    let device = Default::default();
    let request = build_default_module_probe_request(&config).unwrap();

    seed_backend(&device, request.seed);
    let fixture = generate_module_probe_fixture_from_dir::<TestBackend, _>(
        artifact.path(),
        &request,
        &device,
    )
    .unwrap();

    assert_eq!(fixture.probes.len(), 4);
    assert!(matches!(
        fixture.probes[0].target,
        KimiModuleProbeTarget::KdaAttention { layer_idx: 0 }
    ));
    assert!(fixture.probes[0].compare_decode);
    assert_eq!(
        fixture.probes[0].decode_steps.len(),
        KIMI_MODULE_PROBE_DEFAULT_SEQUENCE_LEN
    );
    assert!(matches!(
        fixture.probes[1].target,
        KimiModuleProbeTarget::MlaAttention { layer_idx: 1 }
    ));
    assert!(fixture.probes[1].compare_decode);
    assert_eq!(
        fixture.probes[1].decode_steps.len(),
        KIMI_MODULE_PROBE_DEFAULT_SEQUENCE_LEN
    );
    assert!(matches!(
        fixture.probes[2].target,
        KimiModuleProbeTarget::FinalNorm
    ));
    assert!(!fixture.probes[2].compare_decode);
    assert!(fixture.probes[2].decode_steps.is_empty());
    assert!(matches!(
        fixture.probes[3].target,
        KimiModuleProbeTarget::LmHead
    ));
    assert!(!fixture.probes[3].compare_decode);
    assert!(fixture.probes[3].decode_steps.is_empty());
    assert_eq!(
        fixture.probes[0].fingerprint.shard_paths,
        vec!["model-00001-of-00002.safetensors".to_string()]
    );
    assert_eq!(
        fixture.probes[1].fingerprint.shard_paths,
        vec!["model-00002-of-00002.safetensors".to_string()]
    );

    seed_backend(&device, request.seed);
    compare_module_probe_fixture_from_dir::<TestBackend, _>(
        artifact.path(),
        &request,
        &fixture,
        &device,
    )
    .unwrap();
}

#[test]
fn kimi_rfc_0005_gate2_module_probe_rejects_wrong_attention_kind_for_layer() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let config = builder.config();
    let artifact = builder.write();
    let device = Default::default();
    let mut request = build_default_module_probe_request(&config).unwrap();

    request.probes[0].target = KimiModuleProbeTarget::MlaAttention { layer_idx: 0 };

    let err = generate_module_probe_fixture_from_dir::<TestBackend, _>(
        artifact.path(),
        &request,
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiModuleProbeError::ProbeAttentionKindMismatch {
            layer_idx: 0,
            expected: KimiAttentionLayerKind::FullAttention,
            actual: KimiAttentionLayerKind::LinearAttentionKda,
        }
    ));
}

#[test]
fn kimi_rfc_0005_gate2_module_probe_detects_output_shape_drift() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let config = builder.config();
    let artifact = builder.write();
    let device = Default::default();
    let request = build_default_module_probe_request(&config).unwrap();

    seed_backend(&device, request.seed);
    let mut fixture = generate_module_probe_fixture_from_dir::<TestBackend, _>(
        artifact.path(),
        &request,
        &device,
    )
    .unwrap();
    fixture.probes[0].output.dims[2] += 1;

    seed_backend(&device, request.seed);
    let err = compare_module_probe_fixture_from_dir::<TestBackend, _>(
        artifact.path(),
        &request,
        &fixture,
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiModuleProbeError::TensorShapeMismatch {
            probe_name,
            tensor_label,
            ..
        } if probe_name == "kda_layer_0_seq4" && tensor_label == "output"
    ));
}

#[test]
fn kimi_rfc_0005_gate2_module_probe_detects_decode_cache_token_drift() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let config = builder.config();
    let artifact = builder.write();
    let device = Default::default();
    let request = build_default_module_probe_request(&config).unwrap();

    seed_backend(&device, request.seed);
    let mut fixture = generate_module_probe_fixture_from_dir::<TestBackend, _>(
        artifact.path(),
        &request,
        &device,
    )
    .unwrap();
    match &mut fixture.probes[0].decode_steps[0].cache {
        KimiModuleProbeCache::Kda {
            processed_tokens, ..
        } => *processed_tokens += 1,
        other => panic!("expected KDA cache, got {other:?}"),
    }

    seed_backend(&device, request.seed);
    let err = compare_module_probe_fixture_from_dir::<TestBackend, _>(
        artifact.path(),
        &request,
        &fixture,
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiModuleProbeError::ProcessedTokenMismatch {
            probe_name,
            cache_label,
            expected: 2,
            actual: 1,
        } if probe_name == "kda_layer_0_seq4" && cache_label == "kda"
    ));
}
