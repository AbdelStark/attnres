mod support;

use attnres::kimi::{
    compare_baseline_slice_parity_fixture_from_dir, KimiAttentionLayerKind,
    KimiBaselineSliceParityError, KimiImportError, KimiLayerModuleRef, KimiModuleRef,
};
use serde_json::Value;
use std::process::Command;
use support::kimi_local_artifact::{TestBackend, TinyBaselinePayloadArtifactBuilder};
use support::kimi_slice_parity::{
    build_valid_fixture, write_fixture, write_fixture_value, write_valid_fixture,
};

#[test]
fn kimi_rfc_0005_gate2_consumes_external_baseline_slice_fixture_for_local_sharded_artifact() {
    if std::env::var_os("ATTNRES_KIMI_SLICE_PARITY_CHILD").is_some() {
        run_positive_slice_parity_fixture_comparison();
        return;
    }

    let status = Command::new(std::env::current_exe().unwrap())
        .env("ATTNRES_KIMI_SLICE_PARITY_CHILD", "1")
        .arg("--exact")
        .arg("kimi_rfc_0005_gate2_consumes_external_baseline_slice_fixture_for_local_sharded_artifact")
        .arg("--test-threads=1")
        .status()
        .unwrap();
    assert!(status.success());
}

fn run_positive_slice_parity_fixture_comparison() {
    let builder = TinyBaselinePayloadArtifactBuilder::single_layer_dense_kda();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let fixture = write_valid_fixture(artifact.path(), selection, &[0]);
    let device = Default::default();

    compare_baseline_slice_parity_fixture_from_dir::<TestBackend, _, _>(
        artifact.path(),
        fixture.path(),
        &device,
    )
    .unwrap();
}

#[test]
fn kimi_rfc_0005_gate2_slice_parity_rejects_fixture_kind_mismatch() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let mut fixture = build_valid_fixture(artifact.path(), selection, &[0, 1]);
    fixture.kind = "attnres.kimi.wrong_fixture_kind".to_string();
    let fixture = write_fixture(&fixture);
    let device = Default::default();

    let err = compare_baseline_slice_parity_fixture_from_dir::<TestBackend, _, _>(
        artifact.path(),
        fixture.path(),
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiBaselineSliceParityError::UnexpectedFixtureKind { kind }
            if kind == "attnres.kimi.wrong_fixture_kind"
    ));
}

#[test]
fn kimi_rfc_0005_gate2_slice_parity_rejects_fixture_version_mismatch() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let mut fixture = build_valid_fixture(artifact.path(), selection, &[0, 1]);
    fixture.version += 1;
    let fixture = write_fixture(&fixture);
    let device = Default::default();

    let err = compare_baseline_slice_parity_fixture_from_dir::<TestBackend, _, _>(
        artifact.path(),
        fixture.path(),
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiBaselineSliceParityError::UnsupportedFixtureVersion { version } if version == 2
    ));
}

#[test]
fn kimi_rfc_0005_gate2_slice_parity_rejects_selected_layer_mismatch_against_loaded_config() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let mut fixture = build_valid_fixture(artifact.path(), selection, &[0, 1]);
    fixture.slice.import_selection.layer_indices = vec![0, 2];
    let fixture = write_fixture(&fixture);
    let device = Default::default();

    let err = compare_baseline_slice_parity_fixture_from_dir::<TestBackend, _, _>(
        artifact.path(),
        fixture.path(),
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiBaselineSliceParityError::Import(KimiImportError::SelectedLayerOutOfRange {
            layer_idx,
            num_hidden_layers,
        }) if layer_idx == 2 && num_hidden_layers == 2
    ));
}

#[test]
fn kimi_rfc_0005_gate2_slice_parity_rejects_prompt_token_mismatch() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let mut fixture = build_valid_fixture(artifact.path(), selection, &[0, 1]);
    fixture.prompt_results[0].input_ids = vec![9, 9, 9, 9];
    let fixture = write_fixture(&fixture);
    let device = Default::default();

    let err = compare_baseline_slice_parity_fixture_from_dir::<TestBackend, _, _>(
        artifact.path(),
        fixture.path(),
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiBaselineSliceParityError::PromptTokenMismatch {
            prompt_name,
            actual,
            ..
        } if prompt_name == "ascending_len4" && actual == vec![9, 9, 9, 9]
    ));
}

#[test]
fn kimi_rfc_0005_gate2_slice_parity_rejects_tolerance_metadata_mismatch() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let mut fixture = build_valid_fixture(artifact.path(), selection, &[0, 1]);
    fixture.slice.tolerances.runtime_dtype = "bfloat16".to_string();
    let fixture = write_fixture(&fixture);
    let device = Default::default();

    let err = compare_baseline_slice_parity_fixture_from_dir::<TestBackend, _, _>(
        artifact.path(),
        fixture.path(),
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiBaselineSliceParityError::UnsupportedToleranceMetadata {
            field,
            actual,
            ..
        } if field == "runtime_dtype" && actual == "bfloat16"
    ));
}

#[test]
fn kimi_rfc_0005_gate2_slice_parity_rejects_missing_required_tolerance_fields() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let fixture = build_valid_fixture(artifact.path(), selection, &[0, 1]);
    let mut value: Value = serde_json::to_value(fixture).unwrap();
    value["slice"]["tolerances"]
        .as_object_mut()
        .unwrap()
        .remove("hidden_state_max_abs_diff");
    let fixture = write_fixture_value(&value);
    let device = Default::default();

    let err = compare_baseline_slice_parity_fixture_from_dir::<TestBackend, _, _>(
        artifact.path(),
        fixture.path(),
        &device,
    )
    .unwrap_err();

    match err {
        KimiBaselineSliceParityError::ParseFailed { detail, .. } => {
            assert!(detail.contains("hidden_state_max_abs_diff"));
        }
        other => panic!("expected parse failure for missing tolerance field, got {other:?}"),
    }
}

#[test]
fn kimi_rfc_0005_gate2_slice_parity_rejects_unsupported_module_requests() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let mut fixture = build_valid_fixture(artifact.path(), selection, &[0, 1]);
    fixture
        .slice
        .requested_modules
        .push(KimiModuleRef::DecoderLayer {
            layer_idx: 1,
            component: KimiLayerModuleRef::Attention {
                kind: KimiAttentionLayerKind::LinearAttentionKda,
            },
        });
    let fixture = write_fixture(&fixture);
    let device = Default::default();

    let err = compare_baseline_slice_parity_fixture_from_dir::<TestBackend, _, _>(
        artifact.path(),
        fixture.path(),
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiBaselineSliceParityError::UnsupportedModuleRequest {
            module: KimiModuleRef::DecoderLayer {
                layer_idx: 1,
                component: KimiLayerModuleRef::Attention {
                    kind: KimiAttentionLayerKind::LinearAttentionKda,
                },
            },
        }
    ));
}
