mod support;

use attnres::kimi::{
    compare_baseline_slice_parity_fixture_from_dir,
    compare_baseline_slice_parity_fixture_with_manifest_from_dir, KimiAttentionLayerKind,
    KimiBaselineSliceParityError, KimiImportError, KimiLayerModuleRef, KimiModuleRef,
};
use serde_json::Value;
use std::process::Command;
use support::kimi_local_artifact::{TestBackend, TinyBaselinePayloadArtifactBuilder};
use support::kimi_slice_parity::{
    build_valid_fixture, build_valid_fixture_from_manifest, build_valid_manifest, write_fixture,
    write_fixture_value, write_manifest, write_manifest_value, write_valid_fixture,
    write_valid_manifest,
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
fn kimi_rfc_0005_gate2_round_trips_emitted_manifest_into_fixture_consumer() {
    if std::env::var_os("ATTNRES_KIMI_SLICE_MANIFEST_CHILD").is_some() {
        run_positive_slice_manifest_round_trip();
        return;
    }

    let status = Command::new(std::env::current_exe().unwrap())
        .env("ATTNRES_KIMI_SLICE_MANIFEST_CHILD", "1")
        .arg("--exact")
        .arg("kimi_rfc_0005_gate2_round_trips_emitted_manifest_into_fixture_consumer")
        .arg("--test-threads=1")
        .status()
        .unwrap();
    assert!(status.success());
}

fn run_positive_slice_manifest_round_trip() {
    let builder = TinyBaselinePayloadArtifactBuilder::single_layer_dense_kda();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let manifest = write_valid_manifest(artifact.path(), selection.clone(), &[0]);
    let fixture = write_valid_fixture(artifact.path(), selection, &[0]);
    let device = Default::default();

    compare_baseline_slice_parity_fixture_with_manifest_from_dir::<TestBackend, _, _, _>(
        artifact.path(),
        manifest.path(),
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
fn kimi_rfc_0005_gate2_slice_manifest_rejects_kind_mismatch() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let mut manifest = build_valid_manifest(artifact.path(), selection.clone(), &[0, 1]);
    manifest.kind = "attnres.kimi.wrong_slice_request_kind".to_string();
    let manifest = write_manifest(&manifest);
    let fixture = write_valid_fixture(artifact.path(), selection, &[0, 1]);
    let device = Default::default();

    let err = compare_baseline_slice_parity_fixture_with_manifest_from_dir::<TestBackend, _, _, _>(
        artifact.path(),
        manifest.path(),
        fixture.path(),
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiBaselineSliceParityError::UnexpectedManifestKind { kind }
            if kind == "attnres.kimi.wrong_slice_request_kind"
    ));
}

#[test]
fn kimi_rfc_0005_gate2_slice_manifest_rejects_version_mismatch() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let mut manifest = build_valid_manifest(artifact.path(), selection.clone(), &[0, 1]);
    manifest.version += 1;
    let manifest = write_manifest(&manifest);
    let fixture = write_valid_fixture(artifact.path(), selection, &[0, 1]);
    let device = Default::default();

    let err = compare_baseline_slice_parity_fixture_with_manifest_from_dir::<TestBackend, _, _, _>(
        artifact.path(),
        manifest.path(),
        fixture.path(),
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiBaselineSliceParityError::UnsupportedManifestVersion { version } if version == 2
    ));
}

#[test]
fn kimi_rfc_0005_gate2_slice_manifest_rejects_out_of_range_selected_hidden_layers() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let mut manifest = build_valid_manifest(artifact.path(), selection.clone(), &[0, 1]);
    manifest.slice.selected_hidden_layers = vec![0, 2];
    let manifest = write_manifest(&manifest);
    let fixture = write_valid_fixture(artifact.path(), selection, &[0, 1]);
    let device = Default::default();

    let err = compare_baseline_slice_parity_fixture_with_manifest_from_dir::<TestBackend, _, _, _>(
        artifact.path(),
        manifest.path(),
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
fn kimi_rfc_0005_gate2_slice_manifest_rejects_hidden_layers_outside_import_selection() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let mut manifest = build_valid_manifest(artifact.path(), selection.clone(), &[0, 1]);
    manifest.slice.import_selection.layer_indices = vec![0];
    let manifest = write_manifest(&manifest);
    let fixture = write_valid_fixture(artifact.path(), selection, &[0, 1]);
    let device = Default::default();

    let err = compare_baseline_slice_parity_fixture_with_manifest_from_dir::<TestBackend, _, _, _>(
        artifact.path(),
        manifest.path(),
        fixture.path(),
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiBaselineSliceParityError::SelectedHiddenLayerNotInImportSelection {
            layer_idx,
            ref import_selection_layers,
        } if layer_idx == 1 && import_selection_layers == &vec![0]
    ));
}

#[test]
fn kimi_rfc_0005_gate2_slice_manifest_rejects_duplicate_selected_hidden_layers() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let mut manifest = build_valid_manifest(artifact.path(), selection.clone(), &[0, 1]);
    manifest.slice.selected_hidden_layers = vec![0, 0];
    let manifest = write_manifest(&manifest);
    let fixture = write_valid_fixture(artifact.path(), selection, &[0, 1]);
    let device = Default::default();

    let err = compare_baseline_slice_parity_fixture_with_manifest_from_dir::<TestBackend, _, _, _>(
        artifact.path(),
        manifest.path(),
        fixture.path(),
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiBaselineSliceParityError::Import(KimiImportError::DuplicateSelectedLayer {
            layer_idx,
        }) if layer_idx == 0
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

#[test]
fn kimi_rfc_0005_gate2_slice_manifest_rejects_prompt_metadata_mismatch() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let manifest_value = build_valid_manifest(artifact.path(), selection.clone(), &[0, 1]);
    let mut fixture_value = build_valid_fixture_from_manifest(artifact.path(), &manifest_value);
    fixture_value.prompts[0].name = "ascending_len4_renamed".to_string();
    fixture_value.prompt_results[0].prompt_name = "ascending_len4_renamed".to_string();
    let manifest = write_manifest(&manifest_value);
    let fixture = write_fixture(&fixture_value);
    let device = Default::default();

    let err = compare_baseline_slice_parity_fixture_with_manifest_from_dir::<TestBackend, _, _, _>(
        artifact.path(),
        manifest.path(),
        fixture.path(),
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiBaselineSliceParityError::FixtureManifestMismatch {
            ref field,
            ref actual,
            ..
        } if field == "prompts[0].name" && actual == "\"ascending_len4_renamed\""
    ));
}

#[test]
fn kimi_rfc_0005_gate2_slice_manifest_rejects_tolerance_mismatch() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let manifest_value = build_valid_manifest(artifact.path(), selection.clone(), &[0, 1]);
    let mut fixture_value = build_valid_fixture_from_manifest(artifact.path(), &manifest_value);
    fixture_value.slice.tolerances.hidden_state_max_abs_diff = 1e-4;
    let manifest = write_manifest(&manifest_value);
    let fixture = write_fixture(&fixture_value);
    let device = Default::default();

    let err = compare_baseline_slice_parity_fixture_with_manifest_from_dir::<TestBackend, _, _, _>(
        artifact.path(),
        manifest.path(),
        fixture.path(),
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiBaselineSliceParityError::FixtureManifestMismatch {
            ref field,
            ref actual,
            ..
        } if field == "slice.tolerances.hidden_state_max_abs_diff" && actual == "0.0001"
    ));
}

#[test]
fn kimi_rfc_0005_gate2_slice_manifest_rejects_missing_required_tolerance_fields() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let manifest = build_valid_manifest(artifact.path(), selection.clone(), &[0, 1]);
    let mut value: Value = serde_json::to_value(manifest).unwrap();
    value["slice"]["tolerances"]
        .as_object_mut()
        .unwrap()
        .remove("hidden_state_max_abs_diff");
    let manifest = write_manifest_value(&value);
    let fixture = write_valid_fixture(artifact.path(), selection, &[0, 1]);
    let device = Default::default();

    let err = compare_baseline_slice_parity_fixture_with_manifest_from_dir::<TestBackend, _, _, _>(
        artifact.path(),
        manifest.path(),
        fixture.path(),
        &device,
    )
    .unwrap_err();

    match err {
        KimiBaselineSliceParityError::ManifestParseFailed { detail, .. } => {
            assert!(detail.contains("hidden_state_max_abs_diff"));
        }
        other => panic!("expected parse failure for missing tolerance field, got {other:?}"),
    }
}

#[test]
fn kimi_rfc_0005_gate2_slice_manifest_rejects_module_drift_against_current_import_plan() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let mut manifest_value = build_valid_manifest(artifact.path(), selection.clone(), &[0, 1]);
    manifest_value.slice.requested_modules.pop();
    let manifest = write_manifest(&manifest_value);
    let fixture = write_valid_fixture(artifact.path(), selection, &[0, 1]);
    let device = Default::default();

    let err = compare_baseline_slice_parity_fixture_with_manifest_from_dir::<TestBackend, _, _, _>(
        artifact.path(),
        manifest.path(),
        fixture.path(),
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiBaselineSliceParityError::ManifestFieldMismatch {
            ref field,
            ..
        } if field == "slice.requested_modules"
    ));
}

#[test]
fn kimi_rfc_0005_gate2_slice_manifest_rejects_required_tensor_drift_against_current_import_plan() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let mut manifest_value = build_valid_manifest(artifact.path(), selection.clone(), &[0, 1]);
    manifest_value.slice.required_tensors.pop();
    let manifest = write_manifest(&manifest_value);
    let fixture = write_valid_fixture(artifact.path(), selection, &[0, 1]);
    let device = Default::default();

    let err = compare_baseline_slice_parity_fixture_with_manifest_from_dir::<TestBackend, _, _, _>(
        artifact.path(),
        manifest.path(),
        fixture.path(),
        &device,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        KimiBaselineSliceParityError::ManifestFieldMismatch {
            ref field,
            ..
        } if field == "slice.required_tensors"
    ));
}
