mod support;

use attnres::kimi::{KimiImportCoverageError, KimiImportError};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use support::kimi_baseline_parity::{
    assert_bundle_matches, generate_fixture, load_bundle_from_dir, KimiBaselineParityHarnessError,
    KimiBaselineParityPromptSpec,
};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("kimi")
        .join("tiny-random-baseline")
}

fn read_fixture_text(file_name: &str) -> String {
    fs::read_to_string(fixture_dir().join(file_name)).unwrap()
}

fn unique_temp_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should move forward")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "attnres-kimi-rfc-0005-gate1-{}-{nanos}",
        std::process::id()
    ))
}

fn write_fixture_dir(config_json: &str, index_json: &str, parity_json: &str) -> PathBuf {
    let dir = unique_temp_dir();
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("config.json"), config_json).unwrap();
    fs::write(dir.join("model.safetensors.index.json"), index_json).unwrap();
    fs::write(dir.join("baseline-parity.json"), parity_json).unwrap();
    dir
}

#[test]
fn kimi_rfc_0005_gate1_tiny_random_baseline_fixture_matches_local_baseline_path() {
    let bundle = load_bundle_from_dir(fixture_dir()).unwrap();
    assert_bundle_matches(&bundle, 1e-5);
}

#[test]
fn kimi_rfc_0005_gate1_rejects_unsupported_artifact_dtype_before_parity_claims() {
    let mut config_json: Value = serde_json::from_str(&read_fixture_text("config.json")).unwrap();
    config_json["dtype"] = json!("float16");

    let dir = write_fixture_dir(
        &serde_json::to_string_pretty(&config_json).unwrap(),
        &read_fixture_text("model.safetensors.index.json"),
        &read_fixture_text("baseline-parity.json"),
    );
    let err = load_bundle_from_dir(&dir).unwrap_err();

    assert!(matches!(
        err,
        KimiBaselineParityHarnessError::Import(KimiImportError::UnsupportedArtifactDtype { dtype })
            if dtype == "float16"
    ));
}

#[test]
fn kimi_rfc_0005_gate1_rejects_selected_hidden_layers_outside_artifact_range() {
    let mut parity_json: Value =
        serde_json::from_str(&read_fixture_text("baseline-parity.json")).unwrap();
    parity_json["selected_hidden_layers"] = json!([0, 2]);

    let dir = write_fixture_dir(
        &read_fixture_text("config.json"),
        &read_fixture_text("model.safetensors.index.json"),
        &serde_json::to_string_pretty(&parity_json).unwrap(),
    );
    let err = load_bundle_from_dir(&dir).unwrap_err();

    assert!(matches!(
        err,
        KimiBaselineParityHarnessError::Import(KimiImportError::SelectedLayerOutOfRange {
            layer_idx,
            num_hidden_layers
        }) if layer_idx == 2 && num_hidden_layers == 2
    ));
}

#[test]
fn kimi_rfc_0005_gate1_rejects_missing_required_selected_layer_tensors() {
    let mut index_json: Value =
        serde_json::from_str(&read_fixture_text("model.safetensors.index.json")).unwrap();
    index_json["weight_map"]
        .as_object_mut()
        .unwrap()
        .remove("model.layers.0.self_attn.v_proj.weight");

    let dir = write_fixture_dir(
        &read_fixture_text("config.json"),
        &serde_json::to_string_pretty(&index_json).unwrap(),
        &read_fixture_text("baseline-parity.json"),
    );
    let err = load_bundle_from_dir(&dir).unwrap_err();

    assert!(matches!(
        err,
        KimiBaselineParityHarnessError::Coverage(
            KimiImportCoverageError::MissingRequiredTensor { tensor_name, .. }
        ) if tensor_name == "model.layers.0.self_attn.v_proj.weight"
    ));
}

#[test]
fn kimi_rfc_0005_gate1_rejects_unsupported_selected_layer_tensors() {
    let mut index_json: Value =
        serde_json::from_str(&read_fixture_text("model.safetensors.index.json")).unwrap();
    index_json["weight_map"].as_object_mut().unwrap().insert(
        "model.layers.1.self_attn.kv_a_proj_with_mqa.weight".to_string(),
        json!("model-00002-of-00002.safetensors"),
    );

    let dir = write_fixture_dir(
        &read_fixture_text("config.json"),
        &serde_json::to_string_pretty(&index_json).unwrap(),
        &read_fixture_text("baseline-parity.json"),
    );
    let err = load_bundle_from_dir(&dir).unwrap_err();
    match err {
        KimiBaselineParityHarnessError::Coverage(KimiImportCoverageError::UnsupportedTensor {
            tensor_name,
            ..
        }) => {
            assert_eq!(
                tensor_name,
                "model.layers.1.self_attn.kv_a_proj_with_mqa.weight"
            );
        }
        other => panic!("expected unsupported tensor coverage error, got {other:?}"),
    }
}

#[test]
#[ignore = "developer helper to regenerate the committed tiny-random baseline parity fixture"]
fn kimi_rfc_0005_gate1_print_tiny_random_baseline_fixture() {
    let config =
        attnres::kimi::KimiArtifactConfig::load(fixture_dir().join("config.json")).unwrap();
    let prompt_specs = vec![
        KimiBaselineParityPromptSpec {
            name: "ascending_len4".to_string(),
            input_ids: vec![0, 1, 2, 3],
        },
        KimiBaselineParityPromptSpec {
            name: "repeat_len3".to_string(),
            input_ids: vec![5, 5, 1],
        },
        KimiBaselineParityPromptSpec {
            name: "alternating_len4".to_string(),
            input_ids: vec![7, 3, 7, 3],
        },
    ];
    let fixture = generate_fixture(&config, 20260318, &[0, 1], &prompt_specs);
    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}
