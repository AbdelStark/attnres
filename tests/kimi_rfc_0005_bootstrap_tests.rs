mod support;

use attnres::kimi::{
    KimiArtifactUnderstanding, KimiAttnResBootstrapParityStatus, KimiAttnResBootstrapPolicy,
    KimiAttnResModel,
};
use support::kimi_local_artifact::{
    input_ids, max_abs_diff, seed_backend, TestBackend, TinyBaselinePayloadArtifactBuilder,
};

#[test]
fn kimi_rfc_0005_bootstrap_report_marks_attn_res_import_as_structural_bootstrap_only() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let artifact = builder.write();
    let device = Default::default();
    let understanding = KimiArtifactUnderstanding::load_from_dir(artifact.path()).unwrap();

    let bootstrap = understanding
        .try_bootstrap_attn_res_model_from_dir::<TestBackend, _>(
            artifact.path(),
            2,
            selection.clone(),
            KimiAttnResBootstrapPolicy::BaselineImportWithFreshAttnRes,
            &device,
        )
        .unwrap();

    let plan = understanding.try_slice_plan(selection.clone()).unwrap();
    assert_eq!(
        bootstrap.report.policy,
        KimiAttnResBootstrapPolicy::BaselineImportWithFreshAttnRes
    );
    assert_eq!(bootstrap.report.baseline_selection, selection);
    assert_eq!(
        bootstrap.report.imported_baseline_tensor_count,
        plan.coverage.mapped_tensors.len()
    );
    assert_eq!(
        bootstrap.report.imported_baseline_module_count,
        plan.coverage.module_coverage.len()
    );
    assert_eq!(bootstrap.report.attn_res_operator_count, 4);
    assert_eq!(
        bootstrap.report.parity_status,
        KimiAttnResBootstrapParityStatus::StructuralBootstrapOnly
    );
    assert!(bootstrap.report.requires_training_before_quality_eval);
    assert!(
        bootstrap
            .report
            .notes
            .iter()
            .any(|note| note.contains("structural bootstrap")),
        "bootstrap report should state that this load path is structural bootstrap only",
    );
}

#[test]
fn kimi_rfc_0005_bootstrap_default_loader_matches_explicit_policy_path() {
    let builder = TinyBaselinePayloadArtifactBuilder::new();
    let selection = builder.full_selection();
    let config = builder.config();
    let artifact = builder.write();
    let device = Default::default();
    let prompt = input_ids(&[0, 1, 2, 3], &device);
    let understanding = KimiArtifactUnderstanding::load_from_dir(artifact.path()).unwrap();

    seed_backend(&device, 20260318);
    let explicit = understanding
        .try_bootstrap_attn_res_model_from_dir::<TestBackend, _>(
            artifact.path(),
            2,
            selection.clone(),
            KimiAttnResBootstrapPolicy::BaselineImportWithFreshAttnRes,
            &device,
        )
        .unwrap();

    seed_backend(&device, 20260318);
    let implicit = KimiAttnResModel::<TestBackend>::try_from_artifact_dir(
        artifact.path(),
        2,
        selection,
        &device,
    )
    .unwrap();

    let diff = max_abs_diff(
        explicit.model.forward(prompt.clone()),
        implicit.forward(prompt),
    );
    assert!(
        diff < 1e-6,
        "explicit bootstrap policy path should match the convenience loader, diff={diff}",
    );

    assert_eq!(
        config.default_attn_res_bootstrap_policy(),
        KimiAttnResBootstrapPolicy::BaselineImportWithFreshAttnRes
    );
}
