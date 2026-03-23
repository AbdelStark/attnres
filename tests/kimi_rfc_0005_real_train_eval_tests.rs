mod support;

use attnres::kimi::{
    run_kimi_attn_res_real_train_eval_from_config_path, KimiAttnResAdamOptimizerConfig,
    KimiAttnResModel, KimiAttnResRealTrainEvalConfig, KimiAttnResRealTrainEvalStatus,
    KimiAttnResTrainEvalFailureCriteria, KimiAttnResTrainableScope, KimiTokenSliceFile,
    KIMI_ATTN_RES_REAL_TRAIN_EVAL_CONFIG_KIND, KIMI_ATTN_RES_REAL_TRAIN_EVAL_CONFIG_VERSION,
    KIMI_TOKEN_SLICE_KIND, KIMI_TOKEN_SLICE_VERSION,
};
use burn::backend::{Autodiff, NdArray};
use burn::module::{ModuleVisitor, Param};
use burn::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use support::kimi_local_artifact::TinyBaselinePayloadArtifactBuilder;

type TrainBackend = Autodiff<NdArray>;

static REAL_TRAIN_EVAL_COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn kimi_rfc_0005_real_train_eval_attn_res_only_freeze_keeps_only_attn_res_paths_trainable() {
    let device = Default::default();
    let config = TinyBaselinePayloadArtifactBuilder::single_layer_dense_kda().config();
    let model: KimiAttnResModel<TrainBackend> = config.try_init_attn_res_model(2, &device).unwrap();

    let frozen = model.freeze_baseline_parameters();
    let mut visitor = TrainablePathVisitor::default();
    frozen.visit(&mut visitor);
    visitor.paths.sort();

    assert_eq!(
        visitor.paths,
        vec![
            "layers.0.attn_res.norm.gamma".to_string(),
            "layers.0.attn_res.pseudo_query".to_string(),
            "layers.0.mlp_res.norm.gamma".to_string(),
            "layers.0.mlp_res.pseudo_query".to_string(),
        ]
    );
}

#[test]
fn kimi_rfc_0005_real_train_eval_blocks_on_missing_token_slices() {
    let temp_dir = unique_temp_dir("blocked-missing-data");
    let artifact = TinyBaselinePayloadArtifactBuilder::single_layer_dense_kda().write();
    let config_path = temp_dir.join("config.json");
    let device = Default::default();

    write_config(
        &config_path,
        KimiAttnResRealTrainEvalConfig {
            kind: KIMI_ATTN_RES_REAL_TRAIN_EVAL_CONFIG_KIND.to_string(),
            version: KIMI_ATTN_RES_REAL_TRAIN_EVAL_CONFIG_VERSION,
            run_name: "missing_token_slices".to_string(),
            artifact_dir: artifact.path().display().to_string(),
            repo_id: None,
            revision: None,
            baseline_smoke_report_path: None,
            train_data_path: temp_dir.join("missing-train.json").display().to_string(),
            validation_data_path: temp_dir.join("missing-valid.json").display().to_string(),
            report_path: temp_dir.join("report.json").display().to_string(),
            checkpoint_base_path: temp_dir.join("checkpoint").display().to_string(),
            seed: 20260323,
            num_blocks: 2,
            bootstrap_policy: Default::default(),
            trainable_scope: KimiAttnResTrainableScope::AttnResOnly,
            batch_size: 1,
            max_train_steps: 1,
            max_eval_batches: Some(1),
            optimizer: default_optimizer(),
            failure_criteria: lenient_failure_criteria(),
        },
    );

    let report = run_kimi_attn_res_real_train_eval_from_config_path::<TrainBackend, _>(
        &config_path,
        vec!["cargo".to_string(), "test".to_string()],
        &device,
    )
    .unwrap();

    assert_eq!(
        report.status,
        KimiAttnResRealTrainEvalStatus::BlockedPreflight
    );
    assert!(report.train.is_none());
    assert!(report.eval.is_none());
    assert!(!report.outputs.checkpoint_written);
    assert!(report
        .preflight_blockers
        .iter()
        .any(|blocker| blocker.detail.contains("missing-train.json")));
    assert!(report
        .preflight_blockers
        .iter()
        .any(|blocker| blocker.detail.contains("missing-valid.json")));
}

#[test]
fn kimi_rfc_0005_real_train_eval_local_artifact_path_runs_end_to_end() {
    let temp_dir = unique_temp_dir("local-pass");
    let artifact = TinyBaselinePayloadArtifactBuilder::single_layer_dense_kda().write();
    let train_data_path = temp_dir.join("train.json");
    let validation_data_path = temp_dir.join("valid.json");
    let config_path = temp_dir.join("config.json");
    let report_path = temp_dir.join("report.json");
    let checkpoint_base_path = temp_dir.join("checkpoint");
    let device = Default::default();

    write_token_slice(
        &train_data_path,
        "tiny_train",
        vec![vec![0, 1, 2, 3], vec![3, 2, 1, 0]],
    );
    write_token_slice(
        &validation_data_path,
        "tiny_valid",
        vec![vec![1, 2, 3, 4], vec![4, 3, 2, 1]],
    );
    write_config(
        &config_path,
        KimiAttnResRealTrainEvalConfig {
            kind: KIMI_ATTN_RES_REAL_TRAIN_EVAL_CONFIG_KIND.to_string(),
            version: KIMI_ATTN_RES_REAL_TRAIN_EVAL_CONFIG_VERSION,
            run_name: "tiny_local_attn_res".to_string(),
            artifact_dir: artifact.path().display().to_string(),
            repo_id: None,
            revision: None,
            baseline_smoke_report_path: None,
            train_data_path: train_data_path.display().to_string(),
            validation_data_path: validation_data_path.display().to_string(),
            report_path: report_path.display().to_string(),
            checkpoint_base_path: checkpoint_base_path.display().to_string(),
            seed: 20260323,
            num_blocks: 2,
            bootstrap_policy: Default::default(),
            trainable_scope: KimiAttnResTrainableScope::AttnResOnly,
            batch_size: 1,
            max_train_steps: 2,
            max_eval_batches: Some(2),
            optimizer: KimiAttnResAdamOptimizerConfig {
                learning_rate: 1e-4,
                ..default_optimizer()
            },
            failure_criteria: lenient_failure_criteria(),
        },
    );

    let report = run_kimi_attn_res_real_train_eval_from_config_path::<TrainBackend, _>(
        &config_path,
        vec![
            "cargo".to_string(),
            "run".to_string(),
            "--example".to_string(),
            "kimi_real_model_tools".to_string(),
        ],
        &device,
    )
    .unwrap();

    assert_eq!(report.status, KimiAttnResRealTrainEvalStatus::Passed);
    assert!(report.preflight_blockers.is_empty());
    assert!(report.outputs.checkpoint_written);
    assert!(Path::new(&report.outputs.checkpoint_file_path).is_file());
    assert_eq!(report.train.as_ref().unwrap().steps_completed, 2);
    assert!(report.eval.as_ref().unwrap().mean_loss.unwrap().is_finite());
    assert!(report
        .parameter_summary
        .as_ref()
        .unwrap()
        .trainable_parameter_paths
        .iter()
        .all(|path| path.contains("attn_res") || path.contains("mlp_res")));

    report.write(&report.outputs.report_path).unwrap();
    assert!(report_path.is_file());
}

#[derive(Default)]
struct TrainablePathVisitor {
    path: Vec<String>,
    paths: Vec<String>,
}

impl<B: Backend> ModuleVisitor<B> for TrainablePathVisitor {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if param.is_require_grad() {
            self.paths.push(self.path.join("."));
        }
    }
}

fn default_optimizer() -> KimiAttnResAdamOptimizerConfig {
    KimiAttnResAdamOptimizerConfig {
        learning_rate: 1e-4,
        beta_1: 0.9,
        beta_2: 0.999,
        epsilon: 1e-5,
    }
}

fn lenient_failure_criteria() -> KimiAttnResTrainEvalFailureCriteria {
    KimiAttnResTrainEvalFailureCriteria {
        max_train_loss_growth_factor: 100.0,
        max_train_loss_growth_slack: 100.0,
        max_grad_l2_norm: 1.0e9,
        max_hidden_rms: 1.0e9,
        max_logit_rms: 1.0e9,
        require_validation_batches: true,
    }
}

fn write_token_slice(path: &Path, slice_name: &str, sequences: Vec<Vec<usize>>) {
    let file = KimiTokenSliceFile {
        kind: KIMI_TOKEN_SLICE_KIND.to_string(),
        version: KIMI_TOKEN_SLICE_VERSION,
        slice_name: slice_name.to_string(),
        source: "test".to_string(),
        tokenizer: Some("unit-test".to_string()),
        sequences,
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(
        path,
        format!("{}\n", serde_json::to_string_pretty(&file).unwrap()),
    )
    .unwrap();
}

fn write_config(path: &Path, config: KimiAttnResRealTrainEvalConfig) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(
        path,
        format!("{}\n", serde_json::to_string_pretty(&config).unwrap()),
    )
    .unwrap();
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should move forward")
        .as_nanos();
    let counter = REAL_TRAIN_EVAL_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "attnres-kimi-real-train-eval-{label}-{}-{nanos}-{counter}",
        std::process::id(),
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}
