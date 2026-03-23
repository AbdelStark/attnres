use burn::module::{AutodiffModule, Module, ModuleVisitor, Param};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::AutodiffBackend;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::kimi::bootstrap::{
    KimiAttnResBootstrapLoadResult, KimiAttnResBootstrapPolicy, KimiAttnResBootstrapReport,
};
use crate::kimi::import::{
    KimiArtifactUnderstanding, KimiImportDtypeAction, KimiImportRuntimeDtype, KimiImportSelection,
};
use crate::kimi::KimiAttnResModel;

pub const KIMI_ATTN_RES_REAL_TRAIN_EVAL_CONFIG_KIND: &str =
    "attnres.kimi.attn_res_real_train_eval_config";
pub const KIMI_ATTN_RES_REAL_TRAIN_EVAL_CONFIG_VERSION: u32 = 1;
pub const KIMI_ATTN_RES_REAL_TRAIN_EVAL_REPORT_KIND: &str =
    "attnres.kimi.attn_res_real_train_eval_report";
pub const KIMI_ATTN_RES_REAL_TRAIN_EVAL_REPORT_VERSION: u32 = 1;
pub const KIMI_TOKEN_SLICE_KIND: &str = "attnres.kimi.token_slice";
pub const KIMI_TOKEN_SLICE_VERSION: u32 = 1;
pub const KIMI_BASELINE_SMOKE_REPORT_KIND: &str = "attnres.kimi.baseline_smoke_report";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KimiAttnResTrainableScope {
    AttnResOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiAttnResAdamOptimizerConfig {
    pub learning_rate: f64,
    pub beta_1: f32,
    pub beta_2: f32,
    pub epsilon: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiAttnResTrainEvalFailureCriteria {
    pub max_train_loss_growth_factor: f32,
    pub max_train_loss_growth_slack: f32,
    pub max_grad_l2_norm: f32,
    pub max_hidden_rms: f32,
    pub max_logit_rms: f32,
    pub require_validation_batches: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiAttnResRealTrainEvalConfig {
    pub kind: String,
    pub version: u32,
    pub run_name: String,
    pub artifact_dir: String,
    pub repo_id: Option<String>,
    pub revision: Option<String>,
    pub baseline_smoke_report_path: Option<String>,
    pub train_data_path: String,
    pub validation_data_path: String,
    pub report_path: String,
    pub checkpoint_base_path: String,
    pub seed: u64,
    pub num_blocks: usize,
    pub bootstrap_policy: KimiAttnResBootstrapPolicy,
    pub trainable_scope: KimiAttnResTrainableScope,
    pub batch_size: usize,
    pub max_train_steps: usize,
    pub max_eval_batches: Option<usize>,
    pub optimizer: KimiAttnResAdamOptimizerConfig,
    pub failure_criteria: KimiAttnResTrainEvalFailureCriteria,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiTokenSliceFile {
    pub kind: String,
    pub version: u32,
    pub slice_name: String,
    pub source: String,
    pub tokenizer: Option<String>,
    pub sequences: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiFileFingerprint {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiHostFacts {
    pub os: String,
    pub arch: String,
    pub available_parallelism: usize,
    pub host_total_ram_bytes: Option<u64>,
    pub cpu_brand: Option<String>,
    pub model: Option<String>,
    pub os_product_version: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KimiAttnResPreflightBlockerKind {
    ArtifactReadFailed,
    InvalidFullImportPlan,
    MissingCheckpointShard,
    MissingTrainData,
    InvalidTrainData,
    MissingValidationData,
    InvalidValidationData,
    InvalidBaselineSmokeReport,
    InsufficientHostRamLowerBound,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiAttnResPreflightBlocker {
    pub kind: KimiAttnResPreflightBlockerKind,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KimiAttnResRealTrainEvalStatus {
    Passed,
    BlockedPreflight,
    FailedTrainGate,
    FailedEvalGate,
    FailedRuntimeError,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiAttnResRealTrainEvalTimings {
    pub preflight_seconds: f64,
    pub bootstrap_load_seconds: Option<f64>,
    pub train_seconds: Option<f64>,
    pub eval_seconds: Option<f64>,
    pub checkpoint_save_seconds: Option<f64>,
    pub total_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiAttnResParameterSummary {
    pub total_parameter_count: usize,
    pub trainable_parameter_count: usize,
    pub frozen_parameter_count: usize,
    pub trainable_parameter_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct KimiPreparedTokenSliceSummary {
    pub path: String,
    pub fingerprint: Option<KimiFileFingerprint>,
    pub slice_name: Option<String>,
    pub source: Option<String>,
    pub tokenizer: Option<String>,
    pub sequence_count: Option<usize>,
    pub raw_sequence_length: Option<usize>,
    pub model_input_sequence_length: Option<usize>,
    pub batch_count: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiAttnResTrainStepObservation {
    pub step: usize,
    pub loss: f32,
    pub hidden_rms: f32,
    pub logit_rms: f32,
    pub grad_l2_norm: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiAttnResTrainSummary {
    pub steps_completed: usize,
    pub observations: Vec<KimiAttnResTrainStepObservation>,
    pub initial_loss: Option<f32>,
    pub final_loss: Option<f32>,
    pub max_loss: Option<f32>,
    pub max_hidden_rms: f32,
    pub max_logit_rms: f32,
    pub max_grad_l2_norm: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiAttnResEvalSummary {
    pub batches_evaluated: usize,
    pub batch_losses: Vec<f32>,
    pub mean_loss: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct KimiAttnResArtifactSummary {
    pub artifact_dir: String,
    pub repo_id: Option<String>,
    pub revision: Option<String>,
    pub model_type: Option<String>,
    pub artifact_dtype: Option<String>,
    pub runtime_dtype: Option<String>,
    pub dtype_action: Option<String>,
    pub num_hidden_layers: Option<usize>,
    pub checkpoint_total_size_bytes: Option<u64>,
    pub tensor_count: Option<usize>,
    pub unique_shard_count: Option<usize>,
    pub present_shards: Vec<String>,
    pub missing_shards: Vec<String>,
    pub minimum_runtime_weight_bytes_lower_bound: Option<u64>,
    pub minimum_runtime_weight_bytes_note: Option<String>,
    pub full_import_plan_loadable: Option<bool>,
    pub required_shards: Vec<String>,
    pub artifact_fingerprints: Option<Value>,
    pub artifact_fingerprint_source: Option<String>,
    pub bootstrap_report: Option<KimiAttnResBootstrapReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiAttnResRealTrainEvalOutputs {
    pub report_path: String,
    pub checkpoint_base_path: String,
    pub checkpoint_file_path: String,
    pub checkpoint_written: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiAttnResRealTrainEvalReport {
    pub kind: String,
    pub version: u32,
    pub status: KimiAttnResRealTrainEvalStatus,
    pub run_name: String,
    pub command: Vec<String>,
    pub config_fingerprint: KimiFileFingerprint,
    pub config: KimiAttnResRealTrainEvalConfig,
    pub hardware: KimiHostFacts,
    pub artifact: KimiAttnResArtifactSummary,
    pub train_data: KimiPreparedTokenSliceSummary,
    pub validation_data: KimiPreparedTokenSliceSummary,
    pub parameter_summary: Option<KimiAttnResParameterSummary>,
    pub train: Option<KimiAttnResTrainSummary>,
    pub eval: Option<KimiAttnResEvalSummary>,
    pub outputs: KimiAttnResRealTrainEvalOutputs,
    pub preflight_blockers: Vec<KimiAttnResPreflightBlocker>,
    pub failure_reason: Option<String>,
    pub timings: KimiAttnResRealTrainEvalTimings,
    pub notes: Vec<String>,
}

#[derive(Debug)]
pub enum KimiAttnResRealTrainEvalError {
    ConfigReadFailed { path: String, detail: String },
    ConfigParseFailed { path: String, detail: String },
    ConfigValidation(String),
    ReportWriteFailed { path: String, detail: String },
    CheckpointSaveFailed { path: String, detail: String },
}

impl Display for KimiAttnResRealTrainEvalError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConfigReadFailed { path, detail } => {
                write!(
                    f,
                    "failed to read AttnRes real train/eval config '{path}': {detail}"
                )
            }
            Self::ConfigParseFailed { path, detail } => {
                write!(
                    f,
                    "failed to parse AttnRes real train/eval config '{path}': {detail}"
                )
            }
            Self::ConfigValidation(detail) => {
                write!(f, "invalid AttnRes real train/eval config: {detail}")
            }
            Self::ReportWriteFailed { path, detail } => {
                write!(
                    f,
                    "failed to write AttnRes real train/eval report '{path}': {detail}"
                )
            }
            Self::CheckpointSaveFailed { path, detail } => {
                write!(
                    f,
                    "failed to save AttnRes real train/eval checkpoint '{path}': {detail}"
                )
            }
        }
    }
}

impl std::error::Error for KimiAttnResRealTrainEvalError {}

#[derive(Debug, Clone)]
struct PreparedTokenSlice {
    summary: KimiPreparedTokenSliceSummary,
    batches: Vec<TokenBatch>,
}

#[derive(Debug, Clone)]
struct TokenBatch {
    input_ids: Vec<i64>,
    targets: Vec<i64>,
    batch_size: usize,
    seq_len: usize,
}

#[derive(Debug)]
struct TrainGateFailure {
    reason: String,
    summary: KimiAttnResTrainSummary,
}

#[derive(Debug)]
struct EvalGateFailure {
    reason: String,
    summary: KimiAttnResEvalSummary,
}

#[derive(Debug, Default)]
struct RunContext {
    artifact: KimiAttnResArtifactSummary,
    train_data: KimiPreparedTokenSliceSummary,
    validation_data: KimiPreparedTokenSliceSummary,
    parameter_summary: Option<KimiAttnResParameterSummary>,
    train: Option<KimiAttnResTrainSummary>,
    eval: Option<KimiAttnResEvalSummary>,
    preflight_blockers: Vec<KimiAttnResPreflightBlocker>,
    failure_reason: Option<String>,
    notes: Vec<String>,
}

#[derive(Debug)]
struct PreflightPrepared {
    understanding: Option<KimiArtifactUnderstanding>,
    train_data: Option<PreparedTokenSlice>,
    validation_data: Option<PreparedTokenSlice>,
}

#[derive(Debug)]
struct ReportInput {
    config: KimiAttnResRealTrainEvalConfig,
    config_fingerprint: KimiFileFingerprint,
    command: Vec<String>,
    hardware: KimiHostFacts,
}

#[derive(Debug, Clone)]
struct BaselineSmokeFingerprintSource {
    artifact_fingerprints: Value,
}

impl KimiAttnResRealTrainEvalConfig {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, KimiAttnResRealTrainEvalError> {
        let path = path.as_ref();
        let json = fs::read_to_string(path).map_err(|err| {
            KimiAttnResRealTrainEvalError::ConfigReadFailed {
                path: path.display().to_string(),
                detail: err.to_string(),
            }
        })?;
        let config = serde_json::from_str(&json).map_err(|err| {
            KimiAttnResRealTrainEvalError::ConfigParseFailed {
                path: path.display().to_string(),
                detail: err.to_string(),
            }
        })?;
        Ok(config)
    }

    pub fn try_validate(&self) -> Result<(), KimiAttnResRealTrainEvalError> {
        if self.kind != KIMI_ATTN_RES_REAL_TRAIN_EVAL_CONFIG_KIND {
            return Err(KimiAttnResRealTrainEvalError::ConfigValidation(format!(
                "expected kind '{}' , got '{}'",
                KIMI_ATTN_RES_REAL_TRAIN_EVAL_CONFIG_KIND, self.kind
            )));
        }
        if self.version != KIMI_ATTN_RES_REAL_TRAIN_EVAL_CONFIG_VERSION {
            return Err(KimiAttnResRealTrainEvalError::ConfigValidation(format!(
                "expected version {}, got {}",
                KIMI_ATTN_RES_REAL_TRAIN_EVAL_CONFIG_VERSION, self.version
            )));
        }
        if self.run_name.trim().is_empty() {
            return Err(KimiAttnResRealTrainEvalError::ConfigValidation(
                "run_name must not be empty".to_string(),
            ));
        }
        for (field, value) in [
            ("artifact_dir", self.artifact_dir.trim()),
            ("train_data_path", self.train_data_path.trim()),
            ("validation_data_path", self.validation_data_path.trim()),
            ("report_path", self.report_path.trim()),
            ("checkpoint_base_path", self.checkpoint_base_path.trim()),
        ] {
            if value.is_empty() {
                return Err(KimiAttnResRealTrainEvalError::ConfigValidation(format!(
                    "{field} must not be empty"
                )));
            }
        }
        if self.num_blocks == 0 {
            return Err(KimiAttnResRealTrainEvalError::ConfigValidation(
                "num_blocks must be > 0".to_string(),
            ));
        }
        if self.batch_size == 0 {
            return Err(KimiAttnResRealTrainEvalError::ConfigValidation(
                "batch_size must be > 0".to_string(),
            ));
        }
        if self.max_train_steps == 0 {
            return Err(KimiAttnResRealTrainEvalError::ConfigValidation(
                "max_train_steps must be > 0".to_string(),
            ));
        }
        if self.optimizer.learning_rate <= 0.0 {
            return Err(KimiAttnResRealTrainEvalError::ConfigValidation(
                "optimizer.learning_rate must be > 0".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&self.optimizer.beta_1) {
            return Err(KimiAttnResRealTrainEvalError::ConfigValidation(
                "optimizer.beta_1 must be in [0, 1)".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&self.optimizer.beta_2) {
            return Err(KimiAttnResRealTrainEvalError::ConfigValidation(
                "optimizer.beta_2 must be in [0, 1)".to_string(),
            ));
        }
        if self.optimizer.epsilon <= 0.0 {
            return Err(KimiAttnResRealTrainEvalError::ConfigValidation(
                "optimizer.epsilon must be > 0".to_string(),
            ));
        }
        if self.failure_criteria.max_train_loss_growth_factor <= 0.0 {
            return Err(KimiAttnResRealTrainEvalError::ConfigValidation(
                "failure_criteria.max_train_loss_growth_factor must be > 0".to_string(),
            ));
        }
        if self.failure_criteria.max_grad_l2_norm <= 0.0
            || self.failure_criteria.max_hidden_rms <= 0.0
            || self.failure_criteria.max_logit_rms <= 0.0
        {
            return Err(KimiAttnResRealTrainEvalError::ConfigValidation(
                "failure_criteria caps must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

impl KimiTokenSliceFile {
    fn try_validate(&self, vocab_size: Option<usize>) -> Result<(usize, usize), String> {
        if self.kind != KIMI_TOKEN_SLICE_KIND {
            return Err(format!(
                "expected token slice kind '{}' , got '{}'",
                KIMI_TOKEN_SLICE_KIND, self.kind
            ));
        }
        if self.version != KIMI_TOKEN_SLICE_VERSION {
            return Err(format!(
                "expected token slice version {}, got {}",
                KIMI_TOKEN_SLICE_VERSION, self.version
            ));
        }
        if self.slice_name.trim().is_empty() {
            return Err("slice_name must not be empty".to_string());
        }
        if self.sequences.is_empty() {
            return Err("sequences must not be empty".to_string());
        }
        let raw_len = self.sequences[0].len();
        if raw_len < 2 {
            return Err("each token sequence must contain at least 2 ids".to_string());
        }

        for (sequence_idx, sequence) in self.sequences.iter().enumerate() {
            if sequence.len() != raw_len {
                return Err(format!(
                    "sequence {sequence_idx} length {} does not match the first sequence length {raw_len}",
                    sequence.len()
                ));
            }
            if let Some(vocab_size) = vocab_size {
                if let Some((token_idx, token_id)) = sequence
                    .iter()
                    .enumerate()
                    .find(|(_, token_id)| **token_id >= vocab_size)
                {
                    return Err(format!(
                        "sequence {sequence_idx} token {token_idx} uses id {token_id} outside vocab_size {vocab_size}"
                    ));
                }
            }
        }

        Ok((self.sequences.len(), raw_len))
    }
}

impl KimiAttnResRealTrainEvalReport {
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<(), KimiAttnResRealTrainEvalError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                KimiAttnResRealTrainEvalError::ReportWriteFailed {
                    path: path.display().to_string(),
                    detail: err.to_string(),
                }
            })?;
        }
        let json = serde_json::to_string_pretty(self).map_err(|err| {
            KimiAttnResRealTrainEvalError::ReportWriteFailed {
                path: path.display().to_string(),
                detail: err.to_string(),
            }
        })?;
        fs::write(path, format!("{json}\n")).map_err(|err| {
            KimiAttnResRealTrainEvalError::ReportWriteFailed {
                path: path.display().to_string(),
                detail: err.to_string(),
            }
        })
    }
}

impl KimiAttnResAdamOptimizerConfig {
    fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<burn::optim::Adam, M, B> {
        AdamConfig::new()
            .with_beta_1(self.beta_1)
            .with_beta_2(self.beta_2)
            .with_epsilon(self.epsilon)
            .init()
    }
}

impl KimiAttnResTrainSummary {
    fn new() -> Self {
        Self {
            steps_completed: 0,
            observations: Vec::new(),
            initial_loss: None,
            final_loss: None,
            max_loss: None,
            max_hidden_rms: 0.0,
            max_logit_rms: 0.0,
            max_grad_l2_norm: 0.0,
        }
    }

    fn record(
        &mut self,
        step: usize,
        loss: f32,
        hidden_rms: f32,
        logit_rms: f32,
        grad_l2_norm: f32,
    ) {
        if self.initial_loss.is_none() {
            self.initial_loss = Some(loss);
        }
        self.final_loss = Some(loss);
        self.max_loss = Some(self.max_loss.map_or(loss, |value| value.max(loss)));
        self.max_hidden_rms = self.max_hidden_rms.max(hidden_rms);
        self.max_logit_rms = self.max_logit_rms.max(logit_rms);
        self.max_grad_l2_norm = self.max_grad_l2_norm.max(grad_l2_norm);
        self.steps_completed = step + 1;
        self.observations.push(KimiAttnResTrainStepObservation {
            step,
            loss,
            hidden_rms,
            logit_rms,
            grad_l2_norm,
        });
    }
}

impl KimiAttnResEvalSummary {
    fn empty() -> Self {
        Self {
            batches_evaluated: 0,
            batch_losses: Vec::new(),
            mean_loss: None,
        }
    }

    fn finalize(mut self) -> Self {
        if !self.batch_losses.is_empty() {
            let sum = self.batch_losses.iter().copied().sum::<f32>();
            self.mean_loss = Some(sum / self.batch_losses.len() as f32);
        }
        self
    }
}

impl Default for KimiHostFacts {
    fn default() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            available_parallelism: std::thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(1),
            host_total_ram_bytes: detect_total_ram_bytes(),
            cpu_brand: detect_sysctl_value("machdep.cpu.brand_string"),
            model: detect_sysctl_value("hw.model"),
            os_product_version: detect_sysctl_value("kern.osproductversion"),
        }
    }
}

pub fn run_kimi_attn_res_real_train_eval_from_config_path<B, P: AsRef<Path>>(
    config_path: P,
    command: Vec<String>,
    device: &B::Device,
) -> Result<KimiAttnResRealTrainEvalReport, KimiAttnResRealTrainEvalError>
where
    B: AutodiffBackend<FloatElem = f32>,
    B::InnerBackend: Backend<FloatElem = f32, BoolElem = bool>,
{
    let config_path = config_path.as_ref();
    let config = KimiAttnResRealTrainEvalConfig::load(config_path)?;
    config.try_validate()?;
    let config_fingerprint = fingerprint_file(config_path)?;
    Ok(run_kimi_attn_res_real_train_eval::<B>(
        config,
        config_fingerprint,
        command,
        device,
    ))
}

fn run_kimi_attn_res_real_train_eval<B>(
    config: KimiAttnResRealTrainEvalConfig,
    config_fingerprint: KimiFileFingerprint,
    command: Vec<String>,
    device: &B::Device,
) -> KimiAttnResRealTrainEvalReport
where
    B: AutodiffBackend<FloatElem = f32>,
    B::InnerBackend: Backend<FloatElem = f32, BoolElem = bool>,
{
    let run_started = Instant::now();
    let report_input = ReportInput {
        config,
        config_fingerprint,
        command,
        hardware: KimiHostFacts::default(),
    };
    let checkpoint_file_path = checkpoint_file_path(&report_input.config.checkpoint_base_path);
    let outputs = KimiAttnResRealTrainEvalOutputs {
        report_path: report_input.config.report_path.clone(),
        checkpoint_base_path: report_input.config.checkpoint_base_path.clone(),
        checkpoint_file_path: checkpoint_file_path.clone(),
        checkpoint_written: false,
    };
    let config = &report_input.config;
    let hardware = &report_input.hardware;

    let mut context = RunContext {
        notes: vec![
            "This path always bootstraps from baseline tensors plus freshly initialized AttnRes operators.".to_string(),
            "The current narrow first path freezes imported baseline parameters and trains AttnRes operators only.".to_string(),
            "Blocked preflight states are preserved as report statuses instead of being reinterpreted as quality results.".to_string(),
        ],
        ..Default::default()
    };

    let preflight_started = Instant::now();
    let prepared = preflight(config, hardware, &mut context);
    let preflight_seconds = preflight_started.elapsed().as_secs_f64();

    if !context.preflight_blockers.is_empty() {
        return finalize_report(
            KimiAttnResRealTrainEvalStatus::BlockedPreflight,
            report_input,
            outputs,
            context,
            KimiAttnResRealTrainEvalTimings {
                preflight_seconds,
                bootstrap_load_seconds: None,
                train_seconds: None,
                eval_seconds: None,
                checkpoint_save_seconds: None,
                total_seconds: run_started.elapsed().as_secs_f64(),
            },
        );
    }

    let Some(understanding) = prepared.understanding else {
        context
            .preflight_blockers
            .push(KimiAttnResPreflightBlocker {
                kind: KimiAttnResPreflightBlockerKind::ArtifactReadFailed,
                detail: "artifact understanding unexpectedly missing after preflight".to_string(),
            });
        return finalize_report(
            KimiAttnResRealTrainEvalStatus::BlockedPreflight,
            report_input,
            outputs,
            context,
            KimiAttnResRealTrainEvalTimings {
                preflight_seconds,
                bootstrap_load_seconds: None,
                train_seconds: None,
                eval_seconds: None,
                checkpoint_save_seconds: None,
                total_seconds: run_started.elapsed().as_secs_f64(),
            },
        );
    };
    let Some(train_data) = prepared.train_data else {
        context
            .preflight_blockers
            .push(KimiAttnResPreflightBlocker {
                kind: KimiAttnResPreflightBlockerKind::MissingTrainData,
                detail: "train data unexpectedly missing after preflight".to_string(),
            });
        return finalize_report(
            KimiAttnResRealTrainEvalStatus::BlockedPreflight,
            report_input,
            outputs,
            context,
            KimiAttnResRealTrainEvalTimings {
                preflight_seconds,
                bootstrap_load_seconds: None,
                train_seconds: None,
                eval_seconds: None,
                checkpoint_save_seconds: None,
                total_seconds: run_started.elapsed().as_secs_f64(),
            },
        );
    };
    let Some(validation_data) = prepared.validation_data else {
        context
            .preflight_blockers
            .push(KimiAttnResPreflightBlocker {
                kind: KimiAttnResPreflightBlockerKind::MissingValidationData,
                detail: "validation data unexpectedly missing after preflight".to_string(),
            });
        return finalize_report(
            KimiAttnResRealTrainEvalStatus::BlockedPreflight,
            report_input,
            outputs,
            context,
            KimiAttnResRealTrainEvalTimings {
                preflight_seconds,
                bootstrap_load_seconds: None,
                train_seconds: None,
                eval_seconds: None,
                checkpoint_save_seconds: None,
                total_seconds: run_started.elapsed().as_secs_f64(),
            },
        );
    };

    B::seed(device, config.seed);
    let load_started = Instant::now();
    let bootstrap = match bootstrap_full_attn_res_model::<B>(&understanding, config, device) {
        Ok(result) => result,
        Err(err) => {
            context.failure_reason = Some(err.to_string());
            return finalize_report(
                KimiAttnResRealTrainEvalStatus::FailedRuntimeError,
                report_input,
                outputs,
                context,
                KimiAttnResRealTrainEvalTimings {
                    preflight_seconds,
                    bootstrap_load_seconds: Some(load_started.elapsed().as_secs_f64()),
                    train_seconds: None,
                    eval_seconds: None,
                    checkpoint_save_seconds: None,
                    total_seconds: run_started.elapsed().as_secs_f64(),
                },
            );
        }
    };
    let bootstrap_load_seconds = load_started.elapsed().as_secs_f64();

    context.artifact.bootstrap_report = Some(bootstrap.report.clone());
    let (mut model, parameter_summary) =
        apply_trainable_scope(bootstrap.model, &config.trainable_scope);
    context.parameter_summary = Some(parameter_summary);

    let train_started = Instant::now();
    let train_result = run_train_loop::<B>(config, &mut model, &train_data, device);
    let train_seconds = train_started.elapsed().as_secs_f64();
    match train_result {
        Ok(summary) => context.train = Some(summary),
        Err(failure) => {
            context.failure_reason = Some(failure.reason);
            context.train = Some(failure.summary);
            return finalize_report(
                KimiAttnResRealTrainEvalStatus::FailedTrainGate,
                report_input,
                outputs,
                context,
                KimiAttnResRealTrainEvalTimings {
                    preflight_seconds,
                    bootstrap_load_seconds: Some(bootstrap_load_seconds),
                    train_seconds: Some(train_seconds),
                    eval_seconds: None,
                    checkpoint_save_seconds: None,
                    total_seconds: run_started.elapsed().as_secs_f64(),
                },
            );
        }
    }

    let eval_started = Instant::now();
    let inference_model = model.valid();
    let eval_device = inference_model
        .devices()
        .into_iter()
        .next()
        .expect("AttnRes model should expose at least one device");
    let eval_result = run_eval_loop(config, &inference_model, &validation_data, &eval_device);
    let eval_seconds = eval_started.elapsed().as_secs_f64();
    match eval_result {
        Ok(summary) => context.eval = Some(summary),
        Err(failure) => {
            context.failure_reason = Some(failure.reason);
            context.eval = Some(failure.summary);
            return finalize_report(
                KimiAttnResRealTrainEvalStatus::FailedEvalGate,
                report_input,
                outputs,
                context,
                KimiAttnResRealTrainEvalTimings {
                    preflight_seconds,
                    bootstrap_load_seconds: Some(bootstrap_load_seconds),
                    train_seconds: Some(train_seconds),
                    eval_seconds: Some(eval_seconds),
                    checkpoint_save_seconds: None,
                    total_seconds: run_started.elapsed().as_secs_f64(),
                },
            );
        }
    }

    let save_started = Instant::now();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    let checkpoint_save = inference_model
        .clone()
        .save_file(PathBuf::from(&config.checkpoint_base_path), &recorder)
        .map_err(|err| KimiAttnResRealTrainEvalError::CheckpointSaveFailed {
            path: config.checkpoint_base_path.clone(),
            detail: format!("{err:?}"),
        });
    let checkpoint_save_seconds = save_started.elapsed().as_secs_f64();

    let mut outputs = outputs;
    if let Err(err) = checkpoint_save {
        context.failure_reason = Some(err.to_string());
        return finalize_report(
            KimiAttnResRealTrainEvalStatus::FailedRuntimeError,
            report_input,
            outputs,
            context,
            KimiAttnResRealTrainEvalTimings {
                preflight_seconds,
                bootstrap_load_seconds: Some(bootstrap_load_seconds),
                train_seconds: Some(train_seconds),
                eval_seconds: Some(eval_seconds),
                checkpoint_save_seconds: Some(checkpoint_save_seconds),
                total_seconds: run_started.elapsed().as_secs_f64(),
            },
        );
    }
    outputs.checkpoint_written = Path::new(&checkpoint_file_path).is_file();

    finalize_report(
        KimiAttnResRealTrainEvalStatus::Passed,
        report_input,
        outputs,
        context,
        KimiAttnResRealTrainEvalTimings {
            preflight_seconds,
            bootstrap_load_seconds: Some(bootstrap_load_seconds),
            train_seconds: Some(train_seconds),
            eval_seconds: Some(eval_seconds),
            checkpoint_save_seconds: Some(checkpoint_save_seconds),
            total_seconds: run_started.elapsed().as_secs_f64(),
        },
    )
}

fn finalize_report(
    status: KimiAttnResRealTrainEvalStatus,
    input: ReportInput,
    outputs: KimiAttnResRealTrainEvalOutputs,
    context: RunContext,
    timings: KimiAttnResRealTrainEvalTimings,
) -> KimiAttnResRealTrainEvalReport {
    KimiAttnResRealTrainEvalReport {
        kind: KIMI_ATTN_RES_REAL_TRAIN_EVAL_REPORT_KIND.to_string(),
        version: KIMI_ATTN_RES_REAL_TRAIN_EVAL_REPORT_VERSION,
        status,
        run_name: input.config.run_name.clone(),
        command: input.command,
        config_fingerprint: input.config_fingerprint,
        config: input.config,
        hardware: input.hardware,
        artifact: context.artifact,
        train_data: context.train_data,
        validation_data: context.validation_data,
        parameter_summary: context.parameter_summary,
        train: context.train,
        eval: context.eval,
        outputs,
        preflight_blockers: context.preflight_blockers,
        failure_reason: context.failure_reason,
        timings,
        notes: context.notes,
    }
}

fn preflight(
    config: &KimiAttnResRealTrainEvalConfig,
    hardware: &KimiHostFacts,
    context: &mut RunContext,
) -> PreflightPrepared {
    let artifact_dir = PathBuf::from(&config.artifact_dir);
    let mut understanding = None;
    let mut train_data = None;
    let mut validation_data = None;

    match KimiArtifactUnderstanding::load_from_dir(&artifact_dir) {
        Ok(bundle) => {
            let unique_shards = bundle.shard_index.shard_paths();
            let mut present_shards = Vec::new();
            let mut missing_shards = Vec::new();
            for shard_path in unique_shards {
                if artifact_dir.join(shard_path).is_file() {
                    present_shards.push(shard_path.to_string());
                } else {
                    missing_shards.push(shard_path.to_string());
                    context
                        .preflight_blockers
                        .push(KimiAttnResPreflightBlocker {
                            kind: KimiAttnResPreflightBlockerKind::MissingCheckpointShard,
                            detail: format!(
                            "required full-checkpoint shard '{shard_path}' is missing under '{}'",
                            artifact_dir.display()
                        ),
                        });
                }
            }

            let full_plan = match bundle.try_full_plan() {
                Ok(plan) => {
                    context.artifact.full_import_plan_loadable =
                        Some(plan.coverage.is_fully_loadable());
                    context.artifact.required_shards = plan.required_shards.clone();
                    match plan.try_require_loadable() {
                        Ok(()) => {}
                        Err(err) => context
                            .preflight_blockers
                            .push(KimiAttnResPreflightBlocker {
                                kind: KimiAttnResPreflightBlockerKind::InvalidFullImportPlan,
                                detail: err.to_string(),
                            }),
                    }
                    Some(plan)
                }
                Err(err) => {
                    context
                        .preflight_blockers
                        .push(KimiAttnResPreflightBlocker {
                            kind: KimiAttnResPreflightBlockerKind::InvalidFullImportPlan,
                            detail: err.to_string(),
                        });
                    None
                }
            };

            let minimum_runtime_weight_bytes_lower_bound = minimum_runtime_weight_bytes_lower_bound(
                bundle.report.total_size_bytes,
                bundle.report.dtype_policy.runtime_dtype,
                bundle.report.dtype_policy.action,
                &bundle.config.dtype,
            );
            let minimum_runtime_weight_bytes_note = Some(
                "lower bound covers float runtime weights only; activations, optimizer state, and caches are additional".to_string(),
            );
            if let (Some(host_total_ram_bytes), Some(lower_bound)) = (
                hardware.host_total_ram_bytes,
                minimum_runtime_weight_bytes_lower_bound,
            ) {
                if host_total_ram_bytes < lower_bound {
                    context.preflight_blockers.push(KimiAttnResPreflightBlocker {
                        kind: KimiAttnResPreflightBlockerKind::InsufficientHostRamLowerBound,
                        detail: format!(
                            "host RAM {} bytes is below the runtime-weight lower bound {} bytes for dtype action {:?}",
                            host_total_ram_bytes,
                            lower_bound,
                            bundle.report.dtype_policy.action
                        ),
                    });
                }
            }

            let (artifact_fingerprints, artifact_fingerprint_source) =
                match load_baseline_smoke_fingerprints(config, &artifact_dir) {
                    Ok(Some(source)) => (
                        Some(source.artifact_fingerprints),
                        Some(config.baseline_smoke_report_path.clone().unwrap()),
                    ),
                    Ok(None) => (
                        local_artifact_fingerprints(&artifact_dir, &present_shards),
                        Some("local_config_index_only".to_string()),
                    ),
                    Err(blocker) => {
                        context.preflight_blockers.push(blocker);
                        (None, None)
                    }
                };

            context.artifact = KimiAttnResArtifactSummary {
                artifact_dir: config.artifact_dir.clone(),
                repo_id: config.repo_id.clone(),
                revision: config.revision.clone(),
                model_type: Some(bundle.config.model_type.clone()),
                artifact_dtype: Some(bundle.config.dtype.clone()),
                runtime_dtype: Some(render_runtime_dtype(
                    bundle.report.dtype_policy.runtime_dtype,
                )),
                dtype_action: Some(render_dtype_action(bundle.report.dtype_policy.action)),
                num_hidden_layers: Some(bundle.config.num_hidden_layers),
                checkpoint_total_size_bytes: Some(bundle.report.total_size_bytes),
                tensor_count: Some(bundle.report.tensor_count),
                unique_shard_count: Some(bundle.shard_index.shard_count()),
                present_shards,
                missing_shards,
                minimum_runtime_weight_bytes_lower_bound,
                minimum_runtime_weight_bytes_note,
                full_import_plan_loadable: full_plan
                    .as_ref()
                    .map(|plan| plan.coverage.is_fully_loadable()),
                required_shards: full_plan
                    .as_ref()
                    .map(|plan| plan.required_shards.clone())
                    .unwrap_or_default(),
                artifact_fingerprints,
                artifact_fingerprint_source,
                bootstrap_report: None,
            };

            let vocab_size = Some(bundle.config.vocab_size);
            train_data = prepare_token_slice(
                &config.train_data_path,
                config.batch_size,
                vocab_size,
                KimiAttnResPreflightBlockerKind::MissingTrainData,
                KimiAttnResPreflightBlockerKind::InvalidTrainData,
                &mut context.preflight_blockers,
            );
            validation_data = prepare_token_slice(
                &config.validation_data_path,
                config.batch_size,
                vocab_size,
                KimiAttnResPreflightBlockerKind::MissingValidationData,
                KimiAttnResPreflightBlockerKind::InvalidValidationData,
                &mut context.preflight_blockers,
            );
            context.train_data = train_data
                .as_ref()
                .map(|data| data.summary.clone())
                .unwrap_or_else(|| missing_slice_summary(&config.train_data_path));
            context.validation_data = validation_data
                .as_ref()
                .map(|data| data.summary.clone())
                .unwrap_or_else(|| missing_slice_summary(&config.validation_data_path));
            understanding = Some(bundle);
        }
        Err(err) => {
            context
                .preflight_blockers
                .push(KimiAttnResPreflightBlocker {
                    kind: KimiAttnResPreflightBlockerKind::ArtifactReadFailed,
                    detail: err.to_string(),
                });
            context.artifact = KimiAttnResArtifactSummary {
                artifact_dir: config.artifact_dir.clone(),
                repo_id: config.repo_id.clone(),
                revision: config.revision.clone(),
                model_type: None,
                artifact_dtype: None,
                runtime_dtype: None,
                dtype_action: None,
                num_hidden_layers: None,
                checkpoint_total_size_bytes: None,
                tensor_count: None,
                unique_shard_count: None,
                present_shards: Vec::new(),
                missing_shards: Vec::new(),
                minimum_runtime_weight_bytes_lower_bound: None,
                minimum_runtime_weight_bytes_note: None,
                full_import_plan_loadable: None,
                required_shards: Vec::new(),
                artifact_fingerprints: None,
                artifact_fingerprint_source: None,
                bootstrap_report: None,
            };
            context.train_data = missing_slice_summary(&config.train_data_path);
            context.validation_data = missing_slice_summary(&config.validation_data_path);
        }
    }

    PreflightPrepared {
        understanding,
        train_data,
        validation_data,
    }
}

fn bootstrap_full_attn_res_model<B: AutodiffBackend>(
    understanding: &KimiArtifactUnderstanding,
    config: &KimiAttnResRealTrainEvalConfig,
    device: &B::Device,
) -> Result<KimiAttnResBootstrapLoadResult<B>, String> {
    let selection = KimiImportSelection::full(understanding.config.num_hidden_layers);
    understanding
        .try_bootstrap_attn_res_model_from_dir(
            &config.artifact_dir,
            config.num_blocks,
            selection,
            config.bootstrap_policy.clone(),
            device,
        )
        .map_err(|err| err.to_string())
}

fn apply_trainable_scope<B: AutodiffBackend>(
    model: KimiAttnResModel<B>,
    scope: &KimiAttnResTrainableScope,
) -> (KimiAttnResModel<B>, KimiAttnResParameterSummary) {
    let model = match scope {
        KimiAttnResTrainableScope::AttnResOnly => model.freeze_baseline_parameters(),
    };
    let summary = summarize_trainable_parameters(&model);
    (model, summary)
}

fn run_train_loop<B>(
    config: &KimiAttnResRealTrainEvalConfig,
    model: &mut KimiAttnResModel<B>,
    train_data: &PreparedTokenSlice,
    device: &B::Device,
) -> Result<KimiAttnResTrainSummary, TrainGateFailure>
where
    B: AutodiffBackend<FloatElem = f32>,
    B::InnerBackend: Backend<FloatElem = f32, BoolElem = bool>,
{
    let mut optimizer = config.optimizer.init::<B, KimiAttnResModel<B>>();
    let loss_fn = CrossEntropyLossConfig::new().with_logits(true).init(device);
    let mut summary = KimiAttnResTrainSummary::new();

    for step in 0..config.max_train_steps {
        let batch = &train_data.batches[step % train_data.batches.len()];
        let input_ids = token_batch_input_ids::<B>(batch, device);
        let targets = token_batch_targets::<B>(batch, device);

        let hidden = model.forward_hidden(input_ids.clone()).inner();
        let logits = model.forward(input_ids);
        let logits_inner = logits.clone().inner();

        if !tensor_all_finite(hidden.clone()) {
            return Err(TrainGateFailure {
                reason: format!("step {step} produced non-finite hidden states"),
                summary,
            });
        }
        if !tensor_all_finite(logits_inner.clone()) {
            return Err(TrainGateFailure {
                reason: format!("step {step} produced non-finite logits"),
                summary,
            });
        }

        let hidden_rms = tensor_rms(hidden);
        let logit_rms = tensor_rms(logits_inner);
        let loss = loss_fn
            .forward(
                {
                    let vocab_size = logits.dims()[2];
                    logits.reshape([batch.batch_size * batch.seq_len, vocab_size])
                },
                targets.reshape([batch.batch_size * batch.seq_len]),
            )
            .mean();
        let loss_value: f32 = loss.clone().into_scalar();
        if !loss_value.is_finite() {
            return Err(TrainGateFailure {
                reason: format!("step {step} produced non-finite loss"),
                summary,
            });
        }

        let grads = GradientsParams::from_grads(loss.backward(), model);
        if grads.is_empty() {
            return Err(TrainGateFailure {
                reason: format!("step {step} unexpectedly produced an empty gradient set"),
                summary,
            });
        }
        let (grad_l2_norm, grads_finite) = gradient_l2_norm(model, &grads);
        if !grads_finite || !grad_l2_norm.is_finite() {
            return Err(TrainGateFailure {
                reason: format!("step {step} produced non-finite gradients"),
                summary,
            });
        }

        summary.record(step, loss_value, hidden_rms, logit_rms, grad_l2_norm);
        if let Some(reason) = evaluate_train_gates(&summary, &config.failure_criteria) {
            return Err(TrainGateFailure { reason, summary });
        }

        let next_model = optimizer.step(config.optimizer.learning_rate, model.clone(), grads);
        *model = next_model;
    }

    Ok(summary)
}

fn run_eval_loop<B>(
    config: &KimiAttnResRealTrainEvalConfig,
    model: &KimiAttnResModel<B>,
    validation_data: &PreparedTokenSlice,
    device: &B::Device,
) -> Result<KimiAttnResEvalSummary, EvalGateFailure>
where
    B: Backend<FloatElem = f32, BoolElem = bool>,
{
    let loss_fn = CrossEntropyLossConfig::new().with_logits(true).init(device);
    let mut summary = KimiAttnResEvalSummary::empty();

    for (batch_idx, batch) in validation_data.batches.iter().enumerate() {
        if let Some(limit) = config.max_eval_batches {
            if batch_idx >= limit {
                break;
            }
        }
        let input_ids = token_batch_input_ids::<B>(batch, device);
        let targets = token_batch_targets::<B>(batch, device);
        let logits = model.forward(input_ids);
        let logits_inner = logits.clone();
        if !tensor_all_finite(logits_inner.clone()) {
            return Err(EvalGateFailure {
                reason: format!("validation batch {batch_idx} produced non-finite logits"),
                summary: summary.finalize(),
            });
        }
        let loss = loss_fn
            .forward(
                {
                    let vocab_size = logits.dims()[2];
                    logits.reshape([batch.batch_size * batch.seq_len, vocab_size])
                },
                targets.reshape([batch.batch_size * batch.seq_len]),
            )
            .mean();
        let loss_value: f32 = loss.into_scalar();
        if !loss_value.is_finite() {
            return Err(EvalGateFailure {
                reason: format!("validation batch {batch_idx} produced non-finite loss"),
                summary: summary.finalize(),
            });
        }
        summary.batches_evaluated += 1;
        summary.batch_losses.push(loss_value);
    }

    let summary = summary.finalize();
    if config.failure_criteria.require_validation_batches && summary.batches_evaluated == 0 {
        return Err(EvalGateFailure {
            reason: "validation slice did not yield any batches".to_string(),
            summary,
        });
    }
    Ok(summary)
}

fn prepare_token_slice(
    path: &str,
    batch_size: usize,
    vocab_size: Option<usize>,
    missing_kind: KimiAttnResPreflightBlockerKind,
    invalid_kind: KimiAttnResPreflightBlockerKind,
    blockers: &mut Vec<KimiAttnResPreflightBlocker>,
) -> Option<PreparedTokenSlice> {
    let path_ref = Path::new(path);
    if !path_ref.is_file() {
        blockers.push(KimiAttnResPreflightBlocker {
            kind: missing_kind,
            detail: format!("token slice file '{}' does not exist", path_ref.display()),
        });
        return None;
    }

    let fingerprint = match fingerprint_file(path_ref) {
        Ok(fingerprint) => Some(fingerprint),
        Err(err) => {
            blockers.push(KimiAttnResPreflightBlocker {
                kind: invalid_kind.clone(),
                detail: err.to_string(),
            });
            return None;
        }
    };

    let json = match fs::read_to_string(path_ref) {
        Ok(json) => json,
        Err(err) => {
            blockers.push(KimiAttnResPreflightBlocker {
                kind: invalid_kind,
                detail: format!("failed to read token slice '{}': {err}", path_ref.display()),
            });
            return None;
        }
    };
    let slice: KimiTokenSliceFile = match serde_json::from_str(&json) {
        Ok(slice) => slice,
        Err(err) => {
            blockers.push(KimiAttnResPreflightBlocker {
                kind: invalid_kind,
                detail: format!(
                    "failed to parse token slice '{}': {err}",
                    path_ref.display()
                ),
            });
            return None;
        }
    };
    let (sequence_count, raw_sequence_length) = match slice.try_validate(vocab_size) {
        Ok(value) => value,
        Err(detail) => {
            blockers.push(KimiAttnResPreflightBlocker {
                kind: invalid_kind,
                detail: format!("invalid token slice '{}': {detail}", path_ref.display()),
            });
            return None;
        }
    };

    let batches = slice
        .sequences
        .chunks(batch_size)
        .map(build_token_batch)
        .collect::<Vec<_>>();
    Some(PreparedTokenSlice {
        summary: KimiPreparedTokenSliceSummary {
            path: path.to_string(),
            fingerprint,
            slice_name: Some(slice.slice_name.clone()),
            source: Some(slice.source.clone()),
            tokenizer: slice.tokenizer.clone(),
            sequence_count: Some(sequence_count),
            raw_sequence_length: Some(raw_sequence_length),
            model_input_sequence_length: Some(raw_sequence_length - 1),
            batch_count: Some(batches.len()),
        },
        batches,
    })
}

fn build_token_batch(sequences: &[Vec<usize>]) -> TokenBatch {
    let batch_size = sequences.len();
    let seq_len = sequences[0].len() - 1;
    let mut input_ids = Vec::with_capacity(batch_size * seq_len);
    let mut targets = Vec::with_capacity(batch_size * seq_len);
    for sequence in sequences {
        input_ids.extend(sequence[..seq_len].iter().map(|token| *token as i64));
        targets.extend(sequence[1..].iter().map(|token| *token as i64));
    }
    TokenBatch {
        input_ids,
        targets,
        batch_size,
        seq_len,
    }
}

fn token_batch_input_ids<B: Backend>(batch: &TokenBatch, device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::<B, 1, Int>::from_ints(batch.input_ids.as_slice(), device)
        .reshape([batch.batch_size, batch.seq_len])
}

fn token_batch_targets<B: Backend>(batch: &TokenBatch, device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::<B, 1, Int>::from_ints(batch.targets.as_slice(), device)
        .reshape([batch.batch_size, batch.seq_len])
}

fn evaluate_train_gates(
    summary: &KimiAttnResTrainSummary,
    criteria: &KimiAttnResTrainEvalFailureCriteria,
) -> Option<String> {
    let initial_loss = summary.initial_loss?;
    let max_loss = summary.max_loss?;
    let loss_cap =
        initial_loss * criteria.max_train_loss_growth_factor + criteria.max_train_loss_growth_slack;
    if max_loss > loss_cap {
        return Some(format!(
            "train loss exceeded the explicit cap: initial_loss={initial_loss} max_loss={max_loss} cap={loss_cap}"
        ));
    }
    if summary.max_grad_l2_norm > criteria.max_grad_l2_norm {
        return Some(format!(
            "train gradient L2 norm exceeded the explicit cap: observed={} cap={}",
            summary.max_grad_l2_norm, criteria.max_grad_l2_norm
        ));
    }
    if summary.max_hidden_rms > criteria.max_hidden_rms {
        return Some(format!(
            "train hidden RMS exceeded the explicit cap: observed={} cap={}",
            summary.max_hidden_rms, criteria.max_hidden_rms
        ));
    }
    if summary.max_logit_rms > criteria.max_logit_rms {
        return Some(format!(
            "train logit RMS exceeded the explicit cap: observed={} cap={}",
            summary.max_logit_rms, criteria.max_logit_rms
        ));
    }
    None
}

fn summarize_trainable_parameters<B: Backend>(
    model: &KimiAttnResModel<B>,
) -> KimiAttnResParameterSummary {
    let mut visitor = TrainableParameterVisitor {
        path: Vec::new(),
        total_parameter_count: 0,
        trainable_parameter_count: 0,
        trainable_parameter_paths: Vec::new(),
    };
    model.visit(&mut visitor);
    KimiAttnResParameterSummary {
        total_parameter_count: visitor.total_parameter_count,
        trainable_parameter_count: visitor.trainable_parameter_count,
        frozen_parameter_count: visitor
            .total_parameter_count
            .saturating_sub(visitor.trainable_parameter_count),
        trainable_parameter_paths: visitor.trainable_parameter_paths,
    }
}

struct TrainableParameterVisitor {
    path: Vec<String>,
    total_parameter_count: usize,
    trainable_parameter_count: usize,
    trainable_parameter_paths: Vec<String>,
}

impl<B: Backend> ModuleVisitor<B> for TrainableParameterVisitor {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let numel = param.shape().num_elements();
        self.total_parameter_count += numel;
        if param.is_require_grad() {
            self.trainable_parameter_count += numel;
            self.trainable_parameter_paths.push(self.path.join("."));
        }
    }
}

struct GradientNormVisitor<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    squared_l2_sum: f64,
    all_finite: bool,
    _marker: std::marker::PhantomData<B>,
}

impl<'a, B: AutodiffBackend> GradientNormVisitor<'a, B> {
    fn new(grads: &'a GradientsParams) -> Self {
        Self {
            grads,
            squared_l2_sum: 0.0,
            all_finite: true,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B> ModuleVisitor<B> for GradientNormVisitor<'_, B>
where
    B: AutodiffBackend,
    B::InnerBackend: Backend<FloatElem = f32, BoolElem = bool>,
{
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let Some(grad) = self.grads.get::<B::InnerBackend, D>(param.id) else {
            return;
        };
        let finite: bool = grad.clone().is_finite().all().into_scalar();
        self.all_finite &= finite;
        let sum_sq: f32 = grad.powi_scalar(2).sum().into_scalar();
        self.squared_l2_sum += f64::from(sum_sq);
    }
}

fn gradient_l2_norm<B>(model: &KimiAttnResModel<B>, grads: &GradientsParams) -> (f32, bool)
where
    B: AutodiffBackend,
    B::InnerBackend: Backend<FloatElem = f32, BoolElem = bool>,
{
    let mut visitor = GradientNormVisitor::<B>::new(grads);
    model.visit(&mut visitor);
    ((visitor.squared_l2_sum as f32).sqrt(), visitor.all_finite)
}

fn tensor_all_finite<B: Backend<BoolElem = bool>, const D: usize>(tensor: Tensor<B, D>) -> bool {
    tensor.is_finite().all().into_scalar()
}

fn tensor_rms<B: Backend<FloatElem = f32>, const D: usize>(tensor: Tensor<B, D>) -> f32 {
    tensor.powf_scalar(2.0).mean().sqrt().into_scalar()
}

fn missing_slice_summary(path: &str) -> KimiPreparedTokenSliceSummary {
    KimiPreparedTokenSliceSummary {
        path: path.to_string(),
        fingerprint: None,
        slice_name: None,
        source: None,
        tokenizer: None,
        sequence_count: None,
        raw_sequence_length: None,
        model_input_sequence_length: None,
        batch_count: None,
    }
}

fn render_runtime_dtype(dtype: KimiImportRuntimeDtype) -> String {
    match dtype {
        KimiImportRuntimeDtype::Float32 => "float32".to_string(),
        KimiImportRuntimeDtype::BFloat16 => "bfloat16".to_string(),
    }
}

fn render_dtype_action(action: KimiImportDtypeAction) -> String {
    match action {
        KimiImportDtypeAction::Preserve => "preserve".to_string(),
        KimiImportDtypeAction::PromoteToF32 => "promote_to_f32".to_string(),
    }
}

fn minimum_runtime_weight_bytes_lower_bound(
    checkpoint_total_size_bytes: u64,
    runtime_dtype: KimiImportRuntimeDtype,
    dtype_action: KimiImportDtypeAction,
    artifact_dtype: &str,
) -> Option<u64> {
    match (artifact_dtype, runtime_dtype, dtype_action) {
        ("bfloat16", KimiImportRuntimeDtype::Float32, KimiImportDtypeAction::PromoteToF32) => {
            checkpoint_total_size_bytes.checked_mul(2)
        }
        ("float32", KimiImportRuntimeDtype::Float32, KimiImportDtypeAction::Preserve) => {
            Some(checkpoint_total_size_bytes)
        }
        _ => None,
    }
}

fn load_baseline_smoke_fingerprints(
    config: &KimiAttnResRealTrainEvalConfig,
    artifact_dir: &Path,
) -> Result<Option<BaselineSmokeFingerprintSource>, KimiAttnResPreflightBlocker> {
    let Some(path) = &config.baseline_smoke_report_path else {
        return Ok(None);
    };
    let path_ref = Path::new(path);
    let json = fs::read_to_string(path_ref).map_err(|err| KimiAttnResPreflightBlocker {
        kind: KimiAttnResPreflightBlockerKind::InvalidBaselineSmokeReport,
        detail: format!(
            "failed to read baseline smoke report '{}': {err}",
            path_ref.display()
        ),
    })?;
    let value: Value = serde_json::from_str(&json).map_err(|err| KimiAttnResPreflightBlocker {
        kind: KimiAttnResPreflightBlockerKind::InvalidBaselineSmokeReport,
        detail: format!(
            "failed to parse baseline smoke report '{}': {err}",
            path_ref.display()
        ),
    })?;

    let kind =
        value
            .get("kind")
            .and_then(Value::as_str)
            .ok_or_else(|| KimiAttnResPreflightBlocker {
                kind: KimiAttnResPreflightBlockerKind::InvalidBaselineSmokeReport,
                detail: format!(
                    "baseline smoke report '{}' is missing the kind field",
                    path_ref.display()
                ),
            })?;
    if kind != KIMI_BASELINE_SMOKE_REPORT_KIND {
        return Err(KimiAttnResPreflightBlocker {
            kind: KimiAttnResPreflightBlockerKind::InvalidBaselineSmokeReport,
            detail: format!(
                "baseline smoke report '{}' expected kind '{}' , got '{}'",
                path_ref.display(),
                KIMI_BASELINE_SMOKE_REPORT_KIND,
                kind
            ),
        });
    }

    let report_artifact_dir = value
        .get("artifact_dir")
        .and_then(Value::as_str)
        .ok_or_else(|| KimiAttnResPreflightBlocker {
            kind: KimiAttnResPreflightBlockerKind::InvalidBaselineSmokeReport,
            detail: format!(
                "baseline smoke report '{}' is missing artifact_dir",
                path_ref.display()
            ),
        })?;
    if report_artifact_dir != artifact_dir.display().to_string() {
        return Err(KimiAttnResPreflightBlocker {
            kind: KimiAttnResPreflightBlockerKind::InvalidBaselineSmokeReport,
            detail: format!(
                "baseline smoke report artifact_dir '{}' does not match requested artifact_dir '{}'",
                report_artifact_dir,
                artifact_dir.display()
            ),
        });
    }

    if let Some(expected_repo_id) = &config.repo_id {
        let actual = value.get("repo_id").and_then(Value::as_str);
        if actual != Some(expected_repo_id.as_str()) {
            return Err(KimiAttnResPreflightBlocker {
                kind: KimiAttnResPreflightBlockerKind::InvalidBaselineSmokeReport,
                detail: format!(
                    "baseline smoke report repo_id {:?} does not match config repo_id '{}'",
                    actual, expected_repo_id
                ),
            });
        }
    }

    if let Some(expected_revision) = &config.revision {
        let actual = value.get("revision").and_then(Value::as_str);
        if actual != Some(expected_revision.as_str()) {
            return Err(KimiAttnResPreflightBlocker {
                kind: KimiAttnResPreflightBlockerKind::InvalidBaselineSmokeReport,
                detail: format!(
                    "baseline smoke report revision {:?} does not match config revision '{}'",
                    actual, expected_revision
                ),
            });
        }
    }

    let artifact_fingerprints =
        value
            .get("artifact_fingerprints")
            .cloned()
            .ok_or_else(|| KimiAttnResPreflightBlocker {
                kind: KimiAttnResPreflightBlockerKind::InvalidBaselineSmokeReport,
                detail: format!(
                    "baseline smoke report '{}' is missing artifact_fingerprints",
                    path_ref.display()
                ),
            })?;

    Ok(Some(BaselineSmokeFingerprintSource {
        artifact_fingerprints,
    }))
}

fn local_artifact_fingerprints(artifact_dir: &Path, present_shards: &[String]) -> Option<Value> {
    let config = fingerprint_file(&artifact_dir.join("config.json")).ok()?;
    let index = fingerprint_file(&artifact_dir.join("model.safetensors.index.json")).ok()?;
    let config = serde_json::to_value(config).ok()?;
    let index = serde_json::to_value(index).ok()?;
    Some(serde_json::json!({
        "config": config,
        "index": index,
        "present_shards": present_shards,
    }))
}

fn fingerprint_file(path: &Path) -> Result<KimiFileFingerprint, KimiAttnResRealTrainEvalError> {
    let metadata =
        fs::metadata(path).map_err(|err| KimiAttnResRealTrainEvalError::ConfigReadFailed {
            path: path.display().to_string(),
            detail: err.to_string(),
        })?;
    let sha256 =
        sha256_file(path).map_err(|detail| KimiAttnResRealTrainEvalError::ConfigReadFailed {
            path: path.display().to_string(),
            detail,
        })?;
    Ok(KimiFileFingerprint {
        path: path.display().to_string(),
        size_bytes: metadata.len(),
        sha256,
    })
}

fn sha256_file(path: &Path) -> Result<String, String> {
    for (program, args) in [("shasum", &["-a", "256"][..]), ("sha256sum", &[][..])] {
        let output = Command::new(program).args(args).arg(path).output();
        let Ok(output) = output else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        let stdout = String::from_utf8(output.stdout)
            .map_err(|err| format!("invalid UTF-8 from {program}: {err}"))?;
        let digest = stdout
            .split_whitespace()
            .next()
            .ok_or_else(|| format!("{program} produced empty output"))?;
        return Ok(digest.to_string());
    }
    Err("no supported SHA-256 program found (tried shasum and sha256sum)".to_string())
}

fn detect_total_ram_bytes() -> Option<u64> {
    if cfg!(target_os = "macos") {
        return detect_sysctl_value("hw.memsize").and_then(|value| value.parse().ok());
    }
    None
}

fn detect_sysctl_value(key: &str) -> Option<String> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let output = Command::new("sysctl").args(["-n", key]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn checkpoint_file_path(base_path: &str) -> String {
    format!("{base_path}.mpk")
}
