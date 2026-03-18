use attnres::kimi::{KimiArtifactConfig, KimiAttnResModel, KimiLinearModel};
use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Param};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;

type TrainBackend = Autodiff<NdArray>;
type InnerBackend = NdArray;
type TrainDevice = <TrainBackend as Backend>::Device;

const TRAIN_BATCH_SIZE: usize = 2;
const TRAIN_SEQ_LEN: usize = 6;
const TRAIN_STEPS: usize = 8;
const TRAIN_LR: f64 = 5e-4;
const LOSS_GROWTH_FACTOR_CAP: f32 = 1.35;
const LOSS_GROWTH_SLACK: f32 = 0.5;
const ABSOLUTE_GRAD_NORM_CAP: f32 = 200.0;
const ABSOLUTE_ACTIVATION_RMS_CAP: f32 = 25.0;
const RELATIVE_ATTN_RES_GRAD_NORM_CAP: f32 = 20.0;
const RELATIVE_ATTN_RES_ACTIVATION_CAP: f32 = 6.0;
const RELATIVE_ATTN_RES_FINAL_LOSS_CAP: f32 = 1.2;
const RELATIVE_ATTN_RES_FINAL_LOSS_SLACK: f32 = 0.35;

#[derive(Debug)]
struct TrainingObservation {
    label: String,
    seed: u64,
    losses: Vec<f32>,
    max_hidden_rms: f32,
    max_logit_rms: f32,
    max_grad_l2_norm: f32,
}

impl TrainingObservation {
    fn initial_loss(&self) -> f32 {
        *self
            .losses
            .first()
            .expect("training observation should contain at least one loss")
    }

    fn final_loss(&self) -> f32 {
        *self
            .losses
            .last()
            .expect("training observation should contain at least one loss")
    }

    fn max_loss(&self) -> f32 {
        self.losses
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    }
}

#[derive(Debug)]
struct GradientNormVisitor<'a> {
    grads: &'a GradientsParams,
    squared_l2_sum: f64,
    all_finite: bool,
}

impl<'a> GradientNormVisitor<'a> {
    fn new(grads: &'a GradientsParams) -> Self {
        Self {
            grads,
            squared_l2_sum: 0.0,
            all_finite: true,
        }
    }
}

impl burn::module::ModuleVisitor<TrainBackend> for GradientNormVisitor<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<TrainBackend, D>>) {
        let Some(grad) = self.grads.get::<InnerBackend, D>(param.id) else {
            return;
        };
        let finite: bool = grad.clone().is_finite().all().into_scalar();
        self.all_finite &= finite;
        let sum_sq: f32 = grad.powi_scalar(2).sum().into_scalar();
        self.squared_l2_sum += f64::from(sum_sq);
    }
}

fn reduced_training_config() -> KimiArtifactConfig {
    KimiArtifactConfig::from_json_str(
        r#"{
            "model_type": "kimi_linear",
            "dtype": "float32",
            "vocab_size": 48,
            "hidden_size": 12,
            "intermediate_size": 24,
            "moe_intermediate_size": 20,
            "num_hidden_layers": 3,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "kv_lora_rank": 4,
            "q_lora_rank": null,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 4,
            "mla_use_nope": true,
            "hidden_act": "silu",
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "num_experts": 4,
            "num_experts_per_token": 2,
            "num_shared_experts": 1,
            "moe_renormalize": true,
            "moe_router_activation_func": "sigmoid",
            "routed_scaling_factor": 1.0,
            "use_grouped_topk": true,
            "num_expert_group": 1,
            "topk_group": 1,
            "tie_word_embeddings": false,
            "use_cache": false,
            "rms_norm_eps": 1e-5,
            "linear_attn_config": {
                "full_attn_layers": [2],
                "kda_layers": [1, 3],
                "num_heads": 3,
                "head_dim": 4,
                "short_conv_kernel_size": 3
            }
        }"#,
    )
    .unwrap()
}

fn deterministic_batch(
    step: usize,
    vocab_size: usize,
    device: &TrainDevice,
) -> (Tensor<TrainBackend, 2, Int>, Tensor<TrainBackend, 2, Int>) {
    let mut input_ids = Vec::with_capacity(TRAIN_BATCH_SIZE * TRAIN_SEQ_LEN);
    let mut targets = Vec::with_capacity(TRAIN_BATCH_SIZE * TRAIN_SEQ_LEN);

    for batch_idx in 0..TRAIN_BATCH_SIZE {
        let base = step * 5 + batch_idx * 11;
        let row = (0..TRAIN_SEQ_LEN)
            .map(|position| ((base + position * 3 + position * position) % vocab_size) as i64)
            .collect::<Vec<_>>();
        input_ids.extend(row.iter().copied());
        targets.extend((0..TRAIN_SEQ_LEN).map(|position| row[(position + 1) % TRAIN_SEQ_LEN]));
    }

    let input_ids = Tensor::<TrainBackend, 1, Int>::from_ints(input_ids.as_slice(), device)
        .reshape([TRAIN_BATCH_SIZE, TRAIN_SEQ_LEN]);
    let targets = Tensor::<TrainBackend, 1, Int>::from_ints(targets.as_slice(), device)
        .reshape([TRAIN_BATCH_SIZE, TRAIN_SEQ_LEN]);

    (input_ids, targets)
}

fn tensor_all_finite<const D: usize>(tensor: Tensor<InnerBackend, D>) -> bool {
    tensor.is_finite().all().into_scalar()
}

fn tensor_rms<const D: usize>(tensor: Tensor<InnerBackend, D>) -> f32 {
    tensor.powf_scalar(2.0).mean().sqrt().into_scalar()
}

fn gradient_l2_norm<M>(model: &M, grads: &GradientsParams) -> (f32, bool)
where
    M: AutodiffModule<TrainBackend>,
{
    let mut visitor = GradientNormVisitor::new(grads);
    model.visit(&mut visitor);
    ((visitor.squared_l2_sum as f32).sqrt(), visitor.all_finite)
}

fn run_training_case<M, FHidden, FLogits>(
    label: &str,
    seed: u64,
    vocab_size: usize,
    mut model: M,
    forward_hidden: FHidden,
    forward_logits: FLogits,
    device: &TrainDevice,
) -> TrainingObservation
where
    M: AutodiffModule<TrainBackend>,
    FHidden: Fn(&M, Tensor<TrainBackend, 2, Int>) -> Tensor<TrainBackend, 3>,
    FLogits: Fn(&M, Tensor<TrainBackend, 2, Int>) -> Tensor<TrainBackend, 3>,
{
    let mut optimizer = AdamConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().with_logits(true).init(device);
    let mut observation = TrainingObservation {
        label: label.to_string(),
        seed,
        losses: Vec::with_capacity(TRAIN_STEPS),
        max_hidden_rms: 0.0,
        max_logit_rms: 0.0,
        max_grad_l2_norm: 0.0,
    };

    for step in 0..TRAIN_STEPS {
        let (input_ids, targets) = deterministic_batch(step, vocab_size, device);

        let hidden = forward_hidden(&model, input_ids.clone()).inner();
        let logits = forward_logits(&model, input_ids);
        let logits_inner = logits.clone().inner();

        assert!(
            tensor_all_finite(hidden.clone()),
            "{} seed {} step {} produced non-finite hidden states",
            observation.label,
            observation.seed,
            step,
        );
        assert!(
            tensor_all_finite(logits_inner.clone()),
            "{} seed {} step {} produced non-finite logits",
            observation.label,
            observation.seed,
            step,
        );

        observation.max_hidden_rms = observation.max_hidden_rms.max(tensor_rms(hidden));
        observation.max_logit_rms = observation.max_logit_rms.max(tensor_rms(logits_inner));

        let [batch_size, seq_len, vocab_dim] = logits.dims();
        let loss = loss_fn
            .forward(
                logits.reshape([batch_size * seq_len, vocab_dim]),
                targets.reshape([batch_size * seq_len]),
            )
            .mean();
        let loss_value: f32 = loss.clone().into_scalar();
        assert!(
            loss_value.is_finite(),
            "{} seed {} step {} produced non-finite loss",
            observation.label,
            observation.seed,
            step,
        );
        observation.losses.push(loss_value);

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        assert!(
            !grads.is_empty(),
            "{} seed {} step {} unexpectedly produced an empty gradient set",
            observation.label,
            observation.seed,
            step,
        );
        let (grad_l2_norm, grads_finite) = gradient_l2_norm(&model, &grads);
        assert!(
            grads_finite && grad_l2_norm.is_finite(),
            "{} seed {} step {} produced non-finite gradients",
            observation.label,
            observation.seed,
            step,
        );
        observation.max_grad_l2_norm = observation.max_grad_l2_norm.max(grad_l2_norm);

        model = optimizer.step(TRAIN_LR, model, grads);
    }

    observation
}

fn assert_observation_stable(observation: &TrainingObservation) {
    let loss_growth_cap = observation.initial_loss() * LOSS_GROWTH_FACTOR_CAP + LOSS_GROWTH_SLACK;
    assert!(
        observation.max_loss() <= loss_growth_cap,
        "{} seed {} exceeded the explicit loss-growth cap: initial_loss={} max_loss={} cap={} losses={:?}",
        observation.label,
        observation.seed,
        observation.initial_loss(),
        observation.max_loss(),
        loss_growth_cap,
        observation.losses,
    );
    assert!(
        observation.max_grad_l2_norm <= ABSOLUTE_GRAD_NORM_CAP,
        "{} seed {} exceeded the absolute gradient-norm cap: observed={} cap={}",
        observation.label,
        observation.seed,
        observation.max_grad_l2_norm,
        ABSOLUTE_GRAD_NORM_CAP,
    );
    assert!(
        observation.max_hidden_rms <= ABSOLUTE_ACTIVATION_RMS_CAP,
        "{} seed {} exceeded the hidden-activation RMS cap: observed={} cap={}",
        observation.label,
        observation.seed,
        observation.max_hidden_rms,
        ABSOLUTE_ACTIVATION_RMS_CAP,
    );
    assert!(
        observation.max_logit_rms <= ABSOLUTE_ACTIVATION_RMS_CAP,
        "{} seed {} exceeded the logit RMS cap: observed={} cap={}",
        observation.label,
        observation.seed,
        observation.max_logit_rms,
        ABSOLUTE_ACTIVATION_RMS_CAP,
    );
}

fn assert_attn_res_tracks_baseline(baseline: &TrainingObservation, attn_res: &TrainingObservation) {
    let final_loss_cap = baseline.final_loss() * RELATIVE_ATTN_RES_FINAL_LOSS_CAP
        + RELATIVE_ATTN_RES_FINAL_LOSS_SLACK;
    assert!(
        attn_res.final_loss() <= final_loss_cap,
        "AttnRes final loss drifted too far above baseline for seed {}: baseline_final_loss={} attn_res_final_loss={} cap={}",
        attn_res.seed,
        baseline.final_loss(),
        attn_res.final_loss(),
        final_loss_cap,
    );
    assert!(
        attn_res.max_grad_l2_norm
            <= baseline.max_grad_l2_norm * RELATIVE_ATTN_RES_GRAD_NORM_CAP,
        "AttnRes gradient norm drifted too far above baseline for seed {}: baseline_max_grad={} attn_res_max_grad={} factor_cap={}",
        attn_res.seed,
        baseline.max_grad_l2_norm,
        attn_res.max_grad_l2_norm,
        RELATIVE_ATTN_RES_GRAD_NORM_CAP,
    );
    assert!(
        attn_res.max_hidden_rms
            <= baseline.max_hidden_rms * RELATIVE_ATTN_RES_ACTIVATION_CAP,
        "AttnRes hidden RMS drifted too far above baseline for seed {}: baseline_max_hidden_rms={} attn_res_max_hidden_rms={} factor_cap={}",
        attn_res.seed,
        baseline.max_hidden_rms,
        attn_res.max_hidden_rms,
        RELATIVE_ATTN_RES_ACTIVATION_CAP,
    );
    assert!(
        attn_res.max_logit_rms
            <= baseline.max_logit_rms * RELATIVE_ATTN_RES_ACTIVATION_CAP,
        "AttnRes logit RMS drifted too far above baseline for seed {}: baseline_max_logit_rms={} attn_res_max_logit_rms={} factor_cap={}",
        attn_res.seed,
        baseline.max_logit_rms,
        attn_res.max_logit_rms,
        RELATIVE_ATTN_RES_ACTIVATION_CAP,
    );
}

#[test]
fn kimi_rfc_0005_gate6_reduced_kimi_training_stays_stable_with_attn_res_enabled() {
    let device = Default::default();
    let config = reduced_training_config();
    let vocab_size = config.vocab_size;

    for seed in [20260318_u64, 20260319_u64] {
        TrainBackend::seed(&device, seed);
        let baseline: KimiLinearModel<TrainBackend> = config.try_init_model(&device).unwrap();
        let baseline_observation = run_training_case(
            "baseline_kimi",
            seed,
            vocab_size,
            baseline,
            |model, input_ids| model.forward_hidden(input_ids),
            |model, input_ids| model.forward(input_ids),
            &device,
        );
        assert_observation_stable(&baseline_observation);

        TrainBackend::seed(&device, seed);
        let attn_res: KimiAttnResModel<TrainBackend> =
            config.try_init_attn_res_model(3, &device).unwrap();
        let attn_res_observation = run_training_case(
            "attn_res_kimi",
            seed,
            vocab_size,
            attn_res,
            |model, input_ids| model.forward_hidden(input_ids),
            |model, input_ids| model.forward(input_ids),
            &device,
        );
        assert_observation_stable(&attn_res_observation);
        assert_attn_res_tracks_baseline(&baseline_observation, &attn_res_observation);
    }
}
