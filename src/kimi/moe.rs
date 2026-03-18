use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::IndexingUpdateOp;

use crate::kimi::config::KimiSparseMoeRuntimeConfig;
use crate::kimi::mlp::KimiMlpExpert;
use crate::kimi::payload::{load_param_tensor, KimiBaselinePayloadError, KimiDecodedTensor};

/// Sparse MoE scaffold for RFC 0002.
///
/// This now follows the public Kimi gate surface closely enough for
/// checkpoint-backed parity work: bias-free router projection, the auxiliary
/// expert-score correction bias, grouped top-k selection, and routed-scaling
/// semantics.
#[derive(Module, Debug)]
pub struct KimiSparseMoe<B: Backend> {
    router: Linear<B>,
    router_score_bias_correction: Param<Tensor<B, 1>>,
    experts: Vec<KimiMlpExpert<B>>,
    shared_expert: Option<KimiMlpExpert<B>>,
    hidden_size: usize,
    num_experts: usize,
    num_experts_per_token: usize,
    moe_renormalize: bool,
    moe_router_activation_func: String,
    routed_scaling_factor: f64,
    use_grouped_topk: bool,
    num_expert_group: usize,
    topk_group: usize,
}

impl KimiSparseMoeRuntimeConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> KimiSparseMoe<B> {
        debug_assert_eq!(self.hidden_act, "silu");

        KimiSparseMoe {
            router: LinearConfig::new(self.hidden_size, self.num_experts)
                .with_bias(false)
                .init(device),
            router_score_bias_correction: Param::from_tensor(Tensor::zeros(
                [self.num_experts],
                device,
            )),
            experts: (0..self.num_experts)
                .map(|_| KimiMlpExpert::new(self.hidden_size, self.intermediate_size, device))
                .collect(),
            shared_expert: (self.num_shared_experts > 0).then(|| {
                KimiMlpExpert::new(
                    self.hidden_size,
                    self.intermediate_size * self.num_shared_experts,
                    device,
                )
            }),
            hidden_size: self.hidden_size,
            num_experts: self.num_experts,
            num_experts_per_token: self.num_experts_per_token,
            moe_renormalize: self.moe_renormalize,
            moe_router_activation_func: self.moe_router_activation_func.clone(),
            routed_scaling_factor: self.routed_scaling_factor,
            use_grouped_topk: self.use_grouped_topk,
            num_expert_group: self.num_expert_group,
            topk_group: self.topk_group,
        }
    }
}

impl<B: Backend> KimiSparseMoe<B> {
    pub(crate) fn try_apply_tensor_payload(
        &mut self,
        tensor_name: &str,
        remainder: &str,
        payload: &KimiDecodedTensor,
    ) -> Result<(), KimiBaselinePayloadError> {
        match remainder {
            "gate.weight" => load_param_tensor(&mut self.router.weight, tensor_name, payload),
            "gate.e_score_correction_bias" => load_param_tensor(
                &mut self.router_score_bias_correction,
                tensor_name,
                payload,
            ),
            "shared_experts.gate_proj.weight" => {
                let Some(expert) = self.shared_expert.as_mut() else {
                    return Err(KimiBaselinePayloadError::UnsupportedTensorApplication {
                        tensor_name: tensor_name.to_string(),
                        detail: "shared expert payload loading expects at least one shared expert"
                            .to_string(),
                    });
                };
                expert.try_apply_tensor_payload(tensor_name, "gate_proj.weight", payload)
            }
            "shared_experts.up_proj.weight" => {
                let Some(expert) = self.shared_expert.as_mut() else {
                    return Err(KimiBaselinePayloadError::UnsupportedTensorApplication {
                        tensor_name: tensor_name.to_string(),
                        detail: "shared expert payload loading expects at least one shared expert"
                            .to_string(),
                    });
                };
                expert.try_apply_tensor_payload(tensor_name, "up_proj.weight", payload)
            }
            "shared_experts.down_proj.weight" => {
                let Some(expert) = self.shared_expert.as_mut() else {
                    return Err(KimiBaselinePayloadError::UnsupportedTensorApplication {
                        tensor_name: tensor_name.to_string(),
                        detail: "shared expert payload loading expects at least one shared expert"
                            .to_string(),
                    });
                };
                expert.try_apply_tensor_payload(tensor_name, "down_proj.weight", payload)
            }
            _ => {
                let Some((expert_idx, leaf)) = parse_sparse_expert_leaf(remainder) else {
                    return Err(KimiBaselinePayloadError::UnsupportedTensorApplication {
                        tensor_name: tensor_name.to_string(),
                        detail: format!("unsupported sparse MoE tensor leaf '{remainder}'"),
                    });
                };
                let Some(expert) = self.experts.get_mut(expert_idx) else {
                    return Err(KimiBaselinePayloadError::UnsupportedTensorApplication {
                        tensor_name: tensor_name.to_string(),
                        detail: format!("expert index {expert_idx} is out of range"),
                    });
                };
                let mapped_leaf = match leaf {
                    "w1.weight" => "gate_proj.weight",
                    "w2.weight" => "down_proj.weight",
                    "w3.weight" => "up_proj.weight",
                    other => {
                        return Err(KimiBaselinePayloadError::UnsupportedTensorApplication {
                            tensor_name: tensor_name.to_string(),
                            detail: format!("unsupported sparse expert tensor leaf '{other}'"),
                        });
                    }
                };
                expert.try_apply_tensor_payload(tensor_name, mapped_leaf, payload)
            }
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden_size] = x.dims();
        let device = x.device();

        let gate_logits = self.router.forward(x.clone());
        let scores = match self.moe_router_activation_func.as_str() {
            "sigmoid" => sigmoid(gate_logits.clone()),
            "softmax" => softmax(gate_logits.clone(), 2),
            other => panic!("unsupported Kimi MoE router activation '{other}'"),
        };

        let scores_for_choice = scores.clone()
            + self
                .router_score_bias_correction
                .val()
                .unsqueeze_dim::<2>(0)
                .unsqueeze_dim::<3>(0);
        let grouped_scores = if self.use_grouped_topk {
            restrict_scores_to_selected_groups(
                scores_for_choice,
                self.num_expert_group,
                self.topk_group,
            )
        } else {
            scores_for_choice
        };

        let (_topk_values, topk_indices) =
            grouped_scores.topk_with_indices(self.num_experts_per_token, 2);
        let routing_mask = Tensor::<B, 3>::zeros([batch, seq_len, self.num_experts], &device)
            .scatter(
                2,
                topk_indices,
                Tensor::ones([batch, seq_len, self.num_experts_per_token], &device),
                IndexingUpdateOp::Add,
            );

        let mut routing_weights = scores * routing_mask;
        if self.num_experts_per_token > 1 && self.moe_renormalize {
            let denominator = routing_weights.clone().sum_dim(2).add_scalar(1e-20);
            routing_weights = routing_weights / denominator;
        }
        routing_weights = routing_weights * self.routed_scaling_factor;

        let routed_outputs = Tensor::cat(
            self.experts
                .iter()
                .map(|expert| expert.forward(x.clone()).unsqueeze_dim::<4>(2))
                .collect(),
            2,
        );
        let mut output = (routed_outputs * routing_weights.unsqueeze_dim::<4>(3))
            .sum_dim(2)
            .reshape([batch, seq_len, self.hidden_size]);

        if let Some(shared_expert) = &self.shared_expert {
            output = output + shared_expert.forward(x);
        }

        output
    }
}

fn parse_sparse_expert_leaf(remainder: &str) -> Option<(usize, &str)> {
    let prefix = "experts.";
    let suffix = remainder.strip_prefix(prefix)?;
    let (expert_idx, leaf) = suffix.split_once('.')?;
    Some((expert_idx.parse().ok()?, leaf))
}

fn restrict_scores_to_selected_groups<B: Backend>(
    scores: Tensor<B, 3>,
    num_expert_group: usize,
    topk_group: usize,
) -> Tensor<B, 3> {
    let [batch, seq_len, num_experts] = scores.dims();
    debug_assert!(num_experts.is_multiple_of(num_expert_group));

    let experts_per_group = num_experts / num_expert_group;
    let grouped = scores
        .clone()
        .reshape([batch, seq_len, num_expert_group, experts_per_group]);
    let experts_per_group_topk = experts_per_group.min(2);
    let (top2_values, _) = grouped.topk_with_indices(experts_per_group_topk, 3);
    let group_scores = top2_values
        .sum_dim(3)
        .reshape([batch, seq_len, num_expert_group]);
    let (_topk_group_values, topk_group_indices) =
        group_scores.topk_with_indices(topk_group, 2);
    let group_mask = Tensor::<B, 3>::zeros([batch, seq_len, num_expert_group], &scores.device())
        .scatter(
            2,
            topk_group_indices,
            Tensor::ones([batch, seq_len, topk_group], &scores.device()),
            IndexingUpdateOp::Add,
        );
    let score_mask = group_mask
        .unsqueeze_dim::<4>(3)
        .repeat_dim(3, experts_per_group)
        .reshape([batch, seq_len, num_experts]);

    scores.mask_fill(score_mask.equal_elem(0.0), 0.0)
}
