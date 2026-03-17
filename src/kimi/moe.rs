use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::IndexingUpdateOp;

use crate::kimi::config::KimiSparseMoeRuntimeConfig;
use crate::kimi::mlp::KimiMlpExpert;

/// Sparse MoE scaffold for RFC 0002.
///
/// This is a correctness-first routed mixture that preserves the typed MoE
/// placement and top-k router behavior expected by the local Kimi baseline. It
/// is intentionally not presented as a parity claim for imported public Kimi
/// checkpoints.
#[derive(Debug)]
pub struct KimiSparseMoe<B: Backend> {
    router: Linear<B>,
    experts: Vec<KimiMlpExpert<B>>,
    shared_experts: Vec<KimiMlpExpert<B>>,
    hidden_size: usize,
    num_experts: usize,
    num_experts_per_token: usize,
}

impl KimiSparseMoeRuntimeConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> KimiSparseMoe<B> {
        debug_assert_eq!(self.hidden_act, "silu");

        KimiSparseMoe {
            router: LinearConfig::new(self.hidden_size, self.num_experts).init(device),
            experts: (0..self.num_experts)
                .map(|_| KimiMlpExpert::new(self.hidden_size, self.intermediate_size, device))
                .collect(),
            shared_experts: (0..self.num_shared_experts)
                .map(|_| KimiMlpExpert::new(self.hidden_size, self.intermediate_size, device))
                .collect(),
            hidden_size: self.hidden_size,
            num_experts: self.num_experts,
            num_experts_per_token: self.num_experts_per_token,
        }
    }
}

impl<B: Backend> KimiSparseMoe<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden_size] = x.dims();
        let device = x.device();

        let gate_logits = self.router.forward(x.clone());
        let (_topk_values, topk_indices) = gate_logits
            .clone()
            .topk_with_indices(self.num_experts_per_token, 2);
        let routing_mask = Tensor::<B, 3>::zeros([batch, seq_len, self.num_experts], &device)
            .scatter(
                2,
                topk_indices,
                Tensor::ones([batch, seq_len, self.num_experts_per_token], &device),
                IndexingUpdateOp::Add,
            );
        let sparse_logits = gate_logits.mask_fill(routing_mask.equal_elem(0.0), -1e9);
        let routing_weights = softmax(sparse_logits, 2).unsqueeze_dim::<4>(3);

        let routed_outputs = Tensor::cat(
            self.experts
                .iter()
                .map(|expert| expert.forward(x.clone()).unsqueeze_dim::<4>(2))
                .collect(),
            2,
        );
        let mut output = (routed_outputs * routing_weights).sum_dim(2).reshape([
            batch,
            seq_len,
            self.hidden_size,
        ]);

        if !self.shared_experts.is_empty() {
            let shared_outputs = Tensor::cat(
                self.shared_experts
                    .iter()
                    .map(|expert| expert.forward(x.clone()).unsqueeze_dim::<4>(2))
                    .collect(),
                2,
            )
            .sum_dim(2)
            .reshape([batch, seq_len, self.hidden_size])
                / self.shared_experts.len() as f64;
            output = output + shared_outputs;
        }

        output
    }
}
