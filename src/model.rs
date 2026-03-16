/// A Transformer model with Attention Residuals.
///
/// Combines token embeddings, a stack of AttnRes layers,
/// final RMSNorm, and a language model head.
///
/// Paper reference: Section 3.
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;

use crate::block_state::BlockState;
use crate::config::AttnResConfig;
use crate::layer::AttnResLayer;
use crate::rms_norm::{RmsNorm, RmsNormConfig};
use crate::two_phase::{compute_intra_logit, online_softmax_merge, phase1_batched};

#[derive(Module, Debug)]
pub struct AttnResTransformer<B: Backend> {
    /// Token embedding layer.
    embedding: Embedding<B>,
    /// Stack of transformer layers with AttnRes.
    layers: Vec<AttnResLayer<B>>,
    /// Final RMSNorm.
    final_norm: RmsNorm<B>,
    /// Output projection (language model head).
    lm_head: Linear<B>,
}

impl AttnResConfig {
    /// Initialize the full AttnRes Transformer model.
    ///
    /// # Panics
    /// Panics if the configuration is invalid (see [`AttnResConfig::validate`]).
    pub fn init_model<B: Backend>(&self, device: &B::Device) -> AttnResTransformer<B> {
        self.validate();

        let num_transformer_layers = self.num_transformer_layers();

        let layers = (0..num_transformer_layers)
            .map(|i| self.init_layer(i, device))
            .collect();

        AttnResTransformer {
            embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            layers,
            final_norm: RmsNormConfig::new(self.d_model)
                .with_eps(self.rms_norm_eps)
                .init(device),
            lm_head: LinearConfig::new(self.d_model, self.vocab_size).init(device),
        }
    }
}

impl<B: Backend> AttnResTransformer<B> {
    /// Forward pass through the full model.
    ///
    /// # Arguments
    /// * `input_ids` - Token indices [B, T]
    /// * `mask` - Optional causal attention mask [B, T, T]
    ///
    /// # Returns
    /// * Logits over vocabulary [B, T, vocab_size]
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        mask: Option<&Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        // 1. Token embedding: [B, T] -> [B, T, D]
        let embeddings = self.embedding.forward(input_ids);

        // 2. Initialize block state with embeddings as b_0
        let mut state = BlockState::new(embeddings);

        // 3. Forward through all layers
        for layer in &self.layers {
            state = layer.forward(state, mask);
        }

        // 4. Get final hidden states from partial block
        let output = state
            .partial_block
            .expect("partial_block missing after forward pass; this is a bug in AttnResLayer");

        // 5. Final norm + LM head
        let normed = self.final_norm.forward(output);
        self.lm_head.forward(normed)
    }

    /// Forward pass using two-phase inference optimization.
    ///
    /// Produces identical results to [`forward`](Self::forward) but uses the two-phase
    /// strategy from Algorithm 1 of the paper:
    /// - Phase 1: Batch inter-block attention for all sublayers in each block
    /// - Phase 2: Sequential intra-block attention with online softmax merge
    ///
    /// This is beneficial during inference when blocks are cached, as Phase 1
    /// can be parallelized across sublayers.
    ///
    /// Paper reference: Algorithm 1, Section 4.1.
    pub fn forward_two_phase(
        &self,
        input_ids: Tensor<B, 2, Int>,
        mask: Option<&Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let embeddings = self.embedding.forward(input_ids);
        let mut state = BlockState::new(embeddings);

        // Group layers into blocks based on block boundaries
        let mut block_start = 0;
        while block_start < self.layers.len() {
            // Find the end of this block: layers until next boundary
            let mut block_end = block_start + 1;
            while block_end < self.layers.len() && !self.layers[block_end].is_at_boundary() {
                block_end += 1;
            }

            let block_layers = &self.layers[block_start..block_end];

            if state.blocks.is_empty() {
                // No inter-block context yet; use standard forward
                for layer in block_layers {
                    state = layer.forward(state, mask);
                }
            } else {
                // Two-phase forward for this block of layers
                state = self.forward_block_two_phase(state, block_layers, mask);
            }

            block_start = block_end;
        }

        let output = state
            .partial_block
            .expect("partial_block missing after forward pass; this is a bug in AttnResLayer");

        let normed = self.final_norm.forward(output);
        self.lm_head.forward(normed)
    }

    /// Two-phase forward for a single block of layers.
    ///
    /// Uses Phase 1 (batched inter-block attention) + Phase 2 (sequential intra-block).
    fn forward_block_two_phase(
        &self,
        mut state: BlockState<B>,
        block_layers: &[AttnResLayer<B>],
        mask: Option<&Tensor<B, 3>>,
    ) -> BlockState<B> {
        // Phase 2 setup: handle block boundary first so blocks are correct for Phase 1
        let current_partial = state
            .partial_block
            .take()
            .unwrap_or_else(|| Tensor::zeros_like(state.blocks.last().unwrap()));

        let first_layer = &block_layers[0];
        let at_boundary = first_layer.is_at_boundary();

        if at_boundary {
            state.blocks.push(current_partial.clone());
        }

        let mut partial = if at_boundary {
            Tensor::zeros_like(state.blocks.last().unwrap())
        } else {
            current_partial
        };

        // Collect all AttnResOp references for Phase 1 batching
        // Each layer has 2 ops: attn_res, mlp_res
        let all_ops: Vec<_> = block_layers
            .iter()
            .flat_map(|layer| {
                let (attn_op, mlp_op) = layer.attn_res_ops();
                vec![attn_op, mlp_op]
            })
            .collect();

        // Phase 1: Batch all inter-block attention (now with correct blocks)
        let phase1 = phase1_batched(&all_ops, &state.blocks);

        // Process each sublayer using Phase 1 results + online softmax merge
        for (layer_idx, layer) in block_layers.iter().enumerate() {
            let attn_op_idx = layer_idx * 2;
            let mlp_op_idx = layer_idx * 2 + 1;

            // === AttnRes before self-attention (using two-phase) ===
            let h = if phase1.outputs.is_empty() {
                // No inter-block context: fall back to standard
                let (attn_op, _) = layer.attn_res_ops();
                attn_op.forward(&state.blocks, &partial)
            } else {
                let (attn_op, _) = layer.attn_res_ops();
                let intra_logit = compute_intra_logit(attn_op, &partial);
                online_softmax_merge(
                    phase1.outputs[attn_op_idx].clone(),
                    phase1.max_logits[attn_op_idx].clone(),
                    phase1.sum_exp[attn_op_idx].clone(),
                    intra_logit,
                    partial.clone(),
                )
            };

            // Attention sublayer
            let attn_out = layer.forward_attn_sublayer(h, mask);
            partial = partial + attn_out;

            // === AttnRes before MLP (using two-phase) ===
            let h = if phase1.outputs.is_empty() {
                let (_, mlp_op) = layer.attn_res_ops();
                mlp_op.forward(&state.blocks, &partial)
            } else {
                let (_, mlp_op) = layer.attn_res_ops();
                let intra_logit = compute_intra_logit(mlp_op, &partial);
                online_softmax_merge(
                    phase1.outputs[mlp_op_idx].clone(),
                    phase1.max_logits[mlp_op_idx].clone(),
                    phase1.sum_exp[mlp_op_idx].clone(),
                    intra_logit,
                    partial.clone(),
                )
            };

            // MLP sublayer
            let mlp_out = layer.forward_mlp_sublayer(h);
            partial = partial + mlp_out;
        }

        state.partial_block = Some(partial);
        state
    }

    /// Forward pass returning hidden states (without LM head).
    pub fn forward_hidden(
        &self,
        input_ids: Tensor<B, 2, Int>,
        mask: Option<&Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let embeddings = self.embedding.forward(input_ids);
        let mut state = BlockState::new(embeddings);

        for layer in &self.layers {
            state = layer.forward(state, mask);
        }

        let output = state
            .partial_block
            .expect("partial_block missing after forward pass; this is a bug in AttnResLayer");
        self.final_norm.forward(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_model_forward_shape() {
        let device = Default::default();
        let config = AttnResConfig::new(32, 4, 2)
            .with_num_heads(4)
            .with_vocab_size(100);

        let model = config.init_model::<TestBackend>(&device);

        // Create input token ids [batch=1, seq_len=8]
        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([1, 8], &device);
        let output = model.forward(input_ids, None);

        assert_eq!(output.dims(), [1, 8, 100]);
    }

    #[test]
    fn test_two_phase_matches_standard() {
        let device = Default::default();
        let config = AttnResConfig::new(32, 8, 2)
            .with_num_heads(4)
            .with_vocab_size(100);

        let model = config.init_model::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([1, 8], &device);
        let standard_out = model.forward(input_ids.clone(), None);
        let two_phase_out = model.forward_two_phase(input_ids, None);

        let diff: f32 = (standard_out - two_phase_out).abs().max().into_scalar();
        assert!(
            diff < 1e-3,
            "Two-phase forward should match standard forward, diff={diff}"
        );
    }

    #[test]
    fn test_model_forward_hidden_shape() {
        let device = Default::default();
        let config = AttnResConfig::new(32, 4, 2)
            .with_num_heads(4)
            .with_vocab_size(100);

        let model = config.init_model::<TestBackend>(&device);

        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([1, 8], &device);
        let hidden = model.forward_hidden(input_ids, None);

        assert_eq!(hidden.dims(), [1, 8, 32]);
    }
}
