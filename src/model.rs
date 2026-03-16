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
use crate::config::{AttnResConfig, ConfigError};
use crate::layer::AttnResLayer;
use crate::rms_norm::{RmsNorm, RmsNormConfig};
use crate::two_phase::{
    compute_intra_logit, normalize_inter_output, online_softmax_merge, phase1_batched,
};

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
    /// Initialize the full AttnRes Transformer model, returning a typed error
    /// for invalid user-supplied configuration.
    pub fn try_init_model<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<AttnResTransformer<B>, ConfigError> {
        self.try_validate()?;
        Ok(self.init_model(device))
    }

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
    /// Access the transformer layers for diagnostics and visualization.
    pub fn layers(&self) -> &[AttnResLayer<B>] {
        &self.layers
    }

    /// Embed input token IDs into hidden representations.
    pub fn embed_tokens(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embedding.forward(input_ids)
    }

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
        let mut completed_blocks = vec![embeddings];
        let mut current_block: Option<Tensor<B, 3>> = None;

        let block_size = self.layers[0].block_size();
        let total_sublayers = self.layers.len() * 2;
        let mut block_start = 0;

        while block_start < total_sublayers {
            if let Some(previous_block) = current_block.take() {
                completed_blocks.push(previous_block);
            }

            let block_end = (block_start + block_size).min(total_sublayers);
            let mut ops = Vec::with_capacity(block_end - block_start);

            for sublayer_idx in block_start..block_end {
                let layer = &self.layers[sublayer_idx / 2];
                let (attn_op, mlp_op) = layer.attn_res_ops();
                ops.push(if sublayer_idx % 2 == 0 {
                    attn_op
                } else {
                    mlp_op
                });
            }

            let phase1 = phase1_batched(&ops, &completed_blocks);
            let mut partial: Option<Tensor<B, 3>> = None;

            for (offset, sublayer_idx) in (block_start..block_end).enumerate() {
                let layer = &self.layers[sublayer_idx / 2];
                let op = ops[offset];

                let h = if offset == 0 {
                    normalize_inter_output(
                        phase1.outputs[offset].clone(),
                        phase1.sum_exp[offset].clone(),
                    )
                } else {
                    let partial_ref = partial
                        .as_ref()
                        .expect("missing intra-block partial during two-phase forward");
                    let intra_logit = compute_intra_logit(op, partial_ref);
                    online_softmax_merge(
                        phase1.outputs[offset].clone(),
                        phase1.max_logits[offset].clone(),
                        phase1.sum_exp[offset].clone(),
                        intra_logit,
                        partial_ref.clone(),
                    )
                };

                let sublayer_out = if sublayer_idx % 2 == 0 {
                    layer.forward_attn_sublayer(h, mask)
                } else {
                    layer.forward_mlp_sublayer(h)
                };

                partial = Some(match partial {
                    Some(current_partial) => current_partial + sublayer_out,
                    None => sublayer_out,
                });
            }

            current_block = partial;
            block_start = block_end;
        }

        let output = current_block.expect(
            "missing final block after two-phase forward; this is a bug in AttnResTransformer",
        );

        let normed = self.final_norm.forward(output);
        self.lm_head.forward(normed)
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

    #[test]
    fn test_try_init_model_returns_typed_error() {
        let device = Default::default();
        let config = AttnResConfig::new(32, 4, 2).with_num_heads(0);
        assert!(matches!(
            config.try_init_model::<TestBackend>(&device),
            Err(ConfigError::NumHeadsMustBePositive)
        ));
    }
}
