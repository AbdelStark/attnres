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
