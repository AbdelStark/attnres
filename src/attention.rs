/// Multi-head self-attention module.
///
/// Standard scaled dot-product attention with multiple heads.
/// Used within AttnResLayer for the self-attention sublayer.
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

use crate::config::AttnResConfig;

#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    /// Hidden dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dropout rate.
    #[config(default = 0.0)]
    pub dropout: f64,
}

impl MultiHeadAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        assert!(
            self.d_model.is_multiple_of(self.num_heads),
            "d_model ({}) must be divisible by num_heads ({})",
            self.d_model,
            self.num_heads,
        );
        MultiHeadAttention {
            q_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            k_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            v_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            o_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            num_heads: self.num_heads,
            d_head: self.d_model / self.num_heads,
        }
    }
}

impl AttnResConfig {
    /// Create an attention config from the model config.
    pub fn attention_config(&self) -> MultiHeadAttentionConfig {
        MultiHeadAttentionConfig::new(self.d_model, self.num_heads).with_dropout(self.dropout)
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    dropout: Dropout,
    num_heads: usize,
    d_head: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input tensor [B, T, D]
    /// * `mask` - Optional causal mask [B, T, T] (additive, -inf for masked positions)
    ///
    /// # Returns
    /// * Output tensor [B, T, D]
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<&Tensor<B, 3>>) -> Tensor<B, 3> {
        let [batch, seq_len, _d_model] = x.dims();

        // Project Q, K, V: [B, T, D]
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Reshape to [B, T, H, Dh] then transpose to [B, H, T, Dh]
        let q = q
            .reshape([batch, seq_len, self.num_heads, self.d_head])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.num_heads, self.d_head])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.num_heads, self.d_head])
            .swap_dims(1, 2);

        // Scaled dot-product attention
        // scores: [B, H, T, T]
        let scale = (self.d_head as f64).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)) / scale;

        // Apply mask if provided
        let scores = match mask {
            Some(m) => scores + m.clone().unsqueeze_dim::<4>(1),
            None => scores,
        };

        let attn_weights = softmax(scores, 3); // [B, H, T, T]
        let attn_weights = self.dropout.forward(attn_weights);

        // Weighted sum: [B, H, T, Dh]
        let attn_output = attn_weights.matmul(v);

        // Reshape back: [B, H, T, Dh] -> [B, T, H, Dh] -> [B, T, D]
        let attn_output =
            attn_output
                .swap_dims(1, 2)
                .reshape([batch, seq_len, self.num_heads * self.d_head]);

        self.o_proj.forward(attn_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type TestBackend = NdArray;

    #[test]
    fn test_attention_output_shape() {
        let device = Default::default();
        let attn = MultiHeadAttentionConfig::new(64, 8).init::<TestBackend>(&device);
        let x = Tensor::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device);
        let out = attn.forward(x, None);
        assert_eq!(out.dims(), [2, 16, 64]);
    }

    #[test]
    fn test_attention_with_mask() {
        let device = Default::default();
        let attn = MultiHeadAttentionConfig::new(32, 4).init::<TestBackend>(&device);
        let x = Tensor::random([1, 8, 32], Distribution::Normal(0.0, 1.0), &device);

        // Create causal mask: upper triangle = -inf
        let mask = Tensor::<TestBackend, 2>::ones([8, 8], &device)
            .triu(1)
            .mul_scalar(-1e9)
            .unsqueeze_dim::<3>(0); // [1, 8, 8]

        let out = attn.forward(x, Some(&mask));
        assert_eq!(out.dims(), [1, 8, 32]);
    }
}
