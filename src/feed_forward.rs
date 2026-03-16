/// Feed-forward (MLP) module.
///
/// Standard two-layer MLP with GELU activation used in Transformer layers.
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig};
use burn::prelude::*;

use crate::config::AttnResConfig;

#[derive(Config, Debug)]
pub struct FeedForwardConfig {
    /// Input/output dimension.
    pub d_model: usize,
    /// Intermediate (hidden) dimension.
    pub d_ff: usize,
    /// Dropout rate.
    #[config(default = 0.0)]
    pub dropout: f64,
}

impl FeedForwardConfig {
    /// Initialize the feed-forward MLP with two linear layers and GELU activation.
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        FeedForward {
            linear1: LinearConfig::new(self.d_model, self.d_ff).init(device),
            linear2: LinearConfig::new(self.d_ff, self.d_model).init(device),
            gelu: Gelu::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl AttnResConfig {
    /// Create a feed-forward config from the model config.
    pub fn feed_forward_config(&self) -> FeedForwardConfig {
        FeedForwardConfig::new(self.d_model, self.effective_d_ff()).with_dropout(self.dropout)
    }
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    gelu: Gelu,
    dropout: Dropout,
}

impl<B: Backend> FeedForward<B> {
    /// Forward pass: x -> Linear -> GELU -> Dropout -> Linear -> Dropout.
    ///
    /// # Arguments
    /// * `x` - Input tensor [B, T, D]
    ///
    /// # Returns
    /// * Output tensor [B, T, D]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(x);
        let x = self.gelu.forward(x);
        let x = self.dropout.forward(x);
        let x = self.linear2.forward(x);
        self.dropout.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type TestBackend = NdArray;

    #[test]
    fn test_ff_output_shape() {
        let device = Default::default();
        let ff = FeedForwardConfig::new(64, 256).init::<TestBackend>(&device);
        let x = Tensor::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device);
        let out = ff.forward(x);
        assert_eq!(out.dims(), [2, 16, 64]);
    }
}
