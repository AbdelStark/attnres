/// Root Mean Square Layer Normalization.
///
/// Used in AttnRes to normalize keys before computing attention logits.
/// This prevents layers with large-magnitude outputs from dominating.
///
/// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
///
/// Paper reference: Section 3, applied to block representations before attention.
use burn::config::Config;
use burn::module::{Module, Param};
use burn::prelude::*;

#[derive(Config, Debug)]
pub struct RmsNormConfig {
    /// Hidden dimension.
    pub d_model: usize,
    /// Epsilon for numerical stability.
    #[config(default = 1e-6)]
    pub eps: f64,
}

impl RmsNormConfig {
    /// Initialize RmsNorm with gamma = ones.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RmsNorm<B> {
        RmsNorm {
            gamma: Param::from_tensor(Tensor::ones([self.d_model], device)),
            eps: self.eps,
        }
    }
}

/// RMSNorm module for 3D tensors [B, T, D].
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    /// Learnable scale parameter.
    gamma: Param<Tensor<B, 1>>,
    /// Epsilon for numerical stability.
    eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    pub(crate) fn gamma_param_mut(&mut self) -> &mut Param<Tensor<B, 1>> {
        &mut self.gamma
    }

    /// Forward pass for 3D input [B, T, D].
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // variance: mean(x^2) over last dim -> [B, T, 1]
        let variance = x.clone().powf_scalar(2.0).mean_dim(2); // [B, T, 1]
        let rms = variance.add_scalar(self.eps).sqrt(); // [B, T, 1]
        let normed = x / rms; // [B, T, D]
                              // gamma: [D] -> [1, 1, D] for broadcasting
        normed * self.gamma.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0)
    }

    /// Forward pass for 4D input [N, B, T, D] (stacked blocks).
    pub fn forward_4d(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // variance: mean(x^2) over last dim -> [N, B, T, 1]
        let variance = x.clone().powf_scalar(2.0).mean_dim(3); // [N, B, T, 1]
        let rms = variance.add_scalar(self.eps).sqrt(); // [N, B, T, 1]
        let normed = x / rms; // [N, B, T, D]
                              // gamma: [D] -> [1, 1, 1, D] for broadcasting
        normed
            * self
                .gamma
                .val()
                .unsqueeze_dim::<2>(0)
                .unsqueeze_dim::<3>(0)
                .unsqueeze_dim::<4>(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type TestBackend = NdArray;

    #[test]
    fn test_rmsnorm_output_shape_3d() {
        let device = Default::default();
        let norm = RmsNormConfig::new(64).init::<TestBackend>(&device);
        let x = Tensor::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device);
        let out = norm.forward(x);
        assert_eq!(out.dims(), [2, 16, 64]);
    }

    #[test]
    fn test_rmsnorm_output_shape_4d() {
        let device = Default::default();
        let norm = RmsNormConfig::new(64).init::<TestBackend>(&device);
        let x = Tensor::random([3, 2, 16, 64], Distribution::Normal(0.0, 1.0), &device);
        let out = norm.forward_4d(x);
        assert_eq!(out.dims(), [3, 2, 16, 64]);
    }

    #[test]
    fn test_rmsnorm_normalizes_magnitude() {
        let device = Default::default();
        let norm = RmsNormConfig::new(4).init::<TestBackend>(&device);

        // Large magnitude input
        let x = Tensor::<TestBackend, 3>::from_floats([[[10.0, 20.0, 30.0, 40.0]]], &device);
        let out = norm.forward(x);

        // RMS of [10,20,30,40] = sqrt(mean([100,400,900,1600])) = sqrt(750) ≈ 27.39
        // Normalized values should be much smaller than the input
        let max_val: f32 = out.abs().max().into_scalar();
        assert!(
            max_val < 2.0,
            "RMSNorm should reduce magnitude, got {max_val}"
        );
    }
}
