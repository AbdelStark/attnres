/// A single Attention Residual operation.
///
/// Computes softmax attention over block representations using a learned
/// pseudo-query vector w_l. This is the core building block.
///
/// Paper reference: Equation 2-4, Figure 2 (block_attn_res function).
///
/// Algorithm:
/// ```text
/// V = stack(blocks + [partial_block])   // [N+1, B, T, D]
/// K = RMSNorm(V)                        // [N+1, B, T, D]
/// logits = einsum('d, n b t d -> n b t', w, K)  // [N+1, B, T]
/// alpha = softmax(logits, dim=0)        // [N+1, B, T]  — over DEPTH
/// h = einsum('n b t, n b t d -> b t d', alpha, V) // [B, T, D]
/// ```
use burn::module::{Module, Param};
use burn::prelude::*;
use burn::tensor::activation::softmax;

use crate::config::AttnResConfig;
use crate::rms_norm::{RmsNorm, RmsNormConfig};

#[derive(Module, Debug)]
pub struct AttnResOp<B: Backend> {
    /// Learned pseudo-query vector w_l ∈ R^d.
    /// CRITICAL: Must be initialized to zero for training stability.
    pub pseudo_query: Param<Tensor<B, 1>>,
    /// RMSNorm applied to values before computing attention logits.
    pub norm: RmsNorm<B>,
}

impl AttnResConfig {
    /// Initialize a single AttnResOp with zero pseudo-query.
    ///
    /// The pseudo-query is zero-initialized per the paper's requirement for
    /// training stability. This means the operation starts as uniform averaging
    /// over all available sources.
    pub fn init_op<B: Backend>(&self, device: &B::Device) -> AttnResOp<B> {
        AttnResOp {
            // CRITICAL: zero initialization per paper requirement
            pseudo_query: Param::from_tensor(Tensor::zeros([self.d_model], device)),
            norm: RmsNormConfig::new(self.d_model)
                .with_eps(self.rms_norm_eps)
                .init(device),
        }
    }
}

impl<B: Backend> AttnResOp<B> {
    /// Compute attention residual over any available block representations.
    ///
    /// `partial_block` is optional because the first sublayer of the network
    /// and the first sublayer of each new block attend only over completed
    /// blocks (Eq. 6 in the paper) and therefore have no intra-block partial.
    pub fn forward_optional_partial(
        &self,
        blocks: &[Tensor<B, 3>],
        partial_block: Option<&Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let mut sources: Vec<Tensor<B, 3>> = blocks.to_vec();
        if let Some(partial_block) = partial_block {
            sources.push(partial_block.clone());
        }

        assert!(
            !sources.is_empty(),
            "AttnResOp requires at least one source tensor"
        );

        // Step 1: Stack all sources into value matrix
        // V: [N, B, T, D] or [N+1, B, T, D]
        let v = Tensor::stack(sources, 0);

        // Step 2: Apply RMSNorm to get keys
        // K: same shape as V
        let k = self.norm.forward_4d(v.clone());

        // Step 3: Compute attention logits
        let w = self
            .pseudo_query
            .val()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0); // [1, 1, 1, D]
        let logits = (k * w).sum_dim(3).squeeze_dim::<3>(3);

        // Step 4: Softmax over the depth dimension (dim=0)
        let alpha = softmax(logits, 0);

        // Step 5: Weighted sum of values
        let alpha_expanded = alpha.unsqueeze_dim::<4>(3);
        let weighted = v * alpha_expanded;
        weighted.sum_dim(0).squeeze_dim::<3>(0)
    }

    /// Compute attention residual over block representations.
    ///
    /// # Arguments
    /// * `blocks` - Completed block representations [N tensors of shape [B, T, D]]
    /// * `partial_block` - Current intra-block partial sum [B, T, D]
    ///
    /// # Returns
    /// * Attention-weighted combination of all sources [B, T, D]
    pub fn forward(&self, blocks: &[Tensor<B, 3>], partial_block: &Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_optional_partial(blocks, Some(partial_block))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type TestBackend = NdArray;

    #[test]
    fn test_output_shape() {
        let device = Default::default();
        let config = AttnResConfig::new(64, 12, 4);
        let op = config.init_op::<TestBackend>(&device);

        let blocks = vec![
            Tensor::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device),
            Tensor::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device),
            Tensor::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device),
        ];
        let partial = Tensor::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device);

        let output = op.forward(&blocks, &partial);
        assert_eq!(output.dims(), [2, 16, 64]);
    }

    #[test]
    fn test_zero_init_uniform_weights() {
        let device = Default::default();
        let config = AttnResConfig::new(64, 12, 4);
        let op = config.init_op::<TestBackend>(&device);

        let blocks = vec![
            Tensor::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device),
            Tensor::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device),
        ];
        let partial = Tensor::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device);

        let output = op.forward(&blocks, &partial);

        // With zero pseudo-query, softmax([0,0,0]) = [1/3, 1/3, 1/3]
        // Output should be mean of all sources
        let expected = (blocks[0].clone() + blocks[1].clone() + partial) / 3.0;

        let diff: f32 = (output - expected).abs().max().into_scalar();
        assert!(
            diff < 1e-4,
            "Zero-init should produce uniform weights (mean of sources), diff={diff}"
        );
    }

    #[test]
    fn test_single_block_is_mean() {
        let device = Default::default();
        let config = AttnResConfig::new(32, 4, 4);
        let op = config.init_op::<TestBackend>(&device);

        let blocks = vec![Tensor::random(
            [1, 8, 32],
            Distribution::Normal(0.0, 1.0),
            &device,
        )];
        let partial = Tensor::random([1, 8, 32], Distribution::Normal(0.0, 1.0), &device);

        let output = op.forward(&blocks, &partial);
        // With zero query and 2 sources, output = mean of 2 sources
        let expected = (blocks[0].clone() + partial) / 2.0;
        let diff: f32 = (output - expected).abs().max().into_scalar();
        assert!(diff < 1e-4, "Single block should produce mean, diff={diff}");
    }

    #[test]
    fn test_blocks_only_returns_only_source() {
        let device = Default::default();
        let config = AttnResConfig::new(32, 4, 2);
        let op = config.init_op::<TestBackend>(&device);

        let embedding = Tensor::random([1, 8, 32], Distribution::Normal(0.0, 1.0), &device);
        let output = op.forward_optional_partial(&[embedding.clone()], None);

        let diff: f32 = (output - embedding).abs().max().into_scalar();
        assert!(
            diff < 1e-5,
            "A single completed block should be returned unchanged, diff={diff}"
        );
    }
}
