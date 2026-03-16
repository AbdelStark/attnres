/// A Transformer layer augmented with Block Attention Residuals.
///
/// Each transformer layer has TWO AttnRes operations:
/// - attn_res: applied before the self-attention sublayer
/// - mlp_res: applied before the MLP sublayer
///
/// Paper reference: Figure 2 (forward function).
use burn::module::Module;
use burn::prelude::*;

use crate::attention::MultiHeadAttention;
use crate::attn_res_op::AttnResOp;
use crate::block_state::BlockState;
use crate::config::AttnResConfig;
use crate::feed_forward::FeedForward;
use crate::rms_norm::RmsNorm;

#[derive(Module, Debug)]
pub struct AttnResLayer<B: Backend> {
    /// Layer index (0-based).
    layer_idx: usize,
    /// Block size (number of sublayers per block).
    block_size: usize,

    /// AttnRes before self-attention.
    attn_res: AttnResOp<B>,
    /// AttnRes before MLP.
    mlp_res: AttnResOp<B>,

    /// Pre-attention RMSNorm.
    attn_norm: RmsNorm<B>,
    /// Self-attention module.
    attn: MultiHeadAttention<B>,
    /// Pre-MLP RMSNorm.
    mlp_norm: RmsNorm<B>,
    /// Feed-forward MLP.
    mlp: FeedForward<B>,
}

impl AttnResConfig {
    /// Initialize a single AttnResLayer.
    ///
    /// # Arguments
    /// * `layer_idx` - Zero-based index of this transformer layer. Used to
    ///   determine block boundaries (boundary occurs when `layer_idx > 0` and
    ///   `layer_idx % (block_size / 2) == 0`).
    /// * `device` - Device to allocate tensors on.
    pub fn init_layer<B: Backend>(&self, layer_idx: usize, device: &B::Device) -> AttnResLayer<B> {
        AttnResLayer {
            layer_idx,
            block_size: self.block_size(),
            attn_res: self.init_op(device),
            mlp_res: self.init_op(device),
            attn_norm: crate::rms_norm::RmsNormConfig::new(self.d_model)
                .with_eps(self.rms_norm_eps)
                .init(device),
            attn: self.attention_config().init(device),
            mlp_norm: crate::rms_norm::RmsNormConfig::new(self.d_model)
                .with_eps(self.rms_norm_eps)
                .init(device),
            mlp: self.feed_forward_config().init(device),
        }
    }
}

impl<B: Backend> AttnResLayer<B> {
    fn attn_sublayer_idx(&self) -> usize {
        self.layer_idx * 2
    }

    fn mlp_sublayer_idx(&self) -> usize {
        self.attn_sublayer_idx() + 1
    }

    fn starts_new_block_before_sublayer(&self, sublayer_idx: usize) -> bool {
        sublayer_idx > 0 && sublayer_idx.is_multiple_of(self.block_size)
    }

    pub(crate) fn starts_new_block_before_attn(&self) -> bool {
        self.starts_new_block_before_sublayer(self.attn_sublayer_idx())
    }

    pub(crate) fn starts_new_block_before_mlp(&self) -> bool {
        self.starts_new_block_before_sublayer(self.mlp_sublayer_idx())
    }

    /// Get the layer index.
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Get the block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Check if this layer's attention sublayer starts a new block.
    ///
    /// Block sizing is defined in sublayers, so the MLP sublayer can also
    /// start a new block when `block_size` is odd or when `block_size == 1`
    /// (Full AttnRes). This helper preserves the historical public API by
    /// reporting only the pre-attention boundary.
    pub fn is_at_boundary(&self) -> bool {
        self.starts_new_block_before_attn()
    }

    /// Get references to the AttnRes operations (attn_res, mlp_res).
    pub fn attn_res_ops(&self) -> (&AttnResOp<B>, &AttnResOp<B>) {
        (&self.attn_res, &self.mlp_res)
    }

    /// Execute only the attention sublayer (norm + multi-head attention).
    ///
    /// Used by two-phase inference after AttnRes has been computed externally.
    pub fn forward_attn_sublayer(
        &self,
        h: Tensor<B, 3>,
        mask: Option<&Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let normed = self.attn_norm.forward(h);
        self.attn.forward(normed, mask)
    }

    /// Execute only the MLP sublayer (norm + feed-forward).
    ///
    /// Used by two-phase inference after AttnRes has been computed externally.
    pub fn forward_mlp_sublayer(&self, h: Tensor<B, 3>) -> Tensor<B, 3> {
        let normed = self.mlp_norm.forward(h);
        self.mlp.forward(normed)
    }

    /// Forward pass for a single transformer layer with Block AttnRes.
    ///
    /// Maps directly to the `forward` function in Figure 2 of the paper.
    ///
    /// # Arguments
    /// * `state` - Current block state (completed blocks + partial sum)
    /// * `mask` - Optional causal attention mask [B, T, T]
    ///
    /// # Returns
    /// * Updated block state
    pub fn forward(&self, mut state: BlockState<B>, mask: Option<&Tensor<B, 3>>) -> BlockState<B> {
        // === AttnRes before self-attention ===
        let current_partial = state.partial_block.take();
        let h = self
            .attn_res
            .forward_optional_partial(&state.blocks, current_partial.as_ref());

        let mut partial_for_attn =
            current_partial.unwrap_or_else(|| Tensor::zeros_like(state.blocks.last().unwrap()));
        if self.starts_new_block_before_attn() {
            state.blocks.push(partial_for_attn.clone());
            partial_for_attn = Tensor::zeros_like(state.blocks.last().unwrap());
        }

        // === Self-attention sublayer ===
        let normed = self.attn_norm.forward(h);
        let attn_out = self.attn.forward(normed, mask);

        // Update partial block with attention output
        let partial_after_attn = partial_for_attn + attn_out;

        // === AttnRes before MLP ===
        let h = self
            .mlp_res
            .forward_optional_partial(&state.blocks, Some(&partial_after_attn));

        let mut partial_for_mlp = partial_after_attn;
        if self.starts_new_block_before_mlp() {
            state.blocks.push(partial_for_mlp.clone());
            partial_for_mlp = Tensor::zeros_like(state.blocks.last().unwrap());
        }

        // === MLP sublayer ===
        let normed = self.mlp_norm.forward(h);
        let mlp_out = self.mlp.forward(normed);

        // Update partial block with MLP output
        let partial_after_mlp = partial_for_mlp + mlp_out;

        state.partial_block = Some(partial_after_mlp);
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type TestBackend = NdArray;

    #[test]
    fn test_layer_forward_preserves_shape() {
        let device = Default::default();
        let config = AttnResConfig::new(32, 4, 2).with_num_heads(4);
        let layer = config.init_layer::<TestBackend>(0, &device);

        let emb = Tensor::random([1, 8, 32], Distribution::Normal(0.0, 1.0), &device);
        let state = BlockState::new(emb);
        let new_state = layer.forward(state, None);

        assert!(new_state.partial_block.is_some());
        assert_eq!(new_state.partial_block.unwrap().dims(), [1, 8, 32]);
    }

    #[test]
    fn test_layer_block_boundary() {
        let device = Default::default();
        // 4 sublayers, 2 blocks -> block_size = 2 -> boundary at layer_idx % 1 == 0 for idx > 0
        let config = AttnResConfig::new(32, 4, 2).with_num_heads(4);

        let layer0 = config.init_layer::<TestBackend>(0, &device);
        let layer1 = config.init_layer::<TestBackend>(1, &device);

        let emb = Tensor::random([1, 8, 32], Distribution::Normal(0.0, 1.0), &device);
        let state = BlockState::new(emb);

        // Layer 0: no boundary (first layer)
        let state = layer0.forward(state, None);
        assert_eq!(state.num_blocks(), 1, "Layer 0 should not add a block");

        // Layer 1: boundary (layer_idx=1, block_size/2=1, 1%1==0 and 1>0)
        let state = layer1.forward(state, None);
        assert_eq!(
            state.num_blocks(),
            2,
            "Layer 1 should add a block at boundary"
        );
    }
}
