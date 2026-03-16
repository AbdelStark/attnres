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
    /// Get the layer index.
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Get the block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Check if this layer is at a block boundary.
    pub fn is_at_boundary(&self) -> bool {
        let half_block = self.block_size / 2;
        self.layer_idx > 0 && (half_block == 0 || self.layer_idx.is_multiple_of(half_block))
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
        // Get the current partial block, or zeros if at the start of a new block
        let current_partial = state
            .partial_block
            .take()
            .unwrap_or_else(|| Tensor::zeros_like(state.blocks.last().unwrap()));

        // === Check block boundary ===
        // Block boundary occurs every block_size/2 transformer layers (each layer = 2 sublayers).
        // For Full AttnRes (block_size=1), every layer after the first is a boundary.
        let half_block = self.block_size / 2;
        let at_boundary =
            self.layer_idx > 0 && (half_block == 0 || self.layer_idx.is_multiple_of(half_block));

        if at_boundary {
            // Push the completed partial block as a new block
            state.blocks.push(current_partial.clone());
        }

        // The partial block for AttnRes input: if we just pushed, start fresh; otherwise use current
        let partial_for_attn = if at_boundary {
            Tensor::zeros_like(state.blocks.last().unwrap())
        } else {
            current_partial
        };

        // === AttnRes before self-attention ===
        let h = self.attn_res.forward(&state.blocks, &partial_for_attn);

        // === Self-attention sublayer ===
        let normed = self.attn_norm.forward(h);
        let attn_out = self.attn.forward(normed, mask);

        // Update partial block with attention output
        let partial_after_attn = partial_for_attn + attn_out;

        // === AttnRes before MLP ===
        let h = self.mlp_res.forward(&state.blocks, &partial_after_attn);

        // === MLP sublayer ===
        let normed = self.mlp_norm.forward(h);
        let mlp_out = self.mlp.forward(normed);

        // Update partial block with MLP output
        let partial_after_mlp = partial_after_attn + mlp_out;

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
