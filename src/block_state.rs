/// Tracks the state of block accumulation across layers.
///
/// This is passed between layers during the forward pass.
/// It maintains:
/// - Completed block representations (b_0, b_1, ..., b_{n-1})
/// - The current partial sum within the active block (b_n^i)
///
/// Paper reference: Section 3, Block Attention Residuals.
use burn::prelude::*;

#[derive(Clone, Debug)]
pub struct BlockState<B: Backend> {
    /// Completed block representations.
    /// blocks[0] = token embedding (b_0 = h_1).
    /// blocks[n] = sum of layer outputs in block n.
    pub blocks: Vec<Tensor<B, 3>>, // Each: [batch, seq_len, d_model]

    /// Partial sum of layer outputs within the current (incomplete) block.
    /// None at the start of a new block.
    pub partial_block: Option<Tensor<B, 3>>, // [batch, seq_len, d_model]
}

impl<B: Backend> BlockState<B> {
    /// Initialize with token embeddings as the first block (b_0).
    pub fn new(token_embeddings: Tensor<B, 3>) -> Self {
        Self {
            blocks: vec![token_embeddings],
            partial_block: None,
        }
    }

    /// Number of completed blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_new_state() {
        let device = Default::default();
        let emb = Tensor::<TestBackend, 3>::zeros([2, 16, 64], &device);
        let state = BlockState::new(emb);
        assert_eq!(state.num_blocks(), 1);
        assert!(state.partial_block.is_none());
    }
}
