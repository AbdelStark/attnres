/// Configuration for Attention Residuals.
///
/// Paper reference: Section 3, Block Attention Residuals.
use burn::config::Config;

#[derive(Config, Debug)]
pub struct AttnResConfig {
    /// Hidden dimension (d_model).
    pub d_model: usize,
    /// Total number of sublayers (L). Each transformer layer has 2 sublayers (attn + MLP).
    pub num_layers: usize,
    /// Number of blocks for Block AttnRes. Set to num_layers for Full AttnRes.
    pub num_blocks: usize,
    /// Number of attention heads for multi-head attention.
    #[config(default = 8)]
    pub num_heads: usize,
    /// Intermediate dimension for the MLP feed-forward layer.
    /// Defaults to 4 * d_model.
    #[config(default = 0)]
    pub d_ff: usize,
    /// Vocabulary size for embedding and LM head.
    #[config(default = 32000)]
    pub vocab_size: usize,
    /// Epsilon for RMSNorm.
    #[config(default = 1e-6)]
    pub rms_norm_eps: f64,
    /// Dropout rate.
    #[config(default = 0.0)]
    pub dropout: f64,
}

impl AttnResConfig {
    /// Sublayers per block (S = L / N).
    pub fn block_size(&self) -> usize {
        assert!(self.num_blocks > 0, "num_blocks must be positive");
        assert!(
            self.num_layers.is_multiple_of(self.num_blocks),
            "num_layers ({}) must be divisible by num_blocks ({})",
            self.num_layers,
            self.num_blocks
        );
        self.num_layers / self.num_blocks
    }

    /// Whether this is Full AttnRes (N = L) or Block AttnRes (N < L).
    pub fn is_full(&self) -> bool {
        self.num_blocks == self.num_layers
    }

    /// Effective feed-forward intermediate dimension.
    pub fn effective_d_ff(&self) -> usize {
        if self.d_ff == 0 {
            4 * self.d_model
        } else {
            self.d_ff
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_size() {
        let config = AttnResConfig::new(64, 12, 4);
        assert_eq!(config.block_size(), 3);
    }

    #[test]
    fn test_is_full() {
        let full = AttnResConfig::new(64, 12, 12);
        assert!(full.is_full());
        let block = AttnResConfig::new(64, 12, 4);
        assert!(!block.is_full());
    }

    #[test]
    fn test_effective_d_ff() {
        let config = AttnResConfig::new(64, 12, 4);
        assert_eq!(config.effective_d_ff(), 256);

        let config2 = AttnResConfig::new(64, 12, 4).with_d_ff(128);
        assert_eq!(config2.effective_d_ff(), 128);
    }

    #[test]
    #[should_panic(expected = "num_blocks must be positive")]
    fn test_zero_blocks_panics() {
        let config = AttnResConfig::new(64, 12, 0);
        config.block_size();
    }

    #[test]
    #[should_panic(expected = "divisible")]
    fn test_indivisible_panics() {
        let config = AttnResConfig::new(64, 12, 5);
        config.block_size();
    }
}
