/// Configuration for Attention Residuals.
///
/// `num_layers` counts *sublayers* — each transformer layer has 2 sublayers
/// (attention + MLP), so `num_layers=8` creates 4 transformer layers.
///
/// Supports JSON serialization via [`save`](AttnResConfig::save) and
/// [`load`](AttnResConfig::load) methods.
///
/// Paper reference: Section 3, Block Attention Residuals.
use burn::config::Config;

#[derive(Config, Debug)]
pub struct AttnResConfig {
    /// Hidden dimension (d_model). Must be positive and divisible by `num_heads`.
    pub d_model: usize,
    /// Total number of sublayers (L). Each transformer layer has 2 sublayers (attn + MLP).
    /// Must be positive, even, and divisible by `num_blocks`.
    pub num_layers: usize,
    /// Number of blocks for Block AttnRes. Set to `num_layers` for Full AttnRes.
    /// Must be positive and divide `num_layers` evenly.
    pub num_blocks: usize,
    /// Number of attention heads for multi-head attention.
    /// Must divide `d_model` evenly.
    #[config(default = 8)]
    pub num_heads: usize,
    /// Intermediate dimension for the MLP feed-forward layer.
    /// Defaults to 4 * d_model when set to 0.
    #[config(default = 0)]
    pub d_ff: usize,
    /// Vocabulary size for embedding and LM head.
    #[config(default = 32000)]
    pub vocab_size: usize,
    /// Epsilon for RMSNorm numerical stability.
    #[config(default = 1e-6)]
    pub rms_norm_eps: f64,
    /// Dropout rate (0.0 = no dropout).
    #[config(default = 0.0)]
    pub dropout: f64,
}

impl AttnResConfig {
    /// Validate that this configuration is internally consistent.
    ///
    /// # Panics
    /// Panics with a descriptive message if any constraint is violated.
    pub fn validate(&self) {
        assert!(self.d_model > 0, "d_model must be positive, got 0");
        assert!(self.num_layers > 0, "num_layers must be positive, got 0");
        assert!(self.num_blocks > 0, "num_blocks must be positive, got 0");
        assert!(
            self.num_layers.is_multiple_of(2),
            "num_layers ({}) must be even (each transformer layer = 2 sublayers: attn + MLP)",
            self.num_layers
        );
        assert!(
            self.num_layers.is_multiple_of(self.num_blocks),
            "num_layers ({}) must be divisible by num_blocks ({})",
            self.num_layers,
            self.num_blocks
        );
        assert!(
            self.d_model.is_multiple_of(self.num_heads),
            "d_model ({}) must be divisible by num_heads ({})",
            self.d_model,
            self.num_heads
        );
        assert!(self.vocab_size > 0, "vocab_size must be positive, got 0");
        assert!(
            self.rms_norm_eps > 0.0,
            "rms_norm_eps must be positive, got {}",
            self.rms_norm_eps
        );
        assert!(
            (0.0..=1.0).contains(&self.dropout),
            "dropout must be in [0.0, 1.0], got {}",
            self.dropout
        );
    }

    /// Sublayers per block (S = L / N).
    ///
    /// # Panics
    /// Panics if `num_blocks` is zero or doesn't divide `num_layers`.
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
    /// Returns `d_ff` if explicitly set, otherwise `4 * d_model`.
    pub fn effective_d_ff(&self) -> usize {
        if self.d_ff == 0 {
            4 * self.d_model
        } else {
            self.d_ff
        }
    }

    /// Number of transformer layers (each transformer layer = 2 sublayers).
    pub fn num_transformer_layers(&self) -> usize {
        self.num_layers / 2
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

    #[test]
    fn test_validate_good_config() {
        let config = AttnResConfig::new(64, 12, 4).with_num_heads(8);
        config.validate(); // should not panic
    }

    #[test]
    #[should_panic(expected = "must be even")]
    fn test_validate_odd_num_layers() {
        let config = AttnResConfig::new(64, 11, 1);
        config.validate();
    }

    #[test]
    #[should_panic(expected = "d_model (64) must be divisible by num_heads (5)")]
    fn test_validate_bad_num_heads() {
        let config = AttnResConfig::new(64, 12, 4).with_num_heads(5);
        config.validate();
    }

    #[test]
    fn test_num_transformer_layers() {
        let config = AttnResConfig::new(64, 12, 4);
        assert_eq!(config.num_transformer_layers(), 6);
    }

    #[test]
    fn test_config_save_load_roundtrip() {
        let config = AttnResConfig::new(128, 24, 8)
            .with_num_heads(8)
            .with_d_ff(512)
            .with_vocab_size(50000)
            .with_dropout(0.1);

        let path = std::env::temp_dir().join("attnres_test_config.json");
        config.save(&path).expect("Failed to save config");

        let loaded = AttnResConfig::load(&path).expect("Failed to load config");

        assert_eq!(config.d_model, loaded.d_model);
        assert_eq!(config.num_layers, loaded.num_layers);
        assert_eq!(config.num_blocks, loaded.num_blocks);
        assert_eq!(config.num_heads, loaded.num_heads);
        assert_eq!(config.d_ff, loaded.d_ff);
        assert_eq!(config.vocab_size, loaded.vocab_size);
        assert!((config.dropout - loaded.dropout).abs() < 1e-10);
        assert!((config.rms_norm_eps - loaded.rms_norm_eps).abs() < 1e-15);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_full_attnres_block_size_one() {
        // Full AttnRes: each sublayer is its own block
        let config = AttnResConfig::new(64, 12, 12);
        assert_eq!(config.block_size(), 1);
        assert!(config.is_full());
    }
}
