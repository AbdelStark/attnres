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
use std::fmt::{Display, Formatter};

/// Typed validation failures for [`AttnResConfig`].
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    /// `d_model` must be strictly positive.
    DModelMustBePositive,
    /// `num_layers` must be strictly positive.
    NumLayersMustBePositive,
    /// `num_blocks` must be strictly positive.
    NumBlocksMustBePositive,
    /// `num_heads` must be strictly positive.
    NumHeadsMustBePositive,
    /// `num_layers` counts sublayers and must therefore be even.
    NumLayersMustBeEven { num_layers: usize },
    /// `num_layers` must divide `num_blocks` evenly.
    NumLayersMustBeDivisibleByNumBlocks {
        num_layers: usize,
        num_blocks: usize,
    },
    /// `d_model` must divide `num_heads` evenly.
    DModelMustBeDivisibleByNumHeads { d_model: usize, num_heads: usize },
    /// `vocab_size` must be strictly positive.
    VocabSizeMustBePositive,
    /// `rms_norm_eps` must be strictly positive.
    RmsNormEpsMustBePositive { rms_norm_eps: f64 },
    /// `dropout` must be in the inclusive range `[0.0, 1.0]`.
    DropoutOutOfRange { dropout: f64 },
    /// The default `4 * d_model` MLP expansion overflowed `usize`.
    EffectiveFeedForwardDimOverflow { d_model: usize },
    /// `layer_idx` exceeded the configured layer count.
    LayerIndexOutOfRange {
        layer_idx: usize,
        num_transformer_layers: usize,
    },
}

impl Display for ConfigError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DModelMustBePositive => write!(f, "d_model must be positive, got 0"),
            Self::NumLayersMustBePositive => write!(f, "num_layers must be positive, got 0"),
            Self::NumBlocksMustBePositive => write!(f, "num_blocks must be positive, got 0"),
            Self::NumHeadsMustBePositive => write!(f, "num_heads must be positive, got 0"),
            Self::NumLayersMustBeEven { num_layers } => write!(
                f,
                "num_layers ({num_layers}) must be even (each transformer layer = 2 sublayers: attn + MLP)"
            ),
            Self::NumLayersMustBeDivisibleByNumBlocks {
                num_layers,
                num_blocks,
            } => write!(
                f,
                "num_layers ({num_layers}) must be divisible by num_blocks ({num_blocks})"
            ),
            Self::DModelMustBeDivisibleByNumHeads { d_model, num_heads } => write!(
                f,
                "d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            ),
            Self::VocabSizeMustBePositive => write!(f, "vocab_size must be positive, got 0"),
            Self::RmsNormEpsMustBePositive { rms_norm_eps } => write!(
                f,
                "rms_norm_eps must be positive, got {rms_norm_eps}"
            ),
            Self::DropoutOutOfRange { dropout } => {
                write!(f, "dropout must be in [0.0, 1.0], got {dropout}")
            }
            Self::EffectiveFeedForwardDimOverflow { d_model } => write!(
                f,
                "effective d_ff overflow: 4 * d_model ({d_model}) exceeds usize"
            ),
            Self::LayerIndexOutOfRange {
                layer_idx,
                num_transformer_layers,
            } => write!(
                f,
                "layer_idx ({layer_idx}) must be < num_transformer_layers ({num_transformer_layers})"
            ),
        }
    }
}

impl std::error::Error for ConfigError {}

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
    /// Validate this configuration and return a typed error instead of panicking.
    pub fn try_validate(&self) -> Result<(), ConfigError> {
        if self.d_model == 0 {
            return Err(ConfigError::DModelMustBePositive);
        }
        if self.num_layers == 0 {
            return Err(ConfigError::NumLayersMustBePositive);
        }
        if self.num_blocks == 0 {
            return Err(ConfigError::NumBlocksMustBePositive);
        }
        if self.num_heads == 0 {
            return Err(ConfigError::NumHeadsMustBePositive);
        }
        if !self.num_layers.is_multiple_of(2) {
            return Err(ConfigError::NumLayersMustBeEven {
                num_layers: self.num_layers,
            });
        }
        if !self.num_layers.is_multiple_of(self.num_blocks) {
            return Err(ConfigError::NumLayersMustBeDivisibleByNumBlocks {
                num_layers: self.num_layers,
                num_blocks: self.num_blocks,
            });
        }
        if !self.d_model.is_multiple_of(self.num_heads) {
            return Err(ConfigError::DModelMustBeDivisibleByNumHeads {
                d_model: self.d_model,
                num_heads: self.num_heads,
            });
        }
        if self.vocab_size == 0 {
            return Err(ConfigError::VocabSizeMustBePositive);
        }
        if self.rms_norm_eps <= 0.0 {
            return Err(ConfigError::RmsNormEpsMustBePositive {
                rms_norm_eps: self.rms_norm_eps,
            });
        }
        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(ConfigError::DropoutOutOfRange {
                dropout: self.dropout,
            });
        }

        self.try_effective_d_ff()?;
        Ok(())
    }

    /// Validate that this configuration is internally consistent.
    ///
    /// # Panics
    /// Panics with a descriptive message if any constraint is violated.
    pub fn validate(&self) {
        if let Err(err) = self.try_validate() {
            panic!("{err}");
        }
    }

    /// Sublayers per block (S = L / N).
    ///
    /// # Panics
    /// Panics if `num_blocks` is zero or doesn't divide `num_layers`.
    pub fn try_block_size(&self) -> Result<usize, ConfigError> {
        if self.num_layers == 0 {
            return Err(ConfigError::NumLayersMustBePositive);
        }
        if self.num_blocks == 0 {
            return Err(ConfigError::NumBlocksMustBePositive);
        }
        if !self.num_layers.is_multiple_of(self.num_blocks) {
            return Err(ConfigError::NumLayersMustBeDivisibleByNumBlocks {
                num_layers: self.num_layers,
                num_blocks: self.num_blocks,
            });
        }
        Ok(self.num_layers / self.num_blocks)
    }

    pub fn block_size(&self) -> usize {
        match self.try_block_size() {
            Ok(block_size) => block_size,
            Err(err) => panic!("{err}"),
        }
    }

    /// Whether this is Full AttnRes (N = L) or Block AttnRes (N < L).
    pub fn is_full(&self) -> bool {
        self.num_blocks == self.num_layers
    }

    /// Effective feed-forward intermediate dimension.
    /// Returns `d_ff` if explicitly set, otherwise `4 * d_model`.
    pub fn try_effective_d_ff(&self) -> Result<usize, ConfigError> {
        if self.d_ff == 0 {
            self.d_model
                .checked_mul(4)
                .ok_or(ConfigError::EffectiveFeedForwardDimOverflow {
                    d_model: self.d_model,
                })
        } else {
            Ok(self.d_ff)
        }
    }

    pub fn effective_d_ff(&self) -> usize {
        match self.try_effective_d_ff() {
            Ok(d_ff) => d_ff,
            Err(err) => panic!("{err}"),
        }
    }

    /// Number of transformer layers (each transformer layer = 2 sublayers).
    pub fn num_transformer_layers(&self) -> usize {
        self.num_layers / 2
    }

    /// Validate a zero-based transformer-layer index against this config.
    pub fn try_validate_layer_idx(&self, layer_idx: usize) -> Result<(), ConfigError> {
        self.try_validate()?;
        let num_transformer_layers = self.num_transformer_layers();
        if layer_idx >= num_transformer_layers {
            return Err(ConfigError::LayerIndexOutOfRange {
                layer_idx,
                num_transformer_layers,
            });
        }
        Ok(())
    }

    /// Validate a zero-based transformer-layer index, panicking on failure.
    pub fn validate_layer_idx(&self, layer_idx: usize) {
        if let Err(err) = self.try_validate_layer_idx(layer_idx) {
            panic!("{err}");
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

    #[test]
    fn test_validate_good_config() {
        let config = AttnResConfig::new(64, 12, 4).with_num_heads(8);
        config.validate(); // should not panic
    }

    #[test]
    fn test_try_validate_good_config() {
        let config = AttnResConfig::new(64, 12, 4).with_num_heads(8);
        assert!(config.try_validate().is_ok());
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
    #[should_panic(expected = "num_heads must be positive")]
    fn test_validate_zero_num_heads() {
        let config = AttnResConfig::new(64, 12, 4).with_num_heads(0);
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

    #[test]
    fn test_try_validate_reports_typed_error() {
        let config = AttnResConfig::new(64, 12, 4).with_num_heads(0);
        assert_eq!(
            config.try_validate(),
            Err(ConfigError::NumHeadsMustBePositive)
        );
    }

    #[test]
    fn test_try_effective_d_ff_overflow() {
        let config = AttnResConfig::new(usize::MAX, 2, 1).with_num_heads(1);
        assert_eq!(
            config.try_effective_d_ff(),
            Err(ConfigError::EffectiveFeedForwardDimOverflow {
                d_model: usize::MAX
            })
        );
    }

    #[test]
    fn test_try_validate_layer_idx_rejects_out_of_range() {
        let config = AttnResConfig::new(64, 12, 4).with_num_heads(8);
        assert_eq!(
            config.try_validate_layer_idx(6),
            Err(ConfigError::LayerIndexOutOfRange {
                layer_idx: 6,
                num_transformer_layers: 6
            })
        );
    }
}
