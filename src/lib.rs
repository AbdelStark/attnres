//! # attnres
//!
//! First Rust implementation of Attention Residuals from the MoonshotAI/Kimi paper,
//! built on the [burn](https://burn.dev) deep learning framework.
//!
//! Attention Residuals replace standard fixed-weight residual connections in Transformers
//! with learned softmax attention over depth, enabling selective information routing
//! across layers.
//!
//! The crate also exposes a separate `kimi` module tree for the staged Kimi
//! milestone. This checkout includes RFC 0001 artifact understanding and RFC
//! 0002 baseline Kimi Linear execution scaffolding plus RFC 0003 checkpoint
//! import scaffolding for tensor-name coverage and shard planning, plus RFC
//! 0004 AttnRes-Kimi execution scaffolding via a separate
//! `kimi::KimiAttnResModel` path. This checkout now also includes an executable
//! local RFC 0005 slice: Gate 4 functional validation, reduced-config Gate 5
//! numerical agreement tests, and reduced local benchmark scaffolding. Baseline
//! parity, public-checkpoint parity, and Hugging Face/Python-dependent gates
//! remain deferred.
//!
//! ## Quick Start
//!
//! ```rust
//! use attnres::{AttnResConfig, AttnResTransformer};
//! use burn::prelude::*;
//! use burn::backend::NdArray;
//!
//! type B = NdArray;
//!
//! let device = Default::default();
//! let config = AttnResConfig::new(128, 8, 2)
//!     .with_num_heads(4)
//!     .with_vocab_size(1000);
//!
//! let model: AttnResTransformer<B> = config.init_model(&device);
//! let input_ids = Tensor::<B, 2, Int>::zeros([1, 16], &device);
//! let logits = model.forward(input_ids, None);
//! assert_eq!(logits.dims(), [1, 16, 1000]);
//! ```

pub mod attention;
pub mod attn_res_op;
pub mod block_state;
pub mod config;
pub mod feed_forward;
pub mod kimi;
pub mod layer;
pub mod model;
pub mod rms_norm;
pub mod serialization;
pub mod two_phase;
pub mod utils;

// Public API re-exports
pub use attention::{MultiHeadAttention, MultiHeadAttentionConfig};
pub use attn_res_op::AttnResOp;
pub use block_state::BlockState;
pub use config::{AttnResConfig, ConfigError};
pub use feed_forward::{FeedForward, FeedForwardConfig};
pub use layer::AttnResLayer;
pub use model::AttnResTransformer;
pub use rms_norm::{RmsNorm, RmsNormConfig};
pub use serialization::SerializationError;
pub use utils::causal_mask;
