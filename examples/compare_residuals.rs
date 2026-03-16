//! Compare standard residual connections vs AttnRes.
//!
//! Shows that an AttnRes model with zero-initialized pseudo-queries
//! starts as uniform averaging over all prior blocks,
//! and demonstrates the forward pass works correctly.
//!
//! Run with: `cargo run --example compare_residuals`

use attnres::{AttnResConfig, AttnResOp, AttnResTransformer, BlockState};
use burn::backend::NdArray;
use burn::prelude::*;
use burn::tensor::Distribution;

type B = NdArray;

fn main() {
    let device = Default::default();

    println!("=== AttnRes vs Standard Residuals ===\n");

    // Demo 1: Zero-init produces uniform weights (equivalent to mean)
    println!("1. Zero-initialized AttnRes = uniform averaging");
    println!("   (equal weights over all available sources)\n");

    let config = AttnResConfig::new(32, 4, 2);
    let op: AttnResOp<B> = config.init_op(&device);

    let block0 = Tensor::random([1, 4, 32], Distribution::Normal(0.0, 1.0), &device);
    let block1 = Tensor::random([1, 4, 32], Distribution::Normal(0.0, 1.0), &device);
    let partial = Tensor::random([1, 4, 32], Distribution::Normal(0.0, 1.0), &device);

    let attn_out = op.forward(&[block0.clone(), block1.clone()], &partial);
    let mean_out = (block0 + block1 + partial) / 3.0;

    let diff: f32 = (attn_out - mean_out).abs().max().into_scalar();
    println!("   Max diff between AttnRes and mean: {diff:.2e}");
    println!("   (Should be ~0, confirming uniform weights)\n");

    // Demo 2: Full model forward pass
    println!("2. Full model forward pass");

    let model_config = AttnResConfig::new(64, 8, 2)
        .with_num_heads(4)
        .with_vocab_size(100);

    let model: AttnResTransformer<B> = model_config.init_model(&device);

    let input_ids = Tensor::<B, 2, Int>::zeros([2, 16], &device);
    let logits = model.forward(input_ids, None);
    let [b, t, v] = logits.dims();
    println!("   Input:  [2, 16] (batch=2, seq_len=16)");
    println!("   Output: [{b}, {t}, {v}] (batch, seq_len, vocab)");

    // Demo 3: Block state tracking
    println!("\n3. Block state tracking");
    let emb = Tensor::<B, 3>::random([1, 8, 64], Distribution::Normal(0.0, 1.0), &device);
    let state = BlockState::new(emb);
    println!(
        "   Initial blocks: {} (token embedding)",
        state.num_blocks()
    );
    println!("   Partial block: {}", state.partial_block.is_some());

    println!("\nDone!");
}
