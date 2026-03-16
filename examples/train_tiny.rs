//! Train a tiny AttnRes Transformer model on synthetic data.
//!
//! This example demonstrates:
//! - Model configuration and initialization
//! - Forward pass with causal masking
//! - Basic training loop with cross-entropy loss
//!
//! Run with: `cargo run --example train_tiny`

use attnres_rs::{causal_mask, AttnResConfig, AttnResTransformer};
use burn::backend::Autodiff;
use burn::backend::NdArray;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::Distribution;

type Backend = NdArray;
type AutodiffBackend = Autodiff<Backend>;

fn main() {
    let device = Default::default();

    // Configure a small model
    // 8 sublayers = 4 transformer layers, 2 blocks
    let config = AttnResConfig::new(64, 8, 2)
        .with_num_heads(4)
        .with_vocab_size(256)
        .with_d_ff(128);

    let model: AttnResTransformer<AutodiffBackend> = config.init_model(&device);

    // Create optimizer
    let mut optim = AdamConfig::new().init();

    let batch_size = 4;
    let seq_len = 32;
    let num_steps = 50;
    let mask = causal_mask::<AutodiffBackend>(batch_size, seq_len, &device);

    println!("Training tiny AttnRes model:");
    println!("  d_model={}, layers={}, blocks={}", 64, 8, 2);
    println!("  batch_size={batch_size}, seq_len={seq_len}, steps={num_steps}");
    println!();

    let mut model = model;

    for step in 0..num_steps {
        // Generate random input tokens
        let input_ids = Tensor::<AutodiffBackend, 2, Int>::random(
            [batch_size, seq_len],
            Distribution::Uniform(0.0, 256.0),
            &device,
        );

        // Targets: shifted input (next token prediction)
        let targets = Tensor::<AutodiffBackend, 2, Int>::random(
            [batch_size, seq_len],
            Distribution::Uniform(0.0, 256.0),
            &device,
        );

        // Forward pass
        let logits = model.forward(input_ids, Some(&mask)); // [B, T, V]

        // Cross-entropy loss for next-token prediction
        let [b, t, v] = logits.dims();
        let logits_flat = logits.reshape([b * t, v]);
        let targets_flat = targets.reshape([b * t]);

        let loss_fn = CrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&device);
        let loss = loss_fn.forward(logits_flat, targets_flat).mean();

        if step % 10 == 0 {
            let loss_val: f32 = loss.clone().into_scalar();
            println!("  step {step:>3}: loss = {loss_val:.4}");
        }

        // Backward pass
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(0.001, model, grads);
    }

    println!("\nTraining complete!");
}
