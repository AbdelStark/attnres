//! Visualize learned depth attention weights in an AttnRes model.
//!
//! This example demonstrates how to extract and display the attention
//! patterns over depth (which blocks each layer attends to) from a
//! trained AttnRes model.
//!
//! Run with: `cargo run --example visualize_weights`

use attnres::{AttnResConfig, AttnResOp};
use burn::backend::NdArray;
use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::Distribution;

type B = NdArray;

/// Extract attention weights (alpha) from an AttnResOp given block representations.
///
/// Returns the softmax attention weights over the depth dimension.
/// alpha[i] = how much attention the pseudo-query gives to source i.
fn extract_weights(op: &AttnResOp<B>, blocks: &[Tensor<B, 3>], partial: &Tensor<B, 3>) -> Vec<f32> {
    let mut sources: Vec<Tensor<B, 3>> = blocks.to_vec();
    sources.push(partial.clone());
    let num_sources = sources.len();

    let v = Tensor::stack(sources, 0); // [N+1, B, T, D]

    // Apply the op's own RMSNorm to get keys (must use trained norm, not a fresh one)
    let k = op.norm.forward_4d(v);

    // Compute logits
    let w = op
        .pseudo_query
        .val()
        .unsqueeze_dim::<2>(0)
        .unsqueeze_dim::<3>(0)
        .unsqueeze_dim::<4>(0);
    let logits = (k * w).sum_dim(3).squeeze_dim::<3>(3); // [N+1, B, T]

    // Softmax over depth (dim=0)
    let alpha = softmax(logits, 0); // [N+1, B, T]

    // Average weights across batch and sequence dimensions
    // to get a single weight per source
    let mean_alpha = alpha.mean_dim(2).squeeze_dim::<2>(2); // [N+1, B]
    let mean_alpha = mean_alpha.mean_dim(1).squeeze_dim::<1>(1); // [N+1]

    mean_alpha
        .reshape([num_sources])
        .into_data()
        .to_vec()
        .unwrap()
}

/// Print an ASCII bar chart of attention weights.
fn print_bar_chart(label: &str, weights: &[f32], bar_width: usize) {
    println!("{label}:");
    let max_w = weights.iter().cloned().fold(0.0_f32, f32::max);
    for (i, &w) in weights.iter().enumerate() {
        let name = if i < weights.len() - 1 {
            format!("  block_{i}")
        } else {
            "  partial".to_string()
        };
        let bar_len = if max_w > 0.0 {
            ((w / max_w) * bar_width as f32) as usize
        } else {
            0
        };
        let bar: String = "#".repeat(bar_len);
        println!("  {name:>10} [{bar:<width$}] {w:.4}", width = bar_width);
    }
    println!();
}

fn main() {
    let device = Default::default();

    println!("=== AttnRes Depth Attention Weight Visualization ===\n");

    // Create a model config
    let config = AttnResConfig::new(32, 8, 2).with_num_heads(4);

    // Create some block representations (simulating a mid-inference state)
    let blocks = vec![
        Tensor::random([1, 8, 32], Distribution::Normal(0.0, 1.0), &device),
        Tensor::random([1, 8, 32], Distribution::Normal(0.0, 1.0), &device),
        Tensor::random([1, 8, 32], Distribution::Normal(0.0, 1.0), &device),
    ];
    let partial = Tensor::random([1, 8, 32], Distribution::Normal(0.0, 1.0), &device);

    // === Demo 1: Zero-initialized pseudo-queries (uniform weights) ===
    println!("1. Zero-initialized pseudo-query (should be uniform):\n");
    let op_zero = config.init_op::<B>(&device);
    let weights = extract_weights(&op_zero, &blocks, &partial);
    print_bar_chart("  Layer (zero-init)", &weights, 40);

    // === Demo 2: Manually set pseudo-queries to see non-uniform patterns ===
    println!("2. Non-uniform pseudo-query (manually set to show differentiation):\n");

    // Create multiple ops with different pseudo-query values to simulate
    // what a trained model might look like
    let scenarios: Vec<(&str, Vec<f32>)> = vec![
        ("Early layer (attends to embedding)", vec![1.0; 32]),
        (
            "Middle layer (balanced)",
            (0..32).map(|i| (i as f32 * 0.1).sin()).collect(),
        ),
        ("Late layer (attends to recent)", vec![-1.0; 32]),
    ];

    for (desc, query_vals) in &scenarios {
        let op = config.init_op::<B>(&device);
        // We can't easily set the pseudo-query directly since it's a Param,
        // but we can compute what the weights would be for a given query vector.
        let query = Tensor::<B, 1>::from_floats(query_vals.as_slice(), &device);

        // Compute weights manually for visualization
        let mut sources: Vec<Tensor<B, 3>> = blocks.clone();
        sources.push(partial.clone());
        let v = Tensor::stack(sources, 0);
        let k = op.norm.forward_4d(v);
        let w = query
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0);
        let logits = (k * w).sum_dim(3).squeeze_dim::<3>(3);
        let alpha = softmax(logits, 0);
        let mean_alpha = alpha.mean_dim(2).squeeze_dim::<2>(2);
        let mean_alpha = mean_alpha.mean_dim(1).squeeze_dim::<1>(1);
        let weights: Vec<f32> = mean_alpha.reshape([4]).into_data().to_vec().unwrap();

        print_bar_chart(&format!("  {desc}"), &weights, 40);
    }

    // === Demo 3: Show how weights change across layers in a full model ===
    println!("3. Weight distribution summary:\n");
    println!("  Key insight from the paper:");
    println!("  - Early layers attend more uniformly across all blocks");
    println!("  - Later layers develop preferences for specific blocks");
    println!("  - This allows selective information routing across depth");
    println!();
    println!("  At initialization (zero pseudo-queries), all layers attend");
    println!("  uniformly, equivalent to standard residual connections.");
    println!("  Training gradually differentiates the attention patterns.");

    println!("\nDone!");
}
