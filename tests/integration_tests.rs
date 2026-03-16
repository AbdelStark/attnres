/// Integration tests for attnres-rs.
///
/// Tests end-to-end model behavior including forward passes,
/// autodiff gradient flow, and model configuration.
use attnres_rs::{causal_mask, AttnResConfig, AttnResTransformer};
use burn::backend::Autodiff;
use burn::backend::NdArray;
use burn::prelude::*;

type TestBackend = NdArray;
type AutodiffBackend = Autodiff<TestBackend>;

#[test]
fn test_full_model_forward_backward() {
    let device = Default::default();
    let config = AttnResConfig::new(32, 4, 2)
        .with_num_heads(4)
        .with_vocab_size(50);

    let model: AttnResTransformer<AutodiffBackend> = config.init_model(&device);

    let input_ids = Tensor::<AutodiffBackend, 2, Int>::zeros([1, 8], &device);
    let logits = model.forward(input_ids, None);

    // Compute a simple loss and backward
    let loss = logits.mean();
    let _grads = loss.backward();
    // If we get here without panic, gradients flow correctly
}

#[test]
fn test_model_with_mask_forward_backward() {
    let device = Default::default();
    let config = AttnResConfig::new(32, 4, 2)
        .with_num_heads(4)
        .with_vocab_size(50);

    let model: AttnResTransformer<AutodiffBackend> = config.init_model(&device);

    let mask = causal_mask::<AutodiffBackend>(2, 8, &device);
    let input_ids = Tensor::<AutodiffBackend, 2, Int>::zeros([2, 8], &device);
    let logits = model.forward(input_ids, Some(&mask));
    let loss = logits.mean();
    let _grads = loss.backward();
}

#[test]
fn test_full_attnres_config() {
    // Full AttnRes: num_blocks = num_layers
    let device = Default::default();
    let config = AttnResConfig::new(32, 4, 4)
        .with_num_heads(4)
        .with_vocab_size(50);

    assert!(config.is_full());

    let model: AttnResTransformer<TestBackend> = config.init_model(&device);
    let input = Tensor::<TestBackend, 2, Int>::zeros([1, 4], &device);
    let out = model.forward(input, None);
    assert_eq!(out.dims(), [1, 4, 50]);
}

#[test]
fn test_block_attnres_config() {
    // Block AttnRes: num_blocks < num_layers
    let device = Default::default();
    let config = AttnResConfig::new(32, 8, 2)
        .with_num_heads(4)
        .with_vocab_size(50);

    assert!(!config.is_full());
    assert_eq!(config.block_size(), 4);

    let model: AttnResTransformer<TestBackend> = config.init_model(&device);
    let input = Tensor::<TestBackend, 2, Int>::zeros([1, 8], &device);
    let out = model.forward(input, None);
    assert_eq!(out.dims(), [1, 8, 50]);
}

#[test]
fn test_deterministic_forward() {
    let device = Default::default();
    let config = AttnResConfig::new(32, 4, 2)
        .with_num_heads(4)
        .with_vocab_size(50);

    let model: AttnResTransformer<TestBackend> = config.init_model(&device);

    let input = Tensor::<TestBackend, 2, Int>::zeros([1, 8], &device);

    // Same input should produce same output
    let out1 = model.forward(input.clone(), None);
    let out2 = model.forward(input, None);

    let diff: f32 = (out1 - out2).abs().max().into_scalar();
    assert!(
        diff < 1e-6,
        "Same input should produce identical output, diff={diff}"
    );
}
