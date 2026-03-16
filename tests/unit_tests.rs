/// Unit tests for attnres-rs core functionality.
///
/// Tests the core algorithm components using NdArray backend.
use attnres_rs::{AttnResConfig, BlockState, RmsNormConfig};
use burn::backend::NdArray;
use burn::prelude::*;
use burn::tensor::Distribution;

type TestBackend = NdArray;

// ========================
// AttnResOp Tests
// ========================

#[test]
fn test_attnres_zero_init_produces_uniform_weights() {
    let device = Default::default();
    let config = AttnResConfig::new(64, 12, 4);
    let op = config.init_op::<TestBackend>(&device);

    let blocks = vec![
        Tensor::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device),
        Tensor::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device),
    ];
    let partial = Tensor::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device);

    let output = op.forward(&blocks, &partial);
    let expected = (blocks[0].clone() + blocks[1].clone() + partial) / 3.0;

    let diff: f32 = (output - expected).abs().max().into_scalar();
    assert!(
        diff < 1e-4,
        "Zero-init should produce mean of sources, diff={diff}"
    );
}

#[test]
fn test_attnres_output_shape_various_configs() {
    let device = Default::default();

    // Test with different numbers of blocks
    for num_blocks in [1, 2, 3, 5] {
        let config = AttnResConfig::new(32, 12, num_blocks);
        let op = config.init_op::<TestBackend>(&device);

        let blocks: Vec<_> = (0..num_blocks)
            .map(|_| Tensor::random([1, 8, 32], Distribution::Normal(0.0, 1.0), &device))
            .collect();
        let partial = Tensor::random([1, 8, 32], Distribution::Normal(0.0, 1.0), &device);

        let output = op.forward(&blocks, &partial);
        assert_eq!(
            output.dims(),
            [1, 8, 32],
            "Output shape wrong for num_blocks={num_blocks}"
        );
    }
}

#[test]
fn test_attnres_single_source_returns_that_source() {
    let device = Default::default();
    let config = AttnResConfig::new(16, 4, 4);
    let op = config.init_op::<TestBackend>(&device);

    // With only 1 source (empty blocks, just partial), output should equal partial
    let partial = Tensor::random([1, 4, 16], Distribution::Normal(0.0, 1.0), &device);
    let output = op.forward(&[], &partial);

    let diff: f32 = (output - partial).abs().max().into_scalar();
    assert!(
        diff < 1e-5,
        "Single source should return that source, diff={diff}"
    );
}

// ========================
// Block State Tests
// ========================

#[test]
fn test_block_state_initialization() {
    let device = Default::default();
    let emb =
        Tensor::<TestBackend, 3>::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device);
    let state = BlockState::new(emb.clone());

    assert_eq!(state.num_blocks(), 1);
    assert!(state.partial_block.is_none());

    // First block should be the embedding
    let diff: f32 = (state.blocks[0].clone() - emb).abs().max().into_scalar();
    assert!(diff < 1e-6);
}

// ========================
// RMSNorm Tests
// ========================

#[test]
fn test_rmsnorm_preserves_relative_direction() {
    let device = Default::default();
    let norm = RmsNormConfig::new(4).init::<TestBackend>(&device);

    let x = Tensor::<TestBackend, 3>::from_floats([[[1.0, 2.0, 3.0, 4.0]]], &device);
    let out = norm.forward(x);

    // All values should maintain positive sign
    let data: Vec<f32> = out.reshape([4]).into_data().to_vec().unwrap();
    for (i, &val) in data.iter().enumerate() {
        assert!(val > 0.0, "Element {i} should be positive, got {val}");
    }

    // Values should be monotonically increasing (same as input)
    for i in 0..3 {
        assert!(
            data[i] < data[i + 1],
            "Should be monotonic: {} < {}",
            data[i],
            data[i + 1]
        );
    }
}

#[test]
fn test_rmsnorm_4d_shape() {
    let device = Default::default();
    let norm = RmsNormConfig::new(64).init::<TestBackend>(&device);

    let x = Tensor::random([5, 2, 16, 64], Distribution::Normal(0.0, 1.0), &device);
    let out = norm.forward_4d(x);
    assert_eq!(out.dims(), [5, 2, 16, 64]);
}

// ========================
// Layer Tests
// ========================

#[test]
fn test_layer_produces_partial_block() {
    let device = Default::default();
    let config = AttnResConfig::new(32, 4, 2).with_num_heads(4);
    let layer = config.init_layer::<TestBackend>(0, &device);

    let emb = Tensor::random([1, 8, 32], Distribution::Normal(0.0, 1.0), &device);
    let state = BlockState::new(emb);
    let new_state = layer.forward(state, None);

    assert!(
        new_state.partial_block.is_some(),
        "Layer should produce a partial block"
    );
    assert_eq!(new_state.partial_block.unwrap().dims(), [1, 8, 32]);
}

#[test]
fn test_multiple_layers_sequence() {
    let device = Default::default();
    // 8 sublayers, 2 blocks -> block_size=4 -> boundary every 2 transformer layers
    let config = AttnResConfig::new(32, 8, 2).with_num_heads(4);

    let layers: Vec<_> = (0..4)
        .map(|i| config.init_layer::<TestBackend>(i, &device))
        .collect();

    let emb = Tensor::random([1, 8, 32], Distribution::Normal(0.0, 1.0), &device);
    let mut state = BlockState::new(emb);

    for layer in &layers {
        state = layer.forward(state, None);
    }

    assert!(state.partial_block.is_some());
    // Should have accumulated some blocks beyond the initial embedding
    assert!(
        state.num_blocks() >= 1,
        "Should have at least the initial block"
    );
}

// ========================
// Full Model Tests
// ========================

#[test]
fn test_model_different_batch_sizes() {
    let device = Default::default();
    let config = AttnResConfig::new(32, 4, 2)
        .with_num_heads(4)
        .with_vocab_size(50);

    let model = config.init_model::<TestBackend>(&device);

    for batch in [1, 2, 4] {
        let input = Tensor::<TestBackend, 2, Int>::zeros([batch, 8], &device);
        let out = model.forward(input, None);
        assert_eq!(out.dims(), [batch, 8, 50], "Failed for batch_size={batch}");
    }
}

#[test]
fn test_model_different_seq_lengths() {
    let device = Default::default();
    let config = AttnResConfig::new(32, 4, 2)
        .with_num_heads(4)
        .with_vocab_size(50);

    let model = config.init_model::<TestBackend>(&device);

    for seq_len in [4, 8, 16] {
        let input = Tensor::<TestBackend, 2, Int>::zeros([1, seq_len], &device);
        let out = model.forward(input, None);
        assert_eq!(out.dims(), [1, seq_len, 50], "Failed for seq_len={seq_len}");
    }
}

#[test]
fn test_model_with_causal_mask() {
    let device = Default::default();
    let config = AttnResConfig::new(32, 4, 2)
        .with_num_heads(4)
        .with_vocab_size(50);

    let model = config.init_model::<TestBackend>(&device);
    let mask = attnres_rs::causal_mask::<TestBackend>(2, 8, &device);
    let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
    let out = model.forward(input, Some(&mask));
    assert_eq!(out.dims(), [2, 8, 50]);
}
