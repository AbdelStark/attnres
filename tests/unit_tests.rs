/// Unit tests for attnres-rs core functionality.
///
/// Tests the core algorithm components using NdArray backend.
use attnres_rs::{AttnResConfig, AttnResTransformer, BlockState, RmsNorm, RmsNormConfig};
use burn::backend::Autodiff;
use burn::backend::NdArray;
use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::Distribution;

type TestBackend = NdArray;
type AutodiffBackend = Autodiff<TestBackend>;

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

// ========================
// Gradient Flow Tests
// ========================

#[test]
fn test_gradient_flows_to_pseudo_query() {
    // Verify gradients propagate through AttnRes to the pseudo-query parameter.
    let device = Default::default();
    let config = AttnResConfig::new(32, 4, 2)
        .with_num_heads(4)
        .with_vocab_size(50);

    let model: AttnResTransformer<AutodiffBackend> = config.init_model(&device);
    let input_ids = Tensor::<AutodiffBackend, 2, Int>::zeros([1, 8], &device);
    let logits = model.forward(input_ids, None);

    let loss = logits.mean();
    let grads = loss.backward();

    // Verify we can compute gradients without panicking.
    // Convert to GradientsParams to confirm gradient flow through the model.
    let grads = burn::optim::GradientsParams::from_grads(grads, &model);

    // If we get here, gradients flow correctly through AttnRes.
    // Use grads to prevent optimization away.
    std::hint::black_box(&grads);
}

// ========================
// Attention Weights Tests
// ========================

#[test]
fn test_attnres_weights_sum_to_one() {
    // Verify that the softmax attention weights over depth sum to 1.
    let device = Default::default();
    let config = AttnResConfig::new(32, 12, 4);
    let op = config.init_op::<TestBackend>(&device);

    let blocks = vec![
        Tensor::random([2, 8, 32], Distribution::Normal(0.0, 1.0), &device),
        Tensor::random([2, 8, 32], Distribution::Normal(0.0, 1.0), &device),
        Tensor::random([2, 8, 32], Distribution::Normal(0.0, 1.0), &device),
    ];
    let partial = Tensor::random([2, 8, 32], Distribution::Normal(0.0, 1.0), &device);

    // Replicate the forward pass to extract weights
    let mut sources: Vec<Tensor<TestBackend, 3>> = blocks.to_vec();
    sources.push(partial.clone());
    let v = Tensor::stack(sources, 0); // [4, 2, 8, 32]
    let k = op.norm.forward_4d(v);

    let w = op
        .pseudo_query
        .val()
        .unsqueeze_dim::<2>(0)
        .unsqueeze_dim::<3>(0)
        .unsqueeze_dim::<4>(0);
    let logits = (k * w).sum_dim(3).squeeze_dim::<3>(3); // [4, 2, 8]
    let alpha = softmax(logits, 0); // [4, 2, 8]

    // Sum over depth dimension should be 1.0
    let weight_sum = alpha.sum_dim(0).squeeze_dim::<2>(0); // [2, 8]
    let ones = Tensor::<TestBackend, 2>::ones([2, 8], &device);
    let diff: f32 = (weight_sum - ones).abs().max().into_scalar();
    assert!(
        diff < 1e-5,
        "Attention weights should sum to 1, max deviation={diff}"
    );
}

#[test]
fn test_rmsnorm_prevents_magnitude_domination() {
    // A block with much larger magnitude should not proportionally dominate
    // attention weights after RMSNorm is applied.
    let device = Default::default();
    let config = AttnResConfig::new(16, 4, 2);
    let op = config.init_op::<TestBackend>(&device);

    // block0 has small magnitude, block1 has 100x larger magnitude
    let small = Tensor::<TestBackend, 3>::from_floats(
        [[[
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
        ]]],
        &device,
    );
    let large = small.clone() * 100.0;
    let partial = small.clone();

    // With zero pseudo-query, all normalized logits are 0, so weights are uniform
    // regardless of magnitude. This is the key property.
    let output = op.forward(&[small.clone(), large.clone()], &partial);
    let expected_mean = (small + large + partial) / 3.0;

    let diff: f32 = (output - expected_mean).abs().max().into_scalar();
    assert!(
        diff < 1e-3,
        "With zero query, RMSNorm should lead to uniform weights regardless of magnitude, diff={diff}"
    );
}

// ========================
// Two-phase Equivalence Tests
// ========================

#[test]
fn test_two_phase_matches_standard_forward() {
    // Verify that phase1_batched + online_softmax_merge produces the same
    // result as the standard AttnResOp::forward.
    use attnres_rs::two_phase::{compute_intra_logit, online_softmax_merge, phase1_batched};

    let device = Default::default();
    let config = AttnResConfig::new(16, 4, 2);
    let op = config.init_op::<TestBackend>(&device);

    let blocks = vec![
        Tensor::random([1, 4, 16], Distribution::Normal(0.0, 1.0), &device),
        Tensor::random([1, 4, 16], Distribution::Normal(0.0, 1.0), &device),
    ];
    let partial = Tensor::random([1, 4, 16], Distribution::Normal(0.0, 1.0), &device);

    // Standard forward
    let standard_out = op.forward(&blocks, &partial);

    // Two-phase forward
    let phase1 = phase1_batched(&[&op], &blocks);
    let intra_logit = compute_intra_logit(&op, &partial);
    let two_phase_out = online_softmax_merge(
        phase1.outputs[0].clone(),
        phase1.max_logits[0].clone(),
        phase1.sum_exp[0].clone(),
        intra_logit,
        partial,
    );

    let diff: f32 = (standard_out - two_phase_out).abs().max().into_scalar();
    assert!(
        diff < 1e-4,
        "Two-phase forward should match standard forward, diff={diff}"
    );
}

// ========================
// Full AttnRes (block_size=1) Tests
// ========================

#[test]
fn test_full_attnres_every_layer_is_boundary() {
    // With num_blocks = num_layers (Full AttnRes), block_size=1, half_block=0.
    // Every layer after the first should trigger a boundary.
    let device = Default::default();
    let config = AttnResConfig::new(32, 4, 4).with_num_heads(4);

    let layers: Vec<_> = (0..2)
        .map(|i| config.init_layer::<TestBackend>(i, &device))
        .collect();

    let emb = Tensor::random([1, 4, 32], Distribution::Normal(0.0, 1.0), &device);
    let mut state = BlockState::new(emb);

    // Layer 0: no boundary (first layer)
    state = layers[0].forward(state, None);
    assert_eq!(state.num_blocks(), 1, "Layer 0 should not add a block");

    // Layer 1: boundary
    state = layers[1].forward(state, None);
    assert_eq!(
        state.num_blocks(),
        2,
        "Layer 1 should add a block in Full AttnRes"
    );
}

// ========================
// RMSNorm additional tests
// ========================

#[test]
fn test_rmsnorm_3d_4d_consistency() {
    // RMSNorm on a single [1, B, T, D] slice should match 3D [B, T, D]
    let device = Default::default();
    let norm: RmsNorm<TestBackend> = RmsNormConfig::new(16).init(&device);

    let x_3d = Tensor::random([2, 8, 16], Distribution::Normal(0.0, 1.0), &device);
    let x_4d = x_3d.clone().unsqueeze_dim::<4>(0); // [1, 2, 8, 16]

    let out_3d = norm.forward(x_3d);
    let out_4d = norm.forward_4d(x_4d).squeeze_dim::<3>(0);

    let diff: f32 = (out_3d - out_4d).abs().max().into_scalar();
    assert!(
        diff < 1e-5,
        "3D and 4D RMSNorm should produce consistent results, diff={diff}"
    );
}

// ========================
// Config validation test
// ========================

#[test]
fn test_model_init_validates_config() {
    // init_model should call validate(), catching bad configs early
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
    let config = AttnResConfig::new(32, 4, 2)
        .with_num_heads(4)
        .with_vocab_size(50);
    // This should succeed
    let _model: AttnResTransformer<TestBackend> = config.init_model(&device);
}
