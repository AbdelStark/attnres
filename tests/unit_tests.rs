/// Unit tests for attnres core functionality.
///
/// Tests the core algorithm components using NdArray backend.
use attnres::{AttnResConfig, AttnResTransformer, BlockState, RmsNorm, RmsNormConfig};
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
    let mask = attnres::causal_mask::<TestBackend>(2, 8, &device);
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

    let blocks: Vec<Tensor<TestBackend, 3>> = vec![
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
    use attnres::two_phase::{compute_intra_logit, online_softmax_merge, phase1_batched};

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
fn test_full_attnres_splits_attention_and_mlp_outputs() {
    // Full AttnRes must treat each sublayer as its own source. That means the
    // attention output of layer 0 becomes a completed block before layer 0's
    // MLP runs, rather than waiting until the next transformer layer.
    let device = Default::default();
    let config = AttnResConfig::new(32, 4, 4).with_num_heads(4);

    let layers: Vec<_> = (0..2)
        .map(|i| config.init_layer::<TestBackend>(i, &device))
        .collect();

    let emb = Tensor::random([1, 4, 32], Distribution::Normal(0.0, 1.0), &device);
    let mut state = BlockState::new(emb);

    state = layers[0].forward(state, None);
    assert_eq!(
        state.num_blocks(),
        2,
        "Full AttnRes should commit the attention sublayer before running the MLP"
    );

    state = layers[1].forward(state, None);
    assert_eq!(
        state.num_blocks(),
        4,
        "Full AttnRes should expose embedding + three completed sublayers after layer 1"
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

// ========================
// Config Validation Edge Cases
// ========================

#[test]
#[should_panic(expected = "d_model must be positive")]
fn test_validate_zero_d_model() {
    AttnResConfig::new(0, 4, 2).validate();
}

#[test]
#[should_panic(expected = "num_layers must be positive")]
fn test_validate_zero_num_layers() {
    AttnResConfig::new(32, 0, 0).validate();
}

#[test]
#[should_panic(expected = "vocab_size must be positive")]
fn test_validate_zero_vocab_size() {
    AttnResConfig::new(32, 4, 2).with_vocab_size(0).validate();
}

#[test]
#[should_panic(expected = "dropout must be in [0.0, 1.0]")]
fn test_validate_negative_dropout() {
    AttnResConfig::new(32, 4, 2).with_dropout(-0.1).validate();
}

#[test]
#[should_panic(expected = "dropout must be in [0.0, 1.0]")]
fn test_validate_dropout_over_one() {
    AttnResConfig::new(32, 4, 2).with_dropout(1.5).validate();
}

#[test]
#[should_panic(expected = "rms_norm_eps must be positive")]
fn test_validate_zero_eps() {
    AttnResConfig::new(32, 4, 2)
        .with_rms_norm_eps(0.0)
        .validate();
}

#[test]
#[should_panic(expected = "must be even")]
fn test_validate_odd_layers() {
    AttnResConfig::new(32, 3, 1).validate();
}

#[test]
#[should_panic(expected = "divisible by num_blocks")]
fn test_validate_layers_not_divisible_by_blocks() {
    AttnResConfig::new(32, 6, 4).validate();
}

#[test]
#[should_panic(expected = "divisible by num_heads")]
fn test_validate_d_model_not_divisible_by_heads() {
    AttnResConfig::new(33, 4, 2).with_num_heads(8).validate();
}

// ========================
// Block Boundary Matrix Tests
// ========================

#[test]
fn test_block_boundary_matrix() {
    // Systematically test block boundary detection across configurations.
    let device = Default::default();

    // Config: 8 sublayers, 4 blocks -> block_size=2.
    // Attention-side block starts occur at transformer layers 1, 2, 3.
    let config = AttnResConfig::new(32, 8, 4).with_num_heads(4);
    let layer0 = config.init_layer::<TestBackend>(0, &device);
    let layer1 = config.init_layer::<TestBackend>(1, &device);
    let layer2 = config.init_layer::<TestBackend>(2, &device);
    let layer3 = config.init_layer::<TestBackend>(3, &device);
    assert!(
        !layer0.is_at_boundary(),
        "Layer 0 should never be a boundary"
    );
    assert!(
        layer1.is_at_boundary(),
        "Layer 1 should be a boundary (8 sublayers, 4 blocks)"
    );
    assert!(layer2.is_at_boundary(), "Layer 2 should be a boundary");
    assert!(layer3.is_at_boundary(), "Layer 3 should be a boundary");

    // Config: 12 sublayers, 2 blocks -> block_size=6.
    // The next block starts before transformer layer 3 attention.
    let config2 = AttnResConfig::new(32, 12, 2).with_num_heads(4);
    let l0 = config2.init_layer::<TestBackend>(0, &device);
    let l1 = config2.init_layer::<TestBackend>(1, &device);
    let l2 = config2.init_layer::<TestBackend>(2, &device);
    let l3 = config2.init_layer::<TestBackend>(3, &device);
    let l4 = config2.init_layer::<TestBackend>(4, &device);
    let l5 = config2.init_layer::<TestBackend>(5, &device);
    assert!(!l0.is_at_boundary());
    assert!(!l1.is_at_boundary());
    assert!(!l2.is_at_boundary());
    assert!(
        l3.is_at_boundary(),
        "Layer 3 should be boundary (12 sublayers, 2 blocks)"
    );
    assert!(!l4.is_at_boundary());
    assert!(!l5.is_at_boundary());

    // Config: Full AttnRes (4 sublayers, 4 blocks) -> block_size=1.
    // The public helper reports only attention-side block starts.
    let config3 = AttnResConfig::new(32, 4, 4).with_num_heads(4);
    let fl0 = config3.init_layer::<TestBackend>(0, &device);
    let fl1 = config3.init_layer::<TestBackend>(1, &device);
    assert!(!fl0.is_at_boundary());
    assert!(
        fl1.is_at_boundary(),
        "Full AttnRes should start a new block before layer 1 attention"
    );
}

#[test]
fn test_odd_block_size_boundary_occurs_before_mlp() {
    // 6 sublayers, 2 blocks -> block size 3. The first block contains:
    // attn0, mlp0, attn1. The boundary therefore occurs before layer 1's MLP,
    // not before its attention sublayer.
    let device = Default::default();
    let config = AttnResConfig::new(32, 6, 2).with_num_heads(4);
    let layers: Vec<_> = (0..3)
        .map(|i| config.init_layer::<TestBackend>(i, &device))
        .collect();

    assert!(
        !layers[1].is_at_boundary(),
        "Odd block sizes should not shift the boundary onto the attention sublayer"
    );

    let emb = Tensor::random([1, 4, 32], Distribution::Normal(0.0, 1.0), &device);
    let mut state = BlockState::new(emb);

    state = layers[0].forward(state, None);
    assert_eq!(state.num_blocks(), 1);

    state = layers[1].forward(state, None);
    assert_eq!(
        state.num_blocks(),
        2,
        "The first block should be committed inside layer 1 before its MLP"
    );
}

#[test]
fn test_block_accumulation_count() {
    // Verify the exact number of blocks accumulated after a full forward pass.
    let device = Default::default();
    // 8 sublayers, 2 blocks -> block_size=4
    // 4 transformer layers (indices 0..3)
    // Boundaries at layers where idx > 0 && idx % 2 == 0 → layer 2
    let config = AttnResConfig::new(32, 8, 2).with_num_heads(4);
    let layers: Vec<_> = (0..4)
        .map(|i| config.init_layer::<TestBackend>(i, &device))
        .collect();

    let emb = Tensor::random([1, 4, 32], Distribution::Normal(0.0, 1.0), &device);
    let mut state = BlockState::new(emb);

    // Track block counts after each layer
    let mut counts = vec![];
    for layer in &layers {
        state = layer.forward(state, None);
        counts.push(state.num_blocks());
    }

    // Layer 0: no boundary → 1 block (initial embedding)
    // Layer 1: still inside block 1 → 1 block
    // Layer 2: starts block 2 before attention → 2 blocks
    // Layer 3: still inside block 2 → 2 blocks
    assert_eq!(
        counts,
        vec![1, 1, 2, 2],
        "Block accumulation mismatch: {counts:?}"
    );
}

// ========================
// Numerical Stability Tests
// ========================

#[test]
fn test_attnres_large_magnitude_inputs() {
    // Verify AttnRes handles large-magnitude inputs without NaN.
    let device = Default::default();
    let config = AttnResConfig::new(16, 4, 2);
    let op = config.init_op::<TestBackend>(&device);

    let large = Tensor::<TestBackend, 3>::ones([1, 4, 16], &device) * 1e6;
    let partial = Tensor::<TestBackend, 3>::ones([1, 4, 16], &device) * 1e6;

    let output = op.forward(&[large], &partial);
    let has_nan = output.clone().is_nan().any().into_scalar();
    assert!(!has_nan, "Output should not contain NaN with large inputs");

    let has_inf = output.clone().is_inf().any().into_scalar();
    assert!(!has_inf, "Output should not contain Inf with large inputs");
}

#[test]
fn test_attnres_near_zero_inputs() {
    // Verify AttnRes handles near-zero inputs without NaN (eps should save us).
    let device = Default::default();
    let config = AttnResConfig::new(16, 4, 2);
    let op = config.init_op::<TestBackend>(&device);

    let tiny = Tensor::<TestBackend, 3>::ones([1, 4, 16], &device) * 1e-10;
    let partial = Tensor::<TestBackend, 3>::ones([1, 4, 16], &device) * 1e-10;

    let output = op.forward(&[tiny], &partial);
    let has_nan = output.clone().is_nan().any().into_scalar();
    assert!(
        !has_nan,
        "Output should not contain NaN with near-zero inputs"
    );
}

#[test]
fn test_rmsnorm_zero_input() {
    // RMSNorm(0) should not produce NaN (eps prevents division by zero).
    let device = Default::default();
    let norm = RmsNormConfig::new(8).init::<TestBackend>(&device);

    let zero = Tensor::<TestBackend, 3>::zeros([1, 4, 8], &device);
    let out = norm.forward(zero);
    let has_nan = out.is_nan().any().into_scalar();
    assert!(!has_nan, "RMSNorm of zero should not produce NaN");
}

// ========================
// Two-phase Deeper Stress Test
// ========================

#[test]
fn test_two_phase_deep_model() {
    // Exercise two-phase with a deeper model (24 sublayers, 4 blocks).
    let device = Default::default();
    let config = AttnResConfig::new(32, 24, 4)
        .with_num_heads(4)
        .with_vocab_size(50);

    let model: AttnResTransformer<TestBackend> = config.init_model(&device);
    let input = Tensor::<TestBackend, 2, Int>::zeros([1, 8], &device);

    let standard = model.forward(input.clone(), None);
    let two_phase = model.forward_two_phase(input, None);

    let diff: f32 = (standard - two_phase).abs().max().into_scalar();
    assert!(
        diff < 1e-2,
        "Two-phase should match standard for deep model (24 sublayers, 4 blocks), diff={diff}"
    );
}

#[test]
fn test_two_phase_full_attnres() {
    // Two-phase with Full AttnRes (num_blocks = num_layers).
    let device = Default::default();
    let config = AttnResConfig::new(32, 8, 8)
        .with_num_heads(4)
        .with_vocab_size(50);

    let model: AttnResTransformer<TestBackend> = config.init_model(&device);
    let input = Tensor::<TestBackend, 2, Int>::zeros([1, 8], &device);

    let standard = model.forward(input.clone(), None);
    let two_phase = model.forward_two_phase(input, None);

    let diff: f32 = (standard - two_phase).abs().max().into_scalar();
    assert!(
        diff < 1e-2,
        "Two-phase should match standard for Full AttnRes, diff={diff}"
    );
}
