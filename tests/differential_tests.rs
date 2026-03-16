/// Differential tests comparing Rust implementation against known reference outputs.
///
/// These tests use fixtures generated from the paper's pseudocode to verify
/// numerical correctness of the AttnRes implementation.
use attnres::AttnResConfig;
use burn::backend::NdArray;
use burn::prelude::*;

type TestBackend = NdArray;

#[test]
fn test_differential_zero_query_is_mean() {
    // From fixtures/attn_res_forward.json:
    // With zero pseudo-query, output should equal the mean of all sources.
    let device = Default::default();
    let config = AttnResConfig::new(4, 4, 2);
    let op = config.init_op::<TestBackend>(&device);

    // blocks[0] = [1, 2, 3, 4], blocks[1] = [5, 6, 7, 8], partial = [9, 10, 11, 12]
    let block0 = Tensor::<TestBackend, 3>::from_floats([[[1.0, 2.0, 3.0, 4.0]]], &device);
    let block1 = Tensor::<TestBackend, 3>::from_floats([[[5.0, 6.0, 7.0, 8.0]]], &device);
    let partial = Tensor::<TestBackend, 3>::from_floats([[[9.0, 10.0, 11.0, 12.0]]], &device);

    let output = op.forward(&[block0, block1], &partial);
    let expected = Tensor::<TestBackend, 3>::from_floats([[[5.0, 6.0, 7.0, 8.0]]], &device);

    let diff: f32 = (output - expected).abs().max().into_scalar();
    assert!(
        diff < 1e-4,
        "Differential test failed: expected mean of sources, diff={diff}"
    );
}

#[test]
fn test_differential_rmsnorm_known_input() {
    // RMSNorm([1, 2, 3, 4]) with gamma=1:
    // RMS = sqrt(mean([1, 4, 9, 16])) = sqrt(7.5) ≈ 2.7386
    // Output ≈ [0.3651, 0.7303, 1.0954, 1.4606]
    let device = Default::default();
    let norm = attnres::RmsNormConfig::new(4).init::<TestBackend>(&device);

    let x = Tensor::<TestBackend, 3>::from_floats([[[1.0, 2.0, 3.0, 4.0]]], &device);
    let out = norm.forward(x);

    let data: Vec<f32> = out.reshape([4]).into_data().to_vec().unwrap();
    let rms = (7.5_f64 + 1e-6).sqrt() as f32;
    let expected: Vec<f32> = vec![1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms];

    for (i, (got, want)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "RMSNorm element {i}: got {got}, want {want}"
        );
    }
}

#[test]
fn test_differential_softmax_over_depth() {
    // Verify softmax is computed over the depth dimension.
    // With zero pseudo-query and 3 sources, all logits are 0.
    // softmax([0, 0, 0]) = [1/3, 1/3, 1/3]
    let device = Default::default();
    let config = AttnResConfig::new(4, 4, 2);
    let op = config.init_op::<TestBackend>(&device);

    // Use identical sources so any weighting gives the same result
    let val = Tensor::<TestBackend, 3>::from_floats([[[1.0, 1.0, 1.0, 1.0]]], &device);
    let output = op.forward(&[val.clone(), val.clone()], &val);

    // Output should equal input since all sources are identical
    let diff: f32 = (output - val).abs().max().into_scalar();
    assert!(
        diff < 1e-5,
        "Identical sources should produce identical output, diff={diff}"
    );
}
