/// Property-based tests for attnres-rs using proptest.
///
/// Tests algebraic properties that must hold for any valid input.
use attnres_rs::AttnResConfig;
use burn::backend::NdArray;
use burn::prelude::*;
use burn::tensor::Distribution;
use proptest::prelude::*;

type TestBackend = NdArray;

proptest! {
    /// AttnRes output should be a convex combination of inputs
    /// when pseudo-query is zero (uniform weights).
    ///
    /// For uniform weights, output = mean of sources.
    /// Mean always lies within [min, max] of the sources element-wise.
    #[test]
    fn output_bounded_by_sources(
        num_blocks in 1_usize..5,
        batch in 1_usize..3,
        seq_len in 1_usize..9,
    ) {
        let d_model = 16;
        let device = Default::default();
        let config = AttnResConfig::new(d_model, 12, num_blocks);
        let op = config.init_op::<TestBackend>(&device);

        let mut all_sources: Vec<Tensor<TestBackend, 3>> = Vec::new();
        let blocks: Vec<_> = (0..num_blocks)
            .map(|_| {
                let t = Tensor::random(
                    [batch, seq_len, d_model],
                    Distribution::Uniform(-1.0, 1.0),
                    &device,
                );
                all_sources.push(t.clone());
                t
            })
            .collect();
        let partial = Tensor::random(
            [batch, seq_len, d_model],
            Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        all_sources.push(partial.clone());

        let output = op.forward(&blocks, &partial);

        // Compute element-wise min and max across all sources
        let stacked: Tensor<TestBackend, 4> = Tensor::stack(all_sources, 0); // [N+1, B, T, D]
        let min_vals = stacked.clone().min_dim(0).squeeze_dim::<3>(0); // [B, T, D]
        let max_vals = stacked.max_dim(0).squeeze_dim::<3>(0); // [B, T, D]

        // Output should be >= min and <= max (within tolerance)
        let below_min: f32 = (min_vals - output.clone())
            .clamp_min(0.0)
            .max()
            .into_scalar();
        let above_max: f32 = (output - max_vals)
            .clamp_min(0.0)
            .max()
            .into_scalar();

        prop_assert!(
            below_min < 1e-3,
            "Output below min by {below_min}"
        );
        prop_assert!(
            above_max < 1e-3,
            "Output above max by {above_max}"
        );
    }

    /// Output shape should always match input shape regardless of num_blocks.
    #[test]
    fn output_shape_matches_input(
        num_blocks in 1_usize..6,
        batch in 1_usize..4,
        seq_len in 1_usize..9,
    ) {
        let d_model = 16;
        let device = Default::default();
        let config = AttnResConfig::new(d_model, 12, num_blocks);
        let op = config.init_op::<TestBackend>(&device);

        let blocks: Vec<_> = (0..num_blocks)
            .map(|_| {
                Tensor::random(
                    [batch, seq_len, d_model],
                    Distribution::Uniform(-1.0, 1.0),
                    &device,
                )
            })
            .collect();
        let partial = Tensor::random(
            [batch, seq_len, d_model],
            Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let output = op.forward(&blocks, &partial);
        prop_assert_eq!(output.dims(), [batch, seq_len, d_model]);
    }

    /// Zero-init AttnRes with identical sources should return that source.
    /// This is the identity property: if all inputs are the same, the output
    /// must equal that input regardless of num_blocks.
    #[test]
    fn identical_sources_produce_identical_output(
        num_blocks in 1_usize..5,
        batch in 1_usize..3,
        seq_len in 1_usize..5,
    ) {
        let d_model = 16;
        let device = Default::default();
        let config = AttnResConfig::new(d_model, 12, num_blocks);
        let op = config.init_op::<TestBackend>(&device);

        // All sources are the same tensor
        let source = Tensor::random(
            [batch, seq_len, d_model],
            Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let blocks: Vec<_> = (0..num_blocks).map(|_| source.clone()).collect();

        let output = op.forward(&blocks, &source);
        let diff: f32 = (output - source).abs().max().into_scalar();
        prop_assert!(
            diff < 1e-3,
            "Identical sources should produce that source, diff={diff}"
        );
    }

    /// AttnRes output should be finite (no NaN/Inf) for any reasonable input.
    #[test]
    fn output_is_always_finite(
        num_blocks in 1_usize..4,
        batch in 1_usize..3,
        seq_len in 1_usize..5,
    ) {
        let d_model = 16;
        let device = Default::default();
        let config = AttnResConfig::new(d_model, 12, num_blocks);
        let op = config.init_op::<TestBackend>(&device);

        let blocks: Vec<_> = (0..num_blocks)
            .map(|_| {
                Tensor::random(
                    [batch, seq_len, d_model],
                    Distribution::Uniform(-10.0, 10.0),
                    &device,
                )
            })
            .collect();
        let partial = Tensor::random(
            [batch, seq_len, d_model],
            Distribution::Uniform(-10.0, 10.0),
            &device,
        );

        let output = op.forward(&blocks, &partial);
        let has_nan: bool = output.clone().is_nan().any().into_scalar();
        let has_inf: bool = output.is_inf().any().into_scalar();
        prop_assert!(!has_nan, "Output contains NaN");
        prop_assert!(!has_inf, "Output contains Inf");
    }
}
