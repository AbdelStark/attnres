---
name: testing
description: Testing strategy for attnres covering unit tests, differential tests against PyTorch, property-based tests with proptest, integration tests, and criterion benchmarks. Activate when writing any test, debugging test failures, or validating algorithm correctness.
prerequisites: cargo test, proptest crate, criterion crate
---

# Testing

<purpose>
Defines the testing approach for attnres. Covers five test categories, each with specific patterns and conventions. All tests use NdArray backend for determinism.
</purpose>

<context>
— Unit tests: inline in source files via `#[cfg(test)]` AND in tests/unit_tests.rs
— Differential tests: tests/differential_tests.rs — compare against PyTorch reference outputs in fixtures/
— Property tests: tests/property_tests.rs — proptest for algebraic properties
— Integration tests: tests/integration_tests.rs — training comparison
— Benchmarks: benches/attn_res_benchmark.rs — criterion
— Always use NdArray backend for tests (deterministic, no GPU required)
— Use small dimensions: d_model=64, seq_len=16, batch=2
</context>

<procedure>
Writing a new test:
1. Identify test category (unit/differential/property/integration/bench)
2. Choose the correct file location
3. Use NdArray backend: `type TestBackend = burn::backend::NdArray;`
4. Create tensors with known values or controlled random seeds
5. Assert with floating-point tolerance: `assert!((a - b).abs() < 1e-5)`
6. Run with `cargo test test_name` to verify
7. Run full suite: `cargo test --all-features`

Running tests:
1. `cargo test` — run all tests
2. `cargo test test_name` — run specific test
3. `cargo test -- --nocapture` — show println output
4. `cargo bench` — run criterion benchmarks
</procedure>

<patterns>
<do>
  — Use `burn::backend::NdArray` as the test backend
  — Create helper functions for common test setup (e.g., `make_config`, `make_random_blocks`)
  — Test tensor shapes explicitly: `assert_eq!(output.shape(), [batch, seq_len, d_model])`
  — Test known mathematical properties (softmax sums to 1, convex combination bounds)
  — Use descriptive test names: `test_zero_init_produces_uniform_weights`
  — Add `#[ignore]` to slow tests (training tests) with comment explaining why
</do>
<dont>
  — Don't use exact float equality (`==`) — use tolerance-based comparison
  — Don't use large dimensions in tests — keep fast (d=64, T=16, B=2)
  — Don't skip testing edge cases: single block (N=1), full AttnRes (N=L), first layer, last layer
  — Don't test implementation details — test behavior and contracts
</dont>
</patterns>

<examples>
Example: Unit test for AttnResOp

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_zero_init_uniform_weights() {
        let device = Default::default();
        let config = AttnResConfig::new(64, 12, 4);
        let op: AttnResOp<TestBackend> = config.init_op(&device);

        let blocks = vec![
            Tensor::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device),
            Tensor::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device),
        ];
        let partial = Tensor::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device);

        let output = op.forward(&blocks, &partial);
        // With zero queries, output should approximate mean of all sources
        let expected = (blocks[0].clone() + blocks[1].clone() + partial) / 3.0;
        let diff = (output - expected).abs().max();
        assert!(diff.into_scalar() < 1e-4);
    }
}
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| Test passes locally, fails in CI | Non-deterministic random seed | Use fixed seed or `Tensor::from_data` with known values |
| "thread panicked" with no useful message | Tensor shape mismatch in burn | Add shape assertions before the failing operation |
| Differential test tolerance failure | Floating-point accumulation order differs | Increase tolerance to 1e-4 or use relative tolerance |
| proptest shrinking takes forever | Search space too large | Constrain proptest strategies to small ranges |
</troubleshooting>

<references>
— spec.md §6: Full testing strategy specification
— tests/: Test file locations
— fixtures/: PyTorch reference data
</references>
