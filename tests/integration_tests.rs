/// Integration tests for attnres-rs.
///
/// Tests end-to-end model behavior including forward passes,
/// autodiff gradient flow, and model configuration.
use attnres_rs::{causal_mask, AttnResConfig, AttnResTransformer};
use burn::backend::Autodiff;
use burn::backend::NdArray;
use burn::config::Config;
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

// ========================
// Two-phase Inference Integration Tests
// ========================

#[test]
fn test_two_phase_matches_standard_integration() {
    // Test with a deeper model to exercise multiple block boundaries
    let device = Default::default();
    let config = AttnResConfig::new(32, 12, 3)
        .with_num_heads(4)
        .with_vocab_size(50);

    let model: AttnResTransformer<TestBackend> = config.init_model(&device);
    let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);

    let standard = model.forward(input.clone(), None);
    let two_phase = model.forward_two_phase(input, None);

    let diff: f32 = (standard - two_phase).abs().max().into_scalar();
    assert!(
        diff < 1e-3,
        "Two-phase should match standard for deep model, diff={diff}"
    );
}

#[test]
fn test_two_phase_with_causal_mask() {
    let device = Default::default();
    let config = AttnResConfig::new(32, 8, 2)
        .with_num_heads(4)
        .with_vocab_size(50);

    let model: AttnResTransformer<TestBackend> = config.init_model(&device);
    let mask = causal_mask::<TestBackend>(1, 8, &device);
    let input = Tensor::<TestBackend, 2, Int>::zeros([1, 8], &device);

    let standard = model.forward(input.clone(), Some(&mask));
    let two_phase = model.forward_two_phase(input, Some(&mask));

    let diff: f32 = (standard - two_phase).abs().max().into_scalar();
    assert!(
        diff < 1e-3,
        "Two-phase with mask should match standard, diff={diff}"
    );
}

// ========================
// Serialization Integration Tests
// ========================

#[test]
fn test_save_load_preserves_forward() {
    let device = Default::default();
    let config = AttnResConfig::new(32, 4, 2)
        .with_num_heads(4)
        .with_vocab_size(50);

    let model: AttnResTransformer<TestBackend> = config.init_model(&device);
    let input = Tensor::<TestBackend, 2, Int>::zeros([1, 8], &device);
    let out_before = model.forward(input.clone(), None);

    let path = std::env::temp_dir().join("attnres_integ_save_load");
    let path_str = path.to_str().unwrap();
    model.save(path_str, &device).expect("Failed to save");

    let loaded: AttnResTransformer<TestBackend> =
        AttnResTransformer::load(path_str, &config, &device).expect("Failed to load");
    let out_after = loaded.forward(input, None);

    let diff: f32 = (out_before - out_after).abs().max().into_scalar();
    assert!(
        diff < 1e-6,
        "Loaded model should match saved model, diff={diff}"
    );

    let _ = std::fs::remove_file(format!("{path_str}.mpk"));
}

#[test]
fn test_config_save_load_then_init_model() {
    let device = Default::default();
    let config = AttnResConfig::new(32, 8, 4)
        .with_num_heads(4)
        .with_vocab_size(100)
        .with_dropout(0.1);

    let path = std::env::temp_dir().join("attnres_integ_config.json");
    config.save(&path).expect("Failed to save config");

    let loaded_config = AttnResConfig::load(&path).expect("Failed to load config");
    let _ = loaded_config.init_model::<TestBackend>(&device);

    assert_eq!(config.d_model, loaded_config.d_model);
    assert_eq!(config.num_layers, loaded_config.num_layers);
    assert_eq!(config.num_blocks, loaded_config.num_blocks);

    let _ = std::fs::remove_file(&path);
}
