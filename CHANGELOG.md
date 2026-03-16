# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Improved `SerializationError` with path context in `SaveFailed` and `LoadFailed` variants.
- Better doc comments on `init_op`, `init_layer`, `MultiHeadAttentionConfig::init`,
  and `FeedForwardConfig::init`.
- Documented `phase1_batched` norm sharing behavior.

### Added
- Compact serialization roundtrip test.
- Error path test for loading nonexistent model files.
- `SerializationError::Display` test.

### Fixed
- ROADMAP.md incorrectly described FeedForward as "SwiGLU-style MLP" (it uses GELU).

## [0.2.0] - 2026-03-15

### Added
- Model weight save/load via burn's record system (NamedMpk, binary, compact/half-precision).
- Config save/load via JSON (`AttnResConfig::save` / `AttnResConfig::load`).
- Two-phase inference integrated into model via `forward_two_phase` method.
- Layer accessor methods (`forward_attn_sublayer`, `forward_mlp_sublayer`, `attn_res_ops`).
- `forward_hidden` method for extracting hidden states without LM head.

## [0.1.0] - 2026-03-14

### Added
- Core AttnRes operation (`AttnResOp`) with depth-wise softmax attention.
- Block state tracking (`BlockState`) for cumulative block representations.
- RMSNorm implementation for key normalization.
- Transformer layer with dual AttnRes (`AttnResLayer`).
- Full transformer model (`AttnResTransformer`) with embedding, LM head, causal masking.
- Multi-head self-attention (`MultiHeadAttention`).
- Feed-forward MLP with GELU activation (`FeedForward`).
- Two-phase inference primitives (`phase1_batched`, `online_softmax_merge`).
- Validated configuration with builder pattern (`AttnResConfig`).
- Zero initialization of pseudo-query vectors.
- CI pipeline (test, clippy, fmt, build-examples).
- 3 examples: `train_tiny`, `compare_residuals`, `visualize_weights`.
- Criterion benchmarks.
- Property-based tests with proptest.
- Differential tests against known reference outputs.
- Interactive web demo (WASM + Vite).
