# attnres-rs Roadmap

## Current Phase: Alpha (v0.1.0)

Core algorithm implemented and tested. Suitable for research and experimentation.

---

## v0.1.0 — Core Implementation ✅

- [x] AttnResOp: Block AttnRes forward pass with depth-wise softmax
- [x] BlockState: Cumulative block representation tracking
- [x] RMSNorm: Custom implementation for key normalization
- [x] AttnResLayer: Transformer layer with dual AttnRes (pre-attention + pre-MLP)
- [x] AttnResTransformer: Full model with embedding, LM head, causal masking
- [x] MultiHeadAttention: Standard multi-head self-attention
- [x] FeedForward: SwiGLU-style MLP
- [x] TwoPhase: Two-phase inference optimization (standalone)
- [x] Config: Validated configuration with builder pattern
- [x] Zero initialization of pseudo-query vectors
- [x] CI pipeline (test, clippy, fmt, build-examples)
- [x] 3 examples (train_tiny, compare_residuals, visualize_weights)
- [x] Criterion benchmarks
- [x] Upgrade to burn 0.20

## v0.2.0 — Serialization & Inference ✅

- [x] Model weight save/load (NamedMpk default, binary, compact/half-precision formats)
- [x] Config save/load (JSON via burn's Config trait)
- [x] Integrate two-phase inference into main `forward_two_phase` method
- [x] Layer accessor methods for two-phase inference components
- [x] 66 tests passing (unit, differential, property-based, integration, doctest)
- [ ] Pre-trained weight loading from PyTorch checkpoints
- [ ] Model export utilities

## v0.3.0 — GPU & Performance (Planned)

- [ ] Test and validate wgpu backend
- [ ] Test and validate CUDA backend (via burn-cuda)
- [ ] Test and validate Metal backend (via burn-tch)
- [ ] GPU-specific benchmarks
- [ ] Memory optimization for large models
- [ ] KV-cache support for autoregressive generation

## v0.4.0 — Production Readiness (Planned)

- [ ] Distributed training support
- [ ] Mixed precision (fp16/bf16) training
- [ ] Gradient checkpointing for memory efficiency
- [ ] Comprehensive documentation with examples
- [ ] Publish to crates.io

## Future Ideas

- Full AttnRes mode (per-layer, not per-block) benchmarks at scale
- Integration examples with popular Rust inference frameworks
- ONNX export
- Quantization support (INT8/INT4)
- Streaming/chunked inference for long sequences
