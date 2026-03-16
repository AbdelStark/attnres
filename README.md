# attnres-rs

**The first Rust implementation of [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals) from MoonshotAI/Kimi.**

A drop-in replacement for standard residual connections in Transformers that enables each layer to selectively aggregate earlier representations via learned, input-dependent attention over depth.

Built with [burn](https://github.com/tracel-ai/burn). Runs on CUDA, Metal, wgpu, and CPU.

## Why

Standard residual connections accumulate all layer outputs with fixed unit weights. As depth grows, this dilutes each layer's contribution and causes hidden-state magnitudes to grow unboundedly.

**AttnRes** replaces this with softmax attention over depth: each layer gets selective, content-aware access to all earlier representations. The result is a **1.25x compute advantage** with **< 2% inference overhead**.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
attnres-rs = "0.1"
burn = { version = "0.20", features = ["ndarray"] }
```

```rust
use attnres_rs::{AttnResConfig, AttnResTransformer};
use burn::prelude::*;
use burn::backend::NdArray;

type B = NdArray;

let device = Default::default();
let config = AttnResConfig::new(128, 8, 2)  // d_model=128, 8 sublayers (4 transformer layers), 2 blocks
    .with_num_heads(4)
    .with_vocab_size(1000);

let model: AttnResTransformer<B> = config.init_model(&device);
let input_ids = Tensor::<B, 2, Int>::zeros([1, 16], &device);
let logits = model.forward(input_ids, None);
// logits shape: [1, 16, 1000]
```

## Key Concepts

- **Block AttnRes**: Groups layers into N blocks and computes depth attention over block representations. More practical than Full AttnRes (lower overhead).
- **Full AttnRes**: Set `num_blocks = num_layers` for per-layer depth attention.
- **Zero initialization**: Pseudo-query vectors start at zero, so AttnRes begins as standard residual (uniform averaging) and gradually differentiates during training.
- **Two AttnRes per layer**: Each transformer layer applies AttnRes before both the self-attention and MLP sublayers.

## Architecture

```
AttnResTransformer
  ├── Embedding
  ├── AttnResLayer (x num_transformer_layers)
  │     ├── AttnResOp (before self-attention)
  │     ├── RmsNorm + MultiHeadAttention
  │     ├── AttnResOp (before MLP)
  │     └── RmsNorm + FeedForward
  ├── Final RmsNorm
  └── LM Head (Linear)
```

**Note on `num_layers`**: This parameter counts *sublayers* (each transformer layer = 2 sublayers: attention + MLP). So `num_layers=8` means 4 transformer layers.

## Web Demo

An interactive browser-based demo runs the core AttnRes algorithm via Rust compiled to WASM:

```bash
cd web-demo
npm install
npm run build:wasm   # Compile Rust → WASM (requires wasm-pack)
npm run dev          # Start dev server at localhost:5173
```

Features: configurable model parameters, live depth attention heatmaps, training simulation with loss curves, standard vs AttnRes comparison. No GPU required — runs entirely in the browser.

## Examples

```bash
cargo run --example train_tiny          # Train a small model on synthetic data
cargo run --example compare_residuals   # Compare AttnRes vs standard residuals
cargo run --example visualize_weights   # Visualize depth attention patterns
```

## Development

```bash
cargo build                        # Build
cargo test --all-features          # Run all tests
cargo clippy -- -D warnings        # Lint
cargo fmt                          # Format
cargo bench                        # Benchmarks
```

## Current Status

**Alpha** (v0.2.0). Core algorithm implemented and tested with 87 passing tests (unit, differential, property-based, integration, doctest). Built on burn 0.20. Serialization (NamedMpk, binary, compact/half-precision) and two-phase inference integrated. Suitable for research and experimentation. Not yet suitable for production training at scale.

Known limitations:
- No PyTorch checkpoint import (safetensors format)
- NdArray backend only tested; GPU backends (wgpu, CUDA, Metal) untested
- No distributed training support
- No KV-cache for autoregressive generation

See [ROADMAP.md](ROADMAP.md) for planned features and [CHANGELOG.md](CHANGELOG.md) for release history.

## Paper

> **Attention Residuals** -- Kimi Team (MoonshotAI), 2026
> [Paper PDF](https://github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf) | [Official Repo](https://github.com/MoonshotAI/Attention-Residuals)

## License

MIT
