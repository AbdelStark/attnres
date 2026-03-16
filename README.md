# attnres-rs

**The first Rust implementation of [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals) from MoonshotAI/Kimi.**

A drop-in replacement for standard residual connections in Transformers that enables each layer to selectively aggregate earlier representations via learned, input-dependent attention over depth.

Built with [burn](https://github.com/tracel-ai/burn). Runs on CUDA, Metal, wgpu, and CPU.

## Why

Standard residual connections accumulate all layer outputs with fixed unit weights. As depth grows, this dilutes each layer's contribution and causes hidden-state magnitudes to grow unboundedly.

**AttnRes** replaces this with softmax attention over depth: each layer gets selective, content-aware access to all earlier representations. The result is a **1.25x compute advantage** with **< 2% inference overhead**.

## Quick Start

```rust
use attnres_rs::{AttnResConfig, AttnResTransformer};
use burn::backend::NdArray;

let config = AttnResConfig::new(768, 24, 8);  // d=768, 24 layers, 8 blocks
let model: AttnResTransformer<NdArray> = config.init(&device);
let output = model.forward(input_ids, Some(&mask));
```

## Paper

> **Attention Residuals** — Kimi Team (MoonshotAI), 2026
> [Paper PDF](https://github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf) | [Official Repo](https://github.com/MoonshotAI/Attention-Residuals) | [Announcement Tweet](https://x.com/kimi_moonshot/status/2033378587878072424)

## License

Apache 2.0
