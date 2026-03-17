# attnres

`attnres` is a Rust library that implements Attention Residuals for
[burn](https://github.com/tracel-ai/burn)-based Transformer experiments.

[![CI](https://github.com/AbdelStark/attnres/actions/workflows/ci.yml/badge.svg)](https://github.com/AbdelStark/attnres/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/attnres.svg)](https://crates.io/crates/attnres)
[![docs.rs](https://img.shields.io/docsrs/attnres)](https://docs.rs/attnres)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What This Is

Attention Residuals replace fixed residual additions with learned softmax
attention over prior depth states. This repository provides:

- A reusable Rust library for the core AttnRes components.
- A reference Transformer implementation built on burn.
- Two-phase inference utilities, serialization helpers, benchmarks, and demos.
- A browser demo (`web-demo/`) and a terminal demo (`demo_tui`).

## Why It Exists

The project exists to make the Attention Residuals paper practical in Rust:
readable source, deterministic CPU tests, and enough examples to inspect the
algorithm instead of treating it as a black box.

## Who It Is For

- Researchers validating the paper or adapting the idea to new models.
- Rust engineers experimenting with burn-based Transformer components.
- Contributors who want a small, testable reference implementation.

## Current Status

As of March 16, 2026, `attnres` is **alpha**. It is suitable for research,
examples, local experimentation, and library integration work on trusted
inputs. It is **not yet suitable** for production inference services,
checkpoint interchange with PyTorch ecosystems, or GPU-backed deployments that
require validated performance and operational guarantees.

Known limitations:

- CI exercises the NdArray backend; GPU backends compile via burn but are not
  validated here.
- No general PyTorch checkpoint interchange is shipped for the reference
  `AttnResTransformer` path.
- `src/kimi/` now implements RFC 0001 artifact understanding, RFC 0002
  baseline Kimi Linear scaffolding, and RFC 0003 sharded checkpoint-import
  scaffolding: typed tensor locators, tensor-to-module coverage reports,
  selected-layer/full shard planning, dtype policy, and shard-path resolution
  without fake tensor payload loading.
- Baseline parity is still deferred.
- RFC 0004 AttnRes-Kimi integration is still deferred.
- No reference-parity harness, validated public-checkpoint hot swap, or
  optimized KDA kernels are shipped yet.
- No compatibility promise for a stable 1.0 public API yet.
- No dedicated formal spec document is checked into this repository today.

## Quick Start

```toml
[dependencies]
attnres = "0.1"
burn = { version = "0.20", features = ["ndarray"] }
```

```rust
use attnres::{AttnResConfig, AttnResTransformer};
use burn::backend::NdArray;
use burn::prelude::*;

type B = NdArray;

let device = Default::default();
let config = AttnResConfig::new(128, 8, 2)
    .with_num_heads(4)
    .with_vocab_size(1000);

let model: AttnResTransformer<B> = config
    .try_init_model(&device)
    .expect("hard-coded config should be valid");

let input_ids = Tensor::<B, 2, Int>::zeros([1, 16], &device);
let logits = model.forward(input_ids, None);
assert_eq!(logits.dims(), [1, 16, 1000]);
```

Use `try_validate` / `try_init_model` when configuration can come from user
input, files, or other untrusted sources. The panic-based `validate` /
`init_model` helpers are retained for trusted, hard-coded setups.

## Core Concepts

- `num_layers` counts **sublayers**, not full Transformer blocks.
- Each Transformer layer has **two** AttnRes operations: one before attention,
  one before the MLP.
- The softmax runs over the **depth/block dimension**, not over tokens.
- Pseudo-query vectors must start at zero to recover uniform averaging at
  initialization.
- Block states are cumulative sums inside a block, plus a list of completed
  blocks.

## Architecture

```text
Input IDs
  -> Embedding
  -> BlockState::new(embeddings)
  -> AttnResLayer x N
       -> AttnResOp (pre-attn)
       -> RMSNorm
       -> MultiHeadAttention
       -> AttnResOp (pre-mlp)
       -> RMSNorm
       -> FeedForward
  -> Final RMSNorm
  -> LM head
  -> Logits
```

Repository map:

- `src/config.rs`: configuration, validation, and typed config errors.
- `src/attn_res_op.rs`: the core depth-attention residual operator.
- `src/block_state.rs`: completed blocks plus the current partial block.
- `src/layer.rs`: one Transformer layer with dual AttnRes sublayers.
- `src/model.rs`: end-to-end model and two-phase forward path.
- `src/two_phase.rs`: the paper's two-phase inference primitives.
- `src/serialization.rs`: save/load helpers for burn record formats.
- `src/kimi/`: RFC 0001 artifact understanding plus RFC 0002 baseline Kimi
  architecture scaffolding.
- `tests/`: unit, integration, property, and differential coverage.
- `web-demo/`: WASM crate plus a Vite front-end.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed module map and
invariants.

## RFC 0001 Staging

The real-model milestone is staged rather than presented as one vague "Kimi
support" claim.

- Phase A: artifact understanding. Implemented in this checkout via
  `attnres::kimi` typed config, layer-schedule, shard-index, and import
  planning/report APIs.
- Phase B: baseline Kimi implementation. Implemented in this checkout as
  architecture scaffolding: `KimiLinearModel`, schedule-driven MLA/KDA module
  selection, dense-vs-MoE placement, and separate decode-cache state types.
- RFC 0003: sharded checkpoint-import scaffolding. Implemented in this checkout
  as planning/reporting only: tensor locators, module coverage, unsupported
  tensor reporting, selected-layer/full shard plans, and explicit `bfloat16` to
  local-runtime dtype policy.
- Phase C: baseline parity. Deferred.
- Phase D: AttnRes-Kimi integration. Deferred in RFC 0004.
- Phase E: benchmarks and research validation. Deferred.

See [docs/rfcs/0001-real-model-milestone-scope.md](docs/rfcs/0001-real-model-milestone-scope.md)
for the accepted sequencing and scope boundaries.

## Examples And Demos

Rust examples:

```bash
cargo run --example compare_residuals
cargo run --example train_tiny
cargo run --example visualize_weights
cargo run --example demo_tui --release
```

Web demo:

```bash
cd web-demo
npm install
npm run build
```

`npm run build` invokes the WASM build and the Vite production build. It
requires `wasm-pack` and the `wasm32-unknown-unknown` Rust target.

## Development

The following commands were verified on this checkout during the March 16, 2026
quality pass:

```bash
cargo fmt -- --check
cargo clippy -- -D warnings
cargo test --all-features
cargo build --examples
cd web-demo && npm run build
```

Additional useful commands:

```bash
cargo bench
cargo doc --open
```

## Documentation And Contributor Entry Points

- [ARCHITECTURE.md](ARCHITECTURE.md): module map, data flow, invariants.
- [ROADMAP.md](ROADMAP.md): current status, milestones, known limitations.
- [CONTRIBUTING.md](CONTRIBUTING.md): setup, expectations, verification steps.
- [CHANGELOG.md](CHANGELOG.md): user-visible changes.
- [AGENTS.md](AGENTS.md) / [CLAUDE.md](CLAUDE.md): current agent context.

## Help

Open an issue in the GitHub repository for bugs, incorrect docs, or feature
requests. If you are changing algorithm behavior, include the failing test or
paper reference that motivated the change.

## License

[MIT](LICENSE)
