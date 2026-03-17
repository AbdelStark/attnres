# Architecture

`attnres` is a library-first reference implementation of Attention Residuals
for burn-based Transformer experiments.

## Scope

The repository contains three primary surfaces:

- The core Rust crate under `src/`.
- Rust examples and benchmarks under `examples/` and `benches/`.
- A browser demo under `web-demo/`.

This document is the current architecture reference for the repository. A
dedicated formal `spec.md` is not present in this checkout.

## Forward Path

Standard model forward:

```text
input_ids
  -> embedding
  -> BlockState { blocks: [embedding], partial_block: None }
  -> for each AttnResLayer:
       1. depth attention before self-attention
       2. self-attention sublayer
       3. depth attention before MLP
       4. MLP sublayer
       5. block-state update
  -> final RMSNorm
  -> LM head
  -> logits
```

Two-phase forward:

```text
completed blocks
  -> phase1_batched: inter-block attention statistics for each sublayer
  -> sequential intra-block updates
  -> online_softmax_merge
  -> same final RMSNorm + LM head
```

## Module Map

- `src/config.rs`
  - Owns `AttnResConfig`.
  - Validates user-supplied config.
  - Exposes `ConfigError`, `try_validate`, `try_init_model`, and panic-based
    compatibility helpers.

- `src/attn_res_op.rs`
  - Implements the core depth-attention residual operator.
  - Stacks completed blocks plus the optional partial block.
  - Applies RMSNorm to keys before computing depth logits.

- `src/block_state.rs`
  - Tracks completed blocks and the currently accumulating block.
  - Treats the token embedding as the first completed block.

- `src/layer.rs`
  - Implements one Transformer layer with two AttnRes calls.
  - Handles block-boundary transitions at sublayer granularity.

- `src/model.rs`
  - Builds the full model.
  - Provides standard forward, two-phase forward, and hidden-state forward.

- `src/attention.rs`
  - Standard multi-head self-attention.
  - Assumes additive masks with large negative values for masked positions.

- `src/feed_forward.rs`
  - Two-layer GELU MLP.

- `src/rms_norm.rs`
  - RMSNorm for both `[B, T, D]` and `[N, B, T, D]` tensors.

- `src/two_phase.rs`
  - Batched phase-1 inter-block attention.
  - Online softmax merge for intra-block values.

- `src/serialization.rs`
  - Save/load helpers for burn recorders.
  - Accepts `Path`-like inputs rather than forcing UTF-8 strings.

- `src/kimi/`
  - RFC 0001 Phase A artifact-understanding scaffolding.
  - Parses and validates Hugging Face-style Kimi config metadata.
  - Decodes 1-based layer schedules into typed zero-based internal schedules.
  - Parses shard-index metadata and exposes explicit import planning surfaces.
  - Does not yet provide baseline Kimi execution, checkpoint loading, or
    AttnRes-Kimi model code.

## Invariants

These are the invariants that most directly affect correctness:

- Pseudo-query vectors are zero-initialized.
- Attention weights are normalized over the depth dimension.
- Each Transformer layer performs two AttnRes operations.
- Block boundaries are defined in sublayer space, not Transformer-layer space.
- `BlockState.blocks[0]` is always the token embedding block.
- `partial_block` is expected to exist after at least one sublayer has run.

If one of these invariants is violated by internal code, the crate prefers a
direct panic with a specific message rather than silently returning a wrong
result.

## Error Model

- Invalid caller-supplied configuration should use `ConfigError`.
- Serialization failures return `SerializationError`.
- Kimi artifact-understanding failures return typed `attnres::kimi::*Error`
  enums with explicit phase-gated errors for not-yet-implemented modes.
- Internal invariant breaks still panic because they represent library bugs, not
  recoverable runtime conditions.

## Verification

The current repository-wide baseline checked during the March 16, 2026 quality
pass is:

```bash
cargo fmt -- --check
cargo clippy -- -D warnings
cargo test --all-features
cargo build --examples
cd web-demo && npm run build
```
