# AGENTS.md

## Project Identity

`attnres` is a Rust library that implements Attention Residuals for burn-based
Transformer experiments, plus examples, benchmarks, and a web demo.

## Current State

- Status: alpha as of March 16, 2026.
- Suitable for: research, examples, local experimentation, integration work on
  trusted inputs.
- Not yet suitable for: production inference services, validated GPU claims,
  PyTorch checkpoint interchange, or a stable 1.0 API promise.
- Important gap: there is no dedicated `spec.md` in this checkout. Use
  [ARCHITECTURE.md](ARCHITECTURE.md), README, module docs, and tests as the
  current source of truth.

## Verified Commands

These commands were run successfully during the latest quality pass:

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

## Architecture Map

- `src/config.rs`: `AttnResConfig`, `ConfigError`, validation helpers.
- `src/attn_res_op.rs`: core depth-attention residual operator.
- `src/block_state.rs`: completed blocks + current partial block.
- `src/layer.rs`: one Transformer layer with two AttnRes operations.
- `src/model.rs`: full model, hidden-state forward, two-phase forward.
- `src/two_phase.rs`: batched inter-block pass + online softmax merge.
- `src/attention.rs`: multi-head self-attention.
- `src/feed_forward.rs`: two-layer GELU MLP.
- `src/rms_norm.rs`: RMSNorm for 3D and 4D tensors.
- `src/serialization.rs`: burn-record save/load helpers.
- `tests/`: unit, integration, property, and differential coverage.
- `examples/demo_tui.rs`: terminal demo with live routing visualization.
- `web-demo/`: WASM crate plus Vite frontend.

## Non-Negotiable Invariants

- Pseudo-query vectors start at zero.
- Depth softmax is over block/layer sources, not tokens.
- Each Transformer layer uses two AttnRes operations.
- Block boundaries are defined in sublayer space.
- `BlockState.blocks[0]` is the embedding block.
- Internal invariant failures should panic loudly rather than produce silent
  wrong outputs.

## Conventions

- Prefer `try_validate`, `try_init_model`, `try_init_layer`, and `try_init_op`
  for untrusted config input. Panic-based constructors remain for trusted,
  hard-coded configs.
- Keep tensor shape comments short and accurate when code would otherwise be
  hard to parse.
- Add tests for every algorithm or boundary-condition change.
- Keep README and roadmap claims tied to commands or tests that actually ran.

## Constraints

- Do not modify `LICENSE`.
- Do not change dependency versions in `Cargo.toml` without approval.
- Do not change `.github/workflows/` without approval.
- Do not claim backend support, benchmark numbers, or checkpoint compatibility
  unless the repository validates them.

## Gotchas

- `num_layers` counts sublayers, not full Transformer blocks.
- Full AttnRes means `num_blocks == num_layers`, so block boundaries can occur
  between attention and MLP inside one Transformer layer.
- The web demo is a separate pure-Rust reimplementation for WASM portability;
  do not assume it automatically stays in sync with `src/` without verification.
