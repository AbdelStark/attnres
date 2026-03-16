# CLAUDE.md

## Identity

`attnres` is a Rust implementation of Attention Residuals for burn-based
Transformer experiments. The repository includes a library crate, examples,
benchmarks, and a separate WASM web demo.

## Status

- Alpha as of March 16, 2026.
- Verified locally during the latest quality pass:
  - `cargo fmt -- --check`
  - `cargo clippy -- -D warnings`
  - `cargo test --all-features`
  - `cargo build --examples`
  - `cd web-demo && npm run build`
- Not yet validated:
  - GPU backend correctness/performance claims
  - PyTorch checkpoint interchange
  - Stable 1.0 API guarantees

## Repository Map

- `README.md`: project front door and current status.
- `ARCHITECTURE.md`: current architecture and invariants.
- `ROADMAP.md`: milestones and known limitations.
- `CONTRIBUTING.md`: contributor workflow and required checks.
- `CHANGELOG.md`: user-visible changes.
- `src/config.rs`: config validation and typed config errors.
- `src/model.rs`: full model and two-phase forward entry points.
- `src/layer.rs`: per-layer AttnRes logic and block-boundary handling.
- `src/attn_res_op.rs`: core depth attention operator.
- `web-demo/`: standalone WASM reimplementation plus Vite app.

## Working Rules

- Prefer `try_validate` / `try_init_model` when config is not hard-coded.
- Algorithm changes must be backed by tests or paper-referenced reasoning.
- Keep docs factual. Do not repeat stale test counts or unsupported capability
  claims.
- Avoid touching `Cargo.toml` dependencies or `.github/workflows/` without
  explicit approval.
- There is no local `spec.md` in this checkout. Treat `ARCHITECTURE.md`, module
  docs, and tests as the current source of truth.

## Critical Invariants

- Pseudo-query vectors are zero-initialized.
- Softmax is over depth, not over sequence positions.
- Each Transformer layer has two AttnRes operations.
- Block boundaries are defined in sublayer space.
- The embedding block is always the first completed block.

## Web Demo Design Context

Audience:

- Researchers exploring the paper interactively.
- Rust developers evaluating the library.

Visual direction:

- Academic, restrained, and high-contrast.
- Avoid marketing-site tropes and flashy "AI product" aesthetics.
- Favor typography, diagrams, and visualization clarity over decorative UI.

Implementation note:

- `web-demo/crate/` is not a thin wrapper around the main crate. It is a
  separate pure-Rust reimplementation for WASM portability and must be checked
  explicitly when core algorithm behavior changes.
