# AGENTS.md — AI Agent Technical Context

## Project Overview

**attnres-rs** is the first Rust implementation of Attention Residuals (MoonshotAI/Kimi paper) using the [burn](https://github.com/tracel-ai/burn) deep learning framework. It provides a drop-in replacement for standard residual connections in Transformers.

## Tech Stack

| Component   | Technology       | Version  |
|-------------|-----------------|----------|
| Language    | Rust            | 2021 edition (1.80+) |
| ML Framework| burn            | 0.20     |
| Test Backend| NdArray         | (CPU, deterministic) |
| Testing     | cargo test + proptest + criterion | — |
| Linting     | clippy + rustfmt | —       |
| CI          | GitHub Actions   | test, clippy, fmt, build-examples |

## Project Structure

```
src/
├── lib.rs              # Public API re-exports + module declarations
├── config.rs           # AttnResConfig — validated builder pattern (JSON save/load)
├── attn_res_op.rs      # Core AttnRes operation (depth-wise softmax attention)
├── block_state.rs      # BlockState — cumulative block representation tracking
├── layer.rs            # AttnResLayer — transformer layer with dual AttnRes
├── model.rs            # AttnResTransformer — full model with standard + two-phase forward
├── rms_norm.rs         # RMSNorm implementation
├── serialization.rs    # Model weight save/load (NamedMpk, binary, compact formats)
├── two_phase.rs        # Two-phase inference primitives (phase1_batched, online_softmax_merge)
├── attention.rs        # Multi-head self-attention
├── feed_forward.rs     # Two-layer MLP with GELU activation
└── utils.rs            # Causal mask generation helpers

tests/
├── unit_tests.rs       # Core algorithm correctness tests
├── differential_tests.rs # PyTorch reference comparison tests
├── property_tests.rs   # proptest property-based tests
└── integration_tests.rs # Full model training loop tests

examples/
├── train_tiny.rs       # Train a small model on synthetic data
├── compare_residuals.rs # Compare AttnRes vs standard residuals
└── visualize_weights.rs # Visualize depth attention patterns

benches/
└── attn_res_benchmark.rs # Criterion benchmarks

fixtures/                # Reference outputs from PyTorch
├── attn_res_forward.json
└── block_state_tracking.json
```

## Commands

```bash
cargo build                        # Build the project
cargo test --all-features          # Run all 84 tests
cargo test test_name               # Run specific test
cargo clippy -- -D warnings        # Lint (warnings = errors)
cargo fmt                          # Format code
cargo fmt -- --check               # Check formatting without modifying
cargo bench                        # Run Criterion benchmarks
cargo run --example train_tiny     # Train example
cargo run --example compare_residuals  # Comparison example
cargo run --example visualize_weights  # Visualization example
```

## Architecture Essentials

### Core Algorithm (AttnRes)

Standard residual: `x_{l+1} = x_l + f_l(x_l)` (fixed unit weights)

AttnRes: `x_{l+1} = Σ α_i · v_i` where α = softmax(w_l · RMSNorm(V)) over depth dimension

Key invariants:
1. **Zero-init pseudo-queries** → starts as uniform averaging (standard residual behavior)
2. **Two AttnRes per transformer layer** — one before self-attention, one before MLP
3. **Softmax over depth** (block/layer dimension), NOT over sequence tokens
4. **RMSNorm on keys** to prevent magnitude domination
5. **Block boundaries** at every `block_size/2` sublayers

### Data Flow

```
Input IDs → Embedding → [AttnResLayer × N] → RMSNorm → LM Head → Logits
                              ↓
                    AttnResOp(pre-attn) → RMSNorm → MultiHeadAttention
                    AttnResOp(pre-mlp)  → RMSNorm → FeedForward
```

### Configuration

`AttnResConfig::new(d_model, num_layers, num_blocks)` where:
- `d_model`: Hidden dimension
- `num_layers`: Number of **sublayers** (transformer layers × 2)
- `num_blocks`: Number of blocks for Block AttnRes (set = num_layers for Full AttnRes)

## Boundaries

### Read-Only (never modify)
- `spec.md`, `paper.md`, `research_report.md`, `implementation_plan.md`, `LICENSE`

### Gated (requires approval)
- `Cargo.toml` (dependency changes)
- `.github/workflows/` (CI changes)
- `cargo publish`

## Source of Truth

`spec.md` is the authoritative specification. All algorithm implementations must match the pseudocode and equations defined there.

## Known Gaps

- No PyTorch checkpoint loading (safetensors format)
- GPU backends (wgpu, CUDA, Metal) untested
- No distributed training support
- Pre-trained weight import/export utilities
