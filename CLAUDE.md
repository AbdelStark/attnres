<identity>
attnres: First Rust implementation of Attention Residuals (MoonshotAI/Kimi paper) using the burn deep learning framework. Drop-in replacement for standard residual connections in Transformers.
</identity>

<stack>
| Layer       | Technology    | Version  | Notes                                    |
|-------------|---------------|----------|------------------------------------------|
| Language    | Rust          | 1.80+    | Nightly recommended for some burn features |
| ML Framework| burn          | 0.20     | tracel-ai/burn — multi-backend DL framework |
| Backends    | CUDA, Metal, wgpu, NdArray | — | NdArray for CPU testing, wgpu for cross-platform GPU |
| Testing     | cargo test    | —        | + proptest (property-based), criterion (benchmarks) |
| Serialization | burn record (NamedMpk, bin) | — | Model weight save/load via burn's record system |
| Linting     | clippy + rustfmt | —     | Enforced in CI                           |
| CI          | GitHub Actions | —       | cargo test, clippy, fmt, build-examples  |
</stack>

<status>
PROJECT PHASE: v0.2.0 — serialization and two-phase inference integrated.
All source modules implemented. 87 tests passing (35 inline unit + 34 external unit + 3 differential + 4 property + 9 integration + 2 doctest).
CI configured (test, clippy, fmt, build-examples). Examples and benchmarks functional. burn 0.20.
Model save/load via burn record system (NamedMpk, binary, compact). Config save/load via JSON.
Two-phase inference integrated into model via `forward_two_phase` method.
Known gaps: no PyTorch checkpoint import, GPU backends untested.
</status>

<structure>
Current directory layout:

```
attnres/
├── Cargo.toml                      # Package manifest [agent: CREATE]
├── CLAUDE.md                       # This file
├── AGENTS.md                       # AI agent technical context [agent: MODIFY]
├── ROADMAP.md                      # Feature roadmap and progress [agent: MODIFY]
├── CHANGELOG.md                    # Release history [agent: MODIFY]
├── README.md                       # Project README [agent: MODIFY]
├── LICENSE                         # MIT [agent: READ ONLY]
├── spec.md                         # Technical specification [agent: READ ONLY — source of truth]
├── paper.md                        # Paper digest [agent: READ ONLY]
├── implementation_plan.md          # Build schedule [agent: READ ONLY]
├── research_report.md              # Feasibility analysis [agent: READ ONLY]
├── src/
│   ├── lib.rs                      # Public API re-exports [agent: CREATE/MODIFY]
│   ├── config.rs                   # AttnResConfig [agent: CREATE/MODIFY]
│   ├── attn_res_op.rs              # Core AttnRes operation [agent: CREATE/MODIFY]
│   ├── block_state.rs              # Block state tracking [agent: CREATE/MODIFY]
│   ├── layer.rs                    # AttnResLayer [agent: CREATE/MODIFY]
│   ├── model.rs                    # AttnResTransformer [agent: CREATE/MODIFY]
│   ├── rms_norm.rs                 # RMSNorm implementation [agent: CREATE/MODIFY]
│   ├── serialization.rs            # Model weight save/load [agent: CREATE/MODIFY]
│   ├── two_phase.rs                # Two-phase inference [agent: CREATE/MODIFY]
│   ├── attention.rs                # Multi-head attention [agent: CREATE/MODIFY]
│   ├── feed_forward.rs             # MLP module [agent: CREATE/MODIFY]
│   └── utils.rs                    # Helper functions [agent: CREATE/MODIFY]
├── tests/
│   ├── unit_tests.rs               # Core algorithm tests [agent: CREATE/MODIFY]
│   ├── differential_tests.rs       # PyTorch reference comparison [agent: CREATE/MODIFY]
│   ├── property_tests.rs           # proptest property-based tests [agent: CREATE/MODIFY]
│   └── integration_tests.rs        # Training comparison tests [agent: CREATE/MODIFY]
├── fixtures/                       # Reference outputs from PyTorch [agent: CREATE/MODIFY]
│   ├── attn_res_forward.json
│   └── block_state_tracking.json
├── examples/
│   ├── train_tiny.rs               # Training example [agent: CREATE/MODIFY]
│   ├── compare_residuals.rs        # Standard vs AttnRes [agent: CREATE/MODIFY]
│   └── visualize_weights.rs        # Depth attention visualization [agent: CREATE/MODIFY]
├── benches/
│   └── attn_res_benchmark.rs       # Criterion benchmarks [agent: CREATE/MODIFY]
└── web-demo/                        # Interactive browser demo [agent: CREATE/MODIFY]
    ├── crate/                       # Pure-Rust WASM crate (AttnRes reimplementation)
    │   ├── Cargo.toml
    │   └── src/lib.rs               # wasm-bindgen exports
    ├── src/                         # TypeScript frontend
    │   ├── main.ts                  # App entry + controls
    │   ├── style.css                # Academic-grade CSS
    │   ├── viz.ts                   # Canvas heatmaps/charts
    │   └── diagrams.ts              # Static diagrams
    ├── index.html                   # SPA entry point
    ├── package.json                 # npm scripts (build:wasm, dev, build)
    └── vite.config.ts               # Vite build config
```
</structure>

<commands>
| Task             | Command                          | Notes                              |
|------------------|----------------------------------|------------------------------------|
| Build            | `cargo build`                    | Add `--release` for optimized      |
| Test (all)       | `cargo test --all-features`      | Runs unit + integration + property |
| Test (specific)  | `cargo test test_name`           | Filter by test name                |
| Lint             | `cargo clippy -- -D warnings`    | Treat warnings as errors           |
| Format           | `cargo fmt`                      | Enforce standard formatting        |
| Format (check)   | `cargo fmt -- --check`           | CI check without modifying         |
| Docs             | `cargo doc --open`               | Generate and view documentation    |
| Bench            | `cargo bench`                    | Criterion benchmarks               |
| Run example      | `cargo run --example train_tiny` | CPU (ndarray) by default           |
| Publish          | `cargo publish`                  | ⚠ REQUIRES APPROVAL               |
| Web demo (WASM)  | `cd web-demo && npm run build:wasm` | Requires wasm-pack + wasm32 target |
| Web demo (dev)   | `cd web-demo && npm run dev`     | Vite dev server at localhost:5173  |
| Web demo (build) | `cd web-demo && npm run build`   | Production build (WASM + Vite)     |
</commands>

<conventions>
<code_style>
  Naming: snake_case for functions/variables/modules, PascalCase for types/structs/enums, SCREAMING_SNAKE for constants.
  Files: snake_case.rs — one primary type per file, named after the type.
  Modules: One module per source file. Re-export public API from lib.rs.
  Imports: Group: std → external crates → internal modules. Use `use crate::` for internal.
  Types: Use burn's generic Backend pattern: `struct Foo<B: Backend>`.
  Derive macros: `#[derive(Module, Debug)]` for burn modules, `#[derive(Config, Debug)]` for configs.
</code_style>

<patterns>
  <do>
    — Follow burn framework idioms: Module trait, Config pattern, Backend generic
    — Use `Param<Tensor<B, N>>` for learnable parameters
    — Map paper equations to code with explicit comments citing equation numbers
    — Initialize pseudo-query vectors to zero (CRITICAL for training stability)
    — Use `Tensor::stack` for combining block representations along depth dimension
    — Write doc comments with `///` for all public items, referencing paper sections
    — Use `#[cfg(test)]` for inline unit tests, separate files for integration tests
    — Keep tensor dimension annotations in comments: `// [B, T, D]`
  </do>
  <dont>
    — Don't use raw arrays for tensor ops — use burn's Tensor API exclusively
    — Don't initialize pseudo-queries randomly — must be zero-init per paper
    — Don't hardcode backends — always use `<B: Backend>` generic parameter
    — Don't mix up depth-dimension softmax (dim=0) with token-dimension — AttnRes attends over DEPTH, not sequence
    — Don't forget RMSNorm before computing attention logits — prevents magnitude domination
    — Don't use `println!` for logging — use `tracing` or `log` crate
  </dont>
</patterns>

<commit_conventions>
  Format: `type(scope): description`
  Types: feat, fix, test, docs, bench, refactor, chore
  Scope: core, layer, model, test, example, ci
  Examples:
    feat(core): implement AttnResOp forward pass
    test(unit): add zero-init uniform weights test
    docs(readme): add benchmark results table
</commit_conventions>
</conventions>

<boundaries>
<forbidden>
  DO NOT modify under any circumstances:
  — .env, .env.* (if created — credentials, API keys)
</forbidden>

<gated>
  Modify ONLY with explicit human approval:
  — Cargo.toml (dependency changes affect build and security)
  — .github/workflows/ (CI pipeline changes)
  — Any `cargo publish` operation
  — Database/file system operations in examples that write to disk
</gated>

<safety_checks>
  Before ANY destructive operation:
  1. State what you're about to do
  2. State what could go wrong
  3. Wait for confirmation

  Before modifying core algorithm (attn_res_op.rs):
  1. Verify change aligns with spec.md pseudocode
  2. Run existing tests first to establish baseline
  3. Make change and re-run tests
</safety_checks>
</boundaries>

<critical_implementation_details>
These are the most important technical details from the paper. Getting ANY of these wrong breaks correctness:

1. ZERO INITIALIZATION: Pseudo-query vectors w_l MUST be initialized to zero. This ensures AttnRes starts as standard residual (uniform attention) and gradually learns to differentiate.

2. TWO AttnRes PER LAYER: Each transformer layer has TWO AttnRes operations — one before self-attention, one before MLP. Not one. Two.

3. SOFTMAX OVER DEPTH: The softmax in AttnRes is over the depth/block dimension (dim=0 in stacked tensor), NOT over the sequence dimension. This is attention over layers, not attention over tokens.

4. BLOCK STATE: Block representations are cumulative sums within each block, not individual layer outputs. blocks[n] = sum of all layer outputs in block n.

5. RMSNorm ON KEYS: Apply RMSNorm to the stacked values to get keys before computing attention logits. This prevents blocks with large magnitudes from dominating attention.

6. BLOCK BOUNDARIES: With L layers and N blocks, block_size = L/N. Block boundary occurs every block_size/2 transformer layers (because each transformer layer = 2 sublayers: attn + MLP).
</critical_implementation_details>

<design_context>
## Web Demo Design Context

### Users
ML researchers and enthusiasts exploring the Attention Residuals paper interactively. Secondary audience: Rust developers evaluating the library. Users arrive with mathematical fluency and want to build intuition for how depth attention weights evolve — not to be sold on the concept.

### Brand Personality
**Rigorous, clean, precise.** The demo is a companion to an academic paper. Every visual choice should reinforce credibility and clarity. The interface should feel like a well-typeset paper with interactive figures, not a product.

### Aesthetic Direction
- **Visual tone**: Academic monochrome with a single blue accent. Restrained, high-contrast, typography-forward.
- **Typography**: Source Serif 4 (headings — paper feel), Inter (body — readability), JetBrains Mono (data/code — precision).
- **Palette**: Monochrome grays + blue (#2563eb light / #60a5fa dark). No multi-color branding.
- **Theme**: Light + dark via `prefers-color-scheme`. Both must be first-class.
- **Anti-references**:
  - NOT a marketing landing page (no hero CTAs, testimonials, pricing, sales copy)
  - NOT a generic AI/ML demo (no neon gradients, dark-mode-only gamer aesthetic, "powered by AI" badges)
  - NOT a Jupyter notebook (no raw code cells, unstyled outputs, or academic ugliness)

### Design Principles
1. **Let the math speak.** Visualizations and equations are the content. Chrome should be invisible.
2. **Credibility through restraint.** One accent color. No decorative elements. Every pixel earns its place.
3. **Progressive disclosure.** Start simple (uniform weights), reveal complexity (training simulation) on interaction.
4. **Dark mode is not an afterthought.** Both themes use perceptually appropriate palettes, not inverted colors.
5. **Accessible at WCAG AA.** 4.5:1 contrast minimums, keyboard navigation, screen reader basics. Canvas visualizations should have text alternatives.

### Design Tokens (established)
All colors, spacing, typography, shadows, and motion are defined as CSS custom properties in `web-demo/src/style.css:3-63`. Canvas drawing code should read these tokens at runtime via `getComputedStyle()` rather than hard-coding hex values.
</design_context>
