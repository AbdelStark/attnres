<identity>
attnres-rs: First Rust implementation of Attention Residuals (MoonshotAI/Kimi paper) using the burn deep learning framework. Drop-in replacement for standard residual connections in Transformers.
</identity>

<stack>
| Layer       | Technology    | Version  | Notes                                    |
|-------------|---------------|----------|------------------------------------------|
| Language    | Rust          | 1.80+    | Nightly recommended for some burn features |
| ML Framework| burn          | 0.20     | tracel-ai/burn — multi-backend DL framework |
| Backends    | CUDA, Metal, wgpu, NdArray | — | NdArray for CPU testing, wgpu for cross-platform GPU |
| Testing     | cargo test    | —        | + proptest (property-based), criterion (benchmarks) |
| Serialization | safetensors | —       | For weight loading/saving                |
| Linting     | clippy + rustfmt | —     | Enforced in CI                           |
| CI          | GitHub Actions | —       | cargo test, clippy, fmt, build-examples  |
</stack>

<status>
PROJECT PHASE: Alpha (v0.1.0 — core algorithm implemented, tests passing).
All source modules implemented. 57 tests passing (28 inline unit + 18 external unit + 3 differential + 2 property + 5 integration + 1 doctest).
CI configured (test, clippy, fmt, build-examples). Examples and benchmarks functional. burn upgraded to 0.20.
Known gaps: no safetensors serialization, two-phase inference not integrated into main forward path, GPU backends untested.
</status>

<structure>
Current directory layout:

```
attnres-rs/
├── Cargo.toml                      # Package manifest [agent: CREATE]
├── CLAUDE.md                       # This file
├── AGENTS.md                       # AI agent technical context [agent: MODIFY]
├── ROADMAP.md                      # Feature roadmap and progress [agent: MODIFY]
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
└── benches/
    └── attn_res_benchmark.rs       # Criterion benchmarks [agent: CREATE/MODIFY]
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

<workflows>
<new_module>
  1. Read spec.md for the module's specification (data structures, algorithm, equations)
  2. Create src/module_name.rs with struct and impl
  3. Add `pub mod module_name;` to src/lib.rs
  4. Add public re-export to lib.rs if part of public API
  5. Write inline unit tests in `#[cfg(test)] mod tests {}`
  6. Run `cargo test` — all must pass
  7. Run `cargo clippy -- -D warnings` — zero warnings
  8. Run `cargo fmt` — ensure formatting
</new_module>

<implement_algorithm>
  1. Read the relevant section in spec.md (contains Rust pseudocode with paper references)
  2. Implement step-by-step, commenting each line with the paper equation it maps to
  3. Add tensor shape annotations as comments on key operations
  4. Write unit tests covering: correct output shape, known input/output pairs, edge cases
  5. If differential test data exists in fixtures/, add a differential test
</implement_algorithm>

<add_test>
  1. Determine test type: unit (tests/unit_tests.rs), property (tests/property_tests.rs), differential (tests/differential_tests.rs), integration (tests/integration_tests.rs)
  2. Write test using NdArray backend (CPU, deterministic)
  3. Use small dimensions for speed: d_model=64, seq_len=16, batch=2
  4. Assert with tolerance for floating-point: `assert!((a - b).abs() < 1e-5)`
  5. Run `cargo test test_name` to verify
</add_test>

<bug_fix>
  1. Reproduce with a minimal test case
  2. Check spec.md for the correct algorithm behavior
  3. Compare implementation against paper pseudocode step-by-step
  4. Fix and verify the test passes
  5. Run full test suite to check for regressions
</bug_fix>
</workflows>

<boundaries>
<forbidden>
  DO NOT modify under any circumstances:
  — .env, .env.* (if created — credentials, API keys)
  — LICENSE (legal document)
  — spec.md (source of truth — treat as read-only reference)
  — paper.md (reference material)
  — research_report.md (reference material)
  — implementation_plan.md (reference material)
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

<troubleshooting>
<known_issues>
| Symptom                              | Cause                          | Fix                                    |
|--------------------------------------|--------------------------------|----------------------------------------|
| NaN in attention weights             | Missing RMSNorm before logits  | Ensure `self.norm.forward(v)` is called on values before dot product |
| AttnRes output = simple mean         | Pseudo-queries still at zero   | Expected at init. Train for more steps to see differentiation |
| Shape mismatch in Tensor::stack      | Blocks have inconsistent dims  | Verify all blocks are [B, T, D] before stacking |
| Gradient explosion in deep models    | Missing zero-init on queries   | Set pseudo_query to `Tensor::zeros([d_model])` |
| Block boundary off-by-one            | Incorrect layer_idx counting   | layer_idx is 0-based, boundary at `idx % (block_size/2) == 0` for idx > 0 |
| Backend compilation errors           | Missing feature flags          | Check Cargo.toml features for the target backend |
</known_issues>

<recovery_patterns>
  When stuck:
  1. Read the error message — Rust errors are usually precise and actionable
  2. Check spec.md for the algorithm specification
  3. Verify tensor shapes at each step with shape annotations
  4. Run `cargo test` on a minimal case to isolate the issue
  5. Check burn documentation for API usage
  6. If numerical issues, add debug prints of intermediate tensor values
</recovery_patterns>
</troubleshooting>

<environment>
  Harness: Claude Code / Claude Agent SDK
  File system scope: Full project directory
  Network access: Available (for cargo dependencies)
  Tool access: git, cargo, shell
  Human interaction: Synchronous chat
</environment>

<skills>
Modular skills in .codex/skills/ (symlinked at .claude/skills/ and .agents/skills/).

Available skills:
— _index.md: Skill registry and discovery metadata
— rust-burn-development.md: Burn framework patterns, module creation, backend handling
— testing.md: Testing strategy — unit, differential, property-based, integration, benchmarks
— debugging.md: Debugging tensor operations, numerical issues, burn-specific troubleshooting
— tensor-operations.md: Tensor manipulation patterns for the AttnRes algorithm
</skills>

<memory>
<project_decisions>
  2026-03: Use burn over tch-rs — Multi-backend support (CUDA+Metal+wgpu+CPU), pure Rust, better ergonomics — Rejected: tch-rs (libtorch dependency), candle (less mature module system)
  2026-03: Block AttnRes as primary variant — Paper shows it's more practical (lower overhead) while maintaining benefits — Rejected: Full AttnRes only (quadratic in layer count)
  2026-03: Zero-init pseudo-queries — Paper requirement for training stability, ensures smooth transition from standard residuals — Rejected: Random init (causes training instability)
  2026-03: NdArray backend for testing — Deterministic, no GPU needed, fast for small tensors — Rejected: wgpu for tests (slower startup, non-deterministic)
</project_decisions>

<lessons_learned>
  — [Initial setup] This is a greenfield project. All implementation follows spec.md as the source of truth.
  — [burn 0.16→0.20] Breaking API changes required updates to activation functions, loss computation, and tensor operations. Always check burn changelog when upgrading.
  — [Testing] NdArray backend is deterministic and fast for small tensors. All tests use it. GPU backends remain untested.
  — [Quality audit] Doc comments, config validation, and test coverage were hardened in a dedicated audit pass. Maintain this standard.
</lessons_learned>
</memory>
