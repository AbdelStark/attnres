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

For the Kimi real-model milestone specifically, see
[docs/status/kimi-real-model-status.md](docs/status/kimi-real-model-status.md).
As of March 23, 2026, the repo has code-backed selected-module
public-checkpoint baseline correctness for
`moonshotai/Kimi-Linear-48B-A3B-Instruct`, an honest full-checkpoint smoke
harness with a completed real-checkpoint report, an explicit
baseline-to-AttnRes bootstrap policy, and reduced
optimizer-backed training-stability validation. It still has not crossed the
bar for a meaningful real-model AttnRes result because the public baseline
smoke completion does not validate AttnRes quality, and this checkout still
lacks a real-checkpoint AttnRes train/eval path beyond structural bootstrap
plus reduced Gate 6 stability checks.

Known limitations:

- CI exercises the NdArray backend; GPU backends compile via burn but are not
  validated here.
- No general PyTorch checkpoint interchange is shipped for the reference
  `AttnResTransformer` path.
- `src/kimi/` now implements RFC 0001 artifact understanding, RFC 0002
  baseline Kimi Linear scaffolding, and RFC 0003 sharded checkpoint-import
  scaffolding: typed tensor locators, tensor-to-module coverage reports,
  selected-layer/full shard planning, dtype policy, shard-path resolution, and
  a local payload loader for the currently supported baseline tensor subset.
- `src/kimi/` now also implements RFC 0004 AttnRes-Kimi execution scaffolding:
  a separate `KimiAttnResModel` / `KimiAttnResDecoderLayer` path that keeps the
  RFC 0002 MLA-vs-KDA and dense-vs-MoE sublayer schedule while inserting two
  AttnRes operations per decoder layer.
- RFC 0005 now has an executable in-repo local Gate 1 slice for the baseline
  path only: `fixtures/kimi/tiny-random-baseline/` plus
  `tests/kimi_rfc_0005_gate1_tests.rs` validate a deterministic
  `KimiLinearModel` against a committed tiny-random Kimi-style reference bundle
  by checking logits, selected post-layer hidden states, and MLA/KDA cache
  traces on fixed prompts.
- That Gate 1 slice validates only what the repo can execute today: Kimi-style
  `config.json`, shard-index coverage for the selected baseline modules, and
  committed reference outputs from the local deterministic baseline path.
- RFC 0005 now also has an executable local Gate 2 preparation slice for both
  model families on the same supported baseline tensor subset:
  `KimiLinearModel::try_from_artifact_dir`,
  `KimiArtifactUnderstanding::try_init_baseline_model_from_dir`,
  `KimiAttnResModel::try_from_artifact_dir`, and
  `KimiArtifactUnderstanding::try_init_attn_res_model_from_dir` can load local
  sharded `safetensors` payloads into the baseline and AttnRes-Kimi models,
  but only for tensors the repo can already map and validate today:
  embeddings, supported decoder norms, supported attention projections,
  supported dense and selected sparse-MoE tensors, final norm, and LM head.
  AttnRes operator parameters remain locally initialized because public Kimi
  artifacts do not expose them. The executable harnesses in
  `tests/kimi_rfc_0005_gate2_local_payload_tests.rs` and
  `tests/kimi_rfc_0005_gate2_attn_res_payload_tests.rs` use tiny local sharded
  artifacts with real payload bytes and check deterministic output changes plus
  explicit failures for unsupported tensors, missing shards, unsupported
  dtypes, shape mismatches, and incomplete selected-layer payload coverage;
  the AttnRes path also checks cached decode availability and standard vs
  two-phase agreement after loading.
- RFC 0005 now also has an executable baseline-only Gate 2 fixture-consumption
  slice for external-generator handoff:
  `attnres::kimi::KimiBaselineSliceRequestManifest` plus
  `KimiArtifactUnderstanding::try_build_baseline_slice_request_manifest`
  emit a machine-readable `baseline-slice-request.json` for `KimiLinearModel`
  only. That request manifest captures the artifact/config fingerprint,
  deterministic local-init seed requirement, import selection, selected hidden
  layers, exact selected modules, exact required tensor names, prompt suite,
  and explicit tolerance metadata that an external baseline reference must use
  to generate a matching fixture.
- RFC 0005 also keeps the baseline-only fixture-consumption path:
  `attnres::kimi::KimiBaselineSliceParityFixture`,
  `compare_baseline_slice_parity_fixture_from_dir`, and
  `compare_baseline_slice_parity_fixture_with_manifest_from_dir` consume an
  externally generated `baseline-slice-parity.json`, optionally require exact
  agreement with an emitted `baseline-slice-request.json`, seed the remaining
  locally initialized parameters deterministically, load the supported local
  sharded slice into `KimiLinearModel`, and compare logits plus selected
  post-layer hidden states with explicit max-abs-diff tolerances.
- RFC 0005 now also has a narrow executed external-generator pilot for that
  handoff in `external/kimi_baseline_reference/`: a standalone Python
  reference consumes an attnres-emitted `baseline-slice-request.json` plus a
  minimal `local-init-contract.json` marker, reconstructs the deterministic
  Burn/NdArray local-init tensors itself from the manifest seed plus artifact
  config, loads a local Hugging Face-style artifact, and writes a
  schema-compatible `baseline-slice-parity.json` that the existing
  manifest-aware Rust consumer accepts unchanged.
- That external-generator slice is still intentionally local-only, but it is
  no longer locked to one fixed pilot manifest: the current contract now
  supports both full logits-plus-hidden fixtures and hidden-state-only prefix
  fixtures on the supported local baseline surface.
- RFC 0005 now also has an executed public-checkpoint module-probe path in
  `external/kimi_baseline_reference/` plus
  `examples/kimi_real_model_tools.rs`: official Hugging Face remote code is
  loaded for `moonshotai/Kimi-Linear-48B-A3B-Instruct` revision
  `e1df551a447157d4658b573f9a695d57658590e9`, selected public shards are
  fingerprinted, and the Rust path is validated against one KDA layer, one MLA
  layer, final norm, LM head, and decode/cache traces where applicable.
- RFC 0005 now also has an honest full-checkpoint smoke harness:
  `external/kimi_baseline_reference/run_baseline_smoke.py` reports executed
  smoke assumptions, execution-path details, timings, hardware facts, and
  artifact fingerprints. The current repo now includes a completed public
  report at `docs/reports/kimi-public-baseline-smoke-2026-03-23.json`, and
  the harness still returns a blocked status instead of a false pass when the
  full 48B artifact set or required prerequisites are unavailable.
- RFC 0005 now also has reduced optimizer-backed training-stability coverage in
  `tests/kimi_rfc_0005_gate6_training_stability_tests.rs` for a hybrid
  KDA/MLA plus dense/MoE reduced config, with explicit loss-growth, activation,
  gradient, and non-finite failure criteria against a baseline Kimi control.
- Real-checkpoint AttnRes quality evaluation after training, an honest
  real-checkpoint AttnRes train/eval runner, optimized KDA kernels, and
  reportable benchmark conclusions beyond the reduced/local harnesses remain
  deferred.
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
  as tensor locators, module coverage, unsupported tensor reporting,
  selected-layer/full shard plans, explicit `bfloat16` to local-runtime dtype
  policy, plus local payload loading for the currently supported baseline
  tensor subset.
- Phase C: baseline parity. Implemented in this checkout for two executed
  slices: local Gate 1 deterministic tiny-random parity for the baseline
  `KimiLinearModel` path, plus selected-module public-checkpoint parity against
  the official Hugging Face remote-code path for one KDA layer, one MLA layer,
  final norm, LM head, and decode/cache traces.
- Phase D: AttnRes-Kimi integration. Implemented in this checkout as execution
  scaffolding plus local artifact bootstrap for the supported baseline tensor
  subset: separate AttnRes-Kimi model/layer/state types, explicit
  cache-vs-block-state handling, sublayer-space block boundaries, local shard
  loading for baseline-compatible tensors, and reduced-config two-phase
  equivalence coverage. Real-checkpoint AttnRes quality evaluation and
  optimized kernels remain deferred.
- Phase E: validation and benchmark scaffolding. Partially implemented in this
  checkout for local deterministic fixtures and reduced configs only: baseline
  Gate 1 fixture-backed parity, local Gate 2 payload-loading prep for the
  supported baseline subset into both baseline and AttnRes-Kimi models,
  baseline-only Gate 2 external-generator request manifests plus external
  fixture consumption for locally loadable sharded slices, public-checkpoint
  module-probe parity, an honest Gate 3 smoke harness with blocked-state
  reporting, Gate 4 functional tests, reduced Gate 5 numerical agreement
  checks, reduced Gate 6 training-stability validation, and reduced local
  benchmark groups. Full 48B smoke success and reportable benchmark claims
  remain deferred.

See [docs/rfcs/0001-real-model-milestone-scope.md](docs/rfcs/0001-real-model-milestone-scope.md)
for the accepted sequencing and scope boundaries.
See [docs/status/kimi-real-model-status.md](docs/status/kimi-real-model-status.md)
for the current milestone status and the concrete remaining blockers before a
meaningful real-model AttnRes test.

## Examples And Demos

Rust examples:

```bash
cargo run --example compare_residuals
cargo run --example kimi_real_model_tools -- emit-module-probe-request <artifact-dir> <output-path>
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
