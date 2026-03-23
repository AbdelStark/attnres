# RFC 0005: Validation And Benchmark Plan

Status: Draft  
Date: 2026-03-17  
Depends on: RFC 0001, RFC 0002, RFC 0003, RFC 0004

## Summary

Define the validation ladder for the Kimi real-model milestone. The main goal
is to make incorrect implementations fail early and loudly.

## Implementation Status In This Checkout

The repository now executes a narrow local subset of this RFC:

- Gate 1 is executable in-repo only as a baseline-only local fixture harness:
  `fixtures/kimi/tiny-random-baseline/` provides Kimi-style `config.json`,
  shard-index metadata, and committed reference outputs; the test harness
  validates selected-layer baseline coverage through RFC 0003 planning, then
  checks deterministic `KimiLinearModel` logits, selected post-layer hidden
  states, and cache traces on a fixed prompt suite.
- Gate 2 now has an executable local preparation slice for the baseline path
  only: `tests/kimi_rfc_0005_gate2_local_payload_tests.rs` writes a tiny local
  sharded `safetensors` artifact with real payload bytes, loads the supported
  tensor subset into `KimiLinearModel`, and checks deterministic forward-output
  changes plus explicit failures for unsupported tensors, missing shard files,
  unsupported dtypes, shape mismatches, and incomplete selected-layer payload
  coverage.
- Gate 2 now also has an executable local preparation slice for the
  AttnRes-Kimi path on that same supported baseline tensor subset:
  `tests/kimi_rfc_0005_gate2_attn_res_payload_tests.rs` loads the local shard
  bytes into `KimiAttnResModel` while leaving AttnRes operator parameters
  locally initialized, then checks deterministic output changes, cached decode
  availability, standard-vs-two-phase agreement after loading, and the same
  explicit failures for unsupported tensors, missing shard files, and
  incomplete selected-layer payload coverage.
- Gate 2 now also has an executable baseline-only external-generator handoff
  slice: `src/kimi/slice_parity.rs` defines a machine-readable
  `baseline-slice-request.json` manifest for the supported `KimiLinearModel`
  subset, derives it from `KimiArtifactUnderstanding` plus caller-provided
  slice/prompt inputs, and `tests/kimi_rfc_0005_gate2_slice_parity_tests.rs`
  validates that the repo can reject stale or invalid request manifests,
  consume externally generated `baseline-slice-parity.json` fixtures, require
  exact manifest-to-fixture metadata agreement when requested, seed
  still-unloaded local parameters deterministically, and compare logits plus
  selected post-layer hidden states with explicit tolerances.
- Gate 2 now also has a narrow executed external-generator pilot in
  `external/kimi_baseline_reference/`: a standalone Python reference consumes
  an attnres-emitted local pilot `baseline-slice-request.json` plus a minimal
  `local-init-contract.json` marker, reconstructs the deterministic Burn/NdArray
  local-init tensors for the executed prefix of an attnres-emitted local slice
  request, loads a local sharded artifact, and produces a schema-compatible
  `baseline-slice-parity.json` that passes the existing manifest-aware Rust
  consumer unchanged. That local external path now supports both
  logits-plus-hidden fixtures and hidden-state-only fixtures when
  `compare_logits = false`.
- Gate 2 now also has an executed public-checkpoint module-probe path:
  `src/kimi/module_probe.rs`, `examples/kimi_real_model_tools.rs`, and
  `external/kimi_baseline_reference/module_probe.py` request and validate
  fingerprinted public-checkpoint probes for one KDA layer, one MLA layer,
  final norm, and LM head, including decode/cache traces for the attention
  modules where applicable. The executed public reference path uses official
  Hugging Face remote code and a CPU fallback for the `fla-core` pieces on the
  current macOS CPU environment.
- Gate 3 now has an executable public-checkpoint smoke harness in
  `external/kimi_baseline_reference/run_baseline_smoke.py`. The executed
  report path records assumptions, hardware, execution-path details, timings,
  and artifact fingerprints explicitly instead of turning blockers into soft
  passes.
- Gate 4 is executable in-repo through AttnRes-Kimi tests for dual AttnRes
  placement, mixed MLA/KDA block-state progression, embedding-block retention,
  and loud invariant-panics on corrupted internal state.
- Gate 5 is executable in-repo only for reduced local configs through
  deterministic-seed tests that compare standard and two-phase hidden states
  plus final logits.
- Gate 6 is now executable in-repo through
  `tests/kimi_rfc_0005_gate6_training_stability_tests.rs`: a reduced hybrid
  KDA/MLA plus dense/MoE config trains with Burn autodiff and Adam, compares
  AttnRes-Kimi against a baseline Kimi control on deterministic batches and
  seeds, and fails on explicit loss-growth, gradient-norm, activation-norm, or
  non-finite regressions.
- Gate 7 now has reduced local benchmark scaffolding for baseline Kimi
  forward/cached-forward and AttnRes-Kimi forward/two-phase runs, with
  benchmark ids that encode backend/model/sequence metadata.

The following remain deferred and should not be implied by these executed
results:

- Python/Hugging Face execution against `tiny-random/kimi-linear` for Gate 1.
- real-checkpoint AttnRes quality evaluation after training or continued
  pretraining.
- an honest real-checkpoint train/eval runner in this checkout that continues
  from the structural bootstrap and measures those quality gates.
- Any benchmark claim beyond the reduced local harnesses above.

## Principles

- correctness before throughput
- artifact coverage before numerical benchmarking
- baseline parity before AttnRes benchmarking
- explicit failure criteria instead of optimistic interpretation

## Validation Ladder

### Gate 0: Static artifact validation

Requirements:

- parse `config.json`
- parse `model.safetensors.index.json`
- verify shard count and metadata
- produce a complete tensor coverage report

Pass condition:

- no unknown or silently dropped tensors in supported modules

### Gate 1: Tiny-random baseline parity

Executable repo target:

- `fixtures/kimi/tiny-random-baseline/`

Still-deferred external target:

- `tiny-random/kimi-linear`

Requirements:

- validate `config.json` and `model.safetensors.index.json`
- require selected baseline modules to be fully supported by RFC 0003 coverage
  planning
- run the baseline `KimiLinearModel` path only
- use deterministic seeds and a fixed prompt suite
- compare logits, selected post-layer hidden states, and cache updates against
  the committed local reference bundle

Still deferred:

- same token IDs into a Python/Hugging Face reference
- tensor payload loading from a public artifact
- any public-checkpoint parity claim

Suggested tolerances:

- `max_abs_diff <= 1e-4` in `f32`
- relaxed bf16 tolerance only if documented and justified

Pass condition:

- repeated deterministic parity on at least a small prompt suite

### Gate 2: Public Kimi layer-slice parity

Executable repo sub-slices:

- baseline-only local shard loading for selected supported tensors into
  `KimiLinearModel`
- local shard loading for that same supported baseline tensor subset into
  `KimiAttnResModel`, with AttnRes operator parameters still locally
  initialized
- baseline-only external-generator request manifests for that same supported
  subset via `baseline-slice-request.json`
- baseline-only external slice-fixture consumption for that same supported
  tensor subset via `baseline-slice-parity.json`
- a narrow local external Python pilot that consumes one attnres-emitted
  request bundle plus a minimal `local-init-contract.json` marker, reconstructs
  the deterministic local-init subset externally, and returns a fixture
  accepted by the existing consumer for the requested local slice recipe,
  including hidden-only prefix slices
- an executed public-checkpoint module-probe path for one KDA layer, one MLA
  layer, final norm, and LM head, with decode/cache comparisons for the
  attention modules plus explicit shard fingerprinting
- local negative-path validation for missing shards, unsupported tensors,
  unsupported dtypes, tensor-shape mismatches, incomplete selected-module
  payloads, fixture kind/version drift, selected-layer mismatches,
  prompt/token mismatches, tolerance-metadata drift, and unsupported
  module/tensor requests inside the external fixture

Target:

- `moonshotai/Kimi-Linear-48B-A3B-Instruct`

Requirements:

- selective import of embedding and chosen layers
- parity checks on one MLA layer and one KDA layer
- parity checks on final norm and LM head slices

Executed public artifact:

- `moonshotai/Kimi-Linear-48B-A3B-Instruct`
- revision `e1df551a447157d4658b573f9a695d57658590e9`
- required executed shards:
  `model-00001-of-00020.safetensors`,
  `model-00002-of-00020.safetensors`,
  `model-00020-of-00020.safetensors`

Pass condition:

- targeted module parity within agreed tolerances for one KDA layer, one MLA
  layer, final norm, and LM head
- decode/cache parity for the selected MLA and KDA module probes

Still deferred in this checkout:

- full-model prompt-path parity on the public checkpoint
- any claim that the selected-module parity results imply end-to-end 48B smoke
  success
- any AttnRes parity claim on a real public checkpoint

### Gate 3: End-to-end baseline smoke

Executable repo target:

- `external/kimi_baseline_reference/run_baseline_smoke.py`

Requirements:

- load enough of the public model to execute a real prompt path on suitable
  hardware
- compare next-token logits or top-k ordering against the Hugging Face
  reference

Pass condition:

- consistent prompt completion behavior on a fixed smoke suite

Executed status on 2026-03-23:

- harness executed against
  `moonshotai/Kimi-Linear-48B-A3B-Instruct`
  revision `e1df551a447157d4658b573f9a695d57658590e9`
- result: `passed`
- execution path: `cpu_disk_offload`
- host RAM: `51,539,607,552` bytes
- configured CPU-only minimum RAM: `106,835,463,168` bytes
- checkpoint size: `98,245,528,576` bytes
- local shard state: `20 / 20` present, `0` missing
- load dtype: `float32`
- dtype action:
  `upcast_bfloat16_to_float32_for_cpu_offload`
- `fla-core` backend: `cpu_fallback`
- full report:
  `docs/reports/kimi-public-baseline-smoke-2026-03-23.json`

Interpretation:

- Gate 3 baseline smoke is now closed through an approved equivalent execution
  path on a lower-RAM host.
- This result does not imply kernel parity, throughput parity, or any
  real-checkpoint AttnRes quality claim.

### Gate 4: AttnRes-Kimi functional validation

Requirements:

- verify dual AttnRes placement in every decoder layer
- verify block-state updates across mixed KDA/MLA schedules
- verify embedding block remains `blocks[0]`
- verify internal invariant failures panic clearly

Pass condition:

- new AttnRes-Kimi tests cover both ordinary and edge-case schedules

### Gate 5: AttnRes-Kimi numerical validation

Requirements:

- standard and two-phase AttnRes paths agree on reduced Kimi-style configs
- controlled random seeds
- compare hidden states, not only final logits

Pass condition:

- equivalence within documented tolerance

### Gate 6: Training-stability validation

Executable repo target:

- `tests/kimi_rfc_0005_gate6_training_stability_tests.rs`

Requirements:

- reduced Kimi-style training run with AttnRes enabled
- monitor loss, gradient norms, activation norms, and NaN incidence
- compare against the baseline Kimi-style reduced model

Pass condition:

- no divergence attributable to obvious implementation error

Executed status on 2026-03-18:

- reduced hybrid KDA/MLA plus dense/MoE training test passes for deterministic
  seeds `20260318` and `20260319`
- harness records loss trajectories plus explicit hidden/logit RMS and
  whole-model gradient L2 caps
- AttnRes-Kimi is checked against a baseline Kimi control instead of being
  evaluated in isolation

### Gate 7: Benchmark reporting

Benchmarks must be separated into categories:

- baseline parity throughput
- AttnRes overhead
- training-stability observations
- quality deltas after actual training or continued pretraining

Pass condition:

- each reported number states hardware, backend, dtype, model size, sequence
  length, and prompt/task source

## Benchmark Matrix

Minimum matrix to aim for:

| Category | Sizes | Sequence lengths | Outputs |
| --- | --- | --- | --- |
| Baseline smoke | tiny-random | 32, 128, 512 | logits parity, cache parity |
| Slice parity | public 48B selected layers | 32, 512 | hidden-state parity |
| Overhead | reduced Kimi-style local configs | 512, 4k, 32k | latency, memory |
| Long-context stress | reduced configs first | 32k, 128k, stretch goal 1M | stability, memory |
| Quality | reduced training runs | task-dependent | loss, eval metrics |

## Hardware And Environment Expectations

The plan must distinguish:

- laptop/local correctness work
- workstation slice-loading work
- large-memory or multi-GPU smoke work for public 48B
- Python reference environment using Hugging Face remote code and `fla-core`

If a result only ran in the Python reference environment, say so.

## Required Negative Tests

- wrong layer schedule lists
- missing shard file
- duplicate tensor mapping
- wrong dtype handling
- wrong 1-based to 0-based conversion
- cache reuse across batch-reset boundaries
- block-state corruption across decoder layers

## Stop Conditions

Stop and do not advance benchmark claims if any of the following occur:

- tensor coverage is incomplete
- tiny-random baseline parity fails
- selected-layer public Kimi parity fails
- AttnRes-Kimi training diverges immediately without a clear reason
- long-context cache behavior is not understood well enough to explain failures

## Reporting Format

Every milestone report should include:

- what artifact was used
- what exact command or harness ran
- what hardware and backend were used
- whether the result is baseline, AttnRes, or reference-only
- what failed, if anything

This milestone is only credible if failure cases are preserved instead of
smoothed over.
