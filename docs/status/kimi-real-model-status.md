# Kimi Real-Model Milestone Status

Status: Active  
Date: 2026-03-18  
Branch: `codex/kimi-real-model-rfc`

## Executive Summary

`attnres` has now cleared the public-checkpoint baseline correctness critical
path for the strongest slice this checkout and environment can honestly
support:

- selected-module public-checkpoint parity now passes against the official
  Hugging Face remote-code path for
  `moonshotai/Kimi-Linear-48B-A3B-Instruct`;
- the executed parity surface covers one KDA layer, one MLA layer, final norm,
  LM head, and decode/cache traces for the selected attention modules;
- an honest full-checkpoint smoke harness now exists and was executed;
- baseline-to-AttnRes bootstrap is now explicit and reported as structural
  bootstrap only;
- reduced Kimi-style AttnRes training-stability validation now passes with a
  real Burn optimizer loop and explicit failure criteria.

The repository still cannot honestly claim:

- successful full-model smoke on the public 48B checkpoint in this environment;
- any real-checkpoint AttnRes quality result after training or continued
  pretraining;
- validated `fla-core` kernel parity or performance on this macOS CPU host;
- reportable benchmark conclusions beyond the reduced/local harnesses.

## Executed Evidence

### Public Baseline Correctness

Executed public artifact:

- repo: `moonshotai/Kimi-Linear-48B-A3B-Instruct`
- revision: `e1df551a447157d4658b573f9a695d57658590e9`
- dtype: `bfloat16`
- shard fingerprint coverage:
  - `model-00001-of-00020.safetensors` for KDA layer 0
  - `model-00002-of-00020.safetensors` for MLA layer 3
  - `model-00020-of-00020.safetensors` for final norm and LM head

Executed commands:

- emit request:
  `cargo run --example kimi_real_model_tools -- emit-module-probe-request /tmp/attnres-kimi-public-probe /tmp/attnres-kimi-public-probe/module-probe-request.json`
- generate public fixture:
  `python -m external.kimi_baseline_reference.generate_module_probe_fixture --artifact-dir /tmp/attnres-kimi-public-probe --request-path /tmp/attnres-kimi-public-probe/module-probe-request.json --output-path /tmp/attnres-kimi-public-probe/module-probe-fixture.json --repo-id moonshotai/Kimi-Linear-48B-A3B-Instruct --revision e1df551a447157d4658b573f9a695d57658590e9`
- validate public fixture:
  `cargo run --example kimi_real_model_tools -- validate-module-probe-fixture /tmp/attnres-kimi-public-probe /tmp/attnres-kimi-public-probe/module-probe-request.json /tmp/attnres-kimi-public-probe/module-probe-fixture.json`

Observed result:

- public module-probe parity passed for:
  - `kda_layer_0_seq4`
  - `mla_layer_3_seq4`
  - `final_norm_seq4`
  - `lm_head_seq4`
- decode/cache comparisons also passed for the selected KDA and MLA probes
- the executed public reference path used official Hugging Face remote code
  with the `cpu_fallback` backend for the `fla-core` pieces on this host

### End-to-End Smoke

Executed command:

- `python -m external.kimi_baseline_reference.run_baseline_smoke --artifact-dir /tmp/attnres-kimi-public-probe --output-path /tmp/attnres-kimi-public-probe/baseline-smoke-report.json --repo-id moonshotai/Kimi-Linear-48B-A3B-Instruct --revision e1df551a447157d4658b573f9a695d57658590e9`

Observed result:

- status: `blocked_missing_full_checkpoint`
- host: Apple M4 Max CPU with `51,539,607,552` bytes RAM
- checkpoint size: `98,245,528,576` bytes
- configured minimum CPU RAM for attempted smoke:
  `106,835,463,168` bytes
- local shard state:
  - present: 3 / 20
  - missing: 17 / 20

This is an executed blocker report, not a soft failure and not a hidden gap.

### Bootstrap Policy

The baseline-to-AttnRes transition is now explicit in code:

- policy:
  `KimiAttnResBootstrapPolicy::BaselineImportWithFreshAttnRes`
- entry point:
  `KimiArtifactUnderstanding::try_bootstrap_attn_res_model_from_dir`
- report:
  `KimiAttnResBootstrapReport`

The load report now states plainly that:

- baseline-compatible tensors are imported only into matching baseline modules;
- every AttnRes operator remains freshly initialized;
- zero pseudo-query initialization is preserved;
- the result is structural bootstrap only and does not justify parity, quality,
  or benchmark claims before additional training succeeds.

### Reduced Training-Stability Validation

Executed repo target:

- `tests/kimi_rfc_0005_gate6_training_stability_tests.rs`

Observed result:

- passes on deterministic seeds `20260318` and `20260319`
- trains a reduced hybrid KDA/MLA plus dense/MoE config with Burn autodiff and
  Adam
- compares AttnRes-Kimi against a baseline Kimi control on the same
  deterministic batches
- fails explicitly on:
  - non-finite loss
  - non-finite hidden states or logits
  - non-finite gradients
  - excessive loss growth
  - excessive whole-model gradient L2 norm
  - excessive hidden/logit RMS growth

## Phase Status

### Phase A: Artifact Understanding

Status: implemented

- typed config parsing and validation
- layer-schedule decoding
- shard-index parsing
- selected-layer/full import planning
- explicit dtype-policy handling

### Phase B: Baseline Kimi Implementation

Status: implemented

- separate `KimiLinearModel`
- typed MLA and KDA execution paths
- dense SiLU MLP and sparse MoE execution paths
- decode-cache support for MLA and KDA

### Phase C: Baseline Parity

Status: implemented for the selected-module slice that this environment can
honestly validate

- local deterministic Gate 1 parity remains locked
- public selected-module parity now passes for one KDA layer, one MLA layer,
  final norm, LM head, and decode/cache traces

Still missing:

- full-model prompt-path smoke success on the public 48B checkpoint

### Phase D: AttnRes-Kimi Integration

Status: implemented as a structural bootstrap and reduced-config execution path

- separate `KimiAttnResModel`, `KimiAttnResDecoderLayer`, and block-state path
- dual AttnRes placement per decoder layer
- sublayer-space block boundaries
- explicit baseline-to-AttnRes bootstrap policy and report
- reduced-config standard-vs-two-phase agreement

Still missing:

- any trained real-checkpoint AttnRes result
- any claim that imported baseline tensors plus fresh AttnRes operators are
  meaningful quality evaluation

### Phase E: Validation And Benchmark Work

Status: materially advanced, but not benchmark-complete

- Gate 1 local deterministic parity
- Gate 2 local payload loading for baseline and AttnRes-Kimi
- Gate 2 external slice-request / slice-fixture contract
- Gate 2 public module-probe parity against official remote code
- Gate 3 smoke harness with honest blocked-state reporting
- Gate 4 functional tests
- Gate 5 reduced numerical agreement
- Gate 6 reduced optimizer-backed training stability

Still missing:

- successful full-model public smoke
- real-checkpoint AttnRes quality evaluation
- benchmark numbers suitable for external claims

## Honest Remaining Limitations

1. Public selected-module correctness is not the same as full 48B prompt-path
   success. The repo now has strong evidence that the local loader and module
   math align on key public slices, but the full model still has not run end to
   end on this machine.
2. The executed public reference path used official Hugging Face remote code
   with a CPU fallback for the `fla-core` operators. That is acceptable for
   correctness validation here, but it is not evidence for kernel parity or
   performance parity.
3. The AttnRes real-model story remains a bootstrap plus stability result, not
   a quality result. Real-checkpoint AttnRes evaluation still needs additional
   training and post-training measurement.

## Project-Lead Status

As of 2026-03-18, the repo has closed the selected-module public-checkpoint
baseline correctness path, added an honest full-checkpoint smoke harness,
made the AttnRes bootstrap policy explicit, and executed reduced Kimi-style
training-stability validation.

The strongest honest blocker that remains is full-model public smoke success on
hardware with the full shard set and enough RAM, followed by real-checkpoint
AttnRes quality evaluation after training.
