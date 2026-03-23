# Kimi Real-Model Milestone Status

Status: Active  
Date: 2026-03-23  
Branch: `codex/kimi-real-model-rfc`

## Executive Summary

`attnres` has now cleared the public baseline critical path that this checkout
can honestly support:

- selected-module public-checkpoint parity passes against the official Hugging
  Face remote-code path for
  `moonshotai/Kimi-Linear-48B-A3B-Instruct`;
- full public baseline smoke now passes on the real 20-shard checkpoint using
  an approved equivalent execution path
  (`cpu_disk_offload`) on a host below the CPU-only RAM floor;
- the smoke report records exact commands, hardware facts, timings, execution
  path, dtype handling, attention-backend overrides, device placement, and
  SHA-256 fingerprints for `config.json`, the shard index, remote-code files,
  and all 20 checkpoint shards;
- baseline-to-AttnRes bootstrap is explicit and reported as structural
  bootstrap only;
- reduced Kimi-style AttnRes training-stability validation still passes with a
  real Burn optimizer loop and explicit failure criteria.

The repository still cannot honestly claim:

- any real-checkpoint AttnRes quality result after training or continued
  pretraining;
- any AttnRes quality, parity, or benchmark result on the 48B public
  checkpoint;
- validated `fla-core` kernel parity or performance on this macOS CPU host;
- reportable benchmark conclusions beyond the reduced/local harnesses.

## Executed Evidence

### Public Baseline Correctness

Executed public artifact:

- repo: `moonshotai/Kimi-Linear-48B-A3B-Instruct`
- revision: `e1df551a447157d4658b573f9a695d57658590e9`
- artifact dir:
  `/Users/abdel/.cache/attnres/kimi-linear-48b-e1df551a447157d4658b573f9a695d57658590e9`
- dtype: `bfloat16`
- selected executed shard coverage:
  - `model-00001-of-00020.safetensors` for KDA layer 0
  - `model-00002-of-00020.safetensors` for MLA layer 3
  - `model-00020-of-00020.safetensors` for final norm and LM head

Executed commands:

- emit request:
  `cargo run --example kimi_real_model_tools -- emit-module-probe-request /Users/abdel/.cache/attnres/kimi-linear-48b-e1df551a447157d4658b573f9a695d57658590e9 /tmp/attnres-kimi-public-20260323/module-probe-request.json`
- generate public fixture:
  `.venv-kimi-public/bin/python -m external.kimi_baseline_reference.generate_module_probe_fixture --artifact-dir /Users/abdel/.cache/attnres/kimi-linear-48b-e1df551a447157d4658b573f9a695d57658590e9 --request-path /tmp/attnres-kimi-public-20260323/module-probe-request.json --output-path /tmp/attnres-kimi-public-20260323/module-probe-fixture.json --repo-id moonshotai/Kimi-Linear-48B-A3B-Instruct --revision e1df551a447157d4658b573f9a695d57658590e9`
- validate public fixture:
  `cargo run --example kimi_real_model_tools -- validate-module-probe-fixture /Users/abdel/.cache/attnres/kimi-linear-48b-e1df551a447157d4658b573f9a695d57658590e9 /tmp/attnres-kimi-public-20260323/module-probe-request.json /tmp/attnres-kimi-public-20260323/module-probe-fixture.json`

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

- `.venv-kimi-public/bin/python -m external.kimi_baseline_reference.run_baseline_smoke --artifact-dir /Users/abdel/.cache/attnres/kimi-linear-48b-e1df551a447157d4658b573f9a695d57658590e9 --output-path /Users/abdel/dev/me/machine-learning/attnres/docs/reports/kimi-public-baseline-smoke-2026-03-23.json --repo-id moonshotai/Kimi-Linear-48B-A3B-Instruct --revision e1df551a447157d4658b573f9a695d57658590e9 --execution-path cpu_disk_offload --offload-dir /Users/abdel/.cache/attnres/kimi-linear-48b-e1df551a447157d4658b573f9a695d57658590e9/baseline-smoke-offload --max-cpu-memory-gib 40`

Observed result:

- status: `passed`
- report:
  `docs/reports/kimi-public-baseline-smoke-2026-03-23.json`
- host:
  - platform: `macOS-26.2-arm64-arm-64bit-Mach-O`
  - system: `Darwin`
  - machine: `arm64`
  - CPU: `Apple M4 Max`
  - model: `Mac16,5`
  - Python: `3.14.2`
  - total RAM: `51,539,607,552` bytes
- checkpoint facts:
  - total size: `98,245,528,576` bytes
  - present shards: `20 / 20`
  - missing shards: `0`
  - unique shard count: `20`
  - configured CPU-only minimum RAM:
    `106,835,463,168` bytes
- execution path:
  - `cpu_disk_offload`
  - CPU memory budget: `40.0 GiB`
  - load dtype: `float32`
  - dtype action:
    `upcast_bfloat16_to_float32_for_cpu_offload`
  - `fla-core` backend: `cpu_fallback`
  - attention backend override:
    `flash_attention_2 -> eager_attention_forward`
- timings:
  - load: `128.85063075018115` seconds
  - first forward: `22.32013437501155` seconds
- exact top-5 next-token outputs and logits are recorded in the report
- device map is recorded in the report and places embeddings plus layers `0-4`
  on CPU, with layers `5-26`, final norm, and LM head disk-offloaded
- artifact fingerprints recorded in the report:
  - `config.json`
    `sha256 = a6ac3c2c4b5aa72370f9727f49ffa4432715d20061889acdb37c688be853096e`
  - `model.safetensors.index.json`
    `sha256 = e63dba2b42cfe38ad7f2fd7cf561706c584f8fe7b42016baba16b27fe0d06bab`
  - `configuration_kimi.py`
    `sha256 = 79422aca3ee6c89d201e0c15c4c9a6db517ba83d87ecdc4e41fa0f71297238d9`
  - `modeling_kimi.py`
    `sha256 = d79b365e37378881b9f1585007a56e236ca27a414920943cb85d1dacb75dda99`
  - all `20` shard SHA-256 fingerprints are recorded in the report

This is now an executed completion report, not a blocked-state placeholder.

### Bootstrap Policy

The baseline-to-AttnRes transition is explicit in code:

- policy:
  `KimiAttnResBootstrapPolicy::BaselineImportWithFreshAttnRes`
- entry point:
  `KimiArtifactUnderstanding::try_bootstrap_attn_res_model_from_dir`
- report:
  `KimiAttnResBootstrapReport`

The load report states plainly that:

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

This remains reduced-config stability evidence only. There is still no
executed real-checkpoint AttnRes train/eval result in this checkout.

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

Status: implemented for the strongest public-baseline slice this checkout has
actually run

- local deterministic Gate 1 parity remains locked
- public selected-module parity passes for one KDA layer, one MLA layer, final
  norm, LM head, and decode/cache traces
- full-model public smoke now passes on the real 48B public checkpoint through
  the recorded `cpu_disk_offload` execution path

Still missing:

- any end-to-end Rust-vs-Hugging-Face full-model parity claim on the 48B
  checkpoint
- any claim that this CPU/offload smoke path demonstrates kernel or
  performance parity

### Phase D: AttnRes-Kimi Integration

Status: implemented as a structural bootstrap and reduced-config execution path

- separate `KimiAttnResModel`, `KimiAttnResDecoderLayer`, and block-state path
- dual AttnRes placement per decoder layer
- sublayer-space block boundaries
- explicit baseline-to-AttnRes bootstrap policy and report
- reduced-config standard-vs-two-phase agreement

Still missing:

- any trained real-checkpoint AttnRes result
- any real-checkpoint train/eval harness that continues from the structural
  bootstrap and measures quality gates honestly

### Phase E: Validation And Benchmark Work

Status: materially advanced, but not benchmark-complete

- Gate 1 local deterministic parity
- Gate 2 local payload loading for baseline and AttnRes-Kimi
- Gate 2 external slice-request / slice-fixture contract
- Gate 2 public module-probe parity against official remote code
- Gate 3 full public baseline smoke completion with execution-path reporting,
  host facts, timings, and artifact fingerprints
- Gate 4 functional tests
- Gate 5 reduced numerical agreement
- Gate 6 reduced optimizer-backed training stability

Still missing:

- real-checkpoint AttnRes quality evaluation
- benchmark numbers suitable for external claims

## Honest Remaining Limitations

1. The full public baseline smoke succeeded through a CPU-plus-disk-offload
   path on a host with `51,539,607,552` bytes RAM, below the CPU-only minimum
   of `106,835,463,168` bytes. That is an acceptable executed smoke path, but
   it is not evidence for native bf16 CPU parity or performance.
2. The executed public reference path still needed the official Kimi remote
   code plus an explicit `flash_attention_2` remap to
   `eager_attention_forward`, and the `fla-core` path ran through
   `cpu_fallback`. That is acceptable for correctness validation here, but it
   is not evidence for kernel parity or throughput.
3. The AttnRes real-model story remains a structural bootstrap plus reduced
   stability result, not a quality result. This checkout contains bootstrap
   reporting and reduced Gate 6 training stability, but not an honest
   real-checkpoint train/eval runner that continues from the imported baseline
   bootstrap and measures quality gates on the 48B path.

## Project-Lead Status

As of 2026-03-23, the repo has closed the public baseline milestone through
selected-module parity plus a completed full public baseline smoke on the real
checkpoint, with an evidence report that preserves exact commands, outputs,
timings, hardware facts, and artifact fingerprints.

The strongest honest blocker that remains is no longer baseline smoke. It is
real-checkpoint AttnRes validation after bootstrap: continued training,
explicit evaluation gates, and benchmark-quality reporting still require an
execution path that this checkout does not yet implement.
