# RFC 0005: Validation And Benchmark Plan

Status: Draft  
Date: 2026-03-17  
Depends on: RFC 0001, RFC 0002, RFC 0003, RFC 0004

## Summary

Define the validation ladder for the Kimi real-model milestone. The main goal
is to make incorrect implementations fail early and loudly.

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

Target:

- `tiny-random/kimi-linear`

Requirements:

- same token IDs into Hugging Face reference and local model
- compare logits, selected hidden states, and cache updates

Suggested tolerances:

- `max_abs_diff <= 1e-4` in `f32`
- relaxed bf16 tolerance only if documented and justified

Pass condition:

- repeated deterministic parity on at least a small prompt suite

### Gate 2: Public Kimi layer-slice parity

Target:

- `moonshotai/Kimi-Linear-48B-A3B-Instruct`

Requirements:

- selective import of embedding and chosen layers
- parity checks on one MLA layer and one KDA layer
- parity checks on final norm and LM head slices

Pass condition:

- targeted hidden-state parity within agreed tolerances

### Gate 3: End-to-end baseline smoke

Requirements:

- load enough of the public model to execute a real prompt path on suitable
  hardware
- compare next-token logits or top-k ordering against the Hugging Face
  reference

Pass condition:

- consistent prompt completion behavior on a fixed smoke suite

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

Requirements:

- reduced Kimi-style training run with AttnRes enabled
- monitor loss, gradient norms, activation norms, and NaN incidence
- compare against the baseline Kimi-style reduced model

Pass condition:

- no divergence attributable to obvious implementation error

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
