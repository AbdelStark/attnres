# Kimi Real-Model Milestone Status

Status: Active  
Date: 2026-03-18  
Branch: `codex/kimi-real-model-rfc`

## Executive Summary

`attnres` is now structurally ready for real-model integration work on Kimi
Linear, but it is not yet at the point where the project can honestly claim a
meaningful real-model AttnRes result.

The repository can now:

- represent the Kimi Linear decoder schedule locally, including mixed MLA/KDA
  attention and dense-vs-MoE feed-forward placement;
- parse and validate Hugging Face-style Kimi `config.json` and
  `model.safetensors.index.json` artifacts;
- plan selected-layer imports and report unsupported, duplicate, and missing
  tensors explicitly;
- load the currently supported baseline tensor subset from local sharded
  `safetensors` artifacts into both `KimiLinearModel` and `KimiAttnResModel`;
- validate local deterministic fixture behavior for baseline Kimi, plus
  functional and reduced-config numerical behavior for AttnRes-Kimi.

The repository still cannot honestly claim:

- public-checkpoint baseline parity against Hugging Face remote code;
- end-to-end smoke execution on the public 48B checkpoint;
- meaningful AttnRes quality evaluation on a real Kimi checkpoint without
  continued pretraining or fresh training;
- reportable benchmark conclusions beyond the reduced local harnesses.

## Current State By Phase

### Phase A: Artifact Understanding

Status: implemented

Implemented in-repo:

- typed Kimi config parsing and validation;
- typed layer-schedule decoding;
- sharded safetensors index parsing;
- selected-layer/full import planning;
- explicit dtype policy and coverage reporting.

### Phase B: Baseline Kimi Implementation

Status: implemented

Implemented in-repo:

- separate `KimiLinearModel` path;
- MLA and KDA execution scaffolding;
- dense SiLU MLP and sparse MoE execution scaffolding;
- separate decode-cache types for MLA and KDA.

### Phase C: Baseline Parity

Status: partially implemented

Implemented in-repo:

- local deterministic tiny-random Gate 1 fixture-backed parity for baseline
  Kimi;
- local Gate 2 payload-loading prep for the supported baseline tensor subset;
- baseline-only external slice-request / slice-fixture handoff for the current
  supported local baseline surface, including hidden-state-only slices that do
  not force final norm or LM head loading; the executed regression lock is
  still the narrow local pilot artifact.

Still missing:

- Hugging Face / Python execution against the public Kimi reference;
- selected-layer parity on the public 48B checkpoint;
- stateful decode parity against the public reference on real slices.

### Phase D: AttnRes-Kimi Integration

Status: partially implemented

Implemented in-repo:

- separate `KimiAttnResModel`, `KimiAttnResDecoderLayer`, and block-state path;
- dual AttnRes placement per decoder layer;
- sublayer-space block-boundary handling;
- reduced-config standard-vs-two-phase numerical agreement;
- local shard loading for the same supported baseline tensor subset used by
  `KimiLinearModel`, while keeping AttnRes operator parameters locally
  initialized.

Still missing:

- any public-checkpoint AttnRes parity surface;
- any trained AttnRes-Kimi checkpoint or continued-pretraining result;
- any evidence that imported baseline weights plus locally initialized AttnRes
  operators produce meaningful quality on real prompts.

### Phase E: Benchmark And Training Validation

Status: mostly deferred

Implemented in-repo:

- reduced local benchmark scaffolding;
- written benchmark-gate structure in RFC 0005.

Still missing:

- training-stability validation;
- quality evaluation after training or continued pretraining;
- public-checkpoint throughput or end-to-end smoke measurements;
- benchmark numbers that would support external claims.

## What Counts As A Meaningful Real-Model Test

This milestone now needs two separate definitions of "meaningful":

### 1. Meaningful Baseline Real-Model Test

This means proving that the local baseline Kimi path is numerically aligned
with the public Kimi reference on real checkpoint slices.

Minimum bar:

- load selected tensors from the public `moonshotai/Kimi-Linear-48B-A3B-Instruct`
  artifact;
- compare one MLA layer, one KDA layer, final norm, and LM head against an
  external Hugging Face reference on identical token inputs;
- verify stateful decode parity for the selected slices, not just single-pass
  hidden states;
- document exact tolerances, hardware, and which tensors were actually loaded.

### 2. Meaningful AttnRes Real-Model Test

This means showing that the AttnRes-augmented Kimi path behaves coherently on a
real Kimi-style model after an honest bootstrap/training story, not merely
after loading baseline tensors into a structurally modified model.

Minimum bar:

- start from a stated bootstrap policy for baseline-to-AttnRes transition;
- run continued pretraining or fresh reduced-model training;
- pass training-stability checks;
- report quality and overhead separately from baseline-parity correctness.

## The Current Gap

The remaining gap is not one thing. It is two stacked blockers.

### Blocker 1: Public Baseline Correctness

The repository still needs an external reference harness that can execute the
public Hugging Face Kimi implementation and return comparable slice fixtures.

The local manifest/fixture contract is no longer the main blocker here:

- attnres can now emit slice requests that compare selected hidden states
  without forcing final logits;
- the local external Python path can now consume those manifests and return
  schema-compatible hidden-only fixtures;
- the unchanged Rust consumer can validate those fixtures against local shard
  loading.

Without that, the project cannot answer:

- Are we loading public Kimi tensors correctly?
- Are MLA and KDA cache semantics numerically aligned with the public model?
- Are observed differences import bugs, kernel differences, or real model
  mismatches?

### Blocker 2: AttnRes-Specific Evaluation

Even if baseline public-checkpoint parity lands, that still does not create a
meaningful AttnRes result by itself.

Why:

- the public checkpoint is baseline Kimi, not AttnRes-Kimi;
- zero-initialized pseudo-queries are not equivalent to standard residual
  addition;
- imported baseline tensors plus locally initialized AttnRes operators are only
  a research bootstrap, not a benchmark-ready model.

Therefore a meaningful AttnRes real-model test still requires:

- a clearly documented warm-start or training policy;
- reduced Kimi-style stability validation;
- quality evaluation after training.

## Immediate Next Steps

In priority order:

1. Build or harden the external Hugging Face reference harness for public
   slice-generation on Kimi Linear 48B. The local hidden-only contract is now
   in place, so this step is primarily about public artifacts and remote-code
   execution rather than more local manifest plumbing.
2. Execute Gate 2 public slice parity on at least one MLA layer, one KDA
   layer, final norm, and LM head.
3. Execute Gate 3 end-to-end baseline smoke on suitable hardware.
4. Define the baseline-to-AttnRes bootstrap policy explicitly.
5. Run Gate 6 reduced Kimi-style AttnRes training-stability validation before
   attempting any quality or benchmark claim.

## Honest Project-Lead Status

As of 2026-03-18, the project has cleared the local architecture-and-artifact
plumbing stage for Kimi integration and has enough scaffolding to make public
baseline parity the next serious correctness milestone.

It has not yet cleared the bar for a meaningful real-model AttnRes result.
