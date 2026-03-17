# RFC 0001: Real-Model Milestone Scope

Status: Draft  
Date: 2026-03-17  
Depends on: none

## Summary

Define the real-model milestone as a staged program:

1. baseline Kimi Linear compatibility;
2. sharded checkpoint import;
3. baseline parity validation;
4. AttnRes-Kimi integration;
5. benchmark and training validation.

This RFC exists to stop the project from collapsing several different kinds of
work into one vague promise.

## Problem

The phrase "run Attention Residuals on a real model" hides at least four
separate tasks:

- represent a real architecture;
- load a real checkpoint format;
- prove numerical correctness against a reference implementation;
- evaluate an AttnRes-augmented version of that architecture.

If these are not separated, the project risks shipping an ambiguous result:
perhaps a partially loaded checkpoint, or a Kimi-shaped model without parity,
or an AttnRes variant without a valid baseline comparison.

## Decision

The milestone is accepted only if all of the following become explicit
deliverables:

- baseline Kimi Linear model class;
- Hugging Face config and sharded `safetensors` importer;
- reference parity harness against Hugging Face remote code;
- AttnRes-Kimi model class;
- benchmark and failure criteria.

The milestone is not defined as "port Moonshot's internal AttnRes Kimi code"
because that public artifact does not currently exist.

## Scope

In scope:

- Kimi Linear only
- token-id forward path
- checkpoint inspection and loading
- baseline parity
- AttnRes insertion into Kimi decoder layers
- correctness, stability, and benchmark validation

Out of scope:

- Kimi K2
- production deployment
- unsupported backend claims
- benchmark claims that are not backed by actual runs

## Sequencing

### Phase A: Artifact understanding

Deliverables:

- config parser
- layer schedule decoder
- checkpoint index reader
- tensor-name coverage report

### Phase B: Baseline implementation

Deliverables:

- Kimi Linear local modules
- MLA path
- KDA path
- dense MLP and sparse MoE path
- cache structs

### Phase C: Baseline parity

Deliverables:

- tiny-random end-to-end parity
- selected-layer parity on public Kimi 48B
- cache parity checks

### Phase D: AttnRes-Kimi

Deliverables:

- dual AttnRes per decoder layer
- block-state integration
- two-phase validation on supported configs

### Phase E: Benchmarks and research result

Deliverables:

- architecture overhead measurement
- stability results on reduced training runs
- clearly scoped quality benchmarks

## Rejected Alternatives

### Alternative: Patch the current `AttnResTransformer` until it looks like Kimi

Rejected because it would turn the reference model into a confusing hybrid and
make both code paths harder to reason about.

### Alternative: Skip baseline parity and jump directly to AttnRes-Kimi

Rejected because it removes the ability to distinguish import bugs from
AttnRes-specific behavior.

### Alternative: Treat Kimi K2 and Kimi Linear as one milestone

Rejected because K2 is a different family with DeepSeekV3-style modeling and
FP8 quantization.

## Acceptance

This RFC is successful if later implementation work can answer these questions
unambiguously:

- Are we loading baseline Kimi correctly?
- Are we numerically matching the public reference?
- Are AttnRes-specific differences intentional?
- Which claims are about correctness, which are about speed, and which are
  about quality after training?
