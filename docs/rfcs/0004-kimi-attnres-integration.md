# RFC 0004: AttnRes Integration For Kimi Linear

Status: Draft  
Date: 2026-03-17  
Depends on: RFC 0001, RFC 0002, RFC 0003

## Summary

Define how Attention Residuals will be inserted into the Kimi Linear decoder
without violating the existing AttnRes invariants.

## Problem

Public Kimi Linear uses standard residual additions. The project goal is to run
the AttnRes framework on a real model architecture, ideally the same family
used in the Attention Residuals paper. That requires a Kimi-shaped model whose
residual path is replaced by AttnRes.

## Decision

Create an `AttnResKimiDecoderLayer` that keeps Kimi attention and MLP/MoE
submodules intact but replaces the fixed residual path with two AttnRes
operations:

- one before the attention sublayer;
- one before the MLP/MoE sublayer.

This mirrors the existing project invariant that each decoder layer contributes
two AttnRes sublayers.

## Proposed Forward Structure

For each decoder layer:

1. read the current block state;
2. run AttnRes before the attention path;
3. apply input RMSNorm to the AttnRes output;
4. run KDA or MLA attention;
5. update the partial block;
6. run AttnRes before the MLP/MoE path;
7. apply post-attention RMSNorm to the AttnRes output;
8. run dense MLP or sparse MoE;
9. update the partial block again.

Pseudo-structure:

```text
state -> attn_res -> attn_norm -> {KDA | MLA} -> partial update
      -> mlp_res  -> mlp_norm  -> {MLP | MoE} -> partial update
```

## Invariants That Must Survive

- pseudo-query vectors start at zero
- depth softmax is over prior depth sources, not tokens
- each decoder layer owns two AttnRes operations
- block boundaries are defined in sublayer space
- block `0` is the embedding block

These do not change just because the surrounding architecture is Kimi.

## Kimi-Specific Considerations

### Mixed attention types

AttnRes must treat KDA and MLA layers identically at the residual level. The
attention type changes the sublayer implementation, not the residual invariant.

### MoE layers

AttnRes wraps the input to the MoE block; it does not alter router semantics,
expert selection, or expert aggregation.

### Cache handling

AttnRes block state is separate from Kimi decode cache state. The design must
not conflate:

- depth-state history used by AttnRes
- attention/cache state used by KDA or MLA

## Warm-Start Policy

This RFC makes an explicit negative decision:

- do not claim that baseline Kimi weights are a parity-preserving
  initialization for AttnRes-Kimi.

Reason:

- zero-initialized pseudo-queries yield uniform depth weights, not standard
  residual addition.

Therefore benchmark-quality AttnRes-Kimi results require one of:

- continued pretraining;
- training from scratch on a reduced Kimi-style model;
- an explicit non-paper-faithful bootstrap method proposed in a later RFC.

## Two-Phase Inference

The existing two-phase logic applies to the AttnRes residual operator, not to
the Kimi attention kernels themselves. For AttnRes-Kimi, two-phase validation
must be scoped carefully:

- prove equivalence for the AttnRes depth-attention path;
- do not claim a global Kimi kernel optimization unless separately implemented.

## Validation

- unit tests for mixed KDA/MLA layer schedules
- unit tests for block boundaries in sublayer space
- standard vs two-phase AttnRes equivalence on reduced Kimi-style configs
- no-NaN stability tests for short training runs
- hidden-state magnitude tracking across depth

## Failure Conditions

- AttnRes state accidentally depends on attention type
- block boundaries are implemented in decoder-layer space instead of sublayer
  space
- caches and depth-state history become coupled
- baseline weight transplant is described as equivalent when it is not
