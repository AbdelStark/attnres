# RFC 0003: Sharded Checkpoint Import For Kimi

Status: Implemented in repository as planning/report scaffolding  
Date: 2026-03-17  
Depends on: RFC 0001, RFC 0002

## Summary

Implement a Kimi-specific checkpoint loader for Hugging Face `config.json` and
sharded `safetensors`.

## Problem

Current serialization only supports burn-native record formats. Public Kimi
artifacts are distributed as:

- `config.json`
- `model.safetensors.index.json`
- multiple `model-xxxxx-of-xxxxx.safetensors` files

This means the current crate cannot inspect or load a public Kimi checkpoint.

## Decision

Build a dedicated importer around the Hugging Face artifact layout instead of
trying to force-fit the current burn recorder APIs.

The importer must support three modes:

- `inspect`: parse config and index, report model structure and coverage
- `slice`: load only selected layers or modules
- `full`: load the full supported model

Repository note:

- This checkout implements the typed planning/reporting scaffold for all three
  modes: shard index ingest, tensor-name to module classification, selected
  layer and full shard plans, dtype policy, and explicit unsupported/missing
  tensor reporting.
- It does not yet claim end-to-end public-checkpoint tensor payload loading or
  baseline numerical parity.
- Where the public Kimi tensor surface does not match the current RFC 0002
  local scaffold exactly, the implementation reports the mismatch rather than
  silently dropping or inventing tensors.

## Why A Dedicated Importer

`burn-import` and `SafetensorsFileRecorder` are useful building blocks, but the
public Kimi checkpoint is sharded across 20 files. The local importer therefore
needs explicit support for:

- reading the shard index;
- resolving tensor name to shard path;
- loading only the shards needed for the requested modules;
- validating coverage and duplicates across the whole model.

## Import Contract

### Config ingest

The importer must parse and validate at least:

- hidden size
- head counts
- attention head dimensions
- MLA low-rank settings
- linear-attention kernel settings
- layer count
- layer schedule lists
- MLP and MoE dimensions
- tied-vs-untied embedding policy
- dtype
- cache flags

### Index ingest

The importer must parse:

- shard metadata
- total parameter count
- total byte size
- full tensor-name to shard mapping

### Mapping report

The importer must emit a machine-readable report that classifies every tensor
name as one of:

- mapped
- intentionally ignored
- unsupported
- duplicate
- shape mismatch
- dtype mismatch

No silent dropping is allowed.

## Recommended Architecture

- `KimiArtifactConfig`: parsed `config.json`
- `KimiShardIndex`: parsed `model.safetensors.index.json`
- `KimiTensorLocator`: name to shard resolver
- `KimiImportReport`: coverage and mismatch summary
- `KimiImportPlan`: selected modules and required shards

## Dtype Policy

Public Kimi weights are `bfloat16`. The loader must define a clear policy:

- preserve bf16 if backend supports it;
- otherwise promote to `f32` deliberately;
- never silently narrow precision.

## Slice Loading

Slice loading is mandatory because it enables:

- fast parity checks on a small number of layers;
- validation on machines that cannot hold the full checkpoint;
- debugging name mappings before full import.

Examples:

- embedding + first decoder layer
- one MLA layer
- one KDA layer
- final norm + LM head

## Rejected Alternatives

### Alternative: Convert the checkpoint externally and only support burn-native files

Rejected because it pushes the hardest correctness problem out of the
repository, hides mapping bugs, and weakens reproducibility.

### Alternative: Only support full checkpoint loading

Rejected because the Kimi checkpoint is too large to make full loading the
first validation step.

## Validation

- index parser tests using tiny-random Kimi fixtures
- name coverage tests against public Kimi metadata
- slice import smoke tests
- checksum or manifest consistency tests for required shard files

## Failure Conditions

- any unmapped tensor in a supposedly supported module
- duplicate ownership of a tensor across two local parameters
- silent dtype coercion
- inability to load selected-layer slices independently
