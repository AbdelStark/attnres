# RFC 0002: Baseline Kimi Linear Compatibility

Status: Implemented in repository as baseline scaffolding  
Date: 2026-03-17  
Depends on: RFC 0001

## Summary

Add a new Kimi-specific module tree that can represent the public Kimi Linear
architecture faithfully enough to support checkpoint loading and parity tests.

## Problem

The current crate implements a small reference Transformer with:

- dense multi-head attention;
- GELU feed-forward;
- no external checkpoint format;
- no Kimi cache semantics.

That is too far from public Kimi Linear to support meaningful real-model
experiments.

## Decision

Implement a separate Kimi baseline architecture rather than mutating the
existing `AttnResTransformer`.

Expected module split:

- `src/kimi/config.rs`
- `src/kimi/model.rs`
- `src/kimi/layer.rs`
- `src/kimi/attention/mla.rs`
- `src/kimi/attention/kda.rs`
- `src/kimi/moe.rs`
- `src/kimi/mlp.rs`
- `src/kimi/cache.rs`
- `src/kimi/import.rs`

## Required Baseline Semantics

### Layer schedule

The local config layer schedule must reproduce the public 1-based lists from
Hugging Face config:

- full-attention layers: `[4, 8, 12, 16, 20, 24, 27]`
- KDA layers: all remaining layers

Internally we should convert these to zero-based indices once and keep them
typed, because mixing 1-based external config with 0-based internal indexing is
an easy source of silent bugs.

### Attention parameterization

The baseline implementation must preserve the public attention-related config
surface, including:

- `num_attention_heads = 32`
- `num_key_value_heads = 32`
- `head_dim = 72`
- `kv_lora_rank = 512`
- `q_lora_rank = null`
- `qk_nope_head_dim = 128`
- `qk_rope_head_dim = 64`
- `v_head_dim = 128`
- `mla_use_nope = true`
- `linear_attn_config.num_heads = 32`
- `linear_attn_config.head_dim = 128`
- `linear_attn_config.short_conv_kernel_size = 4`

Even if the first implementation does not optimize every path, these fields
must not be discarded or silently approximated.

### Decoder layer structure

Each baseline Kimi decoder layer must support:

- input RMSNorm
- attention path chosen by layer type
- residual addition
- post-attention RMSNorm
- dense MLP or sparse MoE
- residual addition

### MLP and MoE

The baseline implementation must represent:

- `hidden_act = "silu"`
- `tie_word_embeddings = false`
- dense MLP for layer 0
- sparse MoE from layer 1 onward
- `num_experts = 256`
- `num_experts_per_token = 8`
- `num_shared_experts = 1`

### Cache semantics

The baseline implementation must not fake cache support. Kimi Linear exposes
two distinct cache families:

- MLA full-attention KV cache
- KDA linear-attention conv/recurrent state

These need separate state types and separate tests.

## Design Constraints

- Keep baseline Kimi and AttnRes-Kimi as different types, even if they share
  many submodules.
- Keep current `attnres` invariants intact; do not weaken them to fit baseline
  Kimi.
- Accept a correctness-first implementation before a fast kernel-backed one.

## Open Implementation Question

The public Kimi Linear reference stack uses FlashAttention and `fla-core` KDA
kernels for performance. The first local implementation may need a slower but
semantically correct path. That is acceptable, but the code should make the
boundary between "reference semantics" and "optimized kernel backend" explicit.

## Validation

- config parsing tests for layer types and MoE placement
- cache shape and lifecycle tests
- tensor-shape smoke tests on tiny-random Kimi
- hidden-state parity tests once import exists

## Failure Conditions

- wrong 1-based to 0-based mapping
- MoE activated on the wrong layers
- one cache type silently reused for the other
- baseline layer outputs cannot be matched against the reference model
