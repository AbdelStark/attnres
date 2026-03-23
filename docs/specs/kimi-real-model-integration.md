# Real-Model Integration Specification: Kimi Linear First

Status: Draft
Date: 2026-03-23
Branch: `codex/kimi-real-model-rfc`

## Why This Exists

`attnres` currently proves the Attention Residuals algorithm inside a compact,
burn-native reference Transformer. That is necessary but not sufficient for the
next milestone. The next serious step is to run the framework on a real model
architecture and on real checkpoint artifacts, with enough validation that a
negative result is trusted and a positive result is meaningful.

## Current Status As Of 2026-03-23

What is now implemented in this checkout:

- local Kimi artifact understanding for official-style `config.json` and
  `model.safetensors.index.json`;
- a separate baseline `KimiLinearModel` execution path with typed MLA/KDA,
  dense MLP, sparse MoE, and cache families;
- a separate `KimiAttnResModel` execution path with dual AttnRes placement,
  sublayer-space block boundaries, and reduced-config two-phase agreement
  checks;
- local shard loading for the currently supported baseline tensor subset into
  both model families;
- local deterministic Gate 1 baseline parity on a tiny-random Kimi-style
  fixture;
- local Gate 2 baseline and AttnRes payload-loading preparation slices;
- a local external baseline slice-reference path that now accepts
  attnres-emitted hidden-only slice manifests in addition to full
  logits-plus-hidden fixtures on the supported local surface;
- executed public-checkpoint module-probe parity against the official
  Hugging Face remote-code path for one KDA layer, one MLA layer, final norm,
  and LM head, including decode/cache comparisons for the selected attention
  modules;
- a completed full public baseline smoke on the real 48B checkpoint through an
  approved `cpu_disk_offload` execution path, with an evidence report that
  records timings, host facts, execution-path details, and artifact
  fingerprints;
- an explicit baseline-to-AttnRes bootstrap policy and load report;
- reduced optimizer-backed AttnRes-Kimi training-stability validation on a
  hybrid KDA/MLA plus dense/MoE reduced config.

What is still missing for a meaningful real-model result:

- any AttnRes-Kimi checkpoint or trained warm-start that would make imported
  baseline tensors a meaningful quality test rather than a structural
  bootstrap;
- post-training quality validation for AttnRes-Kimi on a real checkpoint;
- an honest in-checkout train/eval execution path that continues from the
  structural bootstrap and measures quality gates on a real checkpoint.

The current status summary for this milestone lives in
[`docs/status/kimi-real-model-status.md`](../status/kimi-real-model-status.md).

The ideal first target is Kimi Linear because:

- the official Attention Residuals repository reports downstream results on
  Kimi Linear 48B / 3B activated trained on 1.4T tokens;
- Kimi Linear is public, open-weight, and has a published Hugging Face config,
  modeling code, and sharded `safetensors` index;
- the architecture is materially more realistic than the current in-repo model:
  hybrid attention, MoE, large context, and stateful decoding.

## Research Findings

### 1. Public Kimi Linear is the correct first target

The public `moonshotai/Kimi-Linear-48B-A3B-Instruct` checkpoint exposes:

- `model_type = "kimi_linear"`
- `num_hidden_layers = 27`
- `hidden_size = 2304`
- `num_attention_heads = 32`
- `num_key_value_heads = 32`
- `hidden_act = "silu"`
- `num_experts = 256`
- `num_experts_per_token = 8`
- `num_shared_experts = 1`
- `model_max_length = 1_048_576`
- `dtype = "bfloat16"`
- 20 sharded `safetensors` files with `metadata.total_size = 98,245,528,576`
  bytes and `metadata.total_parameters = 49,122,681,728`

The public layer mix is hybrid:

- full-attention layers are 1-based indices `[4, 8, 12, 16, 20, 24, 27]`;
- linear-attention layers are the remaining 20 layers;
- `linear_attn_config.num_heads = 32`, `linear_attn_config.head_dim = 128`,
  and `short_conv_kernel_size = 4`;
- MLA-relevant config includes `kv_lora_rank = 512`, `q_lora_rank = null`,
  `qk_nope_head_dim = 128`, `qk_rope_head_dim = 64`, `v_head_dim = 128`, and
  `mla_use_nope = true`;
- layer 0 uses a dense MLP;
- layers `>= 1` use sparse MoE according to `first_k_dense_replace = 1` and
  `moe_layer_freq = 1`;
- embeddings are not tied because `tie_word_embeddings = false`.

### 2. Public Kimi Linear is not an AttnRes checkpoint

This is the most important research finding.

The public `modeling_kimi.py` still uses standard PreNorm residual additions:

- normalize input
- run attention
- `hidden_states = residual + attention_output`
- normalize again
- run MLP or MoE
- `hidden_states = residual + mlp_or_moe_output`

The official Attention Residuals repository currently provides the paper,
figures, and pseudocode, but not a released Kimi-specific AttnRes checkpoint or
full training/inference code path.

Therefore this milestone cannot honestly be framed as "load the official Kimi
AttnRes model" because that public artifact does not exist today. The correct
framing is:

1. implement baseline Kimi Linear compatibility;
2. implement an AttnRes-augmented Kimi variant;
3. validate the augmentation against strong numerical and training-stability
   gates;
4. only then talk about benchmarking.

This remains true on 2026-03-23. The repo can now import the supported
baseline tensor subset into `KimiAttnResModel`, but that is still a research
bootstrap with locally initialized AttnRes operators, not a meaningful
real-model AttnRes evaluation.

### 3. Kimi K2 is out of initial scope

The public `moonshotai/Kimi-K2-Base` config is a different class of target:

- `architectures = ["DeepseekV3ForCausalLM"]`
- `num_hidden_layers = 61`
- `hidden_size = 7168`
- `num_attention_heads = 64`
- FP8 quantization metadata is present
- rope scaling is YaRN-based

That is a separate DeepSeekV3-derived stack and should not be mixed into the
first milestone. Kimi Linear remains the correct first real-model target.

## Current Gap

The current repository model is intentionally small and clear. Compared with
public Kimi Linear, it is missing or incompatible with:

| Surface | Current `attnres` | Needed for Kimi Linear |
| --- | --- | --- |
| Decoder stack | simple Transformer | Kimi-specific decoder stack |
| Self-attention | dense MHA | hybrid MLA + KDA |
| Feed-forward | GELU MLP | SiLU dense MLP + sparse MoE |
| Checkpoints | burn recorders only | Hugging Face `config.json` + sharded `safetensors` |
| Context/runtime | toy/local | 1M-context architecture and stateful decode semantics |
| Caches | none | KDA conv/recurrent states and full-attention KV caches |
| Tokenizer path | none | Hugging Face tokenizer-compatible path |
| Warm start story | not needed | explicit decision for baseline-to-AttnRes transition |

There is also a subtler gap: the current project assumes the algorithmic
invariants of AttnRes, but not the compatibility invariants of an external
architecture and external checkpoint format.

## Goals

- Add a baseline-compatible Kimi Linear architecture to the project without
  destabilizing the existing reference Transformer.
- Load and validate Hugging Face Kimi configs and sharded `safetensors`
  artifacts.
- Add an AttnRes-augmented Kimi variant that preserves the paper's invariants:
  zero-initialized pseudo-queries, depth softmax over block/layer sources, and
  two AttnRes operations per decoder layer.
- Create a validation harness strong enough to catch wrong tensor mappings,
  wrong cache semantics, wrong layer typing, and silent numerical drift.
- Produce benchmark outputs that distinguish architecture correctness from
  training-quality claims.

## Non-Goals For V1

- Claiming production-grade Kimi inference in burn.
- Claiming parity with Moonshot's internal training recipe or benchmark numbers.
- Adding Kimi K2 support in the first milestone.
- Promising that a public baseline Kimi checkpoint can be hot-swapped into an
  AttnRes Kimi model with no retraining cost.
- Promising validated GPU throughput until the kernels and benchmarks exist.

## Required Decisions

### Decision 1: Add a separate Kimi module tree

The current `AttnResTransformer` should remain the small reference model. Kimi
support should live in a separate module tree, for example `src/kimi/`, so the
existing library remains readable and the new work can evolve without
destabilizing the teaching/reference path.

### Decision 2: Treat baseline Kimi and AttnRes-Kimi as separate model classes

The public checkpoint matches baseline Kimi Linear, not AttnRes-Kimi.
Therefore:

- baseline Kimi compatibility must exist on its own;
- AttnRes-Kimi must be a derived architecture with its own config and tests;
- any baseline weight transplant into AttnRes-Kimi is a research bootstrap,
  not proof of benchmark-ready quality.

### Decision 3: Sharded import is a first-class feature, not a utility

Kimi Linear ships as 20 shards. Import must be designed around:

- parsing `config.json`;
- parsing `model.safetensors.index.json`;
- mapping names to local modules;
- loading lazily per shard;
- reporting unmapped, duplicate, and mismatched tensors clearly.

The local Gate 2 slice-parity contract is now strong enough to request
hidden-state-only prefix slices without also demanding final norm and LM head
parity. That removes one local dead end, but it does not remove the need for a
public Hugging Face reference execution path.

### Decision 4: Validation needs a Python reference harness

For this milestone, burn-only tests are not enough. The public reference model
is the Hugging Face implementation with custom remote code. The Rust
implementation must be compared against that reference at the config, tensor,
layer, and end-to-end levels.

## Milestone Structure

### M1. Baseline Kimi compatibility

Exit criteria:

- local config loader understands official Kimi Linear config files;
- decoder stack can represent the MLA/KDA layer schedule;
- dense MLP and sparse MoE semantics are represented locally;
- unit tests cover layer typing, config validation, and cache shape invariants.

### M2. Sharded checkpoint import

Exit criteria:

- Kimi `config.json` and `model.safetensors.index.json` load successfully;
- import coverage is 100% for the tensors explicitly supported by the local
  model class;
- tiny-random Kimi weights load end-to-end;
- slice loading for selected layers works on the 48B checkpoint.

### M3. Baseline parity

Exit criteria:

- tiny-random deterministic baseline parity remains locked in-repo;
- selected public-checkpoint probes on the 48B checkpoint match the official
  reference for identical inputs;
- stateful decode parity is demonstrated for the selected MLA and KDA probes.

### M4. AttnRes-Kimi integration

Exit criteria:

- each decoder layer uses two AttnRes operations around the existing attention
  and MLP/MoE sublayers;
- block boundaries remain defined in sublayer space;
- internal invariants are tested on mixed MLA/KDA stacks;
- forward and two-phase AttnRes paths agree on supported configs.

### M5. Benchmark and research validation

Exit criteria:

- there is a written distinction between baseline-parity benchmarks,
  architecture-overhead benchmarks, and quality benchmarks;
- no claims are made beyond what was actually run;
- failure conditions are documented and treated as legitimate outcomes.

## Current Critical Gap

The immediate blocker to a meaningful real-model test is no longer selected
module correctness or baseline smoke. Those slices are now executed and
validated.

The remaining blocker is AttnRes-specific quality after training:

- baseline Kimi weights are not an AttnRes checkpoint;
- zero-initialized pseudo-queries do not preserve standard residual behavior;
- meaningful AttnRes-on-real-model claims therefore still require continued
  pretraining or fresh training plus stability checks;
- this checkout does not yet provide a real-checkpoint train/eval runner that
  continues from the structural bootstrap and measures those gates honestly.

## Hard Truths And Risks

### Risk 1: No exact baseline hot swap

The current AttnRes invariant is zero-initialized pseudo-queries, which produce
uniform depth weights at initialization. That is not equivalent to standard
residual addition. Therefore public baseline Kimi weights cannot be presented
as an exact parity-preserving initialization for AttnRes-Kimi.

Implication:

- baseline checkpoint import is a separate deliverable from AttnRes-Kimi
  benchmark claims;
- meaningful AttnRes quality results likely require continued pretraining or
  fresh training on a smaller but real Kimi-style model.

### Risk 2: Kernel mismatch

The public Kimi Linear stack depends on specialized KDA kernels in `fla-core`
for practical speed. A semantically correct Rust path may be possible before a
competitive performance path exists.

Implication:

- correctness-first implementation is acceptable;
- performance claims are forbidden until profiled and validated.

### Risk 3: Memory footprint

The public checkpoint is roughly 98.2 GB of weights before runtime overhead.

Implication:

- full local end-to-end runs need an explicit hardware plan;
- slice loading and tiny-random parity are mandatory early gates;
- an approved disk-offload path can close the baseline smoke gate on a
  lower-RAM host, but that does not convert the result into a performance
  claim;
- benchmark work must separate "can load" from "can serve efficiently."

## Go/No-Go Gates

- If tiny-random baseline parity cannot be achieved, stop before AttnRes-Kimi
  integration.
- If the loader cannot prove complete and correct tensor-name coverage, stop
  before numerical benchmarking.
- If KDA cache semantics remain ambiguous after reference probing, stop before
  long-context claims.
- If AttnRes-Kimi cannot train stably on a reduced Kimi-style config, do not
  present benchmark deltas as evidence for the full model family.

## Deliverables In This Branch

- this source-of-truth specification;
- an RFC set that breaks the milestone into reviewable technical decisions;
- a validation plan strong enough to treat failure as informative, not as
  hidden debt.

## References

- Attention Residuals official repo and README:
  <https://github.com/MoonshotAI/Attention-Residuals>
- Attention Residuals paper:
  <https://arxiv.org/abs/2603.15031>
- Kimi Linear official repo:
  <https://github.com/MoonshotAI/Kimi-Linear>
- Kimi Linear paper:
  <https://arxiv.org/abs/2510.26692>
- Public Kimi Linear checkpoint:
  <https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct>
- Public tiny-random Kimi checkpoint:
  <https://huggingface.co/tiny-random/kimi-linear>
- Public Kimi K2 checkpoint:
  <https://huggingface.co/moonshotai/Kimi-K2-Base>
- Burn import docs:
  <https://docs.rs/burn-import/latest/burn_import/>
- Burn store announcement and docs entry point:
  <https://burn.dev/blog/release-0.19.0/>
