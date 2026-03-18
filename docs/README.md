# Documentation

This folder now contains the source-of-truth planning set for the real-model
milestone.

The current code-backed delivery slice for that plan lives in `src/kimi/` and
now includes RFC 0001 artifact understanding, RFC 0002 baseline Kimi Linear
execution scaffolding, RFC 0003 checkpoint-import scaffolding for shard index
inspection, tensor coverage reporting, selected-layer/full shard planning,
explicit dtype policy, and local payload loading for the supported baseline
tensor subset into both `KimiLinearModel` and `KimiAttnResModel`, plus RFC
0004 AttnRes-Kimi execution scaffolding with a separate model/layer/state
path. RFC 0005 is now partially executable in-repo: baseline-only local Gate 1
parity for a deterministic tiny-random Kimi-style fixture, local Gate 2
payload-loading prep with real sharded payload bytes, baseline-only Gate 2
external-generator request manifests plus externally generated fixture
consumption for the supported `KimiLinearModel` subset, a local external
Python fixture-generator path for that handoff that now supports hidden-only
prefix slices as well as full logits fixtures, Gate 4 functional
validation, reduced-config Gate 5 numerical agreement, and reduced local
benchmark scaffolding. Hugging Face remote-code baseline parity,
public-checkpoint parity, AttnRes-Kimi external parity generation, and other
external validation still remain deferred.

- `status/kimi-real-model-status.md`: current milestone status, the exact gap
  to a meaningful real-model test, and the next critical blockers.

- `specs/kimi-real-model-integration.md`: top-level specification for running
  `attnres` on a real model architecture, with Kimi Linear as the first target.
- `rfcs/0001-real-model-milestone-scope.md`: milestone definition, scope, and
  sequencing.
- `rfcs/0002-kimi-baseline-compatibility.md`: baseline Kimi Linear
  compatibility architecture.
- `rfcs/0003-sharded-checkpoint-import.md`: Hugging Face config and sharded
  `safetensors` import plan.
- `rfcs/0004-kimi-attnres-integration.md`: how AttnRes is inserted into the
  separate AttnRes-Kimi execution path.
- `rfcs/0005-validation-and-benchmark-plan.md`: validation gates, failure
  criteria, and benchmark plan.
