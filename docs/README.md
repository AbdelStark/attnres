# Documentation

This folder now contains the source-of-truth planning set for the real-model
milestone.

The current code-backed delivery slice for that plan lives in `src/kimi/` and
now includes RFC 0001 artifact understanding, RFC 0002 baseline Kimi Linear
execution scaffolding, and RFC 0003 checkpoint-import scaffolding for shard
index inspection, tensor coverage reporting, selected-layer/full shard
planning, and explicit dtype policy. Baseline parity is still deferred, and RFC
0004 AttnRes-Kimi integration is still deferred.

- `specs/kimi-real-model-integration.md`: top-level specification for running
  `attnres` on a real model architecture, with Kimi Linear as the first target.
- `rfcs/0001-real-model-milestone-scope.md`: milestone definition, scope, and
  sequencing.
- `rfcs/0002-kimi-baseline-compatibility.md`: baseline Kimi Linear
  compatibility architecture.
- `rfcs/0003-sharded-checkpoint-import.md`: Hugging Face config and sharded
  `safetensors` import plan.
- `rfcs/0004-kimi-attnres-integration.md`: how AttnRes will be inserted into
  the Kimi architecture.
- `rfcs/0005-validation-and-benchmark-plan.md`: validation gates, failure
  criteria, and benchmark plan.
