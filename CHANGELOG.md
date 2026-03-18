# Changelog

All notable user-visible changes to this project will be documented here.

The repository did not maintain a structured changelog before March 16, 2026.

## [Unreleased]

### Added

- `ConfigError` with typed validation for invalid model configuration.
- Fallible initialization helpers such as `try_validate`, `try_init_model`,
  `try_init_layer`, and `try_init_op`.
- `ARCHITECTURE.md` and `CONTRIBUTING.md`.
- `attnres::kimi` Phase A scaffolding for RFC 0001: typed Kimi artifact config,
  typed layer schedules, shard-index metadata, and import planning/report APIs.
- Baseline-only local sharded-`safetensors` payload loading for the supported
  Kimi tensor subset via `KimiLinearModel::try_from_artifact_dir` and
  `KimiArtifactUnderstanding::try_init_baseline_model_from_dir`, plus explicit
  negative tests for missing shards, unsupported tensors/dtypes, shape
  mismatches, and incomplete selected-layer payload coverage.
- Baseline-only Gate 2 slice parity fixture consumption via
  `attnres::kimi::KimiBaselineSliceParityFixture` and
  `attnres::kimi::compare_baseline_slice_parity_fixture_from_dir`, plus
  negative tests for fixture kind/version drift, selected-layer mismatches,
  prompt/token mismatches, tolerance metadata drift, and unsupported
  module-request drift.
- Baseline-only Gate 2 external-generator handoff manifests via
  `attnres::kimi::KimiBaselineSliceRequestManifest`,
  `KimiArtifactUnderstanding::try_build_baseline_slice_request_manifest`, and
  `compare_baseline_slice_parity_fixture_with_manifest_from_dir`, plus
  negative tests for manifest kind/version drift, invalid selected-hidden-layer
  requests, prompt/tolerance drift against emitted manifests, and stale
  module/tensor manifest metadata.

### Changed

- Serialization APIs now accept `Path`-like inputs instead of only `&str`.
- README, roadmap, and agent context files now reflect the current alpha status
  and verified commands instead of stale or aspirational claims.
- Top-level docs now state that Kimi work is staged and that only Phase A
  artifact understanding is implemented in this checkout.
- Kimi milestone docs now describe the new local baseline-only payload-loading
  slice in executable repo terms and keep public-checkpoint/Hugging Face claims
  deferred.
- Kimi milestone docs now also describe the new external baseline slice-fixture
  consumption path as fixture consumption only, keeping Hugging Face execution,
  fixture generation, and any public-checkpoint parity claim deferred.
- Kimi milestone docs now describe the new baseline-only external-generator
  handoff manifest slice as manifest emission plus external fixture
  consumption, without implying in-repo Hugging Face execution or any
  public-checkpoint parity claim.

### Fixed

- Explicit validation for `num_heads = 0`.
- Explicit validation for out-of-range `layer_idx` values.
