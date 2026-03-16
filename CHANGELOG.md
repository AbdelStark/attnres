# Changelog

All notable user-visible changes to this project will be documented here.

The repository did not maintain a structured changelog before March 16, 2026.

## [Unreleased]

### Added

- `ConfigError` with typed validation for invalid model configuration.
- Fallible initialization helpers such as `try_validate`, `try_init_model`,
  `try_init_layer`, and `try_init_op`.
- `ARCHITECTURE.md` and `CONTRIBUTING.md`.

### Changed

- Serialization APIs now accept `Path`-like inputs instead of only `&str`.
- README, roadmap, and agent context files now reflect the current alpha status
  and verified commands instead of stale or aspirational claims.

### Fixed

- Explicit validation for `num_heads = 0`.
- Explicit validation for out-of-range `layer_idx` values.
