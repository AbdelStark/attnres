# Roadmap

## Current State

As of March 16, 2026, this project is **alpha**. It is suitable for research,
examples, and local experimentation with burn-based Transformer models. It is
not yet suitable for production inference services, backend portability claims,
or ecosystem checkpoint interchange. Breaking changes are still possible before
1.0.

## Next Three Milestones

### 1. API Hardening And Source-Of-Truth Docs

What it means:

- Public APIs fail predictably on bad input.
- Contributors can understand the architecture and invariants without reading
  every file.

Exit criteria:

- All public config entry points have typed failure paths.
- Repository docs stop making unverified claims.
- A dedicated algorithm/spec document exists in-repo.

Scope:

- Config validation, docs, contributor guidance, release notes discipline.

Deferred:

- Performance tuning and new backend support.

Dependencies:

- None beyond current codebase.

Estimated complexity:

- Medium

### 2. Backend Validation And Performance Baselines

What it means:

- The project can make evidence-based statements about supported backends and
  performance.

Exit criteria:

- At least one non-NdArray backend is exercised in CI or documented as
  validated manually with reproducible steps.
- Benchmarks exist for standard forward, two-phase forward, and serialization.
- README statements about performance and backend support cite repository data.

Scope:

- Backend test matrix, benchmark baselines, profiling notes.

Deferred:

- Distributed training and serving infrastructure.

Dependencies:

- Milestone 1 docs discipline to avoid repeating unsupported claims.

Estimated complexity:

- Large

### 3. Checkpoint Interchange And Release Discipline

What it means:

- Users can move model state into and out of this library with less manual glue.

Exit criteria:

- PyTorch or `safetensors` checkpoint story is documented and implemented.
- Release checklist is followed for a tagged version with changelog entries.
- Compatibility expectations for saved artifacts are written down.

Scope:

- Import/export utilities, release documentation, artifact compatibility notes.

Deferred:

- Hosted model registry or remote serving integrations.

Dependencies:

- Milestones 1 and 2.

Estimated complexity:

- Large

## Known Limitations Register

Missing:

- Formal algorithm/spec document in-repo.
- PyTorch checkpoint loading.
- Kimi tensor payload loading/parity beyond the baseline-only local supported
  shard-loading slice.
- Full-model Rust-vs-Hugging-Face parity claims for the public 48B checkpoint.
  The repo now has executed public selected-module parity and a completed
  reference-only full baseline smoke report, but not an end-to-end Rust
  parity result on that checkpoint.
- Real-checkpoint AttnRes quality evaluation after training, plus an honest
  in-checkout train/eval runner that continues from the structural bootstrap
  and measures those gates.
- Public/Hugging Face-dependent RFC 0005 benchmark-quality gates beyond the
  selected-module parity and completed reference-only baseline smoke slices now
  in this repo.
- Stable 1.0 API guarantees.

Fragile:

- GPU backend claims are not backed by automated validation in this repo.
- Two-phase inference has good reduced-config equivalence coverage but limited
  benchmark data.

Performance ceilings:

- Only small local benchmarks exist today, including reduced baseline Kimi and
  AttnRes-Kimi benchmark scaffolding.
- No KV-cache or long-context serving path.

Operational knowledge still implicit:

- Release process is not documented beyond local verification commands.
- Production observability guidance does not apply because this is a library,
  not an operated service.

## Release Readiness Snapshot

Checked now:

- `cargo fmt -- --check`
- `cargo clippy -- -D warnings`
- `cargo test --all-features`
- `cargo build --examples`
- `cd web-demo && npm run build`

Still required before any production-ready claim:

- Formal spec/source-of-truth algorithm document.
- Validated non-CPU backend support.
- Evidence-backed performance envelope.
- Checkpoint interchange story.
