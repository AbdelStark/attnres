# Contributing

## Scope

This repository is a reference implementation. Favor correctness, readable
code, and documented invariants over cleverness.

## Setup

Prerequisites:

- Rust stable 1.80+.
- Node.js and npm for `web-demo/`.
- `wasm-pack` plus the `wasm32-unknown-unknown` target for the web demo build.

Core setup:

```bash
cargo build
cargo test --all-features
```

Web demo setup:

```bash
cd web-demo
npm install
npm run build
```

## Required Checks

Run these before opening a PR:

```bash
cargo fmt -- --check
cargo clippy -- -D warnings
cargo test --all-features
cargo build --examples
cd web-demo && npm run build
```

## Change Rules

- If you change algorithm behavior, add or update tests that would fail without
  the change.
- If config may come from files, CLI flags, or user input, prefer
  `try_validate` / `try_init_model` over panic-only helpers.
- Keep public docs honest. Do not claim support, performance, or validation
  that the repository does not currently verify.
- Do not modify `Cargo.toml` dependencies or `.github/workflows/` without
  maintainer approval.
- Avoid untracked `TODO` / `FIXME` comments. Open an issue or fix the problem.

## Review Checklist

- New behavior is covered by tests.
- Public API changes are reflected in `README.md`, `ARCHITECTURE.md`, or rustdoc
  where appropriate.
- Error messages include enough context for a caller to act on them.
- New examples or docs use commands that work on a clean checkout.

## Commit Style

Preferred format:

```text
type(scope): short description
```

Examples:

- `fix(config): reject zero attention heads`
- `docs(readme): document alpha status and limitations`
- `test(model): cover odd block boundary behavior`
