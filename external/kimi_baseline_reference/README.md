# RFC 0005 External Baseline Reference Generator

This directory contains the smallest executed external-generator slice for RFC
0005 Gate 2.

What it does today:

- reads an attnres-emitted `baseline-slice-request.json` pilot manifest
- validates the manifest contract exactly for the executed pilot slice
- reads a companion `seeded-init-state.json` bundle emitted by attnres for the
  same seed and artifact fingerprint
- uses that seeded-init bundle only for parameters that remain locally
  initialized after the checkpoint slice is loaded
- loads the pilot checkpoint slice from a local Hugging Face-style artifact
  layout (`config.json`, `model.safetensors.index.json`, shard files)
- runs a standalone Python baseline reference for the current pilot
- writes a schema-compatible `baseline-slice-parity.json`

What it does not claim:

- no in-repo Hugging Face remote-code execution for attnres
- no public-checkpoint parity claim
- no AttnRes-Kimi parity claim
- no benchmark claim
- no optimized-kernel claim

## Executed Scope

This slice is intentionally narrow.

- Artifact source: the local pilot artifact written by
  `cargo run --example kimi_rfc_0005_external_pilot -- write-artifact ...`
- Model fingerprint: `kimi_linear`, `float32`, `num_hidden_layers = 2`,
  `hidden_size = 8`, `vocab_size = 16`
- Import slice: full embeddings + layers `[0, 1]` + final norm + LM head
- Captured hidden layers: `[0]`
- Prompt suite:
  - `single_token_0` with tokens `[0]`
  - `single_token_5` with tokens `[5]`
- Tolerance metadata:
  - `metric = "max_abs_diff"`
  - `runtime_dtype = "float32"`
  - `logits_max_abs_diff = 0.5`
  - `hidden_state_max_abs_diff = 1.0`

The generator rejects drift from that executed pilot manifest loudly.

## Environment

- Python: `3.10`
- Packages:
  - `numpy==2.2.6`
  - `safetensors==0.5.3`
- Remote code: none for this pilot
- Hardware: CPU verified

## Commands

Create the pilot artifact and attnres-emitted bundle:

```bash
cargo run --example kimi_rfc_0005_external_pilot -- write-artifact /tmp/attnres-kimi-pilot-artifact
cargo run --example kimi_rfc_0005_external_pilot -- emit-request-bundle /tmp/attnres-kimi-pilot-artifact /tmp/attnres-kimi-pilot-bundle
```

Create a Python environment and install the exact tested packages:

```bash
python3.10 -m venv .venv-kimi-pilot
source .venv-kimi-pilot/bin/activate
python -m pip install -r external/kimi_baseline_reference/requirements.txt
```

Generate the fixture:

```bash
python -m external.kimi_baseline_reference.generate_baseline_slice_parity \
  --artifact-dir /tmp/attnres-kimi-pilot-artifact \
  --manifest-path /tmp/attnres-kimi-pilot-bundle/baseline-slice-request.json \
  --seeded-init-path /tmp/attnres-kimi-pilot-bundle/seeded-init-state.json \
  --output-path /tmp/attnres-kimi-pilot-bundle/baseline-slice-parity.json
```

Validate the generated fixture with the existing manifest-aware consumer path:

```bash
cargo run --example kimi_rfc_0005_external_pilot -- validate-fixture \
  /tmp/attnres-kimi-pilot-artifact \
  /tmp/attnres-kimi-pilot-bundle/baseline-slice-request.json \
  /tmp/attnres-kimi-pilot-bundle/baseline-slice-parity.json
```

## Deferred Work

- Generalize beyond the fixed executed pilot manifest.
- Remove the attnres-emitted `seeded-init-state.json` bridge by reproducing the
  Burn/NdArray seeded local-init path independently in Python.
- Support public-checkpoint artifacts and Hugging Face remote code.
- Support larger Gate 2 selected-layer slices and Gate 3 end-to-end public
  checkpoint smoke paths.
