# RFC 0005 External Baseline Reference Generator

This directory contains the current executed RFC 0005 Gate 2 external-generator
pilot.

## What It Does

- reads an attnres-emitted `baseline-slice-request.json`
- validates the executed pilot manifest exactly
- reads a tiny attnres-emitted `local-init-contract.json`
- reconstructs the pilot's deterministic Burn/NdArray local-init tensors in
  Python from:
  - the manifest seed
  - the artifact `config.json`
  - the local-init contract strategy marker
- loads the pilot checkpoint slice from a local Hugging Face-style artifact
  layout (`config.json`, `model.safetensors.index.json`, shard files)
- runs a standalone Python baseline reference for the executed pilot
- writes a schema-compatible `baseline-slice-parity.json`
- leaves the Rust manifest/fixture consumer path unchanged

No Rust tensor payload bridge is used anymore for the pilot.

## Residual Bridge

The remaining bridge is `local-init-contract.json`:

- `kind`
  Why it remains: lets the external generator reject the wrong bridge schema
  immediately.
- `version`
  Why it remains: versions the bridge schema independently from the slice
  manifest.
- `strategy`
  Why it remains: names the exact deterministic local-init contract the Python
  side must emulate. The manifest already carries the seed, prompt suite,
  slice metadata, tolerances, and artifact fingerprint; the artifact
  `config.json` carries the model dimensions and schedule; this one field is
  the only residual metadata needed to identify the lazy-init algorithm/order.

There are intentionally no tensor payloads, no duplicated seed field, and no
duplicated artifact fingerprint fields in this bridge.

## Executed Scope

This pilot is intentionally narrow.

- Artifact source:
  `cargo run --example kimi_rfc_0005_external_pilot -- write-artifact ...`
- Artifact/config fingerprint:
  - `model_type = "kimi_linear"`
  - `dtype = "float32"`
  - `num_hidden_layers = 2`
  - `hidden_size = 8`
  - `vocab_size = 16`
- Deterministic local-init contract:
  - `strategy = "burn.ndarray.lazy_linear_kaiming_uniform.v1"`
- Import slice:
  - full embeddings
  - layers `[0, 1]`
  - final norm
  - LM head
- Captured hidden layers:
  - `[0]`
- Prompt suite:
  - `single_token_0` with tokens `[0]`
  - `single_token_5` with tokens `[5]`
- Tolerance metadata:
  - `metric = "max_abs_diff"`
  - `runtime_dtype = "float32"`
  - `logits_max_abs_diff = 0.5`
  - `hidden_state_max_abs_diff = 1.0`

The generator rejects drift from that executed pilot loudly: seed drift,
artifact/config drift, selected-layer drift, prompt drift, tolerance drift,
unsupported module/tensor requests, local-init contract field drift, and
reconstruction drift coverage all exist in
`external/kimi_baseline_reference/tests/test_generator.py`.

## External Environment

This is the exact environment used for the executed pilot in this checkout:

- Artifact source: local pilot artifact emitted by the Rust example above
- Python: `python3.10` (`3.10.19` on the executed machine)
- Packages:
  - `numpy==2.2.6`
  - `safetensors==0.5.3`
- Remote code: none
- Hardware assumption: CPU only
- In-repo Hugging Face execution for attnres itself: none

## Exact Commands

Create the pilot artifact and request bundle:

```bash
cargo run --example kimi_rfc_0005_external_pilot -- write-artifact /tmp/attnres-kimi-pilot-artifact
cargo run --example kimi_rfc_0005_external_pilot -- emit-request-bundle /tmp/attnres-kimi-pilot-artifact /tmp/attnres-kimi-pilot-bundle
```

Create the tested Python environment and install the exact packages:

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
  --local-init-contract-path /tmp/attnres-kimi-pilot-bundle/local-init-contract.json \
  --output-path /tmp/attnres-kimi-pilot-bundle/baseline-slice-parity.json
```

Validate the generated fixture with the unchanged manifest-aware consumer:

```bash
cargo run --example kimi_rfc_0005_external_pilot -- validate-fixture \
  /tmp/attnres-kimi-pilot-artifact \
  /tmp/attnres-kimi-pilot-bundle/baseline-slice-request.json \
  /tmp/attnres-kimi-pilot-bundle/baseline-slice-parity.json
```

## Executable Meaning

For the current pilot, "external generation" now means:

- attnres owns the supported slice manifest and the tiny local-init strategy
  marker
- the external Python reference owns deterministic reconstruction of every
  still-local tensor needed for that slice
- the external Python reference also owns fixture generation
- attnres then validates the generated fixture unchanged through the existing
  consumer path

That is the full executed contract for this pilot. It is not a public-checkpoint
claim and not a Hugging Face remote-code claim.

## Deferred Work

Remaining Gate 2 / Gate 3 dependencies are still deferred:

- generalize beyond the fixed executed pilot manifest and config
- support additional local-init strategies beyond
  `burn.ndarray.lazy_linear_kaiming_uniform.v1`
- support public-checkpoint artifacts instead of the local pilot artifact
- support Hugging Face remote code in the external reference path
- execute Gate 2 selected-layer parity against a real public checkpoint
- execute Gate 3 end-to-end public-checkpoint smoke coverage
- support broader selected-layer/module combinations than the current pilot
- keep any future public-checkpoint claim blocked until the unchanged attnres
  consumer accepts the externally generated fixture for that checkpoint path
