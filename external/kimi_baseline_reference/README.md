# RFC 0005 External Baseline Reference Paths

This directory now contains two executable external RFC 0005 reference paths:

- the baseline-slice generator for attnres-emitted local manifests on the
  supported local Kimi baseline surface;
- the public-checkpoint module-probe and smoke harnesses for
  `moonshotai/Kimi-Linear-48B-A3B-Instruct`.

## What It Does

- reads an attnres-emitted `baseline-slice-request.json`
- validates a structurally compatible attnres-emitted manifest on the current
  supported surface
- reads a tiny attnres-emitted `local-init-contract.json`
- reconstructs deterministic Burn/NdArray local-init tensors in Python from:
  - the manifest seed
  - the artifact `config.json`
  - the local-init contract strategy marker
- loads a local Hugging Face-style artifact layout (`config.json`,
  `model.safetensors.index.json`, shard files)
- runs a standalone Python baseline reference for the executed prefix of the
  requested slice
- supports both:
  - full slice fixtures with final logits plus selected hidden states
  - hidden-state-only fixtures when `compare_logits = false`
- writes a schema-compatible `baseline-slice-parity.json`
- leaves the Rust manifest/fixture consumer path unchanged
- can also load official Hugging Face remote code for public Kimi module probes
- can emit an honest full-checkpoint smoke report that records execution-path
  details, timings, host facts, and artifact fingerprints, and still records
  missing-shard or RAM blockers explicitly when prerequisites are not met

No Rust tensor payload bridge is used anymore for this path.

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

## Current Supported Surface

The generator is still intentionally constrained. It currently supports:

- local Hugging Face-style artifacts only
- baseline `model_type = "kimi_linear"` only
- `dtype` values `float32` and `bfloat16`
- `hidden_act = "silu"`
- `q_lora_rank = null`
- `tie_word_embeddings = false`
- `num_shared_experts` in `{0, 1}`
- a one-based MLA/KDA layer schedule that covers every decoder layer exactly
  once
- the deterministic local-init bridge
  `burn.ndarray.lazy_linear_kaiming_uniform.v1`
- attnres-emitted manifests on the currently supported import surface
- optional final-logits comparison via `compare_logits`

The executed regression lock in this checkout is still the original local
pilot:

- artifact source:
  `cargo run --example kimi_rfc_0005_external_pilot -- write-artifact ...`
- artifact/config fingerprint:
  - `model_type = "kimi_linear"`
  - `dtype = "float32"`
  - `num_hidden_layers = 2`
  - `hidden_size = 8`
  - `vocab_size = 16`
- deterministic local-init contract:
  - `strategy = "burn.ndarray.lazy_linear_kaiming_uniform.v1"`
- default emitted request bundle:
  - full embeddings
  - layers `[0, 1]`
  - final norm
  - LM head
  - selected hidden layer `[0]`
  - final logits enabled

`external/kimi_baseline_reference/tests/test_generator.py` now locks both:

- the original pilot deterministic reconstruction digest
- non-pilot but structurally valid attnres-emitted manifests, including
  hidden-only slices without final logits

## External Environment

This is the exact environment used for the executed local regression in this
checkout:

- Artifact source: local pilot artifact emitted by the Rust example above
- Python: `python3.10` (`3.10.19` on the executed machine)
- Packages:
  - `numpy==2.2.6`
  - `safetensors==0.5.3`
- Remote code: none
- Hardware assumption: CPU only
- In-repo Hugging Face execution for attnres itself: none

## Exact Commands

Create the pilot artifact and default request bundle:

```bash
cargo run --example kimi_rfc_0005_external_pilot -- write-artifact /tmp/attnres-kimi-pilot-artifact
cargo run --example kimi_rfc_0005_external_pilot -- emit-request-bundle /tmp/attnres-kimi-pilot-artifact /tmp/attnres-kimi-pilot-bundle
```

Create a request bundle from a caller-provided `KimiBaselineSliceRequestSpec`
JSON instead of the fixed pilot request:

```bash
cargo run --example kimi_rfc_0005_external_pilot -- emit-request-bundle-from-spec \
  /tmp/attnres-kimi-pilot-artifact \
  /tmp/request-spec.json \
  /tmp/attnres-kimi-custom-bundle
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

For the current local compatibility surface, "external generation" now means:

- attnres owns the supported slice manifest and the tiny local-init strategy
  marker
- the external Python reference owns deterministic reconstruction of every
  still-local tensor needed for the executed prefix of that slice
- the external Python reference also owns fixture generation, including
  hidden-state-only fixtures when requested
- attnres then validates the generated fixture unchanged through the existing
  consumer path

That is the full executed local contract in this checkout. It is not a
public-checkpoint claim and not a Hugging Face remote-code claim.

## Public Module-Probe And Smoke Paths

Additional executable surfaces in this directory:

- `module_probe.py` and `generate_module_probe_fixture.py`
  - load official Hugging Face remote code for public Kimi
  - generate fingerprinted probe fixtures for one KDA layer, one MLA layer,
    final norm, and LM head
  - support decode/cache comparisons for the selected attention modules
- `remote_code_support.py`
  - downloads official `configuration_kimi.py` and `modeling_kimi.py`
  - loads them with a CPU fallback for `fla-core` pieces when the current
    machine cannot import the official backend directly
- `run_baseline_smoke.py`
  - supports `cpu_only` and `cpu_disk_offload` execution paths
  - records host facts, explicit dtype handling, attention backend overrides,
    device placement, and SHA-256 fingerprints for the config, shard index,
    remote-code files, and all present shards
  - emits `blocked_missing_full_checkpoint` instead of pretending a smoke run
    passed when shards or prerequisites are missing
  - also supports an approved lower-RAM completion path via
    `--execution-path cpu_disk_offload --offload-dir ... --max-cpu-memory-gib ...`

Executed public smoke in this checkout:

- repo:
  `moonshotai/Kimi-Linear-48B-A3B-Instruct`
- revision:
  `e1df551a447157d4658b573f9a695d57658590e9`
- command:
  `.venv-kimi-public/bin/python -m external.kimi_baseline_reference.run_baseline_smoke --artifact-dir /Users/abdel/.cache/attnres/kimi-linear-48b-e1df551a447157d4658b573f9a695d57658590e9 --output-path /Users/abdel/dev/me/machine-learning/attnres/docs/reports/kimi-public-baseline-smoke-2026-03-23.json --repo-id moonshotai/Kimi-Linear-48B-A3B-Instruct --revision e1df551a447157d4658b573f9a695d57658590e9 --execution-path cpu_disk_offload --offload-dir /Users/abdel/.cache/attnres/kimi-linear-48b-e1df551a447157d4658b573f9a695d57658590e9/baseline-smoke-offload --max-cpu-memory-gib 40`
- result:
  `docs/reports/kimi-public-baseline-smoke-2026-03-23.json`

## Deferred Work

Remaining Gate 2 / Gate 3 dependencies are still deferred:

- support additional local-init strategies beyond
  `burn.ndarray.lazy_linear_kaiming_uniform.v1`
- execute any real-checkpoint AttnRes quality evaluation after training
- add an honest real-checkpoint AttnRes train/eval path beyond structural
  bootstrap plus reduced Gate 6 local stability
- keep any future public-checkpoint claim blocked until the unchanged attnres
  consumer accepts the externally generated fixture for that checkpoint path
