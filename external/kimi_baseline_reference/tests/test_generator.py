from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from external.kimi_baseline_reference import generator as generator_module
from external.kimi_baseline_reference.generator import (
    GeneratorError,
    PILOT_LOCAL_INIT_FLOAT32_DIGEST,
    generate_fixture,
    local_init_tensor_float32_digest,
    reconstruct_local_init_tensors,
)


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="attnres-kimi-external-")
        root = Path(cls.temp_dir.name)
        cls.artifact_dir = root / "artifact"
        cls.bundle_dir = root / "bundle"
        cls.fixture_path = root / "baseline-slice-parity.json"
        cls.repo_root = Path(__file__).resolve().parents[3]

        cls._run(
            "cargo",
            "run",
            "--example",
            "kimi_rfc_0005_external_pilot",
            "--",
            "write-artifact",
            str(cls.artifact_dir),
        )
        cls._run(
            "cargo",
            "run",
            "--example",
            "kimi_rfc_0005_external_pilot",
            "--",
            "emit-request-bundle",
            str(cls.artifact_dir),
            str(cls.bundle_dir),
        )
        cls.manifest_path = cls.bundle_dir / "baseline-slice-request.json"
        cls.local_init_contract_path = cls.bundle_dir / "local-init-contract.json"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    @classmethod
    def _run(cls, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, cwd=cls.repo_root, check=True, capture_output=True, text=True)

    def test_generates_fixture_that_validates_with_attnres_consumer(self) -> None:
        fixture = generate_fixture(
            artifact_dir=self.artifact_dir,
            manifest_path=self.manifest_path,
            local_init_contract_path=self.local_init_contract_path,
        )
        self.fixture_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")

        self._run(
            "cargo",
            "run",
            "--example",
            "kimi_rfc_0005_external_pilot",
            "--",
            "validate-fixture",
            str(self.artifact_dir),
            str(self.manifest_path),
            str(self.fixture_path),
        )

    def test_reconstructs_expected_local_init_tensor_digest(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        config = json.loads((self.artifact_dir / "config.json").read_text(encoding="utf-8"))
        tensors = reconstruct_local_init_tensors(
            config=config,
            seed=manifest["seed"],
            required_tensors=manifest["slice"]["required_tensors"],
        )

        self.assertEqual(len(tensors), 26)
        self.assertEqual(
            local_init_tensor_float32_digest(tensors),
            PILOT_LOCAL_INIT_FLOAT32_DIGEST,
        )

    def test_rejects_manifest_kind_mismatch(self) -> None:
        self._expect_manifest_error(
            "kind",
            "attnres.kimi.wrong_slice_request_kind",
            "expected baseline slice request manifest kind",
        )

    def test_rejects_manifest_version_mismatch(self) -> None:
        self._expect_manifest_error("version", 2, "expected baseline slice request manifest version 1")

    def test_rejects_manifest_seed_drift(self) -> None:
        self._expect_manifest_error("seed", 20260319, "baseline manifest field 'seed'")

    def test_rejects_artifact_fingerprint_mismatch(self) -> None:
        self._expect_nested_manifest_error(
            ("artifact", "hidden_size"),
            9,
            "baseline manifest field 'artifact'",
        )

    def test_rejects_artifact_config_drift(self) -> None:
        artifact_dir = self._tampered_artifact_dir()
        config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
        config["q_lora_rank"] = 2
        (artifact_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(GeneratorError, "artifact config field 'q_lora_rank'"):
            generate_fixture(
                artifact_dir=artifact_dir,
                manifest_path=self.manifest_path,
                local_init_contract_path=self.local_init_contract_path,
            )

    def test_rejects_selected_layer_mismatch(self) -> None:
        self._expect_nested_manifest_error(
            ("slice", "selected_hidden_layers"),
            [0, 1],
            "baseline manifest field 'slice.selected_hidden_layers'",
        )

    def test_rejects_prompt_metadata_drift(self) -> None:
        self._expect_nested_manifest_error(
            ("prompts", 0, "name"),
            "single_token_0_drift",
            "baseline manifest field 'prompts'",
        )

    def test_rejects_tolerance_metadata_drift(self) -> None:
        self._expect_nested_manifest_error(
            ("slice", "tolerances", "runtime_dtype"),
            "bfloat16",
            "baseline manifest field 'slice.tolerances'",
        )

    def test_rejects_unsupported_module_request(self) -> None:
        self._expect_nested_manifest_error(
            ("slice", "requested_modules"),
            ["Embeddings"],
            "baseline manifest field 'slice.requested_modules'",
        )

    def test_rejects_unsupported_tensor_request(self) -> None:
        self._expect_nested_manifest_error(
            ("slice", "required_tensors"),
            ["model.embed_tokens.weight"],
            "baseline manifest field 'slice.required_tensors'",
        )

    def test_rejects_local_init_contract_kind_mismatch(self) -> None:
        self._expect_contract_error(
            lambda contract: contract.__setitem__("kind", "attnres.kimi.wrong_local_init_contract"),
            "local init contract field 'kind'",
        )

    def test_rejects_local_init_contract_version_mismatch(self) -> None:
        self._expect_contract_error(
            lambda contract: contract.__setitem__("version", 2),
            "local init contract field 'version'",
        )

    def test_rejects_missing_local_init_contract_field(self) -> None:
        self._expect_contract_error(
            lambda contract: contract.pop("strategy"),
            "local init contract schema mismatch: missing keys",
        )

    def test_rejects_extra_local_init_contract_field(self) -> None:
        self._expect_contract_error(
            lambda contract: contract.__setitem__("seed", 20260318),
            "local init contract schema mismatch: unexpected keys",
        )

    def test_reconstruction_drift_fails_attnres_validation(self) -> None:
        def drifted_reconstruction(config, seed, required_tensors):
            tensors = reconstruct_local_init_tensors(config, seed, required_tensors)
            tensors["lm_head.bias"] = tensors["lm_head.bias"].copy()
            tensors["lm_head.bias"][0] += np.float32(5.0)
            return tensors

        with patch.object(
            generator_module,
            "reconstruct_local_init_tensors",
            side_effect=drifted_reconstruction,
        ):
            fixture = generate_fixture(
                artifact_dir=self.artifact_dir,
                manifest_path=self.manifest_path,
                local_init_contract_path=self.local_init_contract_path,
            )

        self.fixture_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
        with self.assertRaises(subprocess.CalledProcessError):
            self._run(
                "cargo",
                "run",
                "--example",
                "kimi_rfc_0005_external_pilot",
                "--",
                "validate-fixture",
                str(self.artifact_dir),
                str(self.manifest_path),
                str(self.fixture_path),
            )

    def _expect_manifest_error(self, key: str, replacement, message: str) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest[key] = replacement
        path = self.bundle_dir / f"tampered-{key}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(GeneratorError, message):
            generate_fixture(
                self.artifact_dir,
                path,
                self.local_init_contract_path,
            )

    def _expect_nested_manifest_error(self, path_parts, replacement, message: str) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        target = manifest
        for part in path_parts[:-1]:
            target = target[part]
        target[path_parts[-1]] = replacement
        path = self.bundle_dir / ("tampered-" + "-".join(str(part) for part in path_parts) + ".json")
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(GeneratorError, message):
            generate_fixture(
                self.artifact_dir,
                path,
                self.local_init_contract_path,
            )

    def _expect_contract_error(self, mutate, message: str) -> None:
        contract = json.loads(self.local_init_contract_path.read_text(encoding="utf-8"))
        mutate(contract)
        path = self.bundle_dir / "tampered-local-init-contract.json"
        path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(GeneratorError, message):
            generate_fixture(
                self.artifact_dir,
                self.manifest_path,
                path,
            )

    def _tampered_artifact_dir(self) -> Path:
        artifact_dir = Path(tempfile.mkdtemp(prefix="attnres-kimi-artifact-drift-", dir=self.temp_dir.name))
        shutil.copytree(self.artifact_dir, artifact_dir, dirs_exist_ok=True)
        return artifact_dir


if __name__ == "__main__":
    unittest.main()
