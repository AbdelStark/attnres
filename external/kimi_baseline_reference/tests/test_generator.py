from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
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

    def test_generates_hidden_only_fixture_that_validates_with_attnres_consumer(self) -> None:
        manifest_path, contract_path, bundle_dir = self._emit_bundle_from_request_spec(
            "hidden-only",
            {
                "seed": 20260319,
                "import_selection": {
                    "layer_indices": [0],
                    "include_embeddings": True,
                    "include_final_norm": False,
                    "include_lm_head": False,
                },
                "selected_hidden_layers": [0],
                "compare_logits": False,
                "prompts": [
                    {"name": "ascending_pair", "input_ids": [0, 5]},
                    {"name": "repeat_pair", "input_ids": [5, 5]},
                ],
                "tolerances": {
                    "metric": "max_abs_diff",
                    "runtime_dtype": "float32",
                    "logits_max_abs_diff": 0.5,
                    "hidden_state_max_abs_diff": 1.0,
                },
            },
        )
        fixture = generate_fixture(
            artifact_dir=self.artifact_dir,
            manifest_path=manifest_path,
            local_init_contract_path=contract_path,
        )
        self.assertTrue(all(result["logits"] is None for result in fixture["prompt_results"]))

        fixture_path = bundle_dir / "baseline-slice-parity.json"
        fixture_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
        self._run(
            "cargo",
            "run",
            "--example",
            "kimi_rfc_0005_external_pilot",
            "--",
            "validate-fixture",
            str(self.artifact_dir),
            str(manifest_path),
            str(fixture_path),
        )

    def test_reconstructs_expected_local_init_tensor_digest(self) -> None:
        manifest = self._load_json(self.manifest_path)
        config = self._load_json(self.artifact_dir / "config.json")
        tensors = reconstruct_local_init_tensors(
            config=config,
            seed=manifest["seed"],
            required_tensors=manifest["slice"]["required_tensors"],
            selected_hidden_layers=manifest["slice"]["selected_hidden_layers"],
            compare_logits=manifest["slice"].get("compare_logits", True),
        )

        self.assertEqual(len(tensors), 26)
        self.assertEqual(
            local_init_tensor_float32_digest(tensors),
            PILOT_LOCAL_INIT_FLOAT32_DIGEST,
        )

    def test_accepts_non_pilot_seed_when_manifest_is_structurally_valid(self) -> None:
        manifest = self._load_json(self.manifest_path)
        manifest["seed"] = 7
        fixture = generate_fixture(
            artifact_dir=self.artifact_dir,
            manifest_path=self._write_manifest("seed-7", manifest),
            local_init_contract_path=self.local_init_contract_path,
        )
        self.assertEqual(fixture["seed"], 7)

    def test_accepts_non_pilot_prompt_suite_when_manifest_is_structurally_valid(self) -> None:
        manifest = self._load_json(self.manifest_path)
        manifest["prompts"] = [
            {"name": "triple", "input_ids": [0, 1, 2]},
            {"name": "repeat", "input_ids": [5, 5, 5]},
        ]
        fixture = generate_fixture(
            artifact_dir=self.artifact_dir,
            manifest_path=self._write_manifest("prompt-variation", manifest),
            local_init_contract_path=self.local_init_contract_path,
        )
        self.assertEqual(
            [prompt["prompt_name"] for prompt in fixture["prompt_results"]],
            ["triple", "repeat"],
        )

    def test_rejects_manifest_kind_mismatch(self) -> None:
        self._expect_manifest_error(
            "kind",
            "attnres.kimi.wrong_slice_request_kind",
            "expected baseline slice request manifest kind",
        )

    def test_rejects_manifest_version_mismatch(self) -> None:
        self._expect_manifest_error(
            "version",
            2,
            "expected baseline slice request manifest version 1",
        )

    def test_rejects_artifact_fingerprint_mismatch(self) -> None:
        self._expect_nested_manifest_error(
            ("artifact", "hidden_size"),
            9,
            "baseline manifest artifact field 'hidden_size'",
        )

    def test_rejects_artifact_config_drift(self) -> None:
        artifact_dir = self._tampered_artifact_dir()
        config = self._load_json(artifact_dir / "config.json")
        config["q_lora_rank"] = 2
        (artifact_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(GeneratorError, "artifact config field 'q_lora_rank'"):
            generate_fixture(
                artifact_dir=artifact_dir,
                manifest_path=self.manifest_path,
                local_init_contract_path=self.local_init_contract_path,
            )

    def test_rejects_hidden_only_manifest_without_selected_hidden_layers(self) -> None:
        manifest = self._load_json(self.manifest_path)
        manifest["slice"]["compare_logits"] = False
        manifest["slice"]["selected_hidden_layers"] = []
        with self.assertRaisesRegex(
            GeneratorError,
            "baseline manifest field 'slice.selected_hidden_layers' must be non-empty when compare_logits = false",
        ):
            generate_fixture(
                artifact_dir=self.artifact_dir,
                manifest_path=self._write_manifest("hidden-only-empty-layers", manifest),
                local_init_contract_path=self.local_init_contract_path,
            )

    def test_rejects_compare_logits_true_without_final_norm(self) -> None:
        manifest = self._load_json(self.manifest_path)
        manifest["slice"]["import_selection"]["include_final_norm"] = False
        with self.assertRaisesRegex(
            GeneratorError,
            "baseline manifest field 'slice.import_selection.include_final_norm' must be true when compare_logits = true",
        ):
            generate_fixture(
                artifact_dir=self.artifact_dir,
                manifest_path=self._write_manifest("missing-final-norm", manifest),
                local_init_contract_path=self.local_init_contract_path,
            )

    def test_rejects_duplicate_prompt_names(self) -> None:
        manifest = self._load_json(self.manifest_path)
        manifest["prompts"][1]["name"] = manifest["prompts"][0]["name"]
        with self.assertRaisesRegex(GeneratorError, "duplicate prompt name"):
            generate_fixture(
                artifact_dir=self.artifact_dir,
                manifest_path=self._write_manifest("duplicate-prompts", manifest),
                local_init_contract_path=self.local_init_contract_path,
            )

    def test_rejects_tolerance_metadata_drift(self) -> None:
        self._expect_nested_manifest_error(
            ("slice", "tolerances", "runtime_dtype"),
            "bfloat16",
            "baseline manifest field 'slice.tolerances' field 'runtime_dtype'",
        )

    def test_rejects_unknown_required_tensor(self) -> None:
        manifest = self._load_json(self.manifest_path)
        manifest["slice"]["required_tensors"].append("model.layers.99.self_attn.q_proj.weight")
        with self.assertRaisesRegex(GeneratorError, "artifact shard index is missing required tensors"):
            generate_fixture(
                artifact_dir=self.artifact_dir,
                manifest_path=self._write_manifest("unknown-required-tensor", manifest),
                local_init_contract_path=self.local_init_contract_path,
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
        def drifted_reconstruction(
            config: dict[str, Any],
            seed: int,
            required_tensors: list[str],
            selected_hidden_layers: list[int],
            compare_logits: bool,
        ) -> dict[str, np.ndarray]:
            tensors = reconstruct_local_init_tensors(
                config=config,
                seed=seed,
                required_tensors=required_tensors,
                selected_hidden_layers=selected_hidden_layers,
                compare_logits=compare_logits,
            )
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

    def _expect_manifest_error(self, key: str, replacement: Any, message: str) -> None:
        manifest = self._load_json(self.manifest_path)
        manifest[key] = replacement
        with self.assertRaisesRegex(GeneratorError, message):
            generate_fixture(
                artifact_dir=self.artifact_dir,
                manifest_path=self._write_manifest(key, manifest),
                local_init_contract_path=self.local_init_contract_path,
            )

    def _expect_nested_manifest_error(
        self,
        path_parts: tuple[Any, ...],
        replacement: Any,
        message: str,
    ) -> None:
        manifest = self._load_json(self.manifest_path)
        target = manifest
        for part in path_parts[:-1]:
            target = target[part]
        target[path_parts[-1]] = replacement
        with self.assertRaisesRegex(GeneratorError, message):
            generate_fixture(
                artifact_dir=self.artifact_dir,
                manifest_path=self._write_manifest("-".join(str(part) for part in path_parts), manifest),
                local_init_contract_path=self.local_init_contract_path,
            )

    def _expect_contract_error(self, mutate, message: str) -> None:
        contract = self._load_json(self.local_init_contract_path)
        mutate(contract)
        path = self.bundle_dir / "tampered-local-init-contract.json"
        path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(GeneratorError, message):
            generate_fixture(
                artifact_dir=self.artifact_dir,
                manifest_path=self.manifest_path,
                local_init_contract_path=path,
            )

    def _emit_bundle_from_request_spec(
        self,
        bundle_name: str,
        request_spec: dict[str, Any],
    ) -> tuple[Path, Path, Path]:
        bundle_dir = Path(tempfile.mkdtemp(prefix=f"attnres-kimi-{bundle_name}-", dir=self.temp_dir.name))
        request_spec_path = bundle_dir / "request-spec.json"
        request_spec_path.write_text(json.dumps(request_spec, indent=2) + "\n", encoding="utf-8")
        self._run(
            "cargo",
            "run",
            "--example",
            "kimi_rfc_0005_external_pilot",
            "--",
            "emit-request-bundle-from-spec",
            str(self.artifact_dir),
            str(request_spec_path),
            str(bundle_dir),
        )
        return (
            bundle_dir / "baseline-slice-request.json",
            bundle_dir / "local-init-contract.json",
            bundle_dir,
        )

    def _write_manifest(self, stem: str, manifest: dict[str, Any]) -> Path:
        path = self.bundle_dir / f"tampered-{stem}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return path

    def _load_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _tampered_artifact_dir(self) -> Path:
        artifact_dir = Path(tempfile.mkdtemp(prefix="attnres-kimi-artifact-drift-", dir=self.temp_dir.name))
        shutil.copytree(self.artifact_dir, artifact_dir, dirs_exist_ok=True)
        return artifact_dir


if __name__ == "__main__":
    unittest.main()
