from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from external.kimi_baseline_reference.generator import GeneratorError, generate_fixture


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
        cls.seeded_init_path = cls.bundle_dir / "seeded-init-state.json"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    @classmethod
    def _run(cls, *args: str) -> None:
        subprocess.run(args, cwd=cls.repo_root, check=True, capture_output=True, text=True)

    def test_generates_fixture_that_validates_with_attnres_consumer(self) -> None:
        fixture = generate_fixture(
            artifact_dir=self.artifact_dir,
            manifest_path=self.manifest_path,
            seeded_init_path=self.seeded_init_path,
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

    def test_rejects_manifest_kind_mismatch(self) -> None:
        self._expect_manifest_error("kind", "attnres.kimi.wrong_slice_request_kind", "expected baseline slice request manifest kind")

    def test_rejects_manifest_version_mismatch(self) -> None:
        self._expect_manifest_error("version", 2, "expected baseline slice request manifest version 1")

    def test_rejects_artifact_fingerprint_mismatch(self) -> None:
        self._expect_nested_manifest_error(("artifact", "hidden_size"), 9, "baseline manifest field 'artifact'")

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

    def _expect_manifest_error(self, key: str, replacement, message: str) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest[key] = replacement
        path = self.bundle_dir / f"tampered-{key}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(GeneratorError, message):
            generate_fixture(self.artifact_dir, path, self.seeded_init_path)

    def _expect_nested_manifest_error(self, path_parts, replacement, message: str) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        target = manifest
        for part in path_parts[:-1]:
            target = target[part]
        target[path_parts[-1]] = replacement
        path = self.bundle_dir / ("tampered-" + "-".join(str(part) for part in path_parts) + ".json")
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(GeneratorError, message):
            generate_fixture(self.artifact_dir, path, self.seeded_init_path)


if __name__ == "__main__":
    unittest.main()
