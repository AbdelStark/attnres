from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from external.kimi_baseline_reference.module_probe import (
    KIMI_MODULE_PROBE_FIXTURE_KIND,
    PUBLIC_KIMI_REPO_ID,
    generate_fixture,
    write_fixture,
)
from external.kimi_baseline_reference.remote_code_support import ensure_hf_files
from external.kimi_baseline_reference.run_baseline_smoke import (
    BASELINE_SMOKE_REPORT_KIND,
    run_smoke,
)


class ModuleProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[3]
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="attnres-kimi-module-probe-"))
        cls.artifact_dir = cls.temp_dir / "artifact"
        cls.request_path = cls.temp_dir / "module-probe-request.json"
        cls.fixture_path = cls.temp_dir / "module-probe-fixture.json"
        cls.remote_code_dir = cls.temp_dir / "remote-code"

        cls._run(
            "cargo",
            "run",
            "--example",
            "kimi_rfc_0005_external_pilot",
            "--",
            "write-artifact",
            str(cls.artifact_dir),
        )
        config_path = cls.artifact_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["num_key_value_heads"] = config["num_attention_heads"]
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        cls._run(
            "cargo",
            "run",
            "--example",
            "kimi_real_model_tools",
            "--",
            "emit-module-probe-request",
            str(cls.artifact_dir),
            str(cls.request_path),
        )
        ensure_hf_files(
            cls.remote_code_dir,
            PUBLIC_KIMI_REPO_ID,
            ["configuration_kimi.py", "modeling_kimi.py"],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        for path in cls.temp_dir.rglob("*"):
            if path.is_file():
                path.unlink()
        for path in sorted(cls.temp_dir.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
        cls.temp_dir.rmdir()

    @classmethod
    def _run(cls, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, cwd=cls.repo_root, check=True, capture_output=True, text=True)

    def test_generate_fixture_roundtrips_against_rust_validator(self) -> None:
        fixture, fla_backend = generate_fixture(
            artifact_dir=self.artifact_dir,
            request_path=self.request_path,
            repo_id=None,
            remote_code_dir=self.remote_code_dir,
        )

        self.assertEqual(fixture["kind"], KIMI_MODULE_PROBE_FIXTURE_KIND)
        self.assertEqual(fixture["version"], 1)
        self.assertEqual(len(fixture["probes"]), 4)
        self.assertIn(fla_backend, {"official", "cpu_fallback"})

        write_fixture(self.fixture_path, fixture)
        self._run(
            "cargo",
            "run",
            "--example",
            "kimi_real_model_tools",
            "--",
            "validate-module-probe-fixture",
            str(self.artifact_dir),
            str(self.request_path),
            str(self.fixture_path),
        )

    def test_smoke_harness_reports_missing_full_checkpoint_honestly(self) -> None:
        smoke_dir = self.temp_dir / "public-metadata-only"
        report = run_smoke(
            artifact_dir=smoke_dir,
            repo_id=PUBLIC_KIMI_REPO_ID,
            revision=None,
            prompt="Explain linear attention in one sentence.",
            memory_overhead_gib=8.0,
        )

        self.assertEqual(report["kind"], BASELINE_SMOKE_REPORT_KIND)
        self.assertEqual(report["status"], "blocked_missing_full_checkpoint")
        self.assertGreater(report["missing_shard_count"], 0)


if __name__ == "__main__":
    unittest.main()
