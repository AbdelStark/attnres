from __future__ import annotations

import argparse
import sys
from pathlib import Path

from external.kimi_baseline_reference.generator import GeneratorError, generate_fixture, write_fixture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a baseline-slice-parity.json fixture from the RFC 0005 external pilot manifest."
    )
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--local-init-contract-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        fixture = generate_fixture(
            artifact_dir=args.artifact_dir,
            manifest_path=args.manifest_path,
            local_init_contract_path=args.local_init_contract_path,
        )
        write_fixture(args.output_path, fixture)
        return 0
    except GeneratorError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
