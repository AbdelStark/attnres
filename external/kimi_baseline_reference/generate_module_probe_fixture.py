from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .module_probe import PUBLIC_KIMI_REPO_ID, generate_fixture, write_fixture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate attnres Kimi module-probe fixtures from official Hugging Face remote code.",
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--request-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--repo-id", default=PUBLIC_KIMI_REPO_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--remote-code-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fixture, fla_backend = generate_fixture(
        artifact_dir=args.artifact_dir,
        request_path=args.request_path,
        repo_id=args.repo_id,
        revision=args.revision,
        remote_code_dir=args.remote_code_dir,
    )
    write_fixture(args.output_path, fixture)
    print(
        f"wrote module probe fixture to {args.output_path} using FLA backend '{fla_backend}'",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
