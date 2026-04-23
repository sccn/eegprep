"""CLI entrypoint for parity manifest inspection."""

from __future__ import annotations

import argparse
import json

from .config import load_manifest
from .oracle import detect_oracle_backends


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the EEGPrep parity manifest.")
    parser.add_argument("--manifest", help="Path to a parity manifest TOML file.")
    parser.add_argument("--deviations", help="Path to a known-deviations TOML file.")
    parser.add_argument("--format", choices=("json", "text"), default="text")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest, deviations_path=args.deviations)
    payload = {
        "manifest": manifest.summary(),
        "backends": {name: info.__dict__ for name, info in detect_oracle_backends().items()},
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Parity Manifest Summary")
        print("=======================")
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
