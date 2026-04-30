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
    parser.add_argument("--artifact-root", help="Reference artifact root to include in backend availability.")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest, deviations_path=args.deviations)
    backends = detect_oracle_backends(artifact_root=args.artifact_root)
    payload = {
        "manifest": manifest.summary(),
        "backends": {backend.value: info.__dict__ for backend, info in backends.items()},
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Parity Manifest Summary")
        print("=======================")
        manifest_summary = payload["manifest"]
        print(f"Version: {manifest_summary['version']}")
        print(f"Default backend: {manifest_summary['default_backend']}")
        print(f"Cases: {manifest_summary['case_count']}")
        print(f"Known deviations: {manifest_summary['deviation_count']}")
        print("Tiers:")
        for tier, count in sorted(manifest_summary["tiers"].items()):
            print(f"  {tier}: {count}")
        print("Surfaces:")
        for surface, count in sorted(manifest_summary["surfaces"].items()):
            print(f"  {surface}: {count}")
        print("Backends:")
        for name, info in sorted(payload["backends"].items()):
            status = "available" if info["available"] else "unavailable"
            print(f"  {name}: {status} ({info['detail']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
