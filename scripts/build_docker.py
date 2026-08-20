#!/usr/bin/env python3
"""Build and push the EEGPrep Docker image.

PyPI releases are handled entirely by .github/workflows/release.yml, triggered by
pushing a ``v*`` tag. This script covers the one step CI does not do: publishing
the Docker image and updating the Singularity/Docker pin that the HPC wrapper uses.

Run it after the release workflow has published the tag:

    docker login
    uv run python scripts/build_docker.py

The image version comes from ``__version__`` in src/eegprep/__init__.py, so bump
that (and tag) before running this.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn


PROJECT_ROOT = Path(__file__).parent.parent
VERSION_PATH = PROJECT_ROOT / "src" / "eegprep" / "__init__.py"
DOCKERFILE_NAME = "Dockerfile"
HPC_WRAPPER = PROJECT_ROOT / "tools" / "hpc" / "main.pbs"
DOCKER_REPO = "arnodelorme/eegprep"


def fail(message) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def get_version():
    """Read __version__ from src/eegprep/__init__.py."""
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', VERSION_PATH.read_text(), re.MULTILINE)
    if not match:
        fail(f"could not find __version__ in {VERSION_PATH}")
    return match.group(1)


def run(command):
    print(f"+ {' '.join(command)}")
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        fail(f"command failed: {' '.join(command)}")


def update_hpc_pin(version):
    """Point the HPC wrapper at the image just pushed."""
    text = HPC_WRAPPER.read_text()
    updated = re.sub(rf"{re.escape(DOCKER_REPO)}:[^\s\"']+", f"{DOCKER_REPO}:{version}", text)
    if updated == text:
        print(f"note: no {DOCKER_REPO}:<version> pin found in {HPC_WRAPPER.name}; left unchanged")
        return
    HPC_WRAPPER.write_text(updated)
    print(f"updated {HPC_WRAPPER.relative_to(PROJECT_ROOT)} to {DOCKER_REPO}:{version} (commit this)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build and push the EEGPrep Docker image.")
    parser.add_argument("--version", help="override the version tag (default: __version__)")
    parser.add_argument("--no-push", action="store_true", help="build locally without pushing")
    args = parser.parse_args(argv)

    if shutil.which("docker") is None:
        fail("docker is not installed or not on PATH")

    version = args.version or get_version()
    local_tag = f"eegprep:{version}"
    remote_tag = f"{DOCKER_REPO}:{version}"
    print(f"Building {remote_tag}\n")

    run(["docker", "build", "-t", local_tag, "-f", DOCKERFILE_NAME, "."])
    run(["docker", "tag", local_tag, remote_tag])

    if args.no_push:
        print(f"\nBuilt {remote_tag}; not pushed (--no-push).")
        return 0

    # Fails with a clear message from docker itself if `docker login` is missing.
    run(["docker", "push", remote_tag])
    update_hpc_pin(version)
    print(f"\nPushed {remote_tag}")
    print("Remember to update the default app option on brainlife if this release affects it.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        sys.exit(1)
