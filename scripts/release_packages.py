"""Build and publish eegprep and eegprep-lean under one version.

The two distributions are released together. The command exits before building
when ``src/eegprep/__init__.py``, ``eegprep_lean.__version__``, and the lean
``pyproject.toml`` version are not the same string. ``--publish`` uploads both
directories. Without it, the command builds and stops.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EEGPREP_INIT = Path("src/eegprep/__init__.py")
LEAN_INIT = Path("packages/eegprep-lean/src/eegprep_lean/__init__.py")
LEAN_PROJECT = Path("packages/eegprep-lean/pyproject.toml")
PROJECTS = ("eegprep", "eegprep-lean")
_VERSION_ASSIGNMENT = re.compile(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", re.M)


class VersionMismatch(Exception):
    """The two packages, or a tag, do not name one version."""


def read_assigned_version(path: Path) -> str:
    match = _VERSION_ASSIGNMENT.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise VersionMismatch(f"{path} has no __version__ assignment")
    return match.group(1)


def read_versions(root: Path) -> tuple[str, str, str]:
    """Return eegprep's version, the lean module version, and the lean project version."""
    project = tomllib.loads((root / LEAN_PROJECT).read_text(encoding="utf-8"))
    return (
        read_assigned_version(root / EEGPREP_INIT),
        read_assigned_version(root / LEAN_INIT),
        project["project"]["version"],
    )


def require_one_version(versions: tuple[str, str, str]) -> str:
    eegprep, lean_init, lean_project = versions
    if len({eegprep, lean_init, lean_project}) != 1:
        raise VersionMismatch(
            "eegprep and eegprep-lean must share one version. "
            f"eegprep __version__ is {eegprep}, "
            f"eegprep-lean __version__ is {lean_init}, "
            f"eegprep-lean project version is {lean_project}."
        )
    return eegprep


def release_tag() -> str | None:
    """The version named by ``RELEASE_TAG`` or a ``refs/tags/v*`` GitHub ref."""
    ref = os.environ.get("GITHUB_REF", "")
    if ref.startswith("refs/tags/v"):
        return ref.removeprefix("refs/tags/v")
    tag = os.environ.get("RELEASE_TAG", "")
    if tag.startswith("v"):
        return tag[1:]
    return tag or None


def require_tag(version: str, tag: str | None) -> None:
    if tag is not None and tag != version:
        raise VersionMismatch(f"tag v{tag} does not match version {version}")


def distribution_name(project: str) -> str:
    return project.replace("-", "_")


def artifact_paths(dist: Path, version: str) -> dict[Path, str]:
    paths: dict[Path, str] = {}
    for project in PROJECTS:
        stem = distribution_name(project)
        directory = dist / project
        paths[directory / f"{stem}-{version}-py3-none-any.whl"] = project
        paths[directory / f"{stem}-{version}.tar.gz"] = project
    return paths


def wheel_version(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        metadata = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        text = archive.read(metadata).decode()
    match = re.search(r"^Version: (.+)$", text, re.M)
    if match is None:
        raise VersionMismatch(f"{path} has no Version in METADATA")
    return match.group(1).strip()


def require_artifacts(dist: Path, version: str) -> None:
    expected = artifact_paths(dist, version)
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise VersionMismatch("missing artifacts: " + ", ".join(missing))
    for path, project in expected.items():
        if path.suffix != ".whl":
            continue
        found = wheel_version(path)
        if found != version:
            raise VersionMismatch(f"{project} wheel version is {found}, expected {version}")


def build(root: Path, dist: Path) -> None:
    for project in PROJECTS:
        out = dist / project
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        project_dir = root if project == "eegprep" else root / "packages" / project
        subprocess.run(["uv", "build", "--out-dir", str(out)], cwd=project_dir, check=True)


def publish(dist: Path) -> None:
    for project in PROJECTS:
        subprocess.run(["uv", "publish", str(dist / project)], check=True)


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="Upload both distributions. Omit this to build only.")
    parser.add_argument("--check", action="store_true", help="Compare versions and do not build.")
    args = parser.parse_args(argv)
    root = root or REPO_ROOT
    try:
        version = require_one_version(read_versions(root))
        require_tag(version, release_tag())
    except VersionMismatch as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.check:
        print(f"{version}: versions match")
        return 0
    dist = root / "dist" / "release"
    build(root, dist)
    try:
        require_artifacts(dist, version)
    except VersionMismatch as exc:
        print(exc, file=sys.stderr)
        return 1
    if not args.publish:
        print(f"{version}: built both packages; not publishing")
        return 0
    publish(dist)
    print(f"{version}: published eegprep and eegprep-lean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
