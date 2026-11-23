"""Assert that eegprep's base install (no extras) resolves cleanly under Pyodide.

The question is whether the version Pyodide ships satisfies each declared specifier.
Direct specifiers come from ``[project].dependencies``. Transitive specifiers come from
the ``Requires-Dist`` of every base-closure package at the version pinned in ``uv.lock``.
Markers are evaluated for the pinned Pyodide release. ``extra ==`` requirements are not
part of the base install: packaging treats an empty extra as the base install and those
markers are false. Comparing a shipped version to the version ``uv.lock`` resolved is the
wrong check. The lock runs ahead of Pyodide on purpose, so that comparison fails packages
Pyodide can already install.

A shipped version that satisfies every specifier is ``ok`` with source ``pyodide-lock X.Y.Z``.
When it does not, and some version that satisfies every specifier publishes a universal
wheel (abi ``none``, platform ``any``), micropip can fetch that wheel. That case is a
note, source ``pypi wheel, overrides bundled X.Y.Z``, not a failure. When no such wheel
exists, the package cannot be installed and the check fails. Packages absent from the
Pyodide lock are unchanged: a universal wheel at the locked version is ok, and
``KNOWN_GAPS`` records accepted misses.

``--print-pyodide-packages`` prints the names ``tools/pyodide/run_pyodide.mjs`` should
preload. That list is the base closure plus the picard extra the harness installs,
restricted to packages the Pyodide lock ships.

Needs network access. No workflow runs this script; run it by hand with
``uv run python tools/check_pyodide_base_resolution.py`` after changing
``[project].dependencies`` or the pinned Pyodide version.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[1]
UV_LOCK_PATH = REPO_ROOT / "uv.lock"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

# Pinned Pyodide release used to look up which packages ship prebuilt in the distribution.
# Keep in sync with the pyodide version in tools/pyodide/package.json
# (tests/test_pyodide_harness.py checks that they match).
PYODIDE_VERSION = "0.29.5"
PYODIDE_LOCK_URL = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/pyodide-lock.json"
PYPI_JSON_URL = "https://pypi.org/pypi/{name}/{version}/json"
PYPI_PROJECT_JSON_URL = "https://pypi.org/pypi/{name}/json"
HTTP_TIMEOUT_S = 30

# The harness installs Picard by name on top of the base wheel. Preload the Pyodide
# builds that extra needs; micropip has no WebAssembly wheel for them.
HARNESS_EXTRAS = ("picard",)

KNOWN_GAPS: dict[str, str] = {}


def _fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "eegprep-pyodide-resolution-check"})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
        return json.load(response)


def _pyodide_environment(pyodide_info: dict) -> dict[str, str]:
    """Build a packaging.markers environment for the Pyodide/Emscripten target."""
    python_full_version = pyodide_info["python"]
    python_version = ".".join(python_full_version.split(".")[:2])
    return {
        "implementation_name": "cpython",
        "implementation_version": python_full_version,
        "os_name": "posix",
        "platform_machine": "wasm32",
        "platform_release": "",
        "platform_system": "Emscripten",
        "platform_version": "",
        "python_full_version": python_full_version,
        "platform_python_implementation": "CPython",
        "python_version": python_version,
        "sys_platform": "emscripten",
    }


def _load_uv_lock_packages() -> list[dict]:
    data = tomllib.loads(UV_LOCK_PATH.read_text())
    return data["package"]


def _resolve_dependency(edge: dict, packages_by_key: dict[tuple[str, str], dict], packages_by_name: dict) -> dict:
    if "version" in edge:
        return packages_by_key[(edge["name"], edge["version"])]
    candidates = packages_by_name[edge["name"]]
    if len(candidates) != 1:
        raise RuntimeError(f"Ambiguous uv.lock dependency edge for {edge['name']!r} without a pinned version")
    return candidates[0]


def _package_indexes() -> tuple[dict[tuple[str, str], dict], dict[str, list[dict]]]:
    packages = _load_uv_lock_packages()
    packages_by_key = {(pkg["name"], pkg["version"]): pkg for pkg in packages if "version" in pkg}
    packages_by_name: dict[str, list[dict]] = {}
    for pkg in packages:
        packages_by_name.setdefault(pkg["name"], []).append(pkg)
    return packages_by_key, packages_by_name


def _walk(
    edges: list[dict],
    environment: dict[str, str],
    packages_by_key: dict[tuple[str, str], dict],
    packages_by_name: dict[str, list[dict]],
) -> dict[str, str]:
    closure: dict[str, str] = {}
    stack = list(edges)
    while stack:
        edge = stack.pop()
        marker = edge.get("marker")
        if marker and not Marker(marker).evaluate(environment):
            continue
        pkg = _resolve_dependency(edge, packages_by_key, packages_by_name)
        if pkg["name"] in closure:
            continue
        closure[pkg["name"]] = pkg["version"]
        stack.extend(pkg.get("dependencies", []))
    return closure


def _base_closure(environment: dict[str, str]) -> dict[str, str]:
    """Return {package_name: version} reachable from eegprep's base dependencies under `environment`."""
    packages_by_key, packages_by_name = _package_indexes()
    (eegprep,) = packages_by_name["eegprep"]
    return _walk(list(eegprep.get("dependencies", [])), environment, packages_by_key, packages_by_name)


def _harness_closure(environment: dict[str, str]) -> dict[str, str]:
    packages_by_key, packages_by_name = _package_indexes()
    (eegprep,) = packages_by_name["eegprep"]
    edges = list(eegprep.get("dependencies", []))
    optional = eegprep.get("optional-dependencies", {})
    for extra in HARNESS_EXTRAS:
        edges.extend(optional.get(extra, []))
    return _walk(edges, environment, packages_by_key, packages_by_name)


def _is_universal_wheel(filename: str) -> bool:
    """A wheel usable under Pyodide/micropip regardless of Python minor version: abi=none, platform=any."""
    stem = filename[: -len(".whl")]
    tags = stem.split("-")
    if len(tags) < 3:
        return False
    abi_tag, platform_tag = tags[-2], tags[-1]
    return abi_tag == "none" and platform_tag == "any"


def _universal_wheel_filename(urls: list[dict]) -> str | None:
    for url_info in urls:
        if url_info.get("packagetype") == "bdist_wheel" and _is_universal_wheel(url_info["filename"]):
            return url_info["filename"]
    return None


def _release_metadata(name: str, version: str) -> dict:
    url = PYPI_JSON_URL.format(name=name, version=version)
    try:
        return _fetch_json(url)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"PyPI lookup failed for {name}=={version}: {exc}. "
            f"tools/check_pyodide_base_resolution.py requested {url}. "
            "Retry when PyPI is reachable."
        ) from exc


def _pyodide_versions(lock: dict) -> dict[str, str]:
    versions: dict[str, str] = {}
    for key, info in lock["packages"].items():
        version = info.get("version")
        if not isinstance(version, str) or not version:
            raise RuntimeError(
                f"Pyodide lock entry {key!r} has no version. "
                f"tools/check_pyodide_base_resolution.py reads packages[].version from {PYODIDE_LOCK_URL}. "
                "Point PYODIDE_VERSION at a lock that includes it."
            )
        versions[canonicalize_name(info.get("name", key))] = version
    return versions


@dataclass(frozen=True)
class DeclaredConstraint:
    name: str
    specifier: SpecifierSet
    declared_by: str
    text: str


def _applicable_requirement(raw: str, environment: dict[str, str]) -> Requirement | None:
    requirement = Requirement(raw)
    # Marker.evaluate defaults extra to "" so `extra ==` requirements are not base requirements.
    if requirement.marker is not None and not requirement.marker.evaluate(environment):
        return None
    return requirement


def _constraint(requirement: Requirement, raw: str, declared_by: str) -> DeclaredConstraint:
    return DeclaredConstraint(
        canonicalize_name(requirement.name),
        requirement.specifier,
        declared_by,
        raw.split(";", 1)[0].strip(),
    )


def _direct_constraints(environment: dict[str, str]) -> dict[str, list[DeclaredConstraint]]:
    project = tomllib.loads(PYPROJECT_PATH.read_text())["project"]
    found: dict[str, list[DeclaredConstraint]] = {}
    for raw in project["dependencies"]:
        requirement = _applicable_requirement(raw, environment)
        if requirement is None:
            continue
        constraint = _constraint(requirement, raw, "pyproject.toml")
        found.setdefault(constraint.name, []).append(constraint)
    return found


def _version_meets(version: str, constraints: list[DeclaredConstraint]) -> bool:
    try:
        Version(version)
    except InvalidVersion:
        return False
    return all(constraint.specifier.contains(version, prereleases=True) for constraint in constraints)


def _unsatisfied(version: str, constraints: list[DeclaredConstraint]) -> list[DeclaredConstraint]:
    return [constraint for constraint in constraints if not constraint.specifier.contains(version, prereleases=True)]


def _satisfying_universal_wheel(
    name: str,
    locked_version: str,
    locked_release: dict,
    constraints: list[DeclaredConstraint],
) -> str | None:
    if _version_meets(locked_version, constraints):
        wheel = _universal_wheel_filename(locked_release.get("urls") or [])
        if wheel:
            return wheel
    url = PYPI_PROJECT_JSON_URL.format(name=name)
    try:
        project = _fetch_json(url)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"PyPI lookup failed for {name}: {exc}. "
            f"tools/check_pyodide_base_resolution.py requested {url}. "
            "Retry when PyPI is reachable."
        ) from exc
    for version, files in (project.get("releases") or {}).items():
        if not _version_meets(version, constraints):
            continue
        wheel = _universal_wheel_filename(files or [])
        if wheel:
            return wheel
    return None


def _failure_source(package: str, shipped: str, unsatisfied: list[DeclaredConstraint]) -> str:
    details = "; ".join(f"{constraint.text} declared by {constraint.declared_by}" for constraint in unsatisfied)
    locations = ", ".join(dict.fromkeys(constraint.declared_by for constraint in unsatisfied))
    return (
        f"tools/check_pyodide_base_resolution.py: {package}: Pyodide {PYODIDE_VERSION} ships {shipped}, "
        f"which does not satisfy {details}. "
        f"Lower the floor in {locations} to <= {shipped}, or move to a Pyodide release that ships it."
    )


STATUS_OK = "ok"
STATUS_KNOWN_GAP = "known-gap"
STATUS_FAIL = "FAIL"


@dataclass(frozen=True)
class PackageCheck:
    name: str
    version: str
    status: str
    source: str


def classify_package(name: str, version: str, in_pyodide_lock: bool, universal_wheel: str | None) -> PackageCheck:
    """Decide once whether a base package installs under Pyodide and where it comes from."""
    if in_pyodide_lock:
        return PackageCheck(name, version, STATUS_OK, "pyodide-lock")
    if universal_wheel:
        return PackageCheck(name, version, STATUS_OK, f"pypi wheel: {universal_wheel}")
    if name in KNOWN_GAPS:
        return PackageCheck(name, version, STATUS_KNOWN_GAP, f"NOT FOUND, known gap: {KNOWN_GAPS[name]}")
    return PackageCheck(name, version, STATUS_FAIL, "NOT FOUND (no Pyodide build, no pure-Python wheel)")


def check_base_packages() -> list[PackageCheck]:
    """Classify each base-closure package against the pinned Pyodide release."""
    lock = _fetch_json(PYODIDE_LOCK_URL)
    environment = _pyodide_environment(lock["info"])
    shipped_versions = _pyodide_versions(lock)
    closure = _base_closure(environment)
    closure_names = {canonicalize_name(name) for name in closure}
    releases = {name: _release_metadata(name, version) for name, version in closure.items()}
    constraints = _direct_constraints(environment)
    for name, version in closure.items():
        requires = (releases[name].get("info") or {}).get("requires_dist") or []
        declared_by = f"{name}=={version} requires-dist"
        for raw in requires:
            requirement = _applicable_requirement(raw, environment)
            if requirement is None:
                continue
            constraint = _constraint(requirement, raw, declared_by)
            if constraint.name not in closure_names:
                continue
            constraints.setdefault(constraint.name, []).append(constraint)

    results = []
    for name, version in sorted(closure.items()):
        canonical = canonicalize_name(name)
        shipped = shipped_versions.get(canonical)
        if shipped is None:
            wheel = _universal_wheel_filename(releases[name].get("urls") or [])
            results.append(classify_package(name, version, False, wheel))
            continue
        package_constraints = constraints.get(canonical, [])
        missed = _unsatisfied(shipped, package_constraints)
        if not missed:
            results.append(PackageCheck(name, version, STATUS_OK, f"pyodide-lock {shipped}"))
            continue
        wheel = _satisfying_universal_wheel(name, version, releases[name], package_constraints)
        if wheel:
            results.append(PackageCheck(name, version, STATUS_OK, f"pypi wheel, overrides bundled {shipped}"))
            continue
        results.append(PackageCheck(name, version, STATUS_FAIL, _failure_source(name, shipped, missed)))
    return results


def pyodide_preload_packages() -> list[str]:
    """Return the Pyodide package names the harness should load before micropip."""
    lock = _fetch_json(PYODIDE_LOCK_URL)
    environment = _pyodide_environment(lock["info"])
    shipped = _pyodide_versions(lock)
    closure = _harness_closure(environment)
    return sorted(name for name in closure if canonicalize_name(name) in shipped)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that the base install resolves under Pyodide.")
    parser.add_argument(
        "--print-pyodide-packages",
        action="store_true",
        help="Print the package names tools/pyodide/run_pyodide.mjs should preload.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.print_pyodide_packages:
        for name in pyodide_preload_packages():
            print(name)
        return 0

    results = check_base_packages()
    for result in results:
        print(f"[{result.status}] {result.name}=={result.version}: {result.source}")

    failures = [result for result in results if result.status == STATUS_FAIL]
    known = [result for result in results if result.status == STATUS_KNOWN_GAP]
    notes = [result for result in results if result.source.startswith("pypi wheel, overrides bundled")]
    print(
        f"\nChecked {len(results)} base packages against Pyodide {PYODIDE_VERSION}: "
        f"{len(failures)} failing, {len(known)} known gaps (not blocking), {len(notes)} notes."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
