"""Assert that eegprep's base install (no extras) resolves cleanly under Pyodide.

Gate for issue #374 / epic #324: ``micropip.install("eegprep")`` must not need a
WebAssembly build for anything left in ``[project].dependencies``. Every package reachable
from eegprep's base dependencies in ``uv.lock`` must either ship prebuilt in the target
Pyodide distribution, or publish a pure-Python wheel (abi ``none``, platform ``any``) that
micropip can pull straight from PyPI. This only checks resolvability from lockfile and
package-index metadata; it does not run inside Pyodide (``tools/pyodide/run_pyodide.sh`` is
the live ``micropip.install`` harness).

Needs network access. No workflow runs this script; run it by hand with
``uv run python tools/check_pyodide_base_resolution.py`` after changing
``[project].dependencies`` or the pinned Pyodide version.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import tomllib

from packaging.markers import Marker
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[1]
UV_LOCK_PATH = REPO_ROOT / "uv.lock"

# Pinned Pyodide release used to look up which packages ship prebuilt in the distribution.
# Keep in sync with the pyodide version in tools/pyodide/package.json
# (tests/test_pyodide_harness.py checks that they match).
PYODIDE_VERSION = "0.29.5"
PYODIDE_LOCK_URL = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/pyodide-lock.json"
PYPI_JSON_URL = "https://pypi.org/pypi/{name}/{version}/json"
HTTP_TIMEOUT_S = 30

# The live harness (tools/pyodide/prepare_docopt_wheel.py) builds this exact sdist into a local
# universal wheel before asking micropip to resolve eegprep. Keep the package out of
# ``KNOWN_GAPS`` so the static check cannot accidentally turn a known local transport into a
# silent exception.
KNOWN_GAPS: dict[str, str] = {}
LOCAL_WHEEL_PACKAGES = {"docopt"}


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


def _base_closure(environment: dict[str, str]) -> dict[str, str]:
    """Return {package_name: version} reachable from eegprep's base dependencies under `environment`."""
    packages = _load_uv_lock_packages()
    packages_by_key = {(pkg["name"], pkg["version"]): pkg for pkg in packages if "version" in pkg}
    packages_by_name: dict[str, list[dict]] = {}
    for pkg in packages:
        packages_by_name.setdefault(pkg["name"], []).append(pkg)

    (eegprep,) = packages_by_name["eegprep"]
    closure: dict[str, str] = {}
    stack = list(eegprep.get("dependencies", []))
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


def _is_universal_wheel(filename: str) -> bool:
    """A wheel usable under Pyodide/micropip regardless of Python minor version: abi=none, platform=any."""
    stem = filename[: -len(".whl")]
    tags = stem.split("-")
    if len(tags) < 3:
        return False
    abi_tag, platform_tag = tags[-2], tags[-1]
    return abi_tag == "none" and platform_tag == "any"


def _find_universal_wheel(name: str, version: str) -> str | None:
    try:
        release = _fetch_json(PYPI_JSON_URL.format(name=name, version=version))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"PyPI lookup failed for {name}=={version}: {exc}") from exc
    for url_info in release.get("urls", []):
        if url_info.get("packagetype") == "bdist_wheel" and _is_universal_wheel(url_info["filename"]):
            return url_info["filename"]
    return None


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
    if name in LOCAL_WHEEL_PACKAGES:
        return PackageCheck(
            name, version, STATUS_OK, "local pure-Python wheel built by tools/pyodide/prepare_docopt_wheel.py"
        )
    if name in KNOWN_GAPS:
        return PackageCheck(name, version, STATUS_KNOWN_GAP, f"NOT FOUND, known gap: {KNOWN_GAPS[name]}")
    return PackageCheck(name, version, STATUS_FAIL, "NOT FOUND (no Pyodide build, no pure-Python wheel)")


def main() -> int:
    pyodide_lock = _fetch_json(PYODIDE_LOCK_URL)
    environment = _pyodide_environment(pyodide_lock["info"])
    pyodide_names = {canonicalize_name(name) for name in pyodide_lock["packages"]}

    closure = _base_closure(environment)

    results = []
    for name, version in sorted(closure.items()):
        in_lock = canonicalize_name(name) in pyodide_names
        wheel = None if in_lock else _find_universal_wheel(name, version)
        results.append(classify_package(name, version, in_lock, wheel))

    for result in results:
        print(f"[{result.status}] {result.name}=={result.version}: {result.source}")

    failures = [result for result in results if result.status == STATUS_FAIL]
    known = [result for result in results if result.status == STATUS_KNOWN_GAP]
    print(
        f"\nChecked {len(results)} base packages against Pyodide {PYODIDE_VERSION}: "
        f"{len(failures)} failing, {len(known)} known gaps (not blocking)."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
