"""Keep eegprep installable in the browser.

Pyodide bundles its own build of every compiled scientific package, so a dependency floor
above the version a Pyodide release ships cannot be satisfied in the browser: micropip has
no WebAssembly build to fall back to. That makes ``requires-python`` and these floors a
*ceiling* for the browser target, which is the opposite of how a floor usually reads, and it
is invisible from the code. A routine bump made for an unrelated reason is enough to break it.

``epic/324-pyodide-browser`` already carries ``tools/check_pyodide_base_resolution.py``, but
that gate matches package *names* against the Pyodide lock and never compares versions, so a
floor above what Pyodide ships passes it (sccn/eegprep#400). These tests close that gap on
``develop``, offline and without a network fetch.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"

# The Pyodide release the browser work targets, and the versions it bundles. Read from
# https://cdn.jsdelivr.net/pyodide/v0.29.5/full/pyodide-lock.json on 2026-09-20. Update both
# the version and this table together when the target release moves.
PYODIDE_VERSION = "0.29.5"
PYODIDE_PYTHON = "3.13.2"
PYODIDE_SHIPS = {
    "h5py": "3.13.0",
    "matplotlib": "3.8.4",
    "numpy": "2.2.5",
    "scipy": "1.14.1",
    "threadpoolctl": "3.5.0",
}

# A packaging marker environment describing Pyodide's interpreter, so a dependency guarded by
# a marker is evaluated the way the browser would evaluate it. Pyodide reports
# ``sys.platform == "emscripten"``, which is what makes the darwin-only scipy floor inactive here.
PYODIDE_ENVIRONMENT = {
    "implementation_name": "cpython",
    "implementation_version": PYODIDE_PYTHON,
    "os_name": "posix",
    "platform_machine": "wasm32",
    "platform_python_implementation": "CPython",
    "platform_release": "",
    "platform_system": "Emscripten",
    "platform_version": "",
    "python_full_version": PYODIDE_PYTHON,
    "python_version": ".".join(PYODIDE_PYTHON.split(".")[:2]),
    "sys_platform": "emscripten",
}


def _project() -> dict:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))["project"]


def _requirements_active_in_pyodide() -> dict[str, Requirement]:
    """Base requirements whose markers hold under Pyodide, keyed by normalized name."""
    active: dict[str, Requirement] = {}
    for raw in _project()["dependencies"]:
        requirement = Requirement(raw)
        if requirement.marker is not None and not requirement.marker.evaluate(PYODIDE_ENVIRONMENT):
            continue
        name = requirement.name.lower().replace("_", "-")
        assert name not in active, f"{name} has two requirements active at once under Pyodide: {raw}"
        active[name] = requirement
    return active


def _declared_floor(requirement: Requirement) -> Version:
    floors = [Version(spec.version) for spec in requirement.specifier if spec.operator in {">=", "==", "~="}]
    assert floors, f"{requirement.name} declares no lower bound, so nothing pins it for the browser"
    return max(floors)


@pytest.mark.parametrize("package", sorted(PYODIDE_SHIPS))
def test_floor_does_not_exceed_what_pyodide_ships(package: str) -> None:
    requirement = _requirements_active_in_pyodide().get(package)
    assert requirement is not None, f"{package} is no longer a base dependency; drop it from PYODIDE_SHIPS"

    floor = _declared_floor(requirement)
    shipped = Version(PYODIDE_SHIPS[package])
    assert floor <= shipped, (
        f"{package}>={floor} is above the {shipped} that Pyodide {PYODIDE_VERSION} ships, "
        f"so a browser install cannot satisfy it"
    )


@pytest.mark.parametrize("package", sorted(PYODIDE_SHIPS))
def test_pyodide_version_satisfies_the_whole_specifier(package: str) -> None:
    """The floor is not the only bound; an upper bound can exclude Pyodide's build too."""
    requirement = _requirements_active_in_pyodide().get(package)
    assert requirement is not None, f"{package} is no longer a base dependency; drop it from PYODIDE_SHIPS"

    shipped = PYODIDE_SHIPS[package]
    assert requirement.specifier.contains(shipped, prereleases=True), (
        f"Pyodide {PYODIDE_VERSION} ships {package} {shipped}, which {requirement} excludes"
    )


def test_requires_python_admits_the_pyodide_interpreter() -> None:
    """requires-python is a ceiling here: Pyodide runs one CPython and cannot be told otherwise."""
    requires_python = SpecifierSet(_project()["requires-python"])
    assert requires_python.contains(PYODIDE_PYTHON, prereleases=True), (
        f"requires-python {requires_python} excludes the CPython {PYODIDE_PYTHON} that "
        f"Pyodide {PYODIDE_VERSION} runs, so eegprep cannot be installed in the browser"
    )


def test_the_darwin_scipy_floor_stays_out_of_the_browser() -> None:
    """The macOS-only scipy floor is above what Pyodide ships, and must stay marker-guarded."""
    darwin = dict(PYODIDE_ENVIRONMENT, sys_platform="darwin", platform_system="Darwin")
    darwin_floors = [
        _declared_floor(requirement)
        for raw in _project()["dependencies"]
        if (requirement := Requirement(raw)).name == "scipy"
        and requirement.marker is not None
        and requirement.marker.evaluate(darwin)
    ]
    assert darwin_floors, "the darwin-specific scipy floor is gone; update or remove this test"
    assert max(darwin_floors) > Version(PYODIDE_SHIPS["scipy"]), (
        "the darwin scipy floor no longer exceeds Pyodide's build, so the marker split may be "
        "unnecessary; confirm before removing this test"
    )
    assert all(
        Marker(str(requirement.marker)).evaluate(PYODIDE_ENVIRONMENT) is False
        for raw in _project()["dependencies"]
        if (requirement := Requirement(raw)).name == "scipy"
        and requirement.marker is not None
        and _declared_floor(requirement) > Version(PYODIDE_SHIPS["scipy"])
    ), "a scipy floor above Pyodide's build is active under Pyodide"
