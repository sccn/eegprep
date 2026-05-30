"""Smoke tests for public API docs examples and packaged resources."""

from __future__ import annotations

import runpy
from importlib.resources import files
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_public_api_and_plugins_example_runs() -> None:
    example = REPO_ROOT / "docs/source/examples/plot_public_api_and_plugins.py"

    runpy.run_path(str(example), run_name="__main__")


def test_package_resources_cover_public_workflows() -> None:
    package = files("eegprep")

    assert package.joinpath("resources/help/pop_clean_rawdata.md").is_file()
    assert package.joinpath("resources/headplot/colin27headmesh.mat").is_file()
    assert package.joinpath("resources/montages/standard-10-5-342ch.locs").is_file()
    assert package.joinpath("plugins/ICLabel/netICL.mat").is_file()


def test_project_entry_points_cover_gui_and_console() -> None:
    pyproject = _read_pyproject()

    assert pyproject["project"]["scripts"]["eegprep-gui"] == "eegprep.functions.adminfunc.eeglab:main"
    assert pyproject["project"]["scripts"]["eegprep-console"] == "eegprep.functions.adminfunc.console:main"


def test_setuptools_package_data_covers_runtime_resources() -> None:
    pyproject = _read_pyproject()
    package_root = REPO_ROOT / "src/eegprep"
    patterns = pyproject["tool"]["setuptools"]["package-data"]["eegprep"]
    excluded = pyproject["tool"]["setuptools"]["exclude-package-data"]["eegprep"]

    packaged = {
        path.relative_to(package_root).as_posix()
        for pattern in patterns
        for path in package_root.glob(pattern)
        if path.is_file()
    }

    assert {
        "resources/help/pop_clean_rawdata.md",
        "resources/headplot/colin27headmesh.mat",
        "resources/headplot/mheadnew.transform",
        "resources/headplot/mheadnew.xyz",
        "resources/montages/standard-10-5-342ch.locs",
        "plugins/ICLabel/netICL.mat",
    } <= packaged
    assert "eeglab/**" in excluded
    assert not any(path.startswith("eeglab/") for path in packaged)


def _read_pyproject() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)
