"""Smoke tests for public API docs examples and packaged resources."""

from __future__ import annotations

import runpy
from importlib.resources import files
from pathlib import Path


def test_public_api_and_plugins_example_runs() -> None:
    example = Path("docs/source/examples/plot_public_api_and_plugins.py")

    runpy.run_path(str(example), run_name="__main__")


def test_package_resources_cover_public_workflows() -> None:
    package = files("eegprep")

    assert package.joinpath("resources/help/pop_clean_rawdata.md").is_file()
    assert package.joinpath("resources/headplot/colin27headmesh.mat").is_file()
    assert package.joinpath("resources/montages/standard-10-5-342ch.locs").is_file()
    assert package.joinpath("plugins/ICLabel/netICL.mat").is_file()
