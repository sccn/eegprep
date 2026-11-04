"""Smoke tests for public API docs examples and packaged resources."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QSlider

from tests.eeglab_tests import eeglab_test

import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
MAKEHTML_OUTPUT_WRAPPER = "unittesting_miscfunc/makehtml/output/miscfunc_makehtml_output_wrapperTest.m"


@pytest.mark.gui
@eeglab_test(MAKEHTML_OUTPUT_WRAPPER, "test_eeglab")
def test_reference_generated_documentation_menu(eeglab_backend, eeglab_suite_root, request):
    if request.config.getoption("--eeglab-backend") == "matlab":
        engine = request.getfixturevalue("eeglab_matlab_engine")
        previous = engine.pwd()
        try:
            # Resolve the generated help-menu stub, not the main EEGLAB launcher.
            engine.cd(str(eeglab_suite_root / "unittesting_miscfunc/makehtml/output"), nargout=0)
            eeglab_backend("eeglab", nargout=0)
        finally:
            engine.cd(previous, nargout=0)
    else:
        title = "(Click on blue text for help)"
        # Capture only this test's dialog as the Python equivalent of gcf.
        window = eeglab_backend(
            "textgui",
            np.array([["Contents.m", "eeglab.m"]], dtype=object),
            np.array([["pophelp('Contents.m');", "pophelp('eeglab.m');"]], dtype=object),
            "fontsize",
            15.0,
            "fontname",
            "times",
            "linesperpage",
            18.0,
            "title",
            np.array(["Test".ljust(len(title)), title]),
        )
        defaults = eeglab_backend("icadefs")
        palette = window.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor.fromRgbF(*defaults.BACKCOLOR))
        window.setPalette(palette)
        window.setAutoFillBackground(True)
        slider_color = QColor.fromRgbF(*defaults.GUIBACKCOLOR)
        for slider in window.findChildren(QSlider, options=Qt.FindChildOption.FindDirectChildrenOnly):
            palette = slider.palette()
            palette.setColor(QPalette.ColorRole.Window, slider_color)
            slider.setPalette(palette)
            slider.setAutoFillBackground(True)


def test_public_api_and_plugins_example_runs() -> None:
    example = REPO_ROOT / "docs/source/examples/plot_public_api_and_plugins.py"

    runpy.run_path(str(example), run_name="__main__")


# One illustration script per user guide section. These run for real against
# sample_data/, so a failure here means the documented workflow is broken.
# plot_reject_artifacts.py calls ICLabel, which needs the torch extra and raises
# ImportError without it rather than skipping; install eegprep[torch] to run.
USER_GUIDE_EXAMPLES = (
    "plot_quickstart_tour.py",
    "plot_data_structures.py",
    "plot_import_data.py",
    "plot_dataset_management.py",
    "plot_preprocess_data.py",
    "plot_extract_epochs.py",
    "plot_reject_artifacts.py",
    "plot_data_plotting.py",
    "plot_source_analysis_dipfit.py",
    "plot_group_analysis_study.py",
    "plot_history_to_script.py",
)


@pytest.mark.parametrize("name", USER_GUIDE_EXAMPLES)
def test_user_guide_example_runs(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    example = REPO_ROOT / "docs/source/examples" / name

    assert example.is_file(), f"{name} is referenced by the user guide but missing"
    runpy.run_path(str(example), run_name="__main__")

    # Whatever the example prints becomes gallery page content, so it must not
    # embed machine-specific paths or a build date; those would make the
    # published page change on every build and on every machine.
    printed = capsys.readouterr().out
    leaks = [
        marker for marker in (str(REPO_ROOT), "/var/folders/", "/private/tmp/", "generated on") if marker in printed
    ]
    assert not leaks, f"{name} prints machine-specific output: {leaks}"


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
        "resources/help/eegplot.md",
        "resources/help/pop_eegplot.md",
        "resources/skills/eegprep-cli.md",
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
