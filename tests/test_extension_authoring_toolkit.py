"""Tests for Phase 4 extension authoring assets."""

from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from eegprep.extensions import EXTENSION_ENTRY_POINT_GROUP, ExtensionRegistry, ExtensionStatus
from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace, _console_python_command
from eegprep.functions.guifunc.session import EEGPrepSession

EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples" / "extensions"
EXAMPLE_PACKAGES = (
    "eegprep_ext_template",
    "eegprep_ext_signal_transform",
    "eegprep_ext_file_io",
    "eegprep_ext_gui_dialog",
    "eegprep_ext_plot_browser",
    "eegprep_ext_optional_dependency",
)


class ExampleEntryPoint:
    def __init__(self, package: str) -> None:
        self.name = package.removeprefix("eegprep_ext_")
        self.group = EXTENSION_ENTRY_POINT_GROUP
        self.dist = type("Distribution", (), {"metadata": {"Name": package.replace("_", "-")}})()
        self.package = package

    def load(self) -> Any:
        return importlib.import_module(f"{self.package}.registration").register


class Renderer:
    def __init__(self, result: dict[str, Any] | None) -> None:
        self.result = result
        self.spec = None

    def run(self, spec: Any, initial_values: dict[str, Any] | None = None) -> dict[str, Any] | None:
        self.spec = spec
        return self.result


@pytest.mark.parametrize("package", EXAMPLE_PACKAGES)
def test_example_specs_validate_and_load_lazy_targets(package: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _add_example_package(package, monkeypatch)
    registry = ExtensionRegistry(
        include_bundled=False,
        entry_points_provider=lambda group: (ExampleEntryPoint(package),),
        version_provider=_version_provider,
    )

    records = registry.discover()

    assert len(records) == 1
    record = records[0]
    assert record.status == ExtensionStatus.INSTALLED
    assert record.spec is not None
    assert record.spec.menus
    assert record.spec.actions
    assert record.spec.pop_functions
    assert record.spec.eegprep_requires
    for action in record.spec.actions:
        assert callable(action.load())
    for pop_function in record.spec.pop_functions:
        assert callable(pop_function.load())


def test_template_resources_pop_function_and_console_history(monkeypatch: pytest.MonkeyPatch) -> None:
    _add_example_package("eegprep_ext_template", monkeypatch)
    package = importlib.import_module("eegprep_ext_template")
    spec = package.register()
    eeg = package.load_sample_eeg()

    assert spec.help_resources[0].exists()
    assert "pop_template_gain" in spec.help_resources[0].read_text()
    assert spec.package_data_resources[0].exists()
    assert eeg["data"].shape == (2, 4)

    output, command = package.pop_template_gain(eeg, 2, return_com=True)
    np.testing.assert_allclose(output["data"], eeg["data"] * 2)
    assert command == "EEG = pop_template_gain(EEG, 2);"
    assert _console_python_command(command) == "EEG = pop_template_gain(EEG, 2)"
    assert package.pop_template_gain_dialog_spec().show_help_button is False

    cancelled, cancel_command = package.pop_template_gain(eeg, gui=True, renderer=Renderer(None), return_com=True)
    assert cancelled is eeg
    assert cancel_command == ""

    session = EEGPrepSession()
    session.store_current(package.load_sample_eeg(), new=True, command="EEG = load_sample_eeg();")
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_template_gain": package.pop_template_gain})
    try:
        result = workspace.namespace["pop_template_gain"](session.EEG, 3)
        assert result.command == "EEG = pop_template_gain(EEG, 3);"
        assert session.LASTCOM == result.command
        assert session.ALLCOM[-1] == result.command
        np.testing.assert_allclose(session.EEG["data"], package.load_sample_eeg()["data"] * 3)
    finally:
        workspace.close()


def _import_example(package: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    _add_example_package(package, monkeypatch)
    return importlib.import_module(package)


def _add_example_package(package: str, monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in list(sys.modules):
        if module_name == package or module_name.startswith(f"{package}."):
            del sys.modules[module_name]
    monkeypatch.syspath_prepend(str(EXAMPLES_ROOT / package / "src"))
    importlib.invalidate_caches()


def _version_provider(name: str) -> str:
    if name == "numpy":
        return np.__version__
    if name == "eegprep-template-optional-model":
        raise metadata.PackageNotFoundError(name)
    return "999.0"
