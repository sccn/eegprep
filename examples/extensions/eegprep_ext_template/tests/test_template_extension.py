"""Tests that should pass after copying the template extension."""

from __future__ import annotations

from importlib import metadata

import numpy as np

from eegprep.extensions import ExtensionRegistry, ExtensionStatus, validate_extension_spec
from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace, _console_python_command
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep_ext_template import load_sample_eeg, pop_template_gain, register


class EntryPoint:
    name = "template"
    group = "eegprep.extensions"
    dist = None

    def load(self):
        return register


class CancelRenderer:
    def run(self, spec, initial_values=None):
        return None


def test_template_spec_resources_and_entry_point() -> None:
    spec = register()

    assert validate_extension_spec(spec).ok
    assert spec.menus[0].action == "pop_template_gain"
    loaded_pop = spec.pop_functions[0].load()
    assert loaded_pop.__name__ == pop_template_gain.__name__
    assert loaded_pop.__module__ == pop_template_gain.__module__
    assert "pop_template_gain" in spec.help_resources[0].read_text()
    assert spec.package_data_resources[0].exists()

    registry = ExtensionRegistry(include_bundled=False, entry_points_provider=lambda group: (EntryPoint(),))
    records = registry.discover()
    assert records[0].status == ExtensionStatus.INSTALLED


def test_template_pop_function_sample_data_and_gui_cancel() -> None:
    eeg = load_sample_eeg()

    output, command = pop_template_gain(eeg, 2, return_com=True)
    np.testing.assert_allclose(output["data"], eeg["data"] * 2)
    assert command == "EEG = pop_template_gain(EEG, 2);"
    assert _console_python_command(command) == "EEG = pop_template_gain(EEG, 2)"

    cancelled, cancel_command = pop_template_gain(eeg, gui=True, renderer=CancelRenderer(), return_com=True)
    assert cancelled is eeg
    assert cancel_command == ""


def test_template_console_wrapper_updates_history() -> None:
    session = EEGPrepSession()
    session.store_current(load_sample_eeg(), new=True, command="EEG = load_sample_eeg();")
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_template_gain": pop_template_gain})

    result = workspace.namespace["pop_template_gain"](session.EEG, 3)

    assert result.command == "EEG = pop_template_gain(EEG, 3);"
    assert session.LASTCOM == result.command
    assert session.ALLCOM[-1] == result.command
    np.testing.assert_allclose(session.EEG["data"], load_sample_eeg()["data"] * 3)
    workspace.close()


def test_no_plugins_mode_skips_template_discovery() -> None:
    def provider(*, group: str):
        raise AssertionError(f"entry points should not be loaded in --no-plugins mode: {group}")

    assert ExtensionRegistry(entry_points_provider=provider).discover(include_plugins=False) == ()


def test_optional_dependency_pattern_is_not_required_for_template() -> None:
    def missing_version(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    validation = validate_extension_spec(register(), check_dependencies=False, version_provider=missing_version)
    assert validation.ok
