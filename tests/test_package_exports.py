"""Tests for the public eegprep package export surface."""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap

import pytest

import eegprep


@pytest.mark.parametrize(
    ("name", "module_name", "attr_name"),
    (
        ("gui", "eegprep.functions.adminfunc.eeglab", "gui"),
        ("EEGPrepSession", "eegprep.functions.guifunc.session", "EEGPrepSession"),
        ("pop_study", "eegprep.functions.studyfunc.pop_study", "pop_study"),
        ("pop_clust", "eegprep.functions.studyfunc.pop_clust", "pop_clust"),
        ("plugin_menu", "eegprep.functions.adminfunc.plugin_menu", "plugin_menu"),
    ),
)
def test_representative_lazy_exports_match_direct_import(name: str, module_name: str, attr_name: str) -> None:
    direct = getattr(importlib.import_module(module_name), attr_name)

    assert getattr(eegprep, name) is direct


def test_all_lists_public_exports_once() -> None:
    assert eegprep.__all__ == ["__version__", *eegprep._LAZY_EXPORTS]
    assert len(eegprep.__all__) == len(set(eegprep.__all__))


def test_eegrej_export_matches_eeglab_low_level_function() -> None:
    from eegprep.functions.popfunc.eeg_eegrej import eeg_eegrej
    from eegprep.functions.popfunc.eeg_multieegplot import eeg_multieegplot
    from eegprep.functions.sigprocfunc.eegplot import eegplot as sigproc_eegplot
    from eegprep.functions.sigprocfunc.eegrej import eegrej as sigproc_eegrej
    from eegprep.functions.sigprocfunc.rmbase import rmbase as sigproc_rmbase

    assert eegprep.eegplot is sigproc_eegplot
    assert eegprep.eeg_multieegplot is eeg_multieegplot
    assert eegprep.eegrej is sigproc_eegrej
    assert eegprep.eeg_eegrej is eeg_eegrej
    assert eegprep.eegrej is not eegprep.eeg_eegrej
    assert eegprep.rmbase is sigproc_rmbase


def test_phase_6b_public_exports_are_intentional() -> None:
    expected = {
        "CallbackSpec",
        "ConsoleDatasetResult",
        "ConsolePopResult",
        "ControlSpec",
        "DialogSpec",
        "EEGPrepConsoleWorkspace",
        "EEGPrepSession",
        "CatalogSourceType",
        "ExtensionAction",
        "ExtensionCatalog",
        "ExtensionCatalogEntry",
        "ExtensionDependency",
        "ExtensionLoadError",
        "ExtensionMenu",
        "ExtensionPopFunction",
        "ExtensionRecord",
        "ExtensionRegistry",
        "ExtensionResource",
        "ExtensionSourceType",
        "ExtensionSpec",
        "ExtensionStatus",
        "ExtensionValidationResult",
        "build_safe_install_commands",
        "build_safe_update_commands",
        "bundled_plugins",
        "discover_extensions",
        "eeg_emptyset",
        "eeg_retrieve",
        "eeg_store",
        "firws",
        "firwsord",
        "format_plugin_menu",
        "inputgui",
        "listdlg2",
        "load_extension_catalog",
        "plugin_menu",
        "plugin_status",
        "pop_delset",
        "pop_eegplot",
        "pop_editoptions",
        "pop_newset",
        "pophelp",
        "validate_extension_spec",
    }

    assert expected <= set(eegprep.__all__)


def test_import_eegprep_is_lightweight() -> None:
    code = textwrap.dedent(
        """
        import sys
        import eegprep

        blocked = ["PySide6", "torch", "mne"]
        print(",".join(name for name in blocked if name in sys.modules))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == ""
