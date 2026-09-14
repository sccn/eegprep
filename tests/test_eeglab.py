import socket
import sys
from unittest import mock

import eegprep

from eegprep.functions.adminfunc import eeglab as eeglab_module
from eegprep.functions.guifunc.session import EEGPrepSession
from tests.eeglab_tests import eeglab_test


IS_SCCN_WRAPPER = "unittesting_adminfunc/is_sccn/adminfunc_is_sccn_wrapperTest.m"
IS_DEPLOYED_WRAPPER = "unittesting_adminfunc/iseeglabdeployed/adminfunc_iseeglabdeployed_wrapperTest.m"


def test_eeglab_versions_and_nogui_entry_points():
    session = EEGPrepSession()

    assert eeglab_module.eeglab("versions") == eegprep.__version__
    assert eeglab_module.eeglab("nogui", session=session, show=False) is session


@eeglab_test(IS_SCCN_WRAPPER, "test_pass_general")
def test_eegprep_startup_does_not_depend_on_an_institutional_hostname(monkeypatch):
    # is_sccn only controls behavior tied to an obsolete institutional network,
    # and its MATLAB test accepts either result without an assertion. EEGPrep's
    # standalone startup must not consult DNS at all.
    def fail_hostname_lookup():
        raise AssertionError("portable EEGPrep startup must not inspect the hostname")

    monkeypatch.setattr(socket, "gethostname", fail_hostname_lookup)
    monkeypatch.setattr(socket, "getfqdn", fail_hostname_lookup)

    session = EEGPrepSession()
    assert eeglab_module.eeglab("nogui", session=session, show=False) is session


@eeglab_test(IS_DEPLOYED_WRAPPER, "test_test_iseeglabdeployed")
def test_eegprep_startup_has_one_runtime_path_for_frozen_python(monkeypatch):
    # iseeglabdeployed selects MATLAB Compiler behavior. EEGPrep has no MATLAB
    # runtime branch, so a frozen Python executable uses the same session path.
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    session = EEGPrepSession()
    assert eeglab_module.eeglab("nogui", session=session, show=False) is session


def test_eeglab_full_mode_builds_window_without_showing():
    session = EEGPrepSession()
    window = mock.Mock()

    with mock.patch.object(eeglab_module, "build_main_window", return_value=window) as build:
        returned = eeglab_module.eeglab("full", session=session, show=False, include_plugins=False)

    assert returned is window
    build.assert_called_once_with(
        session,
        all_menus=True,
        include_plugins=False,
        native_menu_bar=None,
        native_file_dialogs=None,
    )
    window.show.assert_not_called()
    window.exec.assert_not_called()


def test_eeglab_show_and_block_paths():
    session = EEGPrepSession()
    window = mock.Mock()
    window.exec.return_value = 7

    with mock.patch.object(eeglab_module, "build_main_window", return_value=window):
        assert eeglab_module.eeglab(session=session, show=True) is window
        assert eeglab_module.eeglab(session=session, block=True) == 7

    assert window.show.call_count == 1
    assert window.exec.call_count == 1


def test_gui_alias_forwards_to_eeglab_launcher():
    session = EEGPrepSession()

    with mock.patch.object(eeglab_module, "eeglab", return_value="window") as eeglab:
        returned = eeglab_module.gui(
            "full",
            session=session,
            show=False,
            include_plugins=False,
            native_menu_bar=False,
            native_file_dialogs=False,
        )

    assert returned == "window"
    eeglab.assert_called_once_with(
        "full",
        session=session,
        show=False,
        block=False,
        all_menus=None,
        include_plugins=False,
        native_menu_bar=False,
        native_file_dialogs=False,
    )


def test_eeglab_main_parses_nogui_and_full_plugin_options():
    with mock.patch.object(eeglab_module, "eeglab") as eeglab:
        assert eeglab_module.main(["--nogui"]) == 0
        eeglab.assert_called_once_with("nogui", show=False)

    with mock.patch.object(eeglab_module, "eeglab") as eeglab:
        assert eeglab_module.main(["--full", "--no-plugins"]) == 0
        eeglab.assert_called_once_with("full", block=True, include_plugins=False, native_menu_bar=None)

    with mock.patch.object(eeglab_module, "eeglab") as eeglab:
        assert eeglab_module.main(["--full", "--window-menu-bar"]) == 0
        eeglab.assert_called_once_with("full", block=True, include_plugins=True, native_menu_bar=False)
