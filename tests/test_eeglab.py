from unittest import mock

import eegprep

from eegprep.functions.adminfunc import eeglab as eeglab_module
from eegprep.functions.guifunc.session import EEGPrepSession


def test_eeglab_versions_and_nogui_entry_points():
    session = EEGPrepSession()

    assert eeglab_module.eeglab("versions") == eegprep.__version__
    assert eeglab_module.eeglab("nogui", session=session, show=False) is session


def test_eeglab_full_mode_builds_window_without_showing():
    session = EEGPrepSession()
    window = mock.Mock()

    with mock.patch.object(eeglab_module, "build_main_window", return_value=window) as build:
        returned = eeglab_module.eeglab("full", session=session, show=False, include_plugins=False)

    assert returned is window
    build.assert_called_once_with(session, all_menus=True, include_plugins=False)
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


def test_eeglab_main_parses_nogui_and_full_plugin_options():
    with mock.patch.object(eeglab_module, "eeglab") as eeglab:
        assert eeglab_module.main(["--nogui"]) == 0
        eeglab.assert_called_once_with("nogui", show=False)

    with mock.patch.object(eeglab_module, "eeglab") as eeglab:
        assert eeglab_module.main(["--full", "--no-plugins"]) == 0
        eeglab.assert_called_once_with("full", block=True, include_plugins=False)
