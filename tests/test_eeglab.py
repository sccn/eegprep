from unittest import mock

import numpy as np


from eegprep.functions.adminfunc import eeglab as eeglab_module
from tests.eeglab_tests import eeglab_test


IS_SCCN_WRAPPER = "unittesting_adminfunc/is_sccn/adminfunc_is_sccn_wrapperTest.m"
IS_DEPLOYED_WRAPPER = "unittesting_adminfunc/iseeglabdeployed/adminfunc_iseeglabdeployed_wrapperTest.m"


@eeglab_test(IS_SCCN_WRAPPER, "test_pass_general")
def test_reference_is_sccn(eeglab_backend):
    # The original prints a message for every result; it makes no assertion.
    result = np.asarray(eeglab_backend("is_sccn"))
    if np.all(np.abs(result - 1) <= 1e-4):
        print("IN SCCN")
    elif np.all(np.abs(result) <= 1e-4):
        print("NOT IN SCCN")
    else:
        print("WHERE AM I?")


@eeglab_test(IS_DEPLOYED_WRAPPER, "test_test_iseeglabdeployed")
def test_reference_iseeglabdeployed(eeglab_backend):
    assert np.all(np.asarray(eeglab_backend("iseeglabdeployed")) == 0)


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
