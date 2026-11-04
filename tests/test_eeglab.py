import os
import subprocess
import sys
from unittest import mock

import numpy as np


from eegprep.functions.adminfunc import eeglab as eeglab_module
from eegprep.functions.adminfunc.is_sccn import is_sccn
from eegprep.functions.adminfunc.iseeglabdeployed import iseeglabdeployed
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


def test_is_sccn_uses_the_unix_hostname(monkeypatch):
    calls = []

    def hostname(text: str, *, returncode: int = 0, error: BaseException | None = None):
        def run(*_args, **kwargs):
            calls.append(kwargs)
            if error is not None:
                raise error
            return subprocess.CompletedProcess(["hostname"], returncode, stdout=text, stderr="")

        return run

    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(subprocess, "run", hostname("lab.ucsd.edu\n"))
    assert is_sccn() == 1
    assert calls[-1]["timeout"] > 0

    monkeypatch.setattr(subprocess, "run", hostname("lab.ucsd.edu"))
    assert is_sccn() == 1

    monkeypatch.setattr(subprocess, "run", hostname("localhost\n"))
    assert is_sccn() == 0

    monkeypatch.setattr(
        subprocess,
        "run",
        hostname("lab.ucsd.edu\n", error=subprocess.TimeoutExpired(["hostname"], 5)),
    )
    assert is_sccn() == 0

    monkeypatch.setattr(subprocess, "run", hostname("lab.ucsd.edu\n", returncode=1))
    assert is_sccn() == 0

    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(subprocess, "run", hostname("lab.ucsd.edu\n"))
    assert is_sccn() == 0


def test_iseeglabdeployed_follows_sys_frozen(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert iseeglabdeployed() == 0
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert iseeglabdeployed() == 1


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
