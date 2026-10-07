from __future__ import annotations

from importlib import resources

import numpy as np
import pytest

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS, EEGOptions
from eegprep.functions.adminfunc.eeg_readoptions import eeg_readoptions
from eegprep.functions.adminfunc.pop_editoptions import pop_editoptions
from tests.eeglab_tests import eeglab_test


def _reference_option_script(eeglab_backend, request, script, *variables):
    if request.config.getoption("--eeglab-backend") == "matlab":
        return eeglab_backend(
            "eegprep_test_run_script", script, np.array([variables], dtype=object), nargout=int(bool(variables))
        )
    # MATLAB initializes caller variables with scripts; Python exposes the
    # actual defaults/settings object instead of executing .m files at runtime.
    return EEGOptions().to_dict() if script == "eeg_optionsbackup" else EEG_OPTIONS


@eeglab_test("unittesting_adminfunc/eeg_options/adminfunc_eeg_options_wrapperTest.m", "test_pass_general")
def test_reference_eeg_options_script(eeglab_backend, eeglab_suite_root, request):
    if (eeglab_suite_root / "eeglab/functions/adminfunc/eeg_options.m").is_file():
        _reference_option_script(eeglab_backend, request, "eeg_options")


@eeglab_test("unittesting_adminfunc/eeg_optionsbackup/adminfunc_eeg_optionsbackup_wrapperTest.m", "test_pass_general")
def test_reference_eeg_optionsbackup_script(eeglab_backend, eeglab_suite_root, request):
    if (eeglab_suite_root / "eeglab/functions/adminfunc/eeg_optionsbackup.m").is_file():
        _reference_option_script(eeglab_backend, request, "eeg_optionsbackup")


@eeglab_test("unittesting_adminfunc/eeglab_options/adminfunc_eeglab_options_wrapperTest.m", "test_pass_general")
def test_reference_eeglab_options_storedisk_exists(eeglab_backend, eeglab_suite_root, request):
    assert (eeglab_suite_root / "eeglab/functions/adminfunc/eeg_options.m").is_file()
    variables = _reference_option_script(eeglab_backend, request, "eeglab_options", "option_storedisk")
    assert "option_storedisk" in variables


@eeglab_test("unittesting_adminfunc/eeg_readoptions/adminfunc_eeg_readoptions_wrapperTest.m", "test_pass_general")
def test_reference_eeg_readoptions_general(eeglab_backend, eeglab_suite_root):
    path = eeglab_suite_root / "eeglab/functions/adminfunc/eeg_options.m"
    if path.is_file():
        _, options = eeglab_backend("eeg_readoptions", str(path), nargout=2)
        assert np.asarray(options).size


@eeglab_test("unittesting_adminfunc/eeg_readoptions/adminfunc_eeg_readoptions_wrapperTest.m", "test_pass_backup")
def test_reference_eeg_readoptions_backup(eeglab_backend, eeglab_suite_root):
    backup = np.array([[("option_storedisk",), ("option_rememberfolder",)]], dtype=[("varname", object)])
    path = eeglab_suite_root / "eeglab/functions/adminfunc/eeg_options.m"
    if path.is_file():
        _, options = eeglab_backend("eeg_readoptions", str(path), backup, nargout=2)
        records = options.ravel(order="F") if isinstance(options, np.ndarray) else options
        assert np.asarray(records[0]["value"]).size
        assert np.asarray(records[1]["value"]).size


@pytest.fixture(autouse=True)
def restore_eeg_options():
    original = dict(EEG_OPTIONS)
    try:
        yield
    finally:
        EEG_OPTIONS.clear()
        EEG_OPTIONS.update(original)


def test_pop_editoptions_updates_known_options_and_returns_history_command():
    command = pop_editoptions(option_computeica=0, option_storedisk=1)

    assert EEG_OPTIONS["option_computeica"] == 0
    assert EEG_OPTIONS["option_storedisk"] == 1
    assert command == "LASTCOM = pop_editoptions();"


def test_eeg_readoptions_parses_packaged_matlab_option_template():
    option_file = resources.files("eegprep.resources").joinpath("eeg_options_64bit.m")

    header, options = eeg_readoptions(option_file)

    values = {option["varname"]: option["value"] for option in options}
    assert "Do not edit or remove this file" in header
    assert values["option_storedisk"] == 0
    assert values["option_single"] == 0
    assert values["option_cachesize"] == 500
    assert all(set(option) == {"varname", "value", "description"} for option in options)


def test_eeg_readoptions_fills_requested_backup_records_only():
    option_file = resources.files("eegprep.resources").joinpath("eeg_options_64bit.m")
    backup = [
        {"varname": "option_storedisk", "description": "stored", "value": None},
        {"varname": "option_rememberfolder", "description": "folder", "value": None},
    ]

    _header, options = eeg_readoptions(option_file, backup)

    assert options == [
        {"varname": "option_storedisk", "description": "stored", "value": 0},
        {"varname": "option_rememberfolder", "description": "folder", "value": 1},
    ]
