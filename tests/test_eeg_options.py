from __future__ import annotations

from importlib import resources

import pytest

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS, EEGOptions
from eegprep.functions.adminfunc.eeg_readoptions import eeg_readoptions
from eegprep.functions.adminfunc.pop_editoptions import pop_editoptions
from tests.eeglab_tests import eeglab_test


@pytest.fixture(autouse=True)
def restore_eeg_options():
    original = dict(EEG_OPTIONS)
    try:
        yield
    finally:
        EEG_OPTIONS.clear()
        EEG_OPTIONS.update(original)


@eeglab_test("unittesting_adminfunc/eeg_options/pass_general.m", "test_pass_general")
@eeglab_test("unittesting_adminfunc/eeg_optionsbackup/pass_general.m", "test_pass_general")
@eeglab_test("unittesting_adminfunc/eeglab_options/pass_general.m", "test_pass_general")
def test_default_options_expose_the_processing_and_storage_contract():
    defaults = EEGOptions().to_dict()

    assert EEG_OPTIONS == defaults
    assert EEG_OPTIONS["option_storedisk"] == 0
    assert EEG_OPTIONS["option_memmapdata"] == 0
    assert EEG_OPTIONS["option_single"] == 1
    assert EEG_OPTIONS["option_computeica"] == 1


@eeglab_test("unittesting_adminfunc/pop_editoptions/test_pop_editoptions.m", "test_test_pop_editoptions")
def test_pop_editoptions_updates_known_options_and_returns_history_command():
    command = pop_editoptions(option_computeica=0, option_storedisk=1)

    assert EEG_OPTIONS["option_computeica"] == 0
    assert EEG_OPTIONS["option_storedisk"] == 1
    assert command == "LASTCOM = pop_editoptions();"


def test_pop_editoptions_rejects_unknown_options_without_partial_mutation():
    original = dict(EEG_OPTIONS)

    with pytest.raises(KeyError, match="Unknown EEGPrep option"):
        pop_editoptions(option_does_not_exist=1)

    assert EEG_OPTIONS == original


@eeglab_test("unittesting_adminfunc/eeg_readoptions/pass_general.m", "test_pass_general")
def test_eeg_readoptions_parses_packaged_matlab_option_template():
    option_file = resources.files("eegprep.resources").joinpath("eeg_options_64bit.m")

    header, options = eeg_readoptions(option_file)

    values = {option["varname"]: option["value"] for option in options}
    assert "Do not edit or remove this file" in header
    assert values["option_storedisk"] == 0
    assert values["option_single"] == 0
    assert values["option_cachesize"] == 500
    assert all(set(option) == {"varname", "value", "description"} for option in options)


@eeglab_test("unittesting_adminfunc/eeg_readoptions/pass_backup.m", "test_pass_backup")
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
