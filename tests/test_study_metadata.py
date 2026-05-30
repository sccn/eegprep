from pathlib import Path

import pytest

from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.functions.studyfunc.pop_loadstudy import pop_loadstudy
from eegprep.functions.studyfunc.pop_savestudy import pop_savestudy
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.functions.studyfunc.pop_studydesign import pop_studydesign
from eegprep.functions.studyfunc.std_checkset import std_checkdatasetinfo, std_checkset
from eegprep.functions.studyfunc.std_editset import std_editset
from eegprep.functions.studyfunc.std_makedesign import std_makedesign
from eegprep.functions.studyfunc.std_selectdesign import std_selectdesign
from tests.fixtures import create_test_eeg


def _eeg(setname="demo", *, subject="", condition="", group="", session=None, run=None, srate=250.0):
    eeg = create_test_eeg(n_channels=4, n_samples=20, srate=srate)
    eeg.update(
        {
            "setname": setname,
            "filename": f"{setname}.set",
            "filepath": "",
            "subject": subject,
            "condition": condition,
            "group": group,
            "session": [] if session is None else session,
            "run": [] if run is None else run,
        }
    )
    return eeg


def test_pop_study_builds_full_metadata_and_default_design():
    first = _eeg("one", subject="S01", condition="target", group="control", session=1, run=1)
    second = _eeg("two", condition="standard")

    study, alleeg, command = pop_study(None, [first, second], name="Oddball", task="button press")

    assert study["name"] == "Oddball"
    assert study["task"] == "button press"
    assert study["datasetinfo"][0]["index"] == 1
    assert study["datasetinfo"][0]["subject"] == "S01"
    assert study["datasetinfo"][1]["subject"] == "S2"
    assert study["subject"] == ["S01", "S2"]
    assert study["condition"] == ["target", "standard"]
    assert study["currentdesign"] == 1
    assert study["design"][0]["variable"][0]["label"] == "condition"
    assert alleeg[1]["setname"] == "two"
    assert command.startswith("STUDY, ALLEEG, LASTCOM = pop_study(")


def test_std_editset_updates_datasetinfo_and_loaded_dataset_metadata():
    study, alleeg, _command = pop_study(None, [_eeg("one")], name="Initial")

    edited, edited_alleeg, command = std_editset(
        study,
        alleeg,
        name="Edited",
        commands=["index", 1, "subject", "S99", "session", 2, "condition", "rare"],
        return_com=True,
    )

    assert edited["name"] == "Edited"
    assert edited["datasetinfo"][0]["subject"] == "S99"
    assert edited["datasetinfo"][0]["session"] == 2
    assert edited_alleeg[0]["subject"] == "S99"
    assert edited_alleeg[0]["condition"] == "rare"
    assert "std_editset" in command


def test_std_makedesign_selects_1_based_design_and_validates_variables():
    study, alleeg, _command = pop_study(
        None,
        [
            _eeg("one", subject="S01", condition="target", group="control"),
            _eeg("two", subject="S02", condition="standard", group="patient"),
        ],
    )

    study, command = std_makedesign(study, alleeg, 2, variable1="group", values1=["control"], return_com=True)

    assert study["currentdesign"] == 2
    assert study["design"][1]["variable"][0]["label"] == "group"
    assert study["design"][1]["variable"][0]["value"] == ["control"]
    assert study["design"][1]["variable"][0]["pairing"] == "off"
    assert "std_makedesign" in command
    with pytest.raises(ValueError, match="Unknown STUDY design variable"):
        std_makedesign(study, alleeg, 1, variable1="missing")
    with pytest.raises(ValueError, match="Cannot select"):
        std_selectdesign(study, alleeg, 3)


def test_pop_studydesign_selects_and_updates_design():
    study, alleeg, _command = pop_study(
        None,
        [_eeg("one", subject="S01", condition="target"), _eeg("two", subject="S02", condition="standard")],
    )

    study, alleeg, command = pop_studydesign(study, alleeg, 1, variable1="condition", values1=["target"])

    assert study["currentdesign"] == 1
    assert study["design"][0]["variable"][0]["value"] == ["target"]
    assert command.startswith("STUDY, LASTCOM = std_makedesign(")


def test_pop_savestudy_and_pop_loadstudy_roundtrip_with_dataset_loading(tmp_path):
    eeg = _eeg("saved", subject="S01", condition="target")
    set_file = tmp_path / "saved.set"
    pop_saveset(eeg, str(set_file))
    loaded = _eeg("saved", subject="S01", condition="target")
    loaded["filename"] = set_file.name
    loaded["filepath"] = str(tmp_path)
    study, alleeg, _command = pop_study(None, [loaded], name="Saved study")

    saved, save_command = pop_savestudy(study, alleeg, filename="saved.study", filepath=tmp_path)
    reloaded, loaded_alleeg, load_command = pop_loadstudy("saved.study", filepath=tmp_path)

    assert saved["saved"] == "yes"
    assert reloaded["name"] == "Saved study"
    assert loaded_alleeg[0]["setname"] == "saved"
    assert Path(reloaded["filepath"]) == tmp_path
    assert "pop_savestudy" in save_command
    assert "pop_loadstudy" in load_command


def test_dataset_consistency_reports_mismatch_without_mutating_files():
    study, alleeg, _command = pop_study(
        None,
        [_eeg("one", subject="S01", srate=250.0), _eeg("two", subject="S02", srate=128.0)],
    )

    checked, checked_alleeg = std_checkset(study, alleeg)
    consistency = std_checkdatasetinfo(checked, checked_alleeg)

    assert consistency["same_srate"] is False
    assert "datasets do not share one sampling rate" in consistency["problems"]
    assert checked["etc"]["eegprep"]["dataset_consistency"]["same_srate"] is False
