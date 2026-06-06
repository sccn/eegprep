from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.functions.studyfunc.pop_loadstudy import pop_loadstudy
from eegprep.functions.studyfunc.pop_savestudy import pop_savestudy
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.functions.studyfunc.pop_studydesign import pop_studydesign
from eegprep.functions.studyfunc.pop_addindepvar import pop_addindepvar
from eegprep.functions.studyfunc.pop_importgroupvar import pop_importgroupvar
from eegprep.functions.studyfunc.pop_listfactors import pop_listfactors
from eegprep.functions.studyfunc.std_addvarlevel import std_addvarlevel
from eegprep.functions.studyfunc.std_builddesignmat import std_builddesignmat
from eegprep.functions.studyfunc.std_checkset import std_checkdatasetinfo, std_checkset
from eegprep.functions.studyfunc.std_editset import std_editset
from eegprep.functions.studyfunc.std_findgroupvars import std_findgroupvars
from eegprep.functions.studyfunc.std_makedesign import std_makedesign
from eegprep.functions.studyfunc.std_rebuilddesign import std_rebuilddesign
from eegprep.functions.studyfunc.std_saveindvar import std_saveindvar
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

    study, alleeg, command = pop_study(None, [first, second], name="Oddball", task="button press", return_com=True)

    assert study["name"] == "Oddball"
    assert study["task"] == "button press"
    assert study["datasetinfo"][0]["index"] == 1
    assert study["datasetinfo"][0]["subject"] == "S01"
    assert study["datasetinfo"][1]["subject"] == "S2"
    assert study["subject"] == ["S01", "S2"]
    assert study["condition"] == ["standard", "target"]
    assert study["currentdesign"] == 1
    assert study["design"][0]["variable"][0]["label"] == "condition"
    assert alleeg[1]["setname"] == "two"
    assert command.startswith("STUDY, ALLEEG = pop_study(")


def test_pop_study_history_preserves_requested_design_name():
    study, _alleeg, command = pop_study(None, [_eeg("one", condition="target")], design="ERP", return_com=True)

    assert study["design"][0]["name"] == "ERP"
    assert "design='ERP'" in command


def test_std_editset_updates_datasetinfo_and_loaded_dataset_metadata():
    study, alleeg = pop_study(None, [_eeg("one")], name="Initial")

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
    assert alleeg[0].get("subject", "") == ""
    assert alleeg[0].get("condition", "") == ""
    assert "std_editset" in command


def test_std_makedesign_selects_1_based_design_and_validates_variables():
    study, alleeg = pop_study(
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
    assert study["design"][1]["variable"][0]["level"] == "two"
    assert study["design"][1]["variable"][0]["levelindex"] == [1]
    assert "std_makedesign" in command
    with pytest.raises(ValueError, match="Unknown STUDY design variable"):
        std_makedesign(study, alleeg, 1, variable1="missing")
    with pytest.raises(ValueError, match="Cannot select"):
        std_selectdesign(study, alleeg, 3)


def test_pop_studydesign_selects_and_updates_design():
    study, alleeg = pop_study(
        None,
        [_eeg("one", subject="S01", condition="target"), _eeg("two", subject="S02", condition="standard")],
    )

    study, alleeg, command = pop_studydesign(
        study, alleeg, 1, variable1="condition", values1=["target"], return_com=True
    )

    assert study["currentdesign"] == 1
    assert study["design"][0]["variable"][0]["value"] == ["target"]
    assert command.startswith("STUDY = std_makedesign(")


def test_std_makedesign_accepts_numpy_subject_selection():
    study, alleeg = pop_study(
        None,
        [
            _eeg("one", subject="S01", condition="target"),
            _eeg("two", subject="S02", condition="standard"),
        ],
    )

    selected, _command = std_makedesign(study, alleeg, 1, subjselect=np.array(["S01"]), return_com=True)
    all_subjects, _command = std_makedesign(study, alleeg, 1, subjselect=np.array([]), return_com=True)

    assert selected["design"][0]["cases"]["value"] == ["S01"]
    assert all_subjects["design"][0]["cases"]["value"] == ["S01", "S02"]


def test_pop_savestudy_and_pop_loadstudy_roundtrip_with_dataset_loading(tmp_path):
    eeg = _eeg("saved", subject="S01", condition="target")
    set_file = tmp_path / "saved.set"
    pop_saveset(eeg, str(set_file))
    loaded = _eeg("saved", subject="S01", condition="target")
    loaded["filename"] = set_file.name
    loaded["filepath"] = str(tmp_path)
    study, alleeg = pop_study(None, [loaded], name="Saved study")

    saved, save_command = pop_savestudy(study, alleeg, filename="saved.study", filepath=tmp_path, return_com=True)
    reloaded, loaded_alleeg, load_command = pop_loadstudy("saved.study", filepath=tmp_path, return_com=True)

    assert saved["saved"] == "yes"
    assert reloaded["name"] == "Saved study"
    assert loaded_alleeg[0]["setname"] == "saved"
    assert Path(reloaded["filepath"]) == tmp_path
    assert "pop_savestudy" in save_command
    assert "pop_loadstudy" in load_command


def test_pop_savestudy_resave_uses_existing_study_path(tmp_path):
    eeg = _eeg("saved", subject="S01", condition="target")
    study, alleeg = pop_study(None, [eeg], name="Initial study")
    saved, _save_command = pop_savestudy(study, alleeg, filename="saved.study", filepath=tmp_path, return_com=True)

    saved["name"] = "Resaved study"
    resaved, command = pop_savestudy(saved, alleeg, savemode="resave", return_com=True)
    reloaded, _loaded_alleeg = pop_loadstudy("saved.study", filepath=tmp_path, load_datasets="off")

    assert resaved["filename"] == "saved.study"
    assert Path(resaved["filepath"]) == tmp_path
    assert reloaded["name"] == "Resaved study"
    assert "savemode='resave'" in command


def test_pop_savestudy_resavedatasets_writes_loaded_dataset_files(tmp_path):
    eeg = _eeg("saved", subject="S01", condition="target")
    eeg["filepath"] = str(tmp_path)
    eeg["saved"] = "no"
    study, alleeg = pop_study(None, [eeg], name="Dataset resave study")

    saved, command = pop_savestudy(
        study, alleeg, filename="saved.study", filepath=tmp_path, resavedatasets="on", return_com=True
    )
    reloaded, loaded_alleeg = pop_loadstudy(saved["filename"], filepath=saved["filepath"])

    assert (tmp_path / "saved.set").exists()
    assert alleeg[0]["saved"] == "no"
    assert loaded_alleeg[0]["setname"] == "saved"
    assert reloaded["name"] == "Dataset resave study"
    assert "resavedatasets='on'" in command


def test_pop_loadstudy_missing_dataset_fails_without_realigning_metadata(tmp_path):
    first = _eeg("first", subject="S01", condition="target", session=1, run=1)
    second = _eeg("second", subject="S02", condition="standard", session=2, run=2)
    first_file = tmp_path / "first.set"
    second_file = tmp_path / "second.set"
    pop_saveset(first, str(first_file))
    pop_saveset(second, str(second_file))
    for eeg, path in ((first, first_file), (second, second_file)):
        eeg["filename"] = path.name
        eeg["filepath"] = str(tmp_path)
    study, alleeg = pop_study(None, [first, second], name="Saved study")
    pop_savestudy(study, alleeg, filename="saved.study", filepath=tmp_path)
    second_file.unlink()

    with pytest.raises(FileNotFoundError, match="second.set"):
        pop_loadstudy("saved.study", filepath=tmp_path)

    unloaded_study, unloaded_alleeg = pop_loadstudy("saved.study", filepath=tmp_path, load_datasets="off")
    assert unloaded_alleeg == []
    assert unloaded_study["datasetinfo"][1]["subject"] == "S02"
    assert unloaded_study["datasetinfo"][1]["session"] == 2


def test_dataset_consistency_reports_mismatch_without_mutating_files():
    study, alleeg = pop_study(
        None,
        [_eeg("one", subject="S01", srate=250.0), _eeg("two", subject="S02", srate=128.0)],
    )

    checked, checked_alleeg = std_checkset(study, alleeg)
    consistency = std_checkdatasetinfo(checked, checked_alleeg)

    assert consistency["same_srate"] is False
    assert "datasets do not share one sampling rate" in consistency["problems"]
    assert checked["etc"]["eegprep"]["dataset_consistency"]["same_srate"] is False


def test_design_variable_helpers_build_factors_and_matrices():
    study, alleeg = pop_study(
        None,
        [
            _eeg("s01_target", subject="S01", condition="target", group="control"),
            _eeg("s01_standard", subject="S01", condition="standard", group="control"),
            _eeg("s02_target", subject="S02", condition="target", group="patient"),
        ],
    )

    study, command = std_makedesign(
        study,
        alleeg,
        1,
        variable1="condition",
        variable2="group",
        return_com=True,
    )
    leveled = std_addvarlevel(study)
    factors = pop_listfactors(leveled, constant="off")
    variable, values, categorical = pop_addindepvar(
        {"factors": ["condition"], "factorvals": [["standard", "target"]], "numerical": [False]},
        var="condition",
        values=["target"],
    )
    matrix, labels, catflag = std_builddesignmat(
        {
            "variable": [
                {"label": "condition", "value": ["target", "standard"], "vartype": "categorical"},
                {"label": "rt", "value": [], "vartype": "continuous"},
            ]
        },
        [{"condition": "target", "rt": 300.0}, {"condition": "standard", "rt": 400.0}],
        expanding=True,
    )

    assert "std_makedesign" in command
    assert leveled["design"][0]["variable"][0]["level"] == "one"
    assert leveled["design"][0]["variable"][1]["level"] == "two"
    assert {factor["description"] for factor in factors} >= {"condition - standard", "condition - target"}
    assert variable == "condition"
    assert values == ["target"]
    assert categorical == 1
    assert labels == ["condition-target", "condition-standard", "rt", "constant"]
    np.testing.assert_allclose(matrix[:, -2:], [[300.0, 1.0], [400.0, 1.0]])
    np.testing.assert_array_equal(catflag, np.asarray([True, True, False, False]))


def test_import_save_and_rebuild_independent_variables():
    study, alleeg = pop_study(
        None,
        [
            _eeg("s01", subject="S01", condition="target", group="control"),
            _eeg("s02", subject="S02", condition="standard", group="patient"),
        ],
    )

    imported, import_command = pop_importgroupvar(
        study,
        1,
        variable="age_group",
        values={"S01": "young", "S02": "older"},
        return_com=True,
    )
    saved, save_command = std_saveindvar(imported, return_com=True)
    imported["datasetinfo"][1]["condition"] = "oddball"
    rebuilt, rebuild_command = std_rebuilddesign(imported, alleeg, return_com=True)

    assert imported["datasetinfo"][0]["age_group"] == "young"
    assert imported["design"][0]["variable"][-1]["label"] == "age_group"
    assert imported["design"][0]["variable"][-1]["level"] == "two"
    assert "pop_importgroupvar" in import_command
    assert saved["etc"]["eegprep"]["independent_variables"]
    assert "std_saveindvar" in save_command
    assert "oddball" in rebuilt["design"][0]["variable"][0]["value"]
    assert "std_rebuilddesign" in rebuild_command
    assert std_findgroupvars(imported) == ["condition", "group", "age_group"]
