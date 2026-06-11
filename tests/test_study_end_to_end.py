from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import pytest

from eegprep.functions.adminfunc import eeglabcompat
from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace
from eegprep.functions.guifunc.menu_actions import action_kind
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.functions.studyfunc.pop_chanplot import pop_chanplot
from eegprep.functions.studyfunc.pop_clust import pop_clust
from eegprep.functions.studyfunc.pop_clustedit import pop_clustedit
from eegprep.functions.studyfunc.pop_limo import pop_limo
from eegprep.functions.studyfunc.pop_loadstudy import pop_loadstudy
from eegprep.functions.studyfunc.pop_preclust import pop_preclust
from eegprep.functions.studyfunc.pop_precomp import pop_precomp
from eegprep.functions.studyfunc.pop_savestudy import pop_savestudy
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.functions.studyfunc.pop_studydesign import pop_studydesign
from tests.fixtures import create_test_eeg_with_ica, matlab_engine_available


def _study_eeg(setname: str, subject: str, condition: str, offset: float) -> dict:
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=32, n_trials=3, srate=128, n_components=3)
    eeg.update(
        {
            "setname": setname,
            "subject": subject,
            "condition": condition,
            "group": "control",
            "session": 1,
            "run": 1,
        }
    )
    eeg["data"] = np.asarray(eeg["data"], dtype=float) + offset
    eeg["icaact"] = np.asarray(eeg["icaact"], dtype=float) + offset * 0.1
    return eeg


def _saved_study_datasets(tmp_path: Path) -> list[dict]:
    np.random.seed(78)
    datasets = [
        _study_eeg("s01_target", "S01", "target", 0.0),
        _study_eeg("s02_standard", "S02", "standard", 1.0),
    ]
    for eeg in datasets:
        path = tmp_path / f"{eeg['setname']}.set"
        pop_saveset(eeg, str(path))
        eeg["filename"] = path.name
        eeg["filepath"] = str(path.parent)
    return datasets


def test_study_end_to_end_workflow_roundtrips_and_syncs_console(tmp_path):
    alleeg = _saved_study_datasets(tmp_path)

    study, alleeg, create_command = pop_study(
        None, alleeg, name="Phase 6c study", task="integration QA", return_com=True
    )
    study, alleeg, design_command = pop_studydesign(
        study,
        alleeg,
        1,
        variable1="condition",
        values1=["target", "standard"],
        return_com=True,
    )
    study, alleeg, channel_command = pop_precomp(
        study,
        alleeg,
        "channels",
        erp="on",
        spec="on",
        return_com=True,
    )
    study, plot_command, channel_figure = pop_chanplot(study, alleeg, channels=[1], measure="erp", return_com=True)
    study, alleeg, component_command = pop_precomp(
        study,
        alleeg,
        "components",
        erp="on",
        spec="on",
        scalp="on",
        return_com=True,
    )
    study, alleeg, preclust_command = pop_preclust(
        study,
        alleeg,
        preproc=[{"measure": "scalp", "npca": 2}],
        return_com=True,
    )
    study, clust_command = pop_clust(study, alleeg, clus_num=2, random_state=78, return_com=True)
    study, cluster_plot_command, cluster_figure = pop_clustedit(
        study, alleeg, action="plot", clusters=[2, 3], return_com=True
    )
    _saved, save_command = pop_savestudy(study, alleeg, filename="phase_6c.study", filepath=tmp_path, return_com=True)
    loaded, loaded_alleeg, load_command = pop_loadstudy(
        "phase_6c.study", filepath=tmp_path, load_datasets=True, return_com=True
    )

    assert loaded["name"] == "Phase 6c study"
    assert loaded["currentdesign"] == 1
    assert loaded["design"][0]["variable"][0]["label"] == "condition"
    assert loaded["changrp"][0]["measureinfo"]["computed"] == ["erp", "spec"]
    assert loaded["cluster"][0]["child"] == [cluster["name"] for cluster in loaded["cluster"][1:]]
    assert sum(len(cluster.get("comps") or []) for cluster in loaded["cluster"][1:]) == 6
    assert len(loaded_alleeg) == 2
    assert all(isinstance(eeg.get("etc"), dict) for eeg in loaded_alleeg)
    assert all(eeg.get("icaweights") is not None and np.asarray(eeg["icaweights"]).size for eeg in loaded_alleeg)
    assert [info["subject"] for info in loaded["datasetinfo"]] == ["S01", "S02"]

    session = EEGPrepSession(STUDY=loaded, ALLEEG=loaded_alleeg, CURRENTSTUDY=1)
    workspace = EEGPrepConsoleWorkspace(
        session,
        exports={"pop_chanplot": pop_chanplot, "pop_clustedit": pop_clustedit},
    )
    console_plot = workspace.namespace["pop_chanplot"](
        workspace.namespace["STUDY"],
        workspace.namespace["ALLEEG"],
        channels=[1],
        measure="spec",
    )
    console_rename = workspace.namespace["pop_clustedit"](
        workspace.namespace["STUDY"],
        workspace.namespace["ALLEEG"],
        action="rename",
        cluster=2,
        name="Console QA",
    )

    assert session.CURRENTSTUDY == 1
    assert workspace.namespace["STUDY"] is session.STUDY
    assert session.STUDY["etc"]["last_chanplot"] == {"measure": "spec", "mode": "channels", "channels": [1]}
    assert session.STUDY["cluster"][1]["name"].startswith("Console QA")
    assert console_plot.command.startswith("STUDY = pop_chanplot(")
    assert console_rename.command.startswith("STUDY = pop_clustedit(")
    assert session.ALLCOM[-2:] == [console_plot.command, console_rename.command]

    for command in (
        create_command,
        design_command,
        channel_command,
        plot_command,
        component_command,
        preclust_command,
        clust_command,
        cluster_plot_command,
        save_command,
        load_command,
    ):
        assert command

    plt.close(channel_figure)
    plt.close(cluster_figure)
    plt.close("all")


def test_study_menu_actions_owned_by_epic_are_implemented():
    study_actions = {
        "pop_study:edit",
        "pop_studydesign",
        "pop_precomp:channels",
        "pop_chanplot",
        "pop_precomp:components",
        "pop_preclust",
        "pop_clust",
        "pop_clustedit",
        "select_study_set",
        "clear_study",
    }

    assert {action: action_kind(action) for action in study_actions} == {
        action: "implemented" for action in study_actions
    }


def test_limo_entry_points_report_standalone_limitation():
    with pytest.raises(NotImplementedError, match="does not implement EEGLAB's external LIMO toolbox"):
        pop_limo({}, [])


def test_eeglabcompat_requires_external_reference_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv(eeglabcompat.EEGLAB_ROOT_ENV, raising=False)
    monkeypatch.setattr(eeglabcompat, "REPO_ROOT", tmp_path / "repo")

    with pytest.raises(ImportError, match=eeglabcompat.EEGLAB_ROOT_ENV):
        eeglabcompat._resolve_eeglab_root()


@pytest.mark.matlab
@pytest.mark.parity
def test_study_metadata_matlab_parity_when_reference_is_available():
    if not matlab_engine_available():
        pytest.skip("MATLAB engine not available or skipped")
    eeglab_root = _eeglab_reference_root()
    if eeglab_root is None:
        pytest.skip("EEGLAB reference checkout not available; set EEGPREP_EEGLAB_ROOT")

    try:
        import matlab.engine
    except ImportError as exc:
        pytest.skip(f"MATLAB engine not available: {exc}")

    python_study, _python_alleeg, _command = pop_study(
        None,
        [
            _study_eeg("s01_target", "S01", "target", 0.0),
            _study_eeg("s02_standard", "S02", "standard", 1.0),
        ],
        name="Parity study",
        return_com=True,
    )

    engine = matlab.engine.start_matlab()
    try:
        engine.addpath(engine.genpath(str(eeglab_root)), nargout=0)
        engine.eval(_matlab_study_metadata_script(), nargout=0)
        matlab_subjects = list(engine.workspace["parity_subjects"])
        matlab_conditions = list(engine.workspace["parity_conditions"])
        matlab_dataset_count = int(engine.workspace["parity_dataset_count"])
    finally:
        engine.quit()

    assert matlab_dataset_count == len(python_study["datasetinfo"])
    assert matlab_subjects == python_study["subject"]
    assert matlab_conditions == python_study["condition"]


def _eeglab_reference_root() -> Path | None:
    candidates = []
    if os.environ.get(eeglabcompat.EEGLAB_ROOT_ENV):
        candidates.append(Path(os.environ[eeglabcompat.EEGLAB_ROOT_ENV]).expanduser())
    candidates.append(Path(__file__).resolve().parents[1].parent / "eeglab")
    for candidate in candidates:
        if (candidate / "eeglab.m").is_file():
            return candidate
    return None


def _matlab_study_metadata_script() -> str:
    return """
clear STUDY ALLEEG parity_subjects parity_conditions parity_dataset_count;
ALLEEG(1).setname = 's01_target';
ALLEEG(1).filename = 's01_target.set';
ALLEEG(1).filepath = '';
ALLEEG(1).subject = 'S01';
ALLEEG(1).condition = 'target';
ALLEEG(1).group = 'control';
ALLEEG(1).session = 1;
ALLEEG(1).run = 1;
ALLEEG(1).data = zeros(4, 32, 3);
ALLEEG(1).nbchan = 4;
ALLEEG(1).pnts = 32;
ALLEEG(1).trials = 3;
ALLEEG(1).srate = 128;
ALLEEG(1).xmin = 0;
ALLEEG(1).xmax = (ALLEEG(1).pnts - 1) / ALLEEG(1).srate;
ALLEEG(1).chanlocs = struct('labels', {'Ch1' 'Ch2' 'Ch3' 'Ch4'});
ALLEEG(1).event = struct([]);
ALLEEG(1).urevent = struct([]);
ALLEEG(1).epoch = struct([]);
ALLEEG(1).icaweights = eye(3, 4);
ALLEEG(1).icasphere = eye(4);
ALLEEG(1).icawinv = pinv(ALLEEG(1).icaweights * ALLEEG(1).icasphere);
ALLEEG(2) = ALLEEG(1);
ALLEEG(2).setname = 's02_standard';
ALLEEG(2).filename = 's02_standard.set';
ALLEEG(2).subject = 'S02';
ALLEEG(2).condition = 'standard';
[STUDY, ALLEEG] = std_editset([], ALLEEG, 'name', 'Parity study');
[STUDY, ALLEEG] = std_checkset(STUDY, ALLEEG);
parity_subjects = STUDY.subject;
parity_conditions = STUDY.condition;
parity_dataset_count = numel(STUDY.datasetinfo);
"""
