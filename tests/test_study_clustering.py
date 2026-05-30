from __future__ import annotations

import numpy as np
import pytest

from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.studyfunc.pop_clust import pop_clust, pop_clust_dialog_spec
from eegprep.functions.studyfunc.pop_clustedit import pop_clustedit, pop_clustedit_dialog_spec
from eegprep.functions.studyfunc.pop_preclust import pop_preclust, pop_preclust_dialog_spec
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.functions.studyfunc.std_mergeclust import std_mergeclust
from eegprep.functions.studyfunc.std_moveoutlier import std_moveoutlier
from eegprep.functions.studyfunc.std_preclust import std_preclust


class _Renderer:
    def __init__(self, result):
        self.result = result

    def run(self, spec, initial_values=None):
        self.spec = spec
        return self.result


def _ica_eeg(setname: str, offset: float = 0.0) -> dict:
    data = np.arange(40, dtype=float).reshape(4, 10) + offset
    return {
        "setname": setname,
        "subject": setname.upper(),
        "data": data,
        "nbchan": 4,
        "pnts": 10,
        "trials": 1,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": 0.09,
        "chanlocs": [{"labels": label} for label in ("Fz", "Cz", "Pz", "Oz")],
        "icaweights": np.eye(3, 4),
        "icasphere": np.eye(4),
        "icawinv": np.array(
            [
                [1.0 + offset, 0.1, 0.2],
                [0.2, 1.2 + offset, 0.3],
                [0.1, 0.4, 1.4 + offset],
                [0.0, 0.2, 0.5],
            ]
        ),
        "icachansind": [0, 1, 2, 3],
        "event": [],
        "urevent": [],
        "epoch": [],
    }


def _study_with_ica() -> tuple[dict, list[dict]]:
    return pop_study(None, [_ica_eeg("s1"), _ica_eeg("s2", 4.0)], name="Cluster study")


def _preclustered_study() -> tuple[dict, list[dict]]:
    study, alleeg = _study_with_ica()
    return pop_preclust(
        study,
        alleeg,
        preproc=[{"measure": "scalp", "npca": 2, "norm": 1, "weight": 1}],
    )


def test_std_preclust_builds_parent_component_matrix_from_scalp_maps():
    study, alleeg = _study_with_ica()

    study, alleeg, command = std_preclust(
        study,
        alleeg,
        1,
        [{"measure": "scalp", "npca": 2, "norm": 1, "weight": 2}],
        return_com=True,
    )

    preclust = study["etc"]["preclust"]
    assert np.asarray(preclust["preclustdata"]).shape == (6, 2)
    assert preclust["clustlevel"] == 1
    assert len(study["cluster"][0]["comps"]) == 6
    assert command.startswith("STUDY, ALLEEG = std_preclust(")


def test_std_preclust_reads_phase_5b_component_measure_contract():
    study, alleeg = _study_with_ica()
    study.setdefault("etc", {}).setdefault("eegprep", {})["component_measures"] = {
        "erp": {
            "times": [-100.0, 0.0, 100.0, 200.0],
            "data": {
                "1": np.arange(12, dtype=float).reshape(3, 4),
                "2": np.arange(12, 24, dtype=float).reshape(3, 4),
            },
        }
    }

    study, _alleeg = std_preclust(
        study,
        alleeg,
        1,
        [{"measure": "erp", "npca": 2, "timewindow": [0, 200]}],
    )

    assert np.asarray(study["etc"]["preclust"]["preclustdata"]).shape == (6, 2)


def test_preclust_missing_component_measure_and_missing_ica_errors():
    study, alleeg = _study_with_ica()
    with pytest.raises(ValueError, match="component measure 'erp' is missing"):
        std_preclust(study, alleeg, 1, [{"measure": "erp"}])

    no_ica = dict(_ica_eeg("bad"))
    no_ica.pop("icaweights")
    no_ica.pop("icawinv")
    study, alleeg = pop_study(None, [no_ica], name="No ICA")
    with pytest.raises(ValueError, match="no ICA components"):
        std_preclust(study, alleeg)


def test_pop_clust_creates_deterministic_child_clusters():
    study, alleeg = _preclustered_study()

    clustered, command = pop_clust(study, alleeg, clus_num=2, random_state=11, return_com=True)

    assert command.startswith("STUDY = pop_clust(")
    assert len(clustered["cluster"]) == 3
    assert sum(len(cluster["comps"]) for cluster in clustered["cluster"][1:]) == 6
    assert clustered["cluster"][0]["child"] == [cluster["name"] for cluster in clustered["cluster"][1:]]


def test_cluster_edit_rename_merge_moveoutlier_and_plot():
    study, alleeg = _preclustered_study()
    study = pop_clust(study, alleeg, clus_num=2, random_state=11)

    renamed, command, _figure = pop_clustedit(
        study,
        alleeg,
        action="rename",
        cluster=2,
        name="Alpha",
        return_com=True,
    )
    assert renamed["cluster"][1]["name"].startswith("Alpha")
    assert "action='rename'" in command

    merged = std_mergeclust(renamed, alleeg, [2, 3], "Merged")
    assert merged["cluster"][-1]["name"].startswith("Merged")
    assert set(merged["cluster"][-1]["parent"]) == {renamed["cluster"][1]["name"], renamed["cluster"][2]["name"]}

    moved = std_moveoutlier(renamed, alleeg, 2, [1])
    assert moved["cluster"][1]["comps"] != renamed["cluster"][1]["comps"]
    assert moved["cluster"][-1]["name"].startswith("Outliers")

    plotted, _command, figure = pop_clustedit(renamed, alleeg, action="plot", clusters=[2, 3], return_com=True)
    assert plotted["cluster"][1]["name"].startswith("Alpha")
    assert figure is not None


def test_pop_preclust_clust_and_clustedit_history_replays():
    study, alleeg = _study_with_ica()
    study, alleeg, preclust_command = pop_preclust(
        study,
        alleeg,
        preproc=[{"measure": "scalp", "npca": 2}],
        return_com=True,
    )
    namespace = {
        "STUDY": study,
        "ALLEEG": alleeg,
        "pop_preclust": pop_preclust,
        "pop_clust": pop_clust,
        "pop_clustedit": pop_clustedit,
    }

    exec(preclust_command, namespace)
    assert namespace["STUDY"]["etc"]["preclust"]["clustlevel"] == 1

    clustered, _command = pop_clust(namespace["STUDY"], namespace["ALLEEG"], clus_num=2, return_com=True)
    namespace["STUDY"] = clustered
    _renamed, rename_command, _figure = pop_clustedit(
        clustered,
        namespace["ALLEEG"],
        action="rename",
        cluster=2,
        name="Replay",
        return_com=True,
    )
    exec(rename_command, namespace)
    assert namespace["STUDY"]["cluster"][1]["name"].startswith("Replay")


def test_console_bare_pop_clust_updates_study_workspace():
    study, alleeg = _preclustered_study()
    session = EEGPrepSession()
    session.ALLEEG = alleeg
    session.STUDY = study
    session.CURRENTSTUDY = 1
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_clust": pop_clust})

    result = workspace.namespace["pop_clust"](workspace.namespace["STUDY"], workspace.namespace["ALLEEG"], clus_num=2)

    assert session.STUDY is workspace.namespace["STUDY"]
    assert len(session.STUDY["cluster"]) == 3
    assert result.command.startswith("STUDY = pop_clust(")
    assert session.ALLCOM[-1].startswith("STUDY = pop_clust(")


def test_cluster_gui_specs_and_cancel_paths_are_stable():
    study, alleeg = _preclustered_study()
    preclust_spec = pop_preclust_dialog_spec(study)
    clust_spec = pop_clust_dialog_spec(study)
    edit_spec = pop_clustedit_dialog_spec(study)

    assert controls_by_tag(preclust_spec)["scalp_on"].value == 0
    assert controls_by_tag(clust_spec)["clus_num"].value
    assert controls_by_tag(edit_spec)["action"].value == 1

    assert pop_preclust(study, alleeg, gui=True, renderer=_Renderer(None), return_com=True)[2] == ""
    assert pop_clust(study, alleeg, gui=True, renderer=_Renderer(None), return_com=True)[1] == ""
    assert pop_clustedit(study, alleeg, gui=True, renderer=_Renderer(None), return_com=True)[1] == ""
