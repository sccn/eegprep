from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from eegprep import (
    std_checkconsist,
    std_checkdesign,
    std_combtrialinfo,
    std_findsameica,
    std_getindvar,
    std_gettrialsind,
    std_indvarmatch,
    std_maketrialinfo,
    std_rmalldatafields,
    std_rmdat,
    std_selectdataset,
    std_selsubject,
    std_substudy,
)
from eegprep.functions.studyfunc.pop_precomp import pop_precomp
from eegprep.functions.studyfunc.pop_study import pop_study
from tests.fixtures import create_test_eeg, create_test_eeg_with_ica


def _long_tail_study():
    first = create_test_eeg(n_channels=3, n_samples=24, n_trials=3, srate=128)
    first.update({"setname": "s01_target", "subject": "S01", "condition": "target", "group": "control"})
    first["event"] = [
        {"type": "rare", "latency": 1, "epoch": 1, "rt": 320.0},
        {"type": "standard", "latency": 25, "epoch": 2, "rt": 410.0},
        {"type": "rare", "latency": 49, "epoch": 3, "rt": 370.0},
    ]
    second = create_test_eeg(n_channels=3, n_samples=24, n_trials=3, srate=128)
    second.update({"setname": "s02_standard", "subject": "S02", "condition": "standard", "group": "control"})
    second["event"] = [
        {"type": "standard", "latency": 1, "epoch": 1, "rt": 390.0},
        {"type": "standard", "latency": 25, "epoch": 2, "rt": 430.0},
        {"type": "rare", "latency": 49, "epoch": 3, "rt": 350.0},
    ]
    third = create_test_eeg(n_channels=3, n_samples=24, n_trials=2, srate=128)
    third.update({"setname": "s03_target", "subject": "S03", "condition": "target", "group": "patient"})
    third["event"] = [
        {"type": "rare", "latency": 1, "epoch": 1, "rt": 300.0},
        {"type": "rare", "latency": 25, "epoch": 2, "rt": 360.0},
    ]
    return pop_study(None, [first, second, third], name="Long tail")


def test_independent_variable_selection_and_trialinfo_queries_are_1_based():
    study, alleeg = _long_tail_study()
    study, alltrialinfo = std_maketrialinfo(study, alleeg)

    factors, factorvals, subjects, paired = std_getindvar(study)
    condition_index = factors.index("condition")
    trial_type_index = factors.index("type")
    selected_datasets, selected_trials = std_selectdataset(study, alleeg, "condition", ["target"])
    direct_datasets, direct_trials = std_selectdataset({}, alleeg, "condition", "target")
    rare_datasets, rare_trials = std_selectdataset(study, alleeg, "type", "rare")
    rt_trials, rt_values = std_gettrialsind(alltrialinfo[0], "rt", "300<380", return_values=True)
    combined = std_combtrialinfo(study["datasetinfo"], "S01")

    assert factors[:2] == ["condition", "group"]
    assert factorvals[condition_index] == ["standard", "target"]
    assert subjects[condition_index] == [["S02"], ["S01", "S03"]]
    assert paired[condition_index] == "off"
    assert factorvals[trial_type_index] == ["rare", "standard"]
    assert selected_datasets == [1, 3]
    assert direct_datasets == [1, 3]
    assert selected_trials == [[1, 2, 3], [1, 2, 3], [1, 2]]
    assert direct_trials == [[1, 2, 3], [1, 2, 3], [1, 2]]
    assert rare_datasets == [1, 2, 3]
    assert rare_trials == [[1, 3], [3], [1, 2]]
    assert rt_trials == [1, 3]
    assert rt_values == [[320.0, 370.0]]
    assert [row["condition"] for row in combined] == ["target", "target", "target"]
    assert [row["type"] for row in combined] == ["rare", "standard", "rare"]


def test_indvarmatch_and_gettrialsind_validate_standalone_inputs():
    assert std_indvarmatch("target", ["standard", "target", "target"]) == [2, 3]
    assert std_indvarmatch([2, 3], [1, 2, 3]) == [2, 3]
    assert std_indvarmatch([2, 3], [1, 2, 3, [2, 3]]) == [4]
    assert std_indvarmatch(["S01", "S03"], ["S01", "S02", "S03"]) == [1, 3]
    selected, trials = std_selectdataset(
        {
            "datasetinfo": [
                {"trialinfo": [{"type": "rare"}, {"rt": 320.0}, {"type": "rare"}]},
            ]
        },
        None,
        "type",
        "rare",
    )

    assert selected == [1]
    assert trials == [[1, 3]]

    with pytest.raises(ValueError, match="not a MATLAB filename"):
        std_gettrialsind("external_trialinfo.mat", "type", "rare")


def test_study_design_consistency_and_subject_selection_helpers():
    study, alleeg = _long_tail_study()
    study, _alltrialinfo = std_maketrialinfo(study, alleeg)
    study["session"] = [1, 2]
    study["datasetinfo"][0]["session"] = 1
    study["datasetinfo"][1]["session"] = 1
    study["datasetinfo"][2]["session"] = 2
    study["design"][0]["variable"].append({"label": "rt", "value": [], "vartype": "continuous"})

    cells = [np.arange(12, dtype=float).reshape(3, 4), np.arange(12, 24, dtype=float).reshape(3, 4)]
    selected = std_selsubject(cells, "S02", [np.array([1, 2, 3]), np.array([3, 2, 1])], ["S01", "S02", "S03"])

    assert std_checkconsist(study, "uniform", "condition") == 0
    assert std_checkconsist(study, uniform="group") == 0
    assert std_checkconsist(study, uniform="session") == 0
    assert std_checkdesign(study, 1) == 0
    np.testing.assert_array_equal(selected[0], cells[0][:, [1]])
    np.testing.assert_array_equal(selected[1], cells[1][:, [1]])
    with pytest.raises(ValueError, match="Cannot select subject"):
        std_selsubject(cells, "S99", [np.array([1, 2, 3]), np.array([3, 2, 1])], ["S01", "S02", "S03"])


def test_same_ica_groups_are_subject_scoped():
    first = create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=2)
    first.update({"subject": "S01"})
    second = deepcopy(first)
    second.update({"subject": "S01"})
    third = deepcopy(first)
    third.update({"subject": "S02"})

    clusters, indices = std_findsameica([first, second, third])

    assert clusters == [[1, 2], [3]]
    assert indices == [1, 1, 2]


def test_substudy_and_rmdat_remap_dataset_membership_and_clear_caches():
    study, alleeg = _long_tail_study()
    study, alleeg = pop_precomp(study, alleeg, "channels", erp="on")
    assert "erpdata" in study["changrp"][0]

    subset, subset_alleeg, command = std_substudy(study, alleeg, subject=["S01", "S03"], return_com=True)
    direct_subset, direct_subset_alleeg = std_substudy({}, alleeg, subject="S02")
    kept, kept_alleeg, removed = std_rmdat(study, alleeg, keepvarvalues=("condition", "target"))
    direct_kept, _direct_alleeg, direct_removed = std_rmdat({}, alleeg, keepvarvalues=("group", "control"))
    unchanged, unchanged_alleeg, unchanged_removed = std_rmdat(study, alleeg, trialrange=[1, 3])

    assert [info["subject"] for info in subset["datasetinfo"]] == ["S01", "S03"]
    assert [eeg["subject"] for eeg in subset_alleeg] == ["S01", "S03"]
    assert [info["subject"] for info in direct_subset["datasetinfo"]] == ["S02"]
    assert [eeg["subject"] for eeg in direct_subset_alleeg] == ["S02"]
    assert subset["changrp"] and "erpdata" not in subset["changrp"][0]
    assert "std_substudy" in command
    assert removed == [2]
    assert direct_removed == [3]
    assert [info["condition"] for info in kept["datasetinfo"]] == ["target", "target"]
    assert [info["group"] for info in direct_kept["datasetinfo"]] == ["control", "control"]
    assert [eeg["condition"] for eeg in kept_alleeg] == ["target", "target"]
    assert [info["subject"] for info in unchanged["datasetinfo"]] == ["S01", "S02", "S03"]
    assert [eeg["subject"] for eeg in unchanged_alleeg] == ["S01", "S02", "S03"]
    assert "erpdata" in unchanged["changrp"][0]
    assert unchanged_removed == []


def test_rmdat_can_remove_by_event_counts_and_rmalldatafields_targets():
    study, alleeg = _long_tail_study()
    study, alleeg = pop_precomp(study, alleeg, "channels", erp="on")
    study["cluster"][0]["erpdata"] = [[[1.0]]]

    event_filtered, _alleeg, removed, command = std_rmdat(
        study,
        alleeg,
        checkeventtype="standard",
        numeventrange=[1, 1],
        return_com=True,
    )
    channels_only = std_rmalldatafields(study, "chan")
    clusters_only = std_rmalldatafields(study, "clust")

    assert removed == [2, 3]
    assert [info["subject"] for info in event_filtered["datasetinfo"]] == ["S01"]
    assert command.startswith("STUDY, ALLEEG, RMDATS = std_rmdat(")
    assert "erpdata" not in channels_only["changrp"][0]
    assert "erpdata" in channels_only["cluster"][0]
    assert "erpdata" in clusters_only["changrp"][0]
    assert "erpdata" not in clusters_only["cluster"][0]
