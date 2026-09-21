"""Regression coverage for STUDY designs whose factors vary within datasets."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from eegprep import pop_study, std_erpplot, std_erspplot, std_itcplot, std_precomp, std_specplot
from eegprep.functions.studyfunc.std_makedesign import std_makedesign


def _trial_factor_study() -> tuple[dict, list[dict]]:
    datasets = []
    srate = 64.0
    pnts = 64
    trials = 6
    seconds = np.arange(pnts, dtype=float) / srate
    conditions = ["standard", "target"] * 3
    for group_index, group in enumerate(("control", "patient")):
        for subject_index in range(2):
            subject = f"{group[0].upper()}{subject_index + 1:02d}"
            data = np.empty((2, pnts, trials), dtype=float)
            activations = np.empty((1, pnts, trials), dtype=float)
            for trial, condition in enumerate(conditions):
                condition_index = int(condition == "target")
                amplitude = 1.0 + 0.25 * group_index + 0.1 * subject_index + 0.4 * condition_index
                phase = 0.07 * trial
                data[0, :, trial] = amplitude * np.sin(2 * np.pi * 8 * seconds + phase) + condition_index
                data[1, :, trial] = (amplitude + 0.2) * np.cos(2 * np.pi * 12 * seconds + phase)
                activations[0, :, trial] = amplitude * np.sin(2 * np.pi * 10 * seconds + phase)
            datasets.append(
                {
                    "setname": subject,
                    "subject": subject,
                    "condition": "",
                    "group": group,
                    "session": 1,
                    "run": 1,
                    "data": data,
                    "nbchan": 2,
                    "pnts": pnts,
                    "trials": trials,
                    "srate": srate,
                    "xmin": 0.0,
                    "xmax": float(seconds[-1]),
                    "times": seconds * 1000.0,
                    "chanlocs": [{"labels": "Cz"}, {"labels": "Pz"}],
                    "icaact": activations,
                    "icawinv": np.asarray([[1.0], [0.0]]),
                    "icaweights": np.asarray([[1.0, 0.0]]),
                    "icasphere": np.eye(2),
                    "icachansind": [0, 1],
                    "trialinfo": [{"condition": condition} for condition in conditions],
                    "event": [],
                    "urevent": [],
                    "epoch": [{"condition": condition} for condition in conditions],
                    "etc": {},
                }
            )
    study, alleeg = pop_study(None, datasets, name="Trial-factor study")
    study = std_makedesign(
        study,
        alleeg,
        1,
        variable1="condition",
        values1=["standard", "target"],
        variable2="group",
        values2=["control", "patient"],
    )
    return study, alleeg


def _precompute_all(study: dict, alleeg: list[dict]) -> tuple[dict, list[dict]]:
    tf_params = {"cycles": 0, "nfreqs": 4, "timesout": 5, "baseline": np.nan}
    study, alleeg = std_precomp(
        study,
        alleeg,
        [1],
        erp="on",
        spec="on",
        ersp="on",
        itc="on",
        savetrials="on",
        recompute="on",
        erspparams=tf_params,
    )
    return std_precomp(
        study,
        alleeg,
        "components",
        erp="on",
        spec="on",
        ersp="on",
        itc="on",
        savetrials="on",
        recompute="on",
        erspparams=tf_params,
    )


def _expected_cells(cache: dict, datatype: str, *, component: bool = False) -> list[list[np.ndarray]]:
    field = f"{datatype}datatrials"
    info_field = f"{datatype}trialinfo"
    output = []
    for condition in ("standard", "target"):
        row = []
        for dataset_indices in ((0, 1), (2, 3)):
            cases = []
            for dataset_index in dataset_indices:
                values = cache[field][dataset_index]
                if component:
                    values = values[0]
                values = np.asarray(values, dtype=float)
                mask = np.asarray(
                    [item["condition"] == condition for item in cache[info_field][dataset_index]], dtype=bool
                )
                selected = values[..., mask]
                if datatype in {"spec", "ersp"}:
                    case = 10 * np.log10(np.mean(selected, axis=-1))
                elif datatype == "itc":
                    case = np.abs(np.mean(np.exp(1j * selected), axis=-1))
                else:
                    case = np.mean(selected, axis=-1)
                cases.append(case)
            row.append(np.stack(cases, axis=-1))
        output.append(row)
    return output


@pytest.mark.parametrize(
    ("datatype", "plotter"),
    [
        ("erp", std_erpplot),
        ("spec", std_specplot),
        ("ersp", std_erspplot),
        ("itc", std_itcplot),
    ],
)
def test_trial_factor_channel_cells_reconstruct_from_single_trial_caches(datatype, plotter):
    study, alleeg = _precompute_all(*_trial_factor_study())
    cache = study["changrp"][0]

    result = plotter(study, alleeg, channels=[1], noplot="on")
    cells = result[1]

    expected = _expected_cells(cache, datatype)
    assert [[values.shape[-1] for values in row] for row in cells] == [[2, 2], [2, 2]]
    for actual_row, expected_row in zip(cells, expected):
        for actual, expected_values in zip(actual_row, expected_row):
            np.testing.assert_allclose(actual, expected_values, atol=1e-12)


@pytest.mark.parametrize(
    ("datatype", "plotter"),
    [
        ("erp", std_erpplot),
        ("spec", std_specplot),
        ("ersp", std_erspplot),
        ("itc", std_itcplot),
    ],
)
def test_trial_factor_component_cluster_preserves_membership_in_every_cell(datatype, plotter):
    study, alleeg = _precompute_all(*_trial_factor_study())
    study = deepcopy(study)
    study["cluster"].append({"name": "IC1", "sets": [[1, 2, 3, 4]], "comps": [1, 1, 1, 1], "child": []})

    result = plotter(study, alleeg, clusters=2, noplot="on")
    cells = result[1]

    expected = _expected_cells(study["cluster"][0], datatype, component=True)
    for actual_row, expected_row in zip(cells, expected):
        for actual, expected_values in zip(actual_row, expected_row):
            np.testing.assert_allclose(actual, expected_values, atol=1e-12)


def test_trial_factor_plot_requires_single_trial_precompute():
    study, alleeg = _trial_factor_study()
    study, alleeg = std_precomp(study, alleeg, [1], erp="on")

    with pytest.raises(ValueError, match="savetrials='on'"):
        std_erpplot(study, alleeg, channels=[1], noplot="on")
