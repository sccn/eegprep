from __future__ import annotations

import ast
from copy import deepcopy
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eegprep
from eegprep.functions.studyfunc.pop_chanplot import pop_chanplot
from eegprep.functions.studyfunc.pop_preclust import pop_preclust
from eegprep.functions.studyfunc.pop_precomp import pop_precomp
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.functions.studyfunc.std_interp import std_interp
from eegprep.functions.studyfunc.std_limodesign import std_limodesign
from eegprep.functions.studyfunc.std_precomp import std_precomp
from eegprep.functions.studyfunc.std_readdata import (
    std_readerp,
)
from eegprep.functions.studyfunc.std_uniformsetinds import std_uniformsetinds
from tests.fixtures import create_test_eeg, create_test_eeg_with_ica


class _Renderer:
    def __init__(self, result):
        self.result = result
        self.spec = None

    def run(self, spec, initial_values=None):
        self.spec = spec
        return self.result


def _study_pair():
    first = create_test_eeg(n_channels=3, n_samples=64, n_trials=4, srate=128)
    first.update({"setname": "one", "subject": "S01", "condition": "target"})
    second = deepcopy(first)
    second.update({"setname": "two", "subject": "S02", "condition": "standard"})
    second["data"] = second["data"] * 2.0
    return pop_study(None, [first, second], name="Measure study")


def test_std_precomp_recompute_off_preserves_cached_channel_measures():
    study, alleeg = _study_pair()

    study, alleeg = std_precomp(study, alleeg, [1], erp="on", spec="on")
    original_erp = deepcopy(study["changrp"][0]["erpdata"])
    # Overwrite the cached value with a sentinel the recompute path would never produce.
    sentinel = (np.asarray(original_erp) + 1000.0).tolist()
    study["changrp"][0]["erpdata"] = sentinel

    # recompute='off' must keep the cached measure rather than recomputing it.
    study, alleeg = std_precomp(study, alleeg, [1], erp="on", spec="on", recompute="off")
    np.testing.assert_allclose(np.asarray(study["changrp"][0]["erpdata"]), np.asarray(sentinel))

    # recompute='on' must overwrite the sentinel with a freshly computed measure.
    study, alleeg = std_precomp(study, alleeg, [1], erp="on", spec="on", recompute="on")
    np.testing.assert_allclose(np.asarray(study["changrp"][0]["erpdata"]), np.asarray(original_erp))


def test_std_precomp_baseline_and_design_contract(caplog):
    study, alleeg = _study_pair()
    alleeg[0]["times"] = np.asarray([-100.0, 0.0, 100.0, 200.0])
    alleeg[0]["xmin"] = -0.1
    alleeg[0]["xmax"] = 0.2
    alleeg[0]["pnts"] = 4
    alleeg[0]["data"] = np.asarray([[[1.0, 3.0], [5.0, 7.0], [20.0, 30.0], [40.0, 50.0]]])
    alleeg[0]["nbchan"] = 1
    alleeg[0]["chanlocs"] = [{"labels": "Cz"}]
    alleeg[1] = deepcopy(alleeg[0])
    alleeg[1]["data"] = alleeg[1]["data"] + 10.0

    with caplog.at_level(logging.WARNING, logger="eegprep.functions.studyfunc.std_precomp"):
        study, _alleeg, _command = std_precomp(
            study,
            alleeg,
            [1],
            erp="on",
            design=2,
            erpparams={"rmbase": [-100, 0]},
            interp="on",
            return_com=True,
        )

    erpdata = np.asarray(study["changrp"][0]["erpdata"])
    assert study["changrp"][0]["measureinfo"]["design"] == 2
    assert study["etc"]["eegprep"]["study_measures"]["design"] == 2
    np.testing.assert_allclose(np.mean(erpdata[:, :2], axis=1), 0.0)
    assert "ignoring EEGLAB-only option(s): interp" in caplog.text


def test_component_measure_reads_map_requested_component_ids_to_cached_axis():
    eeg = create_test_eeg_with_ica(n_channels=5, n_samples=72, n_trials=3, n_components=4)
    eeg.update({"setname": "ica1", "subject": "S01", "condition": "target"})
    study, alleeg = pop_study(None, [eeg], name="Component subset study")
    study["datasetinfo"][0]["comps"] = [2, 4]
    study, alleeg = pop_precomp(study, alleeg, "components", erp="on", allcomps="off")
    cluster = study["cluster"][0]
    raw = np.asarray(cluster["erpdata"], dtype=float)

    _study, selected, _times, _freqs = std_readerp(study, alleeg, clusters=1, components=[2])

    assert cluster["measureinfo"]["components"] == [2, 4]
    np.testing.assert_allclose(selected[0], raw[:, [0], :])
    with pytest.raises(ValueError, match="available component IDs: 2, 4"):
        std_readerp(study, alleeg, clusters=1, components=[1])

    study, _command, figure = pop_chanplot(
        study, alleeg, components=[4], measure="erp", mode="components", return_com=True
    )

    assert study["etc"]["last_chanplot"]["components"] == [4]
    assert figure.axes[0].get_legend().get_texts()[0].get_text() == "IC 4"
    plt.close(figure)


def test_component_precompute_preserves_per_dataset_component_pairs():
    first = create_test_eeg_with_ica(n_channels=5, n_samples=72, n_trials=3, n_components=4)
    first.update({"setname": "ica1", "subject": "S01", "condition": "target"})
    second = deepcopy(first)
    second.update({"setname": "ica2", "subject": "S02", "condition": "standard"})
    study, alleeg = pop_study(None, [first, second], name="Component pair study")
    study["datasetinfo"][0]["comps"] = [1, 2]
    study["datasetinfo"][1]["comps"] = [3, 4]

    study, alleeg = pop_precomp(study, alleeg, "components", erp="on", allcomps="off")
    cluster = study["cluster"][0]
    study, alleeg = pop_preclust(study, alleeg, preproc=[{"measure": "erp", "npca": 2, "norm": 0}])

    assert cluster["sets"] == [[1, 1, 2, 2]]
    assert cluster["comps"] == [1, 2, 3, 4]
    assert np.asarray(cluster["erpdata"], dtype=float).shape == (2, 4, first["pnts"])
    assert study["etc"]["preclust"]["preclustcomps"] == [
        {"set": 1, "comp": 1},
        {"set": 1, "comp": 2},
        {"set": 2, "comp": 3},
        {"set": 2, "comp": 4},
    ]


def test_std_uniformsetinds_treats_nan_as_matching_missing_dataset() -> None:
    study = {"changrp": [{"sets": [1, np.nan]}, {"sets": [1, np.nan]}]}
    assert std_uniformsetinds(study) == 1

    study["changrp"][1]["sets"] = [1, 2]
    assert std_uniformsetinds(study) == 0


def test_pop_chanplot_gui_component_mode_uses_cached_measures():
    first = create_test_eeg_with_ica(n_channels=4, n_samples=64, n_trials=3, n_components=2)
    first.update({"subject": "S01", "condition": "target"})
    study, alleeg = pop_study(None, [first], name="Component study")
    study, alleeg = pop_precomp(study, alleeg, "components", erp="on")
    renderer = _Renderer({"mode": 4, "channels": "", "components": "1", "measure": 1})

    study, command, figure = pop_chanplot(study, alleeg, gui=True, renderer=renderer, return_com=True)

    assert renderer.spec is not None
    assert study["etc"]["last_chanplot"] == {"measure": "erp", "mode": "components", "components": [1]}
    assert "mode='components'" in command
    plt.close(figure)


def test_std_limodesign_builds_categorical_continuous_and_split_exports(tmp_path):
    factors = [
        {"label": "condition", "value": "target", "vartype": "categorical"},
        {"label": "condition", "value": "standard", "vartype": "categorical"},
        {"label": "group", "value": "control", "vartype": "categorical"},
        {"label": "group", "value": "patient", "vartype": "categorical"},
        {"label": "rt", "vartype": "continuous"},
    ]
    trialinfo = [
        {"condition": "target", "group": "control", "rt": 1.0},
        {"condition": "standard", "group": "control", "rt": 2.0},
        {"condition": "target", "group": "patient", "rt": 3.0},
        {"condition": "standard", "group": "patient", "rt": 4.0},
    ]

    catmat, contmat, limodesign, command = std_limodesign(
        factors,
        trialinfo,
        interaction="on",
        splitreg="on",
        filepath=tmp_path,
        return_com=True,
    )

    np.testing.assert_allclose(catmat.ravel(), [1, 3, 2, 4])
    assert contmat.shape == (4, 4)
    assert np.count_nonzero(contmat[0]) == 0
    assert len(limodesign["categorical"][0]) == 4
    assert len(limodesign["continuous"]) == 4
    assert (tmp_path / "categorical_variables.txt").is_file()
    assert (tmp_path / "continuous_variables.txt").is_file()
    assert command.startswith("catMat, contMat, limodesign = std_limodesign(")
    ast.parse(command)


def test_std_interp_adds_requested_missing_channels_without_dropping_existing():
    study, alleeg = _study_pair()
    reduced = deepcopy(alleeg[1])
    reduced["data"] = reduced["data"][:2]
    reduced["nbchan"] = 2
    reduced["chanlocs"] = reduced["chanlocs"][:2]
    alleeg[1] = reduced

    study, interpolated, command = std_interp(study, alleeg, ["Ch3"], return_com=True)

    assert interpolated[1]["data"].shape[0] == 3
    assert [loc["labels"] for loc in interpolated[1]["chanlocs"]] == ["Ch1", "Ch2", "Ch3"]
    assert study["etc"]["eegprep"]["std_interp"]["channels"] == ["Ch3"]
    assert study["etc"]["eegprep"]["std_interp"]["changed_datasets"] == [2]
    assert command.startswith("STUDY, ALLEEG = std_interp(")
    assert eegprep.std_interp is std_interp
