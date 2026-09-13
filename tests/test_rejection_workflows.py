import copy
import os
from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.io

import eegprep.functions.popfunc._eegplot_rejection as eegplot_rejection_module
import eegprep.functions.popfunc.pop_rejcont as pop_rejcont_module
from eegprep.functions.adminfunc.console import _console_python_command
from eegprep.functions.popfunc.pop_eegplot import DEFAULT_REJECTION_COLORS
from eegprep.functions.popfunc.eeg_rejsuperpose import eeg_rejsuperpose
from eegprep.functions.popfunc._rejection import (
    jointprob,
    jointprob_marks,
    kurtosis_marks,
    rejkurt,
    trend_marks,
)
from eegprep.functions.popfunc.pop_autorej import pop_autorej
from eegprep.functions.popfunc.pop_eegthresh import pop_eegthresh
from eegprep.functions.popfunc.pop_jointprob import pop_jointprob
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_rejchan import pop_rejchan
from eegprep.functions.popfunc.pop_rejcont import pop_rejcont
from eegprep.functions.popfunc.pop_rejepoch import pop_rejepoch
from eegprep.functions.popfunc.pop_rejkurt import pop_rejkurt
from eegprep.functions.popfunc.pop_rejmenu import pop_rejmenu
from eegprep.functions.popfunc.pop_rejspec import pop_rejspec
from eegprep.functions.popfunc.pop_rejtrend import pop_rejtrend
from eegprep.functions.popfunc.pop_selectcomps import pop_selectcomps
from eegprep.plugins.ICLabel.pop_viewprops import pop_viewprops
from tests.eeglab_tests import eeglab_test
from tests.fixtures import SAMPLE_DATASET_PATH, create_test_eeg


def _epoched_eeg() -> dict:
    rng = np.random.default_rng(4)
    eeg = create_test_eeg(n_channels=4, n_samples=80, n_trials=5, srate=100)
    data = rng.normal(0, 0.05, (4, 80, 5))
    data[0, 10:20, 1] = 25
    data[1, :, 2] += np.linspace(0, 8, 80)
    data[2, :, 3] += 4 * np.sin(2 * np.pi * 25 * np.arange(80) / 100)
    eeg["data"] = data
    eeg["icaweights"] = np.eye(4)
    eeg["icasphere"] = np.eye(4)
    eeg["icawinv"] = np.eye(4)
    eeg["icachansind"] = np.arange(4)
    eeg["icaact"] = None
    eeg["reject"] = {}
    return eeg


def _reference_trend_marks(
    data: np.ndarray, selected_rows: list[int], *, winsize: int, maxslope: float, min_r: float
) -> np.ndarray:
    row_marks = np.zeros((data.shape[0], data.shape[2]), dtype=bool)
    x = np.linspace(1 / winsize, 1, winsize)
    tolerance = 1000 * winsize * 1.1921e-7
    for row_index in selected_rows:
        for trial in range(data.shape[2]):
            for start in range(0, data.shape[1] - winsize + 1, winsize):
                y = data[row_index, start : start + winsize, trial]
                slope, intercept = np.polyfit(x, y, 1)
                fit = slope * x + intercept
                sst = max(float(np.sum((y - y.mean()) ** 2)), tolerance)
                r2 = 1 - float(np.sum((y - fit) ** 2)) / sst
                if abs(slope) >= maxslope and r2 > min_r:
                    row_marks[row_index, trial] = True
                    break
    return row_marks


@eeglab_test(
    "unittesting_popfunc/pop_eegthresh/popfunc_pop_eegthresh_wrapperTest.m",
    "test_test_pop_eegthresh",
)
def test_pop_eegthresh_marks_epochs_and_emits_replayable_python():
    eeg = _epoched_eeg()

    out, com = pop_eegthresh(eeg, 1, [1], -10, 10, 0, 0.79, 0, 0, return_com=True)
    component_out, component_rejected = pop_eegthresh(eeg, 0, [1], -10, 10, 0, 0.79, 1, 0)

    assert out["reject"]["rejthresh"].tolist() == [False, True, False, False, False]
    assert out["reject"]["rejthreshE"][0].tolist() == [False, True, False, False, False]
    assert component_rejected == [2]
    assert component_out["reject"]["icarejthresh"].tolist() == [False, True, False, False, False]
    assert _console_python_command(com) == (
        "EEG = pop_eegthresh(EEG, icacomp=1, elecrange=[1], negthresh=[-10], "
        "posthresh=[10], starttime=[0], endtime=[0.79], superpose=0, reject=0)"
    )


@eeglab_test(
    "unittesting_popfunc/pop_jointprob/popfunc_pop_jointprob_wrapperTest.m",
    "test_test_pop_jointprob",
)
@eeglab_test(
    "unittesting_popfunc/pop_rejkurt/popfunc_pop_rejkurt_wrapperTest.m",
    "test_test_pop_rejkurt",
)
@eeglab_test(
    "unittesting_popfunc/pop_rejspec/popfunc_pop_rejspec_wrapperTest.m",
    "test_test_pop_rejspec",
)
@eeglab_test(
    "unittesting_popfunc/pop_rejtrend/popfunc_pop_rejtrend_wrapperTest.m",
    "test_test_pop_rejtrend",
)
def test_rejection_statistics_store_data_and_component_marks():
    eeg = _epoched_eeg()

    prob_out, _local, _global, prob_count = pop_jointprob(eeg, 1, [1, 2, 3, 4], 1.2, 1.2, 0, 0)
    kurt_out, _local, _global, kurt_count = pop_rejkurt(eeg, 1, [1, 2, 3, 4], 1.2, 1.2, 0, 0)
    trend_out = pop_rejtrend(eeg, 1, [2], 80, 0.2, 0.3, 0, 0)
    spec_out, spec_indices = pop_rejspec(
        eeg,
        1,
        "method",
        "multitaper",
        "elecrange",
        [3],
        "threshold",
        [-10, 10],
        "freqlimits",
        [20, 30],
        "eegplotreject",
        0,
    )
    fft_spec_out, fft_spec_indices = pop_rejspec(
        eeg,
        1,
        "method",
        "fft",
        "elecrange",
        [3],
        "threshold",
        [-10, 10],
        "freqlimits",
        [20, 30],
        "eegplotreject",
        0,
    )
    comp_out, _local, _global, comp_count = pop_jointprob(eeg, 0, [1, 2, 3, 4], 1.2, 1.2, 0, 0)

    assert prob_count >= 1
    assert kurt_count >= 0
    assert prob_out["reject"]["rejjpE"].shape == (4, 5)
    assert kurt_out["reject"]["rejkurtE"].shape == (4, 5)
    assert trend_out["reject"]["rejconst"][2]
    assert spec_indices
    assert fft_spec_indices
    assert spec_out["specdata"].shape[:2] == (4, 40)
    assert fft_spec_out["specdata"].shape == spec_out["specdata"].shape
    assert not np.allclose(fft_spec_out["specdata"], spec_out["specdata"])
    assert comp_count >= 1
    assert "icarejjp" in comp_out["reject"]


@pytest.mark.parametrize(
    ("runner", "field"),
    [
        (
            lambda eeg, callback: pop_eegthresh(
                eeg,
                1,
                [1],
                -10,
                10,
                0,
                0.79,
                0,
                0,
                topcommand="update",
                command_callback=callback,
                return_com=True,
            ),
            "rejthresh",
        ),
        (
            lambda eeg, callback: pop_jointprob(
                eeg, 1, [1, 2, 3, 4], 1.2, 1.2, 0, 0, 1, command_callback=callback, return_com=True
            ),
            "rejjp",
        ),
        (
            lambda eeg, callback: pop_rejkurt(
                eeg, 1, [1, 2, 3, 4], 1.2, 1.2, 0, 0, 1, command_callback=callback, return_com=True
            ),
            "rejkurt",
        ),
        (
            lambda eeg, callback: pop_rejtrend(
                eeg, 1, [2], 80, 0.2, 0.3, 0, 0, 1, command_callback=callback, return_com=True
            ),
            "rejconst",
        ),
        (
            lambda eeg, callback: pop_rejspec(
                eeg,
                1,
                "method",
                "fft",
                "elecrange",
                [3],
                "threshold",
                [-10, 10],
                "freqlimits",
                [20, 30],
                "eegplotplotallrej",
                2,
                "eegplotreject",
                0,
                command_callback=callback,
                return_com=True,
            ),
            "rejfreq",
        ),
    ],
)
def test_epoched_rejection_display_paths_open_browser_and_accept_marks(monkeypatch, runner, field):
    calls = []
    accepted = []

    def fake_eegplot(data, *args, **kwargs):
        del args
        calls.append((np.asarray(data), kwargs))
        kwargs["command_callback"](kwargs["winrej"])
        return "window"

    monkeypatch.setattr(eegplot_rejection_module, "eegplot", fake_eegplot)
    eeg = _epoched_eeg()

    out, command = runner(eeg, lambda eeg_out, accept_command: accepted.append((eeg_out, accept_command)))

    assert command
    assert out["trials"] == eeg["trials"]
    assert calls
    assert calls[0][0].ndim == 3
    assert accepted
    assert accepted[0][1] == command
    assert accepted[0][0]["reject"][field].shape == (eeg["trials"],)


def test_component_rejection_browser_accept_updates_ica_marks(monkeypatch):
    calls = []
    accepted = []

    def fake_eegplot(data, *args, **kwargs):
        del args
        calls.append((np.asarray(data), kwargs))
        kwargs["command_callback"](kwargs["winrej"])
        return "window"

    monkeypatch.setattr(eegplot_rejection_module, "eegplot", fake_eegplot)
    eeg = _epoched_eeg()

    out, _command = pop_jointprob(
        eeg,
        0,
        [1],
        1.2,
        1.2,
        0,
        0,
        1,
        command_callback=lambda eeg_out, command: accepted.append((eeg_out, command)),
        return_com=True,
    )

    assert calls[0][0].shape[0] == 1
    assert out["trials"] == eeg["trials"]
    assert "icarejjp" in accepted[0][0]["reject"]
    assert accepted[0][0]["reject"]["icarejjpE"].shape == (4, 5)


def test_reject_on_browser_accept_removes_epochs_without_immediate_rejection(monkeypatch):
    accepted = []

    def fake_eegplot(_data, *args, **kwargs):
        del args
        kwargs["command_callback"](kwargs["winrej"])
        return "window"

    monkeypatch.setattr(eegplot_rejection_module, "eegplot", fake_eegplot)
    eeg = _epoched_eeg()

    out, _command = pop_eegthresh(
        eeg,
        1,
        [1],
        -10,
        10,
        0,
        0.79,
        0,
        1,
        topcommand="reject",
        command_callback=lambda eeg_out, command: accepted.append(eeg_out),
        return_com=True,
    )

    assert out["trials"] == eeg["trials"]
    assert accepted[0]["trials"] == eeg["trials"] - 1


def test_superposed_browser_winrej_includes_existing_family_marks(monkeypatch):
    calls = []

    def fake_eegplot(_data, *args, **kwargs):
        del args
        calls.append(kwargs)
        return "window"

    monkeypatch.setattr(eegplot_rejection_module, "eegplot", fake_eegplot)
    eeg = _epoched_eeg()
    eeg["reject"]["rejthresh"] = np.array([True, False, False, False, False])
    eeg["reject"]["rejthreshE"] = np.zeros((4, 5), dtype=bool)
    eeg["reject"]["rejthreshE"][0, 0] = True
    eeg["reject"]["disprej"] = ["thresh"]

    pop_jointprob(eeg, 1, [1, 2, 3, 4], 1.2, 1.2, 2, 0, 1)

    rows = calls[0]["winrej"]
    assert any(np.allclose(row[2:5], DEFAULT_REJECTION_COLORS["thresh"]) for row in rows)


def test_pop_rejcont_display_accept_removes_continuous_regions(monkeypatch):
    accepted = []
    eeg = create_test_eeg(n_channels=2, n_samples=120, n_trials=1, srate=100)
    eeg["chanlocs"] = np.asarray(eeg["chanlocs"], dtype=object)
    time = np.arange(120) / 100
    eeg["data"][0] = 100 * np.sin(2 * np.pi * 30 * time)

    def fake_eegplot(data, *args, **kwargs):
        del args
        assert np.asarray(data).shape[0] == 1
        assert kwargs["eloc_file"][0]["labels"] == "Ch1"
        kwargs["command_callback"](kwargs["winrej"])
        return "window"

    monkeypatch.setattr(pop_rejcont_module, "eegplot", fake_eegplot)

    out, selected = pop_rejcont(
        eeg,
        "elecrange",
        [1],
        "freqlimit",
        [20, 40],
        "threshold",
        0,
        "epochlength",
        0.2,
        "contiguous",
        1,
        "eegplot",
        "on",
        command_callback=lambda eeg_out, command: accepted.append((eeg_out, command)),
    )

    assert selected.size
    assert out["pnts"] == eeg["pnts"]
    assert accepted[0][0]["pnts"] < eeg["pnts"]


def test_pop_rejcont_display_defers_history_command_until_browser_accept(monkeypatch):
    accepted = []
    eeg = create_test_eeg(n_channels=2, n_samples=120, n_trials=1, srate=100)
    time = np.arange(120) / 100
    eeg["data"][0] = 100 * np.sin(2 * np.pi * 30 * time)

    def fake_eegplot(_data, *args, **kwargs):
        del args
        kwargs["command_callback"](kwargs["winrej"])
        return "window"

    monkeypatch.setattr(pop_rejcont_module, "eegplot", fake_eegplot)

    out, command = pop_rejcont(
        eeg,
        "elecrange",
        [1],
        "freqlimit",
        [20, 40],
        "threshold",
        0,
        "epochlength",
        0.2,
        "contiguous",
        1,
        "eegplot",
        "on",
        command_callback=lambda eeg_out, accept_command: accepted.append((eeg_out, accept_command)),
        return_com=True,
    )

    assert out is eeg
    assert command == ""
    assert accepted[0][1].startswith("EEG = pop_rejcont(EEG, ")


def test_pop_autorej_display_marks_original_epochs_before_browser_accept(monkeypatch):
    calls = []
    accepted = []

    def fake_eegplot(_data, *args, **kwargs):
        del args
        calls.append(kwargs)
        kwargs["command_callback"](kwargs["winrej"])
        return "window"

    monkeypatch.setattr(eegplot_rejection_module, "eegplot", fake_eegplot)
    eeg = _epoched_eeg()

    out, command = pop_autorej(
        eeg,
        "threshold",
        10,
        "startprob",
        20,
        "maxrej",
        40,
        "nogui",
        "on",
        "eegplot",
        "on",
        command_callback=lambda eeg_out, accept_command: accepted.append((eeg_out, accept_command)),
        return_com=True,
    )

    assert command
    assert calls
    assert out["trials"] == eeg["trials"]
    assert out["reject"]["rejauto"].shape == (eeg["trials"],)
    assert accepted[0][1] == command


def test_jointprob_global_marks_match_eeglab_trial_rows_for_duplicate_channels():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(3, 12, 4))
    elecrange = [3, 1, 3]

    reject, row_marks, local_scores, global_scores = jointprob_marks(data, elecrange, 0.8, 0.8)

    selected = [2, 0, 2]
    expected_local_scores, expected_local = jointprob(data[selected], 0.8, normalize=1)
    global_data = data[selected].transpose(2, 0, 1).reshape(data.shape[2], -1)
    expected_global_scores, expected_global = jointprob(global_data, 0.8, normalize=1)
    expected_reject = expected_local.any(axis=0) | expected_global.ravel()

    np.testing.assert_allclose(local_scores, expected_local_scores)
    np.testing.assert_allclose(global_scores, expected_global_scores.ravel())
    np.testing.assert_array_equal(reject, expected_reject)
    np.testing.assert_array_equal(row_marks[0], expected_local[1])
    np.testing.assert_array_equal(row_marks[2], expected_local[2])


@eeglab_test(
    "unittesting_sigprocfunc/jointprob/sigprocfunc_jointprob_wrapperTest.m",
    "test_pass_1d_row",
)
@eeglab_test(
    "unittesting_sigprocfunc/jointprob/sigprocfunc_jointprob_wrapperTest.m",
    "test_pass_1d_col",
)
@eeglab_test(
    "unittesting_sigprocfunc/jointprob/sigprocfunc_jointprob_wrapperTest.m",
    "test_pass_general",
)
def test_jointprob_vector_orientation_and_defaults_match_upstream():
    expected = -np.sum(np.log([2 / 3, 2 / 3, 1 / 3]))

    for signal in (np.asarray([1, 1, 3]), np.asarray([[1], [1], [3]])):
        scores, rejected = jointprob(signal)
        np.testing.assert_allclose(scores, [[expected]])
        np.testing.assert_array_equal(rejected, [[False]])


@eeglab_test(
    "unittesting_sigprocfunc/jointprob/sigprocfunc_jointprob_wrapperTest.m",
    "test_pass_3d",
)
def test_jointprob_three_dimensional_scores_match_upstream():
    signal = np.empty((3, 4, 2), dtype=float)
    signal[:, :, 0] = [[1, 1, 3, 4], [1, 2, 1, 4], [1, 2, 3, 4]]
    signal[:, :, 1] = [[1, 2, 3, 4], [1, 2, 1, 4], [2, 2, 3, 4]]
    expected = np.asarray(
        [
            [-np.sum(np.log([3 / 8, 3 / 8, 2 / 8, 2 / 8])), -np.sum(np.log([3 / 8, 1 / 8, 2 / 8, 2 / 8]))],
            [-np.sum(np.log([4 / 8, 2 / 8, 4 / 8, 2 / 8])), -np.sum(np.log([4 / 8, 2 / 8, 4 / 8, 2 / 8]))],
            [-np.sum(np.log([1 / 8, 3 / 8, 2 / 8, 2 / 8])), -np.sum(np.log([3 / 8, 3 / 8, 2 / 8, 2 / 8]))],
        ]
    )

    scores, rejected = jointprob(signal)

    np.testing.assert_allclose(scores, expected)
    np.testing.assert_array_equal(rejected, np.zeros((3, 2), dtype=bool))


@eeglab_test(
    "unittesting_sigprocfunc/jointprob/sigprocfunc_jointprob_wrapperTest.m",
    "test_pass_threshold",
)
@eeglab_test(
    "unittesting_sigprocfunc/jointprob/sigprocfunc_jointprob_wrapperTest.m",
    "test_pass_jp_threshold",
)
def test_jointprob_computed_and_precomputed_thresholds_match_upstream():
    signal = np.asarray([[1, 1, 5], [1, 2, 1], [1, 2, 5]])
    expected = np.asarray(
        [
            -np.sum(np.log([2 / 3, 2 / 3, 1 / 3])),
            -np.sum(np.log([2 / 3, 1 / 3, 2 / 3])),
            -np.sum(np.log([1 / 3, 1 / 3, 1 / 3])),
        ]
    )[:, np.newaxis]

    scores, rejected = jointprob(signal, 2)
    reused, reused_rejected = jointprob(signal, 2, expected)

    np.testing.assert_allclose(scores, expected)
    np.testing.assert_allclose(reused, expected)
    np.testing.assert_array_equal(rejected, [[False], [False], [True]])
    np.testing.assert_array_equal(reused_rejected, rejected)


@eeglab_test(
    "unittesting_sigprocfunc/jointprob/sigprocfunc_jointprob_wrapperTest.m",
    "test_pass_normalize_2d",
)
@eeglab_test(
    "unittesting_sigprocfunc/jointprob/sigprocfunc_jointprob_wrapperTest.m",
    "test_pass_normalize_3d",
)
def test_jointprob_uses_matlab_sample_standard_deviation_for_normalization():
    signal = np.asarray([[1, 1, 3], [1, 2, 1], [1, 2, 3]])
    raw, _ = jointprob(signal)

    normalized, rejected = jointprob(signal, 0, normalize=1)
    expected = (raw - raw.mean()) / raw.std(ddof=1)

    np.testing.assert_allclose(normalized, expected)
    np.testing.assert_array_equal(rejected, np.zeros((3, 1), dtype=bool))

    trials = np.stack([signal, signal[:, ::-1]], axis=2)
    raw_3d, _ = jointprob(trials)
    normalized_3d, _ = jointprob(trials, 0, normalize=1)
    expected_3d = raw_3d - raw_3d.mean(axis=1, keepdims=True)
    std_3d = raw_3d.std(axis=1, ddof=1, keepdims=True)
    std_3d[std_3d == 0] = 1
    expected_3d /= std_3d
    assert normalized_3d.shape == (3, 2)
    np.testing.assert_allclose(normalized_3d, expected_3d)


def test_jointprob_global_threshold_can_reject_when_local_threshold_does_not():
    data = np.array(
        [
            [
                [4.0, 3.0],
                [2.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 4.0],
                [3.0, 4.0],
                [2.0, 3.0],
                [4.0, 3.0],
                [3.0, 2.0],
                [2.0, 4.0],
                [1.0, 4.0],
                [3.0, 0.0],
            ],
            [
                [1.0, 4.0],
                [2.0, 0.0],
                [3.0, 3.0],
                [4.0, 0.0],
                [0.0, 4.0],
                [0.0, 2.0],
                [0.0, 1.0],
                [2.0, 2.0],
                [2.0, 0.0],
                [0.0, 0.0],
                [0.0, 3.0],
                [2.0, 3.0],
            ],
        ]
    )

    reject, row_marks, _local_scores, global_scores = jointprob_marks(data, [1, 2], 10, 0.5)

    np.testing.assert_array_equal(row_marks.any(axis=0), [False, False])
    np.testing.assert_allclose(global_scores, [1 / np.sqrt(2), -1 / np.sqrt(2)])
    np.testing.assert_array_equal(reject, [True, True])


def test_kurtosis_global_marks_match_eeglab_trial_rows_for_duplicate_channels():
    rng = np.random.default_rng(7)
    data = rng.normal(size=(3, 16, 4))
    elecrange = [2, 1, 2]

    reject, row_marks, local_scores, global_scores = kurtosis_marks(data, elecrange, 0.6, 0.6)

    selected = [1, 0, 1]
    expected_local_scores, expected_local = rejkurt(data[selected], 0.6, normalize=1)
    global_data = data[selected].transpose(2, 0, 1).reshape(data.shape[2], -1)
    expected_global_scores, expected_global = rejkurt(global_data, 0.6, normalize=1)
    expected_reject = expected_local.any(axis=0) | expected_global.ravel()

    np.testing.assert_allclose(local_scores, expected_local_scores)
    np.testing.assert_allclose(global_scores, expected_global_scores.ravel())
    np.testing.assert_array_equal(reject, expected_reject)
    np.testing.assert_array_equal(row_marks[0], expected_local[1])
    np.testing.assert_array_equal(row_marks[1], expected_local[2])


@eeglab_test(
    "unittesting_sigprocfunc/rejkurt/sigprocfunc_rejkurt_wrapperTest.m",
    "test_test_rejkurt",
)
def test_rejkurt_upstream_parameter_combinations_return_finite_trial_marks():
    rng = np.random.default_rng(8)
    signal = rng.normal(size=(8, 80, 12))

    calls = (
        (0, None, 0),
        (2, None, 0),
        (1.5, np.ones((8, 12)), 0),
        (0.5, None, 1),
        (0.5, None, 2),
    )
    for threshold, old_scores, normalize in calls:
        scores, rejected = rejkurt(signal, threshold, old_scores, normalize)
        assert scores.shape == rejected.shape == (8, 12)
        assert np.isfinite(scores).all()
        np.testing.assert_array_equal(rejected, np.abs(scores) > threshold if threshold else np.zeros_like(rejected))


def test_kurtosis_global_threshold_can_reject_when_local_threshold_does_not():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2, 12, 2))
    data[:, :, 1] *= 0.1

    reject, row_marks, _local_scores, global_scores = kurtosis_marks(data, [1, 2], 10, 0.5)

    np.testing.assert_array_equal(row_marks.any(axis=0), [False, False])
    np.testing.assert_allclose(global_scores, [1 / np.sqrt(2), -1 / np.sqrt(2)])
    np.testing.assert_array_equal(reject, [True, True])


def test_trend_marks_match_reference_window_loop():
    data = np.zeros((2, 12, 3), dtype=float)
    data[0, :, 0] = np.arange(12, dtype=float)
    data[0, :, 1] = 0.1
    data[1, :, 2] = np.r_[np.arange(6, dtype=float), np.zeros(6)]

    reject, row_marks = trend_marks(data, [1, 2], winsize=6, maxslope=0.3, min_r=0.8)
    expected = _reference_trend_marks(data, [0, 1], winsize=6, maxslope=0.3, min_r=0.8)

    np.testing.assert_array_equal(row_marks, expected)
    np.testing.assert_array_equal(reject, expected.any(axis=0))


@eeglab_test(
    "unittesting_popfunc/pop_rejepoch/popfunc_pop_rejepoch_wrapperTest.m",
    "test_test_pop_rejepoch",
)
def test_eeg_rejsuperpose_and_pop_rejepoch_remove_marked_epochs():
    eeg = _epoched_eeg()
    eeg["reject"]["rejmanual"] = np.array([False, True, False, False, True])
    eeg["reject"]["rejmanualE"] = np.zeros((4, 5), dtype=bool)
    eeg["reject"]["rejmanualE"][0, 1] = True
    eeg["reject"]["rejmanualE"][1, 4] = True

    marked, com = eeg_rejsuperpose(eeg, 1, 1, 0, 0, 0, 0, 0, 0, return_com=True)

    assert marked["reject"]["rejglobal"].tolist() == [False, True, False, False, True]
    assert marked["reject"]["rejglobalE"].shape == (4, 5)
    removed, reject_com = pop_rejepoch(copy.deepcopy(marked), marked["reject"]["rejglobal"], 0, return_com=True)
    assert removed["trials"] == 3
    assert _console_python_command(com) == "EEG = eeg_rejsuperpose(EEG, 1, 1, 0, 0, 0, 0, 0, 0)"
    assert _console_python_command(reject_com) == "EEG = pop_rejepoch(EEG, tmprej=[2, 5], confirm=0)"


def test_eeg_rejsuperpose_only_crosses_trial_marks_between_data_and_ica_families():
    eeg = _epoched_eeg()
    eeg["icaweights"] = np.ones((2, 4))
    eeg["icawinv"] = np.ones((4, 2))
    eeg["reject"] = {
        "rejmanual": np.array([False, True, False, False, False]),
        "rejmanualE": np.zeros((4, 5), dtype=bool),
        "icarejmanual": np.array([False, False, True, False, False]),
        "icarejmanualE": np.ones((2, 5), dtype=bool),
    }

    marked = eeg_rejsuperpose(eeg, 1, 1, 0, 0, 0, 0, 0, 1)

    assert marked["reject"]["rejglobal"].tolist() == [False, True, True, False, False]
    assert marked["reject"]["rejglobalE"].shape == (4, 5)
    assert not marked["reject"]["rejglobalE"].any()


def _suite_rejection_eeg(*, trials=5, components=False):
    eeg = create_test_eeg(n_channels=2, n_samples=100, n_trials=trials)
    eeg["reject"] = {}
    if components:
        eeg["icachansind"] = np.array([0, 1])
    return eeg


@eeglab_test("unittesting_popfunc/eeg_rejsuperpose/popfunc_eeg_rejsuperpose_wrapperTest.m", "test_pass_empty")
def test_eeg_rejsuperpose_current_suite_empty_marks():
    eeg = _suite_rejection_eeg()

    output = eeg_rejsuperpose(eeg, 1, 0, 0, 0, 0, 0, 0, 0)

    np.testing.assert_array_equal(output["reject"]["rejglobal"], np.zeros(5, dtype=bool))
    np.testing.assert_array_equal(output["reject"]["rejglobalE"], np.zeros((2, 5), dtype=bool))


@eeglab_test("unittesting_popfunc/eeg_rejsuperpose/popfunc_eeg_rejsuperpose_wrapperTest.m", "test_pass_zero")
def test_eeg_rejsuperpose_current_suite_zero_marks():
    eeg = _suite_rejection_eeg()
    eeg["reject"].update(
        {
            "rejmanual": np.zeros(5, dtype=bool),
            "rejfreq": np.zeros(5, dtype=bool),
            "rejmanualE": np.zeros((2, 5), dtype=bool),
            "rejfreqE": np.zeros((2, 5), dtype=bool),
        }
    )

    output = eeg_rejsuperpose(eeg, 1, 1, 0, 0, 0, 0, 1, 0)

    np.testing.assert_array_equal(output["reject"]["rejglobal"], np.zeros(5, dtype=bool))
    np.testing.assert_array_equal(output["reject"]["rejglobalE"], np.zeros((2, 5), dtype=bool))


@eeglab_test("unittesting_popfunc/eeg_rejsuperpose/popfunc_eeg_rejsuperpose_wrapperTest.m", "test_pass_general")
def test_eeg_rejsuperpose_current_suite_selected_mark_families():
    eeg = _suite_rejection_eeg()
    eeg["reject"].update(
        {
            "rejmanual": np.array([0, 0, 0, 1, 0], dtype=bool),
            "rejfreq": np.array([1, 0, 0, 1, 0], dtype=bool),
            "rejmanualE": np.array([[0, 0, 0, 1, 0], [0, 1, 0, 0, 0]], dtype=bool),
            "rejfreqE": np.array([[0, 0, 0, 0, 1], [0, 0, 0, 0, 1]], dtype=bool),
        }
    )

    output = eeg_rejsuperpose(eeg, 1, 1, 0, 0, 0, 0, 1, 0)

    np.testing.assert_array_equal(output["reject"]["rejglobal"], [1, 0, 0, 1, 0])
    np.testing.assert_array_equal(output["reject"]["rejglobalE"], [[0, 0, 0, 1, 1], [0, 1, 0, 0, 1]])


def _all_rejection_marks(prefix=""):
    marks = {}
    for index, name in enumerate(("rejmanual", "rejthresh", "rejconst", "rejjp", "rejkurt", "rejfreq")):
        trial_marks = np.zeros(6, dtype=bool)
        trial_marks[index] = True
        row_marks = np.zeros((2, 6), dtype=bool)
        row_marks[0, index] = True
        row_marks[1, 5 - index] = True
        marks[f"{prefix}{name}"] = trial_marks
        marks[f"{prefix}{name}E"] = row_marks
    return marks


@eeglab_test("unittesting_popfunc/eeg_rejsuperpose/popfunc_eeg_rejsuperpose_wrapperTest.m", "test_pass_all")
def test_eeg_rejsuperpose_current_suite_all_data_mark_families():
    eeg = _suite_rejection_eeg(trials=6)
    eeg["reject"] = _all_rejection_marks()

    output = eeg_rejsuperpose(eeg, 1, 1, 1, 1, 1, 1, 1, 0)

    np.testing.assert_array_equal(output["reject"]["rejglobal"], np.ones(6, dtype=bool))
    np.testing.assert_array_equal(output["reject"]["rejglobalE"], np.ones((2, 6), dtype=bool))


@eeglab_test("unittesting_popfunc/eeg_rejsuperpose/popfunc_eeg_rejsuperpose_wrapperTest.m", "test_pass_all_ica")
def test_eeg_rejsuperpose_current_suite_all_component_mark_families():
    eeg = _suite_rejection_eeg(trials=6, components=True)
    eeg["reject"] = _all_rejection_marks("ica")

    output = eeg_rejsuperpose(eeg, 0, 1, 1, 1, 1, 1, 1, 0)

    np.testing.assert_array_equal(output["reject"]["rejglobal"], np.ones(6, dtype=bool))
    np.testing.assert_array_equal(output["reject"]["rejglobalE"], np.ones((2, 6), dtype=bool))


@pytest.mark.matlab
def test_eeg_rejsuperpose_matches_eeglab_for_deterministic_marks(tmp_path):
    if os.environ.get("EEGPREP_SKIP_MATLAB") == "1":
        pytest.skip("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
    matlab = shutil.which("matlab")
    if matlab is None:
        pytest.skip("MATLAB executable not available")
    eeglab_root = _eeglab_root()
    if eeglab_root is None:
        pytest.skip("EEGLAB source not available for parity reference")

    eeg = _epoched_eeg()
    reject = {
        "rejmanual": np.array([False, True, False, False, False]),
        "rejmanualE": np.array(
            [
                [False, True, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        ),
        "rejthresh": np.array([False, False, True, False, False]),
        "rejthreshE": np.array(
            [
                [False, False, True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        ),
        "rejconst": np.zeros(5, dtype=bool),
        "rejconstE": np.zeros((4, 5), dtype=bool),
        "rejjp": np.array([False, False, False, True, False]),
        "rejjpE": np.array(
            [
                [False, False, False, True, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        ),
        "rejkurt": np.zeros(5, dtype=bool),
        "rejkurtE": np.zeros((4, 5), dtype=bool),
        "rejfreq": np.array([True, False, False, False, False]),
        "rejfreqE": np.array(
            [
                [True, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        ),
    }
    eeg["reject"] = reject
    py_out = eeg_rejsuperpose(eeg, 1, 1, 1, 1, 1, 1, 1, 0)

    script = tmp_path / "eeg_rejsuperpose_parity.m"
    output = tmp_path / "out.mat"
    script.write_text(_matlab_rejsuperpose_script(eeglab_root, output), encoding="utf-8")
    result = subprocess.run(
        [matlab, "-batch", f"run('{script.as_posix()}')"], check=False, capture_output=True, text=True
    )
    if result.returncode:
        pytest.fail(result.stdout + result.stderr)
    matlab_out = scipy.io.loadmat(output)

    np.testing.assert_array_equal(
        np.asarray(py_out["reject"]["rejglobal"], dtype=bool), matlab_out["rejglobal"].ravel()
    )
    np.testing.assert_array_equal(np.asarray(py_out["reject"]["rejglobalE"], dtype=bool), matlab_out["rejglobalE"])


def test_pop_rejmenu_can_combine_marks_without_browser():
    eeg = _epoched_eeg()
    eeg["reject"]["rejthresh"] = np.array([False, True, False, False, False])
    eeg["reject"]["rejthreshE"] = np.zeros((4, 5), dtype=bool)

    out, com = pop_rejmenu(eeg, 1, gui=False, return_com=True)

    assert out["reject"]["rejglobal"].tolist() == [False, True, False, False, False]
    assert _console_python_command(com) == "EEG = eeg_rejsuperpose(EEG, 1, 1, 1, 1, 1, 1, 1, 1)"


def test_pop_autorej_preserves_original_epoch_numbers_during_iterative_rejection():
    eeg = _epoched_eeg()

    out, rejected = pop_autorej(eeg, "threshold", 10, "startprob", 20, "maxrej", 40, "nogui", "on")

    assert eeg["trials"] - out["trials"] == len(rejected)
    assert rejected == sorted(set(rejected))


def test_channel_and_continuous_rejection_work_on_sample_data_without_ica():
    sample = pop_loadset(SAMPLE_DATASET_PATH)

    _, rejected_channels, measure = pop_rejchan(sample, "measure", "std", "threshold", 1e9, "indexonly", "on")
    _, selected_regions = pop_rejcont(
        sample,
        "elecrange",
        [1],
        "threshold",
        1e9,
        "epochlength",
        0.5,
        "contiguous",
        1,
        "onlyreturnselection",
        "on",
    )

    assert rejected_channels == []
    assert measure.shape == (32,)
    assert selected_regions.shape == (0, 2)
    with pytest.raises(ValueError, match="ICA decomposition is required"):
        pop_eegthresh(sample, 0, [1], -10, 10, 0, 1)


@eeglab_test(
    "unittesting_popfunc/pop_rejchan/popfunc_pop_rejchan_wrapperTest.m",
    "test_test_pop_rejchan",
)
def test_pop_rejchan_current_suite_probability_and_kurtosis_options():
    rng = np.random.default_rng(91)
    eeg = create_test_eeg(n_channels=5, n_samples=40, n_trials=3, srate=100)
    eeg["data"] = rng.normal(size=(5, 40, 3))
    options = (
        ([2, 4, 5], [5], "kurt", "off"),
        ([1, 2, 3, 4, 5], [5], "kurt", "off"),
        ([2, 4, 5], [5, 5, 5], "kurt", "off"),
        ([2, 4, 5], [5], "prob", "off"),
        ([1, 2, 3, 4, 5], [5], "kurt", "on"),
        ([1, 2, 3, 4, 5], [5], "prob", "off"),
        ([1, 2, 3, 4, 5], [5, 1], "kurt", "off"),
        ([1, 2, 3, 4], [5], "kurt", "on"),
    )

    for channels, threshold, measure_name, norm in options:
        out, rejected, measure = pop_rejchan(
            eeg,
            "elec",
            channels,
            "threshold",
            threshold,
            "measure",
            measure_name,
            "norm",
            norm,
            "indexonly",
            "on",
        )
        assert measure.shape == (len(channels),)
        assert np.isfinite(measure).all()
        assert set(rejected).issubset(channels)
        assert out["nbchan"] == eeg["nbchan"]

    removal_eeg = create_test_eeg(n_channels=2, n_samples=20, n_trials=1, srate=100)
    removal_eeg["data"] = np.zeros((2, 20))
    removal_eeg["data"][0, 10] = 100
    removed, rejected, _measure = pop_rejchan(removal_eeg, "measure", "std", "threshold", 5)
    assert rejected == [1]
    assert removed["nbchan"] == 1


def test_rejection_component_threshold_recomputes_stale_stored_icaact():
    eeg = _epoched_eeg()
    eeg["icaweights"] = 2.0 * np.eye(4)
    eeg["icasphere"] = np.eye(4)
    eeg["icaact"] = np.zeros((4, eeg["pnts"], eeg["trials"]))

    out, rejected = pop_eegthresh(eeg, 0, [1], -40, 40, 0, 0.79, 0, 0)

    assert rejected == [2]
    assert out["reject"]["icarejthresh"].tolist() == [False, True, False, False, False]


def test_pop_rejchan_scripted_default_threshold_matches_eeglab():
    eeg = create_test_eeg(n_channels=2, n_samples=20, n_trials=1, srate=100)
    eeg["data"] = np.zeros((2, 20))
    eeg["data"][0, 10] = 100

    _out, rejected_channels, _measure = pop_rejchan(eeg, "measure", "std", "indexonly", "on")

    assert rejected_channels == []


def test_pop_rejcont_history_replays_effectful_mode_and_overlap_options():
    sample = pop_loadset(SAMPLE_DATASET_PATH)

    _out, command = pop_rejcont(
        sample,
        "elecrange",
        [1],
        "freqlimit",
        [20, 40],
        "threshold",
        1e9,
        "epochlength",
        0.5,
        "overlap",
        0.1,
        "mode",
        "mean",
        "onlyreturnselection",
        "on",
        return_com=True,
    )

    assert _console_python_command(command) == (
        "EEG = pop_rejcont(EEG, elecrange=[1], freqlimit=[20, 40], threshold=1000000000, "
        "epochlength=0.5, overlap=0.1, mode='mean', onlyreturnselection='on')"
    )


def test_component_selection_and_viewprops_are_replayable_without_scrolling_browser():
    eeg = _epoched_eeg()

    selected, select_com = pop_selectcomps(eeg, [1, 3], reject=[2], plot=False, return_com=True)
    figures, props_com = pop_viewprops(eeg, 0, [1, 2], plot=False, return_com=True)

    assert selected["reject"]["gcompreject"].tolist() == [False, True, False, False]
    assert figures == []
    assert _console_python_command(select_com) == "EEG = pop_selectcomps(EEG, compnum=[1, 3], reject=[2])"
    assert _console_python_command(props_com) == (
        "pop_viewprops(EEG, typecomp=0, chanorcomp=[1, 2], spec_opt=[], erp_opt=[], scroll_event=1, classifier_name='')"
    )


def test_gui_cancel_paths_leave_datasets_unchanged():
    class CancelRenderer:
        def run(self, spec, initial_values=None):
            return None

    eeg = _epoched_eeg()
    out, com = pop_eegthresh(eeg, gui=True, renderer=CancelRenderer(), return_com=True)
    rejchan_out, rejchan_com = pop_rejchan(copy.deepcopy(eeg), gui=True, renderer=CancelRenderer(), return_com=True)

    assert out is eeg
    assert com == ""
    assert rejchan_out["data"].shape == eeg["data"].shape
    assert rejchan_com == ""
    plt.close("all")


def _eeglab_root() -> Path | None:
    candidates = []
    if os.environ.get("EEGPREP_EEGLAB_ROOT"):
        candidates.append(Path(os.environ["EEGPREP_EEGLAB_ROOT"]))
    candidates.append(Path(__file__).resolve().parents[1] / "src" / "eegprep" / "eeglab")
    for candidate in candidates:
        if (candidate / "functions" / "popfunc" / "eeg_rejsuperpose.m").exists():
            return candidate
    return None


def _matlab_rejsuperpose_script(eeglab_root: Path, output: Path) -> str:
    return f"""
addpath(fullfile('{eeglab_root.as_posix()}', 'functions', 'popfunc'));
EEG = struct();
EEG.trials = 5;
EEG.nbchan = 4;
EEG.reject = struct();
EEG.reject.rejmanual = logical([0 1 0 0 0]);
EEG.reject.rejmanualE = logical([0 1 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0]);
EEG.reject.rejthresh = logical([0 0 1 0 0]);
EEG.reject.rejthreshE = logical([0 0 1 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0]);
EEG.reject.rejconst = logical([0 0 0 0 0]);
EEG.reject.rejconstE = logical(zeros(4, 5));
EEG.reject.rejjp = logical([0 0 0 1 0]);
EEG.reject.rejjpE = logical([0 0 0 1 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0]);
EEG.reject.rejkurt = logical([0 0 0 0 0]);
EEG.reject.rejkurtE = logical(zeros(4, 5));
EEG.reject.rejfreq = logical([1 0 0 0 0]);
EEG.reject.rejfreqE = logical([1 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0]);
EEG = eeg_rejsuperpose(EEG, 1, 1, 1, 1, 1, 1, 1, 0);
rejglobal = EEG.reject.rejglobal;
rejglobalE = EEG.reject.rejglobalE;
save('{output.as_posix()}', 'rejglobal', 'rejglobalE');
"""
