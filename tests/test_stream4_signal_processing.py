"""Regression tests for release-readiness stream 4 (signal processing)."""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from eegprep.functions.popfunc.eeg_interp import eeg_interp
from eegprep.functions.popfunc.pop_interp import pop_interp
from eegprep.functions.popfunc.pop_newcrossf import pop_newcrossf, pop_newcrossf_dialog_spec
from eegprep.functions.popfunc.pop_newtimef import pop_newtimef, pop_newtimef_dialog_spec
from eegprep.functions.sigprocfunc.runica import _kurtosis_columns, runica
from eegprep.functions.timefreqfunc.newcrossf import newcrossf
from eegprep.functions.timefreqfunc.newtimef import newtimef
from eegprep.plugins.firfilt._filtering import design_eegfiltnew
from eegprep.plugins.firfilt.plotfresp import plotfresp
from eegprep.plugins.firfilt.pop_eegfiltnew import pop_eegfiltnew
from eegprep.functions.guifunc.spec import controls_by_tag
from tests.eeglab_tests import assert_matlab_equal, assert_matlab_near

ROOT = Path(__file__).resolve().parents[1]


def _matlab_scalar_struct(**fields: object) -> np.ndarray:
    result = np.empty((1, 1), dtype=[(name, object) for name in fields])
    for name, value in fields.items():
        result[name][0, 0] = value
    return result


def _matlab_chanlocs(n_channels: int) -> np.ndarray:
    fields = (
        "labels",
        "theta",
        "radius",
        "X",
        "Y",
        "Z",
        "sph_theta",
        "sph_phi",
        "sph_radius",
        "type",
        "ref",
    )
    result = np.empty((1, n_channels), dtype=[(field, object) for field in fields])
    for index, location in enumerate(_locations(n_channels)):
        x = float(location["X"])
        y = float(location["Y"])
        z = float(location["Z"])
        horizontal = float(np.hypot(x, y))
        values = {
            "labels": location["labels"],
            "theta": float(location["theta"]),
            "radius": float(location["radius"]),
            "X": x,
            "Y": y,
            "Z": z,
            "sph_theta": float(np.degrees(np.arctan2(y, x))),
            "sph_phi": float(np.degrees(np.arctan2(z, horizontal))),
            "sph_radius": float(np.sqrt(x * x + y * y + z * z)),
            "type": "",
            "ref": "",
        }
        for field, value in values.items():
            if isinstance(value, str):
                result[field][0, index] = value
            else:
                result[field][0, index] = np.array([[value]], dtype=float)
    return result


def _locations(n_channels: int) -> list[dict]:
    locations = []
    for index in range(n_channels):
        angle = 2 * np.pi * index / n_channels
        locations.append(
            {
                "labels": f"Ch{index + 1}",
                "X": float(np.cos(angle)),
                "Y": float(np.sin(angle)),
                "Z": 0.5,
                "theta": float(np.degrees(np.arctan2(np.cos(angle), np.sin(angle)))),
                "radius": 0.5,
            }
        )
    return locations


def _continuous(n_channels: int = 8, n_points: int = 100, srate: float = 100.0) -> dict:
    xmin = 0.0
    xmax = (n_points - 1) / srate
    return {
        "data": np.random.default_rng(9).normal(size=(n_channels, n_points)),
        "nbchan": n_channels,
        "pnts": n_points,
        "trials": 1,
        "srate": srate,
        "xmin": xmin,
        "xmax": xmax,
        "times": np.linspace(xmin * 1000.0, xmax * 1000.0, n_points),
        "chanlocs": _locations(n_channels),
    }


def _epoch(n_channels: int = 4, n_points: int = 384, trials: int = 4, srate: float = 128.0) -> dict:
    xmin = -1.0
    xmax = xmin + (n_points - 1) / srate
    return {
        "data": np.random.default_rng(4).normal(size=(n_channels, n_points, trials)),
        "nbchan": n_channels,
        "pnts": n_points,
        "trials": trials,
        "srate": srate,
        "xmin": xmin,
        "xmax": xmax,
        "times": np.linspace(xmin * 1000.0, xmax * 1000.0, n_points),
        "chanlocs": _locations(n_channels),
    }


def test_f58_plotfresp_accepts_windowed_sinc_onepass_zerophase():
    coefficients, _metadata = design_eegfiltnew(128.0, locutoff=1.0)
    assert not np.all(coefficients == coefficients[::-1])
    figure, _axes, response = plotfresp(coefficients, 1, 512, 128, "onepass-zerophase", show=False)
    assert response["magnitude_db"].size > 0
    plt.close(figure)


def test_f58_pop_eegfiltnew_frequency_response_plot_opens(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    recording = {
        "data": np.random.default_rng(1).normal(size=(2, 256)),
        "nbchan": 2,
        "pnts": 256,
        "trials": 1,
        "srate": 128.0,
        "xmin": 0.0,
        "xmax": 255 / 128,
    }
    pop_eegfiltnew(recording, locutoff=1, plotfreqz=True, gui=False)
    plt.close("all")


def test_f58_plotfresp_names_the_function_for_a_nonlinear_filter():
    with pytest.raises(ValueError, match="plotfresp"):
        plotfresp(np.array([1.0, 0.2, 0.0, -0.4]), 1, 64, 256, "onepass-zerophase", show=False)


def test_f59_full_range_list_does_not_restore_the_second_to_last_sample():
    eeg = _continuous()
    listed = eeg_interp(eeg, [0], method="spherical", t_range=[eeg["xmin"], eeg["xmax"]])
    whole = eeg_interp(eeg, [0], method="spherical")
    np.testing.assert_array_equal(listed["data"], whole["data"])
    assert not np.array_equal(listed["data"][0, -2], eeg["data"][0, -2])


def test_f59_gui_empty_time_field_matches_unrestricted_interpolation():
    class Renderer:
        def run(self, spec, initial_values=None):
            return {"chanlist": {"chans": [0], "chanstr": "[1]"}, "method": 1, "timerange": ""}

    eeg = _continuous()
    gui = pop_interp(eeg, gui=True, renderer=Renderer())
    whole = pop_interp(eeg, [1], "spherical")
    np.testing.assert_array_equal(gui["data"], whole["data"])


def test_eeg_interp_time_range_requires_sample_times():
    eeg = _continuous()
    del eeg["times"]
    with pytest.raises(ValueError, match="eeg_checkset"):
        eeg_interp(eeg, [0], method="spherical", t_range=(0.2, 0.5))
    eeg = _continuous()
    eeg["times"] = eeg["times"][:-1]
    with pytest.raises(ValueError, match="eeg_checkset"):
        eeg_interp(eeg, [0], method="spherical", t_range=(0.2, 0.5))


def test_f59_partial_range_includes_both_endpoints_and_keeps_the_final_sample():
    eeg = _continuous()
    start, end = 0.2, 0.5
    partial = eeg_interp(eeg, [0], method="spherical", t_range=(start, end))
    whole = eeg_interp(eeg, [0], method="spherical")
    times = eeg["times"]
    outside = (times < start * 1000.0) | (times > end * 1000.0)
    inside = ~outside
    np.testing.assert_array_equal(partial["data"][0, outside], eeg["data"][0, outside])
    np.testing.assert_array_equal(partial["data"][0, inside], whole["data"][0, inside])
    assert times[np.flatnonzero(inside)[0]] == pytest.approx(start * 1000.0)
    assert times[np.flatnonzero(inside)[-1]] == pytest.approx(end * 1000.0)
    assert outside[-1]


def test_f59_epoched_range_restores_every_trial():
    eeg = _continuous()
    eeg["trials"] = 3
    eeg["data"] = np.random.default_rng(3).normal(size=(8, eeg["pnts"], 3))
    start, end = 0.2, 0.5
    partial = eeg_interp(eeg, [0], method="spherical", t_range=(start, end))
    outside = (eeg["times"] < start * 1000.0) | (eeg["times"] > end * 1000.0)
    np.testing.assert_array_equal(partial["data"][0, outside, :], eeg["data"][0, outside, :])


def test_f59_channel_location_structure_is_not_restored_outside_the_range():
    eeg = _continuous()
    added = {"labels": "Nz", "X": 0.2, "Y": 0.2, "Z": 0.9, "theta": 10.0, "radius": 0.2}
    partial = eeg_interp(eeg, [added], method="spherical", t_range=(0.2, 0.5))
    assert not np.all(partial["data"][-1, :5] == 0.0)


def test_f60_pop_interp_honours_a_scripted_time_range():
    """Pinned pop_interp.m:61 sets t_range='' and discards the fourth argument."""
    eeg = _continuous()
    ranged = pop_interp(eeg, [1], "spherical", t_range=[0.2, 0.5])
    whole = pop_interp(eeg, [1], "spherical")
    outside = (eeg["times"] < 200.0) | (eeg["times"] > 500.0)
    np.testing.assert_array_equal(ranged["data"][0, outside], eeg["data"][0, outside])
    assert not np.array_equal(whole["data"][0, outside], eeg["data"][0, outside])
    help_text = (ROOT / "src/eegprep/resources/help/pop_interp.md").read_text()
    assert "pop_interp.m" in help_text
    assert "line 61" in help_text


def test_pop_interp_channel_indices_are_one_based():
    eeg = _continuous()
    interpolated = pop_interp(eeg, [2], "spherical")
    np.testing.assert_array_equal(interpolated["data"][0], eeg["data"][0])
    assert not np.array_equal(interpolated["data"][1], eeg["data"][1])
    with pytest.raises(ValueError, match="pop_interp: channel index 0"):
        pop_interp(eeg, [0], "spherical")


def test_f69_kurtosis_subset_uses_fix_rand_on_the_twister_stream():
    got = _kurtosis_columns(np.random.RandomState(5489), 30504, 17)
    expected = np.array(
        [
            24852,
            27630,
            3873,
            27861,
            19289,
            2975,
            8495,
            16682,
            29207,
            29432,
            4807,
            29606,
            29197,
            14805,
            24411,
            4328,
            12865,
        ]
    )
    np.testing.assert_array_equal(got + 1, expected)


def test_f91_newcrossf_default_frequencies_stop_at_50_hz():
    frames = 384
    trials = np.random.default_rng(1).normal(size=(frames, 4))
    result = newcrossf(trials, trials, frames, [-1000, 1992], 128, [3, 0.5], padratio=1, plot="off")
    assert result.freqs[0] == pytest.approx(6.0)
    assert result.freqs[-1] == pytest.approx(50.0)
    assert result.freqs.size == 23


def test_f91_newcrossf_log_scale_lifts_a_zero_lower_edge_to_3_hz():
    frames = 384
    trials = np.random.default_rng(1).normal(size=(frames, 4))
    result = newcrossf(
        trials,
        trials,
        frames,
        [-1000, 1992],
        128,
        [3, 0.5],
        padratio=1,
        freqscale="log",
        plot="off",
    )
    assert result.freqs[0] == pytest.approx(3.0)
    assert result.freqs[-1] == pytest.approx(50.0)


def test_f92_dialogs_truncate_epoch_limits_toward_zero():
    eeg = _epoch()
    cross = controls_by_tag(pop_newcrossf_dialog_spec(eeg))
    timef = controls_by_tag(pop_newtimef_dialog_spec(eeg))
    assert cross["tlimits"].value == "-1000 1992"
    assert timef["tlimits"].value == "-1000 1992"


def test_f126_pop_newtimef_keeps_the_last_sample_and_a_prestimulus_baseline():
    eeg = _epoch()
    common = dict(
        tlimits=[-1000, 1992],
        cycles=[3, 0.8],
        padratio=1,
        plot="off",
        plotersp="off",
        plotitc="off",
        verbose="off",
    )
    full = pop_newtimef(eeg, 1, 1, baseline=0, **common)
    zeroed = copy.deepcopy(eeg)
    zeroed["data"][:, -1, :] = 0.0
    without_last = pop_newtimef(zeroed, 1, 1, baseline=0, **common)
    assert not np.allclose(full.ersp, without_last.ersp)
    earlier = pop_newtimef(eeg, 1, 1, baseline=-500, **common)
    assert not np.allclose(full.ersp, earlier.ersp)
    assert full.times[0] < 0.0


def test_f126_pop_newcrossf_keeps_the_end_sample_pinned_eeglab_drops():
    eeg = _epoch()
    common = dict(tlimits=[-1000, 1992], cycles=[3, 0.5], padratio=1, plot="off")
    base = pop_newcrossf(eeg, 1, 1, 2, **common)
    last = copy.deepcopy(eeg)
    last["data"][:, -1, :] = 0.0
    changed = pop_newcrossf(last, 1, 1, 2, **common)
    assert not np.allclose(base.coherence, changed.coherence)
    help_text = (ROOT / "src/eegprep/resources/help/pop_newcrossf.md").read_text()
    assert "pop_newcrossf.m" in help_text
    assert "line 189" in help_text


@pytest.mark.matlab_oracle
def test_f69_runica_short_run_matches_pythoncompat(eeglab_backend):
    data = np.random.default_rng(0).normal(size=(4, 300))
    for extended in (0, 1):
        weights, sphere, *_rest = runica(data, extended=extended, rndreset="off", maxsteps=2, verbose="off")
        matlab_weights, matlab_sphere = eeglab_backend(
            "runica",
            data,
            "extended",
            float(extended),
            "rndreset",
            "off",
            "maxsteps",
            2.0,
            "verbose",
            "off",
            "pythoncompat",
            "on",
            nargout=2,
        )
        assert_matlab_near(weights, np.asarray(matlab_weights))
        assert_matlab_near(sphere, np.asarray(matlab_sphere))


def test_newtimef_keeps_repeated_sample_times():
    frames = 128
    signal = np.random.default_rng(2).normal(size=(frames, 3))
    result = newtimef(
        signal,
        frames,
        [-500, 492],
        128,
        [3, 0.8],
        padratio=1,
        baseline=0,
        plotersp="off",
        plotitc="off",
        verbose="off",
    )
    assert result.ersp.shape == (4, 111)
    assert result.itc.shape == (4, 111)
    assert result.times.size == 111
    assert np.unique(result.times).size < result.times.size


@pytest.mark.matlab_oracle
def test_f91_f126_time_frequency_matches_eeglab(eeglab_backend):
    frames = 128
    signal = np.random.default_rng(2).normal(size=(frames, 3))
    python_cross = newcrossf(signal, signal, frames, [-500, 500], 128, [3, 0.5], padratio=1, plot="off")
    vector = signal.reshape(1, -1, order="F")
    _coherence, _baseline, _times, freqs = eeglab_backend(
        "newcrossf",
        vector,
        vector,
        float(frames),
        np.array([[-500.0, 500.0]]),
        128.0,
        np.array([[3.0, 0.5]]),
        "padratio",
        1.0,
        "plotamp",
        "off",
        "plotphase",
        "off",
        nargout=4,
    )
    assert_matlab_near(python_cross.freqs, np.asarray(freqs).reshape(-1))

    python_time = newtimef(
        signal,
        frames,
        [-500, 492],
        128,
        [3, 0.8],
        padratio=1,
        baseline=0,
        plotersp="off",
        plotitc="off",
        verbose="off",
    )
    ersp, itc = eeglab_backend(
        "newtimef",
        signal,
        float(frames),
        np.array([[-500.0, 492.0]]),
        128.0,
        np.array([[3.0, 0.8]]),
        "padratio",
        1.0,
        "baseline",
        0.0,
        "plotersp",
        "off",
        "plotitc",
        "off",
        "verbose",
        "off",
        nargout=2,
    )
    assert_matlab_near(python_time.ersp, np.asarray(ersp))
    assert_matlab_near(python_time.itc, np.asarray(itc))


@pytest.mark.matlab_oracle
def test_f60_pinned_pop_interp_discards_the_fourth_argument(eeglab_backend):
    """Pinned pop_interp.m:61 clears t_range, so the ranged call matches the full one."""
    eeg = eeglab_backend("eeg_emptyset")
    recording = _continuous(n_channels=6, n_points=40, srate=100.0)
    eeg["data"] = np.asarray(recording["data"], dtype=float)
    eeg["nbchan"] = float(recording["nbchan"])
    eeg["pnts"] = float(recording["pnts"])
    eeg["trials"] = float(recording["trials"])
    eeg["srate"] = float(recording["srate"])
    eeg["xmin"] = float(recording["xmin"])
    eeg["xmax"] = float(recording["xmax"])
    eeg["times"] = np.asarray(recording["times"], dtype=float).reshape(1, -1)
    eeg["chanlocs"] = _matlab_chanlocs(int(recording["nbchan"]))
    eeg["chaninfo"] = _matlab_scalar_struct(nosedir="+X", nodatchans=np.array([]))
    eeg["reject"] = _matlab_scalar_struct(rejjp=np.array([]))
    eeg["etc"] = _matlab_scalar_struct(eeglabvers="")
    eeg = eeglab_backend("eeg_checkset", eeg)
    full = eeglab_backend("pop_interp", eeg, np.array([[1.0]]), "spherical", nargout=1)
    ranged = eeglab_backend(
        "pop_interp",
        eeg,
        np.array([[1.0]]),
        "spherical",
        np.array([[0.1, 0.2]]),
        nargout=1,
    )
    assert_matlab_equal(np.asarray(full["data"]), np.asarray(ranged["data"]))
