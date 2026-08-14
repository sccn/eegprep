"""Unit and property tests for EEGPrep ``envtopo`` component-envelope math.

These run without MATLAB and always run in CI. They pin the EEGLAB-parity
contract of ``envtopo``: the component-ranking metric, the selection/ordering of
plotted components, the peak-variance frame per component, and the
``EnvtopoResult`` return shape. Bit-for-bit numerical parity with EEGLAB on real
data is covered separately, MATLAB-gated, in ``test_envtopo_parity.py``.

The independent metric recomputed here is EEGLAB's default ``sortvar='mp'``:
``mp(c) = max_t( mean_chans( (icawinv[:,c] * (weights[c,:] @ data))**2 ) )`` over
the limcontrib window. Ranking that metric is the core the plot depends on; the
other three modes (``pv``/``pp``/``rp``) are exercised against the live oracle.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
import numpy as np
import pytest

from eegprep.functions.sigprocfunc.envtopo import envtopo
from tests.fixtures import create_test_eeg_with_ica

pytestmark = pytest.mark.parity

# Contract: the six EEGLAB numeric outputs, plus the EEGPrep figure.
EXPECTED_FIELDS = (
    "compvarorder",
    "compvars",
    "compframes",
    "comptimes",
    "compsplotted",
    "sortvar",
    "figure",
)

# Deterministic orthonormal scalp maps (unit-norm columns) so each component's
# mean-square back-projection reduces to its activation power: mean_chans(proj**2)
# = ||map_c||**2 / nchan * act(t)**2 = act(t)**2 / nchan.
_Q3 = np.linalg.qr(np.array([[1.0, 0.2, 0.1], [0.3, 1.0, 0.4], [0.2, 0.5, 1.0]]))[0]


def _mp_metric(mean_data, weights, icawinv, limmask):
    """Independent EEGLAB ``mp`` metric and peak (0-based) frame per component."""
    acts = weights @ mean_data
    n_components = weights.shape[0]
    metric = np.empty(n_components)
    frame = np.empty(n_components, dtype=int)
    window = np.flatnonzero(limmask)
    for c in range(n_components):
        proj = np.outer(icawinv[:, c], acts[c])
        mean_square = np.mean(proj[:, limmask] ** 2, axis=0)
        peak = int(np.argmax(mean_square))
        metric[c] = mean_square[peak]
        frame[c] = int(window[peak])
    return metric, frame


def _ica_dataset(seed, *, n_components=4):
    np.random.seed(seed)
    eeg = create_test_eeg_with_ica(n_channels=6, n_samples=40, n_trials=3, n_components=n_components)
    data = np.asarray(eeg["data"], dtype=float)
    mean_data = data.mean(axis=2) if data.ndim == 3 else data
    weights = np.asarray(eeg["icaweights"], dtype=float) @ np.asarray(eeg["icasphere"], dtype=float)
    icawinv = np.asarray(eeg["icawinv"], dtype=float)
    timerange = [float(eeg["xmin"]) * 1000.0, float(eeg["xmax"]) * 1000.0]
    times_ms = np.linspace(timerange[0], timerange[1], mean_data.shape[1])
    return eeg, mean_data, weights, icawinv, timerange, times_ms


# --------------------------------------------------------------------------- #
# Closed-form anchor: hand-built input whose ranking is obvious by design.
# --------------------------------------------------------------------------- #
def test_closed_form_ranking_and_peak_frames():
    """Three orthonormal maps with activation powers 9:4:1 rank as [1, 2, 3]."""
    frames = 5
    acts = np.zeros((3, frames))
    acts[0, 2] = 3.0  # IC1 peaks at frame 2, power 9
    acts[1, 3] = 2.0  # IC2 peaks at frame 3, power 4
    acts[2, 1] = 1.0  # IC3 peaks at frame 1, power 1
    data = _Q3 @ acts
    weights = _Q3.T  # orthonormal -> pinv(icawinv) == icawinv.T
    timerange = [0.0, 400.0]  # -> times_ms = [0, 100, 200, 300, 400]

    res = envtopo(data, weights, chanlocs=None, icawinv=_Q3, timerange=timerange, sortvar="mp")

    # Named-tuple contract (checked here rather than as a standalone type-only test).
    assert res._fields == EXPECTED_FIELDS
    assert isinstance(res.figure, Figure)

    np.testing.assert_array_equal(res.compvarorder, [1, 2, 3])
    np.testing.assert_array_equal(res.compsplotted, [1, 2, 3])
    # compvars are the metric in ranked (descending) order: act_peak**2 / nchan.
    np.testing.assert_allclose(res.compvars, [9 / 3, 4 / 3, 1 / 3], rtol=0, atol=1e-12)
    # compframes are 0-based and aligned with compvarorder; comptimes in ms.
    peak_frame = dict(zip(res.compvarorder.tolist(), res.compframes.tolist()))
    peak_time = dict(zip(res.compvarorder.tolist(), res.comptimes.tolist()))
    assert peak_frame == {1: 2, 2: 3, 3: 1}
    assert peak_time == {1: 200.0, 2: 300.0, 3: 100.0}
    plt.close(res.figure)


# --------------------------------------------------------------------------- #
# Property/invariant tests over seeded synthetic ICA datasets.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_ranking_matches_independent_mp_metric(seed):
    """envtopo ranks components by the independently recomputed ``mp`` metric."""
    _, mean_data, weights, icawinv, timerange, times_ms = _ica_dataset(seed)
    metric, _ = _mp_metric(mean_data, weights, icawinv, np.ones(times_ms.shape, dtype=bool))
    expected_order = (np.argsort(metric)[::-1] + 1).astype(int)  # 1-based, descending

    res = envtopo(mean_data, weights, chanlocs=None, icawinv=icawinv, timerange=timerange, sortvar="mp")

    np.testing.assert_array_equal(res.compvarorder, expected_order)
    # compvars is the metric in ranked order and is therefore non-increasing.
    assert np.all(np.diff(res.compvars) <= 1e-12)
    np.testing.assert_allclose(np.sort(res.sortvar), np.sort(metric), rtol=1e-9, atol=1e-12)
    plt.close(res.figure)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_peak_frame_and_time_alignment(seed):
    """compframes fall on the metric peak; comptimes are times_ms[compframes]."""
    _, mean_data, weights, icawinv, timerange, times_ms = _ica_dataset(seed)
    _, expected_frame = _mp_metric(mean_data, weights, icawinv, np.ones(times_ms.shape, dtype=bool))

    res = envtopo(mean_data, weights, chanlocs=None, icawinv=icawinv, timerange=timerange, sortvar="mp")

    for order_pos, ic in enumerate(res.compvarorder.tolist()):
        frame = int(res.compframes[order_pos])
        assert frame == int(expected_frame[ic - 1])
        np.testing.assert_allclose(res.comptimes[order_pos], times_ms[frame], rtol=0, atol=1e-9)
    plt.close(res.figure)


def test_limcontrib_window_restricts_peak_frames():
    """With a limcontrib window, every peak frame lies inside that window."""
    _, mean_data, weights, icawinv, timerange, times_ms = _ica_dataset(0)
    span = timerange[1] - timerange[0]
    limcontrib = [timerange[0] + 0.3 * span, timerange[1] - 0.3 * span]
    mask = (times_ms >= limcontrib[0]) & (times_ms <= limcontrib[1])
    metric, expected_frame = _mp_metric(mean_data, weights, icawinv, mask)
    expected_order = (np.argsort(metric)[::-1] + 1).astype(int)

    res = envtopo(
        mean_data, weights, chanlocs=None, icawinv=icawinv, timerange=timerange, limcontrib=limcontrib, sortvar="mp"
    )

    np.testing.assert_array_equal(res.compvarorder, expected_order)
    assert np.all((res.comptimes >= limcontrib[0] - 1e-9) & (res.comptimes <= limcontrib[1] + 1e-9))
    plt.close(res.figure)


@pytest.mark.parametrize("compsplot,n_components", [(2, 4), (7, 4), (3, 10)])
def test_compsplotted_count(compsplot, n_components):
    """compsplotted length is min(compsplot, n_candidates), capped at MAXTOPOS=20."""
    _, mean_data, weights, icawinv, timerange, _ = _ica_dataset(3, n_components=n_components)

    res = envtopo(mean_data, weights, chanlocs=None, icawinv=icawinv, timerange=timerange, compsplot=compsplot)

    assert res.compsplotted.size == min(compsplot, n_components, 20)
    # The plotted set is the top of the full ranking.
    np.testing.assert_array_equal(res.compsplotted, res.compvarorder[: res.compsplotted.size])
    plt.close(res.figure)


def test_subcomps_are_subtracted_and_excluded_from_selection():
    """Subtracted components get zero contribution and drop out of the top set."""
    _, mean_data, weights, icawinv, timerange, _ = _ica_dataset(1)

    res = envtopo(mean_data, weights, chanlocs=None, icawinv=icawinv, timerange=timerange, compsplot=2, subcomps=[1])

    assert 1 not in res.compsplotted.tolist()
    plt.close(res.figure)


def test_unknown_sortvar_raises():
    _, mean_data, weights, icawinv, timerange, _ = _ica_dataset(0)
    with pytest.raises(ValueError, match="sortvar"):
        envtopo(mean_data, weights, chanlocs=None, icawinv=icawinv, timerange=timerange, sortvar="nope")


def test_envmode_rms_runs_and_preserves_ranking():
    """envmode only changes the drawn envelope, not the component ranking."""
    _, mean_data, weights, icawinv, timerange, _ = _ica_dataset(7)

    avg = envtopo(mean_data, weights, chanlocs=None, icawinv=icawinv, timerange=timerange, envmode="avg")
    rms = envtopo(mean_data, weights, chanlocs=None, icawinv=icawinv, timerange=timerange, envmode="rms")

    np.testing.assert_array_equal(avg.compvarorder, rms.compvarorder)
    plt.close(avg.figure)
    plt.close(rms.figure)


@pytest.mark.parametrize("mode,label", [("mp", "ppaf"), ("pv", "pvaf"), ("rp", "rp")])
def test_summed_metric_label_matches_mode(mode, label):
    """The envelope panel prints the summed metric with EEGLAB's label per mode."""
    _, mean_data, weights, icawinv, timerange, _ = _ica_dataset(0)
    res = envtopo(mean_data, weights, chanlocs=None, icawinv=icawinv, timerange=timerange, sortvar=mode)
    texts = [t.get_text() for t in res.figure.axes[0].texts]
    assert any(t.startswith(f"{label} ") and t.endswith("%") for t in texts)
    plt.close(res.figure)


def test_sumenv_modes_control_the_summed_envelope():
    _, mean_data, weights, icawinv, timerange, _ = _ica_dataset(0)
    common = dict(chanlocs=None, icawinv=icawinv, timerange=timerange)
    fill = envtopo(mean_data, weights, sumenv="fill", **common)
    lines_on = envtopo(mean_data, weights, sumenv="on", **common)
    off = envtopo(mean_data, weights, sumenv="off", **common)

    assert any(isinstance(c, PolyCollection) for c in fill.figure.axes[0].collections)
    assert not any(isinstance(c, PolyCollection) for c in off.figure.axes[0].collections)
    # 'on' draws the summed envelope as two extra lines (max and min) vs 'off'.
    assert len(lines_on.figure.axes[0].lines) == len(off.figure.axes[0].lines) + 2
    for result in (fill, lines_on, off):
        plt.close(result.figure)
    with pytest.raises(ValueError, match="sumenv"):
        envtopo(mean_data, weights, sumenv="nope", **common)


def test_vert_draws_marker_lines():
    _, mean_data, weights, icawinv, timerange, _ = _ica_dataset(0)
    latency = timerange[0] + 0.4 * (timerange[1] - timerange[0])
    res = envtopo(mean_data, weights, chanlocs=None, icawinv=icawinv, timerange=timerange, vert=[latency])
    verticals = [ln.get_xdata()[0] for ln in res.figure.axes[0].lines if len(np.unique(ln.get_xdata())) == 1]
    assert any(abs(x - latency / 1000.0) < 1e-9 for x in verticals)
    plt.close(res.figure)
