"""Tests for the EEGLAB-style ``erpimage`` plot and its EEGLAB reference contracts.

Numerical parity with MATLAB lives in ``tests/test_erpimage_engine.py``; these
tests cover argument handling, the returned outputs and what is drawn.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pytest

import eegprep
from eegprep.functions.sigprocfunc import erpimage as erpimage_module
from eegprep.functions.sigprocfunc.erpimage import ErpimageResult, erpimage
from eegprep.functions.sigprocfunc.erpimage_engine import welch_psd
from tests.eeglab_tests import eeglab_test
from tests.eeglab_tests.assertions import assert_matlab_struct_near


_ERPIMAGE_SOURCE = "unittesting_sigprocfunc/erpimage/sigprocfunc_erpimage_wrapperTest.m"
SRATE = 128.0


def _reference_erpimage_data():
    return np.array(
        [[21, 24, 25, 28, 31, 37], [22, 25, 26, 29, 32, 38], [23, 26, 27, 30, 33, 39], [24, 27, 28, 31, 34, 40]],
        dtype=float,
    )


def _expect_reference_failure(eeglab_backend, request, *args) -> None:
    # EEGLAB prints help and returns on these inputs; the legacy wrappers pass on that return.
    # EEGPrep raises instead (an accepted deviation: fail fast rather than print and return).
    if request.config.getoption("--eeglab-backend") == "matlab":
        eeglab_backend("erpimage", *args, nargout=0)
        return
    with pytest.raises((TypeError, ValueError)):
        eeglab_backend("erpimage", *args, nargout=0)


@pytest.mark.gui
@eeglab_test(_ERPIMAGE_SOURCE, "test_fail_continuous")
def test_reference_erpimage_continuous(eeglab_backend, request):
    _expect_reference_failure(eeglab_backend, request, np.arange(21.0, 41.0)[None, :])


@pytest.mark.gui
@eeglab_test(_ERPIMAGE_SOURCE, "test_fail_no_arg")
def test_reference_erpimage_no_arg(eeglab_backend, request):
    _expect_reference_failure(eeglab_backend, request)


@pytest.mark.gui
@eeglab_test(_ERPIMAGE_SOURCE, "test_fail_one_trial")
def test_reference_erpimage_one_trial(eeglab_backend, request):
    _expect_reference_failure(eeglab_backend, request, _reference_erpimage_data(), np.array([[1.0]]))


@pytest.mark.gui
@eeglab_test(_ERPIMAGE_SOURCE, "test_fail_trial_div")
def test_reference_erpimage_trial_div(eeglab_backend, request):
    _expect_reference_failure(eeglab_backend, request, _reference_erpimage_data(), np.arange(1.0, 6.0)[None, :])


def _as_matlab_outputs(outputs) -> list:
    """Explicit Python-to-MATLAB conventions: rows, 1-based sortidx, [] and NaN handles."""
    converted = []
    for index, value in enumerate(outputs):
        if index == 4:
            converted.append(value)
        elif value is None:
            converted.append(np.empty((0, 0)))
        elif index == 13:
            converted.append(np.asarray(value)[np.newaxis, :] + 1)
        elif index in {1, 2, 3}:
            converted.append(np.asarray(value)[np.newaxis, :])
        else:
            converted.append(np.asarray(value))
    return converted


def _reference_erpimage_outputs(eeglab_backend, request, *args, remaining_axes=4):
    if request.config.getoption("--eeglab-backend") == "matlab":
        outputs = eeglab_backend("eegprep_test_erpimage_outputs", "call", float(6 - remaining_axes), *args)

        def assert_axis():
            assert eeglab_backend("eegprep_test_erpimage_outputs", "axis_type").lower() == "axes"

        return tuple(outputs.ravel(order="F")), assert_axis
    outputs = _as_matlab_outputs(eeglab_backend("erpimage", *args, nargout=15))
    axes = outputs[4]
    compared = [np.nan if axis is None else axis for axis in axes[5 - remaining_axes : 5]]
    all_missing = all(axis is np.nan for axis in compared)
    outputs[4] = np.array([compared], dtype=float if all_missing else object)

    def assert_axis():
        assert isinstance(axes[0], Axes)

    return outputs, assert_axis


def _assert_reference_erpimage_outputs(outputs, assert_axis, *, limits, erp, remaining_axes):
    empty = np.empty((0, 0))
    indices = np.arange(1.0, 7.0)[None, :]
    expected = (
        _reference_erpimage_data(),
        indices,
        indices,
        np.array([limits]),
        np.full((1, remaining_axes), np.nan),
        erp,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        indices,
        empty,
    )
    for index in range(4):
        assert_matlab_struct_near(expected[index], outputs[index])
    # The source compares handles 2:end, or 4:end in the three-axis case.
    # Modern MATLAB returns that tail as a cell of NaN placeholders. Numeric
    # tails still go through near(); this does not relax any data comparison.
    handles = np.asarray(outputs[4])
    placeholder = handles.dtype == object and all(
        np.asarray(value).dtype.kind in "fc" and np.isnan(np.asarray(value)).all() for value in handles.reshape(-1)
    )
    if placeholder:
        assert handles.size == remaining_axes
    else:
        assert_matlab_struct_near(expected[4], outputs[4])
    assert_axis()
    for index in range(5, 15):
        assert_matlab_struct_near(expected[index], outputs[index])


@pytest.mark.gui
@eeglab_test(_ERPIMAGE_SOURCE, "test_pass_general")
def test_reference_erpimage_general(eeglab_backend, request):
    outputs, assert_axis = _reference_erpimage_outputs(eeglab_backend, request, _reference_erpimage_data())
    _assert_reference_erpimage_outputs(
        outputs, assert_axis, limits=[0, 3, *([np.nan] * 8)], erp=np.empty((0, 0)), remaining_axes=4
    )


@pytest.mark.gui
@eeglab_test(_ERPIMAGE_SOURCE, "test_pass_many_args")
def test_reference_erpimage_many_args(eeglab_backend, request):
    outputs, assert_axis = _reference_erpimage_outputs(
        eeglab_backend,
        request,
        _reference_erpimage_data(),
        np.empty((0, 0)),
        np.empty((0, 0)),
        "testcase",
        1.0,
        1.0,
        "erp",
        "cbar",
        "noxlabel",
        remaining_axes=2,
    )
    _assert_reference_erpimage_outputs(
        outputs,
        assert_axis,
        limits=[1, 4, 36, 39.9, *([np.nan] * 6)],
        erp=np.array([[27 + 2 / 3, 28 + 2 / 3, 29 + 2 / 3, 30 + 2 / 3]]),
        remaining_axes=2,
    )


@pytest.mark.gui
@eeglab_test(_ERPIMAGE_SOURCE, "test_pass_times")
def test_reference_erpimage_times(eeglab_backend, request):
    outputs, assert_axis = _reference_erpimage_outputs(
        eeglab_backend, request, _reference_erpimage_data(), np.empty((0, 0)), np.array([[0.0, 4.0, 1.0]])
    )
    _assert_reference_erpimage_outputs(
        outputs, assert_axis, limits=[0, 3000, *([np.nan] * 8)], erp=np.empty((0, 0)), remaining_axes=4
    )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _trials(points: int = 64, trials: int = 30, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    times = -200 + np.arange(points) * 1000 / SRATE
    phase = rng.uniform(0, 2 * np.pi, trials)
    data = 10 * np.sin(2 * np.pi * 10 * times[:, None] / 1000 + phase) + rng.standard_normal((points, trials))
    return data, times


def _image_axis(result: ErpimageResult) -> Axes:
    return result.axhndls[0]


def test_erpimage_is_a_lazy_eegprep_export() -> None:
    assert eegprep.erpimage is erpimage


def test_scalar_caxis_sets_symmetric_fraction_of_max() -> None:
    data, _times = _trials()
    fraction = 2.0 / 3.0
    result = erpimage(data, caxis=fraction)
    magnitude = fraction * float(np.max(np.abs(result.outdata)))
    vmin, vmax = _image_axis(result).get_images()[0].get_clim()
    np.testing.assert_allclose([vmin, vmax], [-magnitude, magnitude], rtol=1e-9)


def test_erpimage_sorts_trials_and_draws_zero_and_vertical_lines() -> None:
    data = np.asarray([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0], [12.0, 22.0, 32.0], [13.0, 23.0, 33.0]])
    result = erpimage(data, [30.0, 10.0, 20.0], [-100.0, 0.0, 100.0, 200.0], vert=[100.0])

    np.testing.assert_array_equal(result.outdata, data[:, [1, 2, 0]])
    np.testing.assert_array_equal(result.sortidx, [1, 2, 0])
    lines = _image_axis(result).lines
    segments = [line for line in lines if len(line.get_xdata()) == 2]
    zero = [line for line in segments if np.allclose(line.get_xdata(), [0.0, 0.0])]
    vertical = [line for line in segments if np.allclose(line.get_xdata(), [100.0, 100.0])]
    assert zero and zero[0].get_linewidth() == erpimage_module.ZEROWIDTH
    assert vertical and vertical[0].get_linestyle() == "--"
    sortvar_trace = [line for line in lines if line.get_linestyle() == "-" and len(line.get_xdata()) == 3]
    np.testing.assert_allclose(sortvar_trace[0].get_xdata(), [10.0, 20.0, 30.0])


def test_eeglab_tokens_and_keywords_are_equivalent() -> None:
    data, times = _trials()
    sortvar = np.linspace(400, 100, data.shape[1])
    tokens = erpimage(data, sortvar, times, "tokens", 3, 1, "erp", "cbar", "on", "yerplabel", "\\muV")
    keywords = erpimage(data, sortvar, times, "keywords", 3, 1, erp=True, cbar=True, yerplabel="\\muV")
    np.testing.assert_array_equal(tokens.outdata, keywords.outdata)
    np.testing.assert_array_equal(tokens.limits, keywords.limits)
    assert isinstance(tokens, tuple) and len(tokens) == 16
    image_ax, colorbar_ax, erp_ax, amp_ax, coh_ax = tokens.axhndls
    assert colorbar_ax is not None and erp_ax is not None and amp_ax is None and coh_ax is None
    assert erp_ax.get_ylabel() == "µV"
    assert image_ax.get_title() == "tokens"
    # EEGLAB's blue ERP trace, below an image whose time labels are hidden.
    assert erp_ax.lines[0].get_color() == (0.0, 0.0, 1.0)
    assert not any(label.get_visible() and label.get_text() for label in image_ax.get_xticklabels())


def test_erpimage_draws_into_a_target_subfigure() -> None:
    data, times = _trials()
    figure = plt.figure()
    target = figure.add_subfigure(figure.add_gridspec(1, 2)[0, 1])
    before = plt.get_fignums()
    result = erpimage(data, None, times, "", 1, 1, erp=True, cbar=True, target=target)
    assert plt.get_fignums() == before
    assert result.figure is target
    assert all(axis.get_figure() is target for axis in result.axhndls if axis is not None)


def test_scalp_map_inset_stays_round_in_a_subfigure() -> None:
    data, times = _trials()
    chanlocs = [{"labels": "C3", "theta": -90.0, "radius": 0.25}, {"labels": "C4", "theta": 90.0, "radius": 0.25}]
    figure = plt.figure(figsize=(10, 4))
    target = figure.subfigures(1, 2)[1]
    result = erpimage(data, None, times, topo=(1, chanlocs), target=target)
    box = result.axhndls[5].get_window_extent()
    assert box.width == pytest.approx(box.height, rel=1e-6)


def test_target_must_be_a_figure_and_cannot_be_combined_with_noshow() -> None:
    data, times = _trials()
    figure = plt.figure()
    with pytest.raises(TypeError, match="target must be a matplotlib Figure or SubFigure, got Axes"):
        erpimage(data, None, times, target=figure.add_subplot())
    with pytest.raises(ValueError, match="'NoShow' draws no figure"):
        erpimage(data, None, times, "", 1, 1, "NoShow", "on", target=figure)


def test_a_drawing_error_closes_the_new_figure() -> None:
    data, times = _trials()
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        erpimage(data, None, times, auxvar=(np.zeros((1, 30)), "notacolor"))
    assert plt.get_fignums() == before


def test_noshow_returns_outputs_without_a_figure() -> None:
    data, times = _trials()
    before = plt.get_fignums()
    result = erpimage(data, None, times, "", 3, 2, "NoShow", "on", "erp")
    assert plt.get_fignums() == before
    assert result.figure is None and result.axhndls == (None, None, None, None, None)
    # Mirrors EEGLAB's no-plot call: no axis limits are computed.
    assert np.isnan(result.limits[2]) and np.isnan(result.limits[3])


def test_noplot_hides_only_the_sortvar_trace() -> None:
    data, times = _trials()
    sortvar = np.linspace(100, 300, data.shape[1])
    shown = erpimage(data, sortvar, times)
    hidden = erpimage(data, sortvar, times, "", 1, 1, "noplot", "on")
    np.testing.assert_array_equal(shown.outdata, hidden.outdata)
    widths = [[line.get_linewidth() for line in _image_axis(result).lines] for result in (shown, hidden)]
    assert erpimage_module.SORTWIDTH in widths[0] and erpimage_module.SORTWIDTH not in widths[1]


def test_spectral_panels_and_insets() -> None:
    data, times = _trials()
    chanlocs = [{"labels": "C3", "theta": -90.0, "radius": 0.25}, {"labels": "C4", "theta": 90.0, "radius": 0.25}]
    result = erpimage(
        data,
        np.arange(data.shape[1], dtype=float),
        times,
        "phase",
        3,
        1,
        phasesort=[0, 25, 10],
        coher=[10, 10, 0.05],
        erp=True,
        cbar=True,
        cbar_title="\\muV",
        showwin=True,
        topo=(2, chanlocs),
        spec=[2, 40],
        specaxis="lin",
        vert=[50.0],
        rng=np.random.RandomState(1),
    )
    image_ax, colorbar_ax, erp_ax, amp_ax, coh_ax, topo_ax, spec_ax = result.axhndls
    assert image_ax.get_ylabel() == "Phase-sorted Trials"
    assert colorbar_ax.get_title() == "µV"
    assert not amp_ax.axison  # EEGLAB turns the amplitude axes off and writes the values
    assert {text.get_text() for text in amp_ax.texts} >= {"ERSP", f"{result.limits[8]:.4g} dB"}
    assert "ITC" in {text.get_text() for text in coh_ax.texts}
    assert any(text.get_text() == "10 Hz" for text in coh_ax.texts)
    assert erp_ax.patches  # the sort window behind the ERP ('showwin')
    assert coh_ax.patches  # the coherence significance band
    assert topo_ax.collections and spec_ax.get_xscale() == "linear"
    assert result.phsangls is not None and result.outamps is not None and result.cohsig is not None


def test_component_map_inset_and_plotamps() -> None:
    data, times = _trials()
    chanlocs = [
        {"labels": label, "theta": theta, "radius": 0.4}
        for label, theta in (("F3", -40.0), ("F4", 40.0), ("P3", -140.0), ("P4", 140.0))
    ]
    result = erpimage(
        data,
        None,
        times,
        "amps",
        3,
        1,
        "plotamps",
        "coher",
        [10, 10, 0.05],
        "topo",
        ([0.5, -0.2, 0.1, 0.9], chanlocs),
        rng=np.random.RandomState(2),
    )
    assert result.outamps is result.outdata
    image = _image_axis(result).get_images()[0]
    peak = 0.8 * np.max(np.abs(result.outdata))
    np.testing.assert_allclose(image.get_clim(), [-peak, peak])
    assert result.axhndls[5] is not None


def _legend_first(result: ErpimageResult) -> str:
    return [text.get_text() for text in result.axhndls[2].get_legend().get_texts()][0]


@pytest.mark.parametrize(
    ("options", "check"),
    [
        ({"erp": 3}, lambda r: _legend_first(r) == "Lowest 33%"),
        (
            {"erp": True, "erpstd": True, "erp_grid": True},
            lambda r: r.axhndls[2].yaxis.get_gridlines()[0].get_visible(),
        ),
        ({"erp": True, "erpalpha": 0.05}, lambda r: r.erpsig.shape == (2, 64) and r.axhndls[2].patches),
        ({"erp": True, "erp_vltg_ticks": [-5, 0, 5]}, lambda r: list(r.axhndls[2].get_yticks()) == [-5, 0, 5]),
        ({"erp": True, "noxlabel": True}, lambda r: r.axhndls[2].get_xlabel() == ""),
        ({"align": np.inf, "erp": True}, lambda r: np.isfinite(r.limits[2])),
        (
            {"auxvar": np.vstack([np.linspace(0, 100, 30)] * 2), "sortvarpercent": [0.1, 0.9]},
            lambda r: isinstance(r.outvar, tuple),
        ),
        ({"horz": [10, 20]}, lambda r: any(list(line.get_ydata()) == [10, 10] for line in r.axhndls[0].lines)),
        ({"img_trialax_label": "RT (ms)", "horz": [200]}, lambda r: r.axhndls[0].get_ylabel() == "RT (ms)"),
        ({"img_trialax_ticks": [5, 15, 25]}, lambda r: list(r.axhndls[0].get_yticks()) == [5, 15, 25]),
        ({"sortvar_limits": [150, 250], "erp": True}, lambda r: r.axhndls[0].get_ylim()[1] < 30),
        ({"limits": [-100, 200, -20, 20]}, lambda r: tuple(r.axhndls[0].get_xlim()) == (-100.0, 200.0)),
        ({"plotmode": "topo", "nosort": True}, lambda r: not r.axhndls[0].axison),
        ({"valsort": [0, 50, -1]}, lambda r: r.axhndls[0].get_ylabel() == "Voltage-sorted Trials"),
        ({"ampsort": [0, 0, 10]}, lambda r: r.axhndls[0].get_ylabel() == "Amplitude-sorted Trials"),
        (
            {"vert": np.tile(np.linspace(50, 150, 30)[:, None], (1, 2)), "erp": True},
            lambda r: len(r.axhndls[2].lines) > 3,
        ),
        (
            {"timewarp": (np.column_stack([np.linspace(40, 80, 30), np.linspace(120, 180, 30)]),), "erp": True},
            lambda r: r.outdata.shape == (64, 28),
        ),
        ({"filt": [0, 20], "baseline": [-100, 0]}, lambda r: r.outdata.shape == (64, 28)),
    ],
)
def test_erpimage_options_draw(options, check) -> None:
    data, times = _trials()
    sortvar = np.linspace(100, 300, data.shape[1])
    result = erpimage(data, sortvar, times, "", 3, 1, **options)
    assert check(result)


def test_trial_axis_ticks_follow_eeglab_order_of_magnitude() -> None:
    data, times = _trials(trials=40)
    result = erpimage(data, np.arange(40.0), times, "", 5, 1)
    np.testing.assert_array_equal(_image_axis(result).get_yticks(), [10.0, 20.0, 30.0])


def test_trace_ticks_and_hanning_helpers() -> None:
    np.testing.assert_allclose(erpimage_module._trace_ticks(-10.0, 10.0), [-8, -4, 0, 4, 8])
    np.testing.assert_allclose(erpimage_module._trace_ticks(2.0, 10.0), [1.6, 3.6, 5.6, 7.6])
    np.testing.assert_allclose(erpimage_module._trace_ticks(-0.2, 0.2), [-0.16, -0.08, 0, 0.08, 0.16])
    np.testing.assert_allclose(erpimage_module._hanning(4), [0.3454915, 0.9045085, 0.9045085, 0.3454915], rtol=1e-6)
    np.testing.assert_allclose(erpimage_module._hanning(3), [0.5, 1.0, 0.5])


def test_butterworth_filter_modes() -> None:
    signal = np.random.default_rng(3).standard_normal((256, 2))
    for edges in ([0, 20], [10, 64], [-20, -30], [10, 20]):
        filtered = erpimage_module.butterworth_filter(signal, np.asarray(edges, dtype=float), 128.0)
        assert filtered.shape == signal.shape and np.all(np.isfinite(filtered))


def test_log_spectrum_axis_starts_at_the_first_analyzed_frequency() -> None:
    data, times = _trials()
    result = erpimage(data, None, times, spec=[0, 50])
    _psd, freqs = welch_psd(data.reshape(-1, order="F"), data.shape[0], SRATE)
    assert result.axhndls[5].get_xscale() == "log"
    assert result.axhndls[5].get_xlim() == pytest.approx((freqs[1], 50.0))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"topo": (3, [{"labels": "C3"}, {"labels": "C4"}])}, "'topo' channel 3 is not a channel number from 1 to 2"),
        ({"topo": (1.5, [{"labels": "C3"}, {"labels": "C4"}])}, "'topo' channel 1.5"),
        ({"topo": ([0.1, 0.2, 0.3], [{"labels": "C3"}, {"labels": "C4"}])}, "'topo' map has 3 values for 2 channels"),
        ({"spec": [70, 90]}, "starts above the Nyquist frequency"),
    ],
)
def test_insets_that_would_draw_nothing_raise_before_a_figure_exists(options, message) -> None:
    data, times = _trials()
    before = plt.get_fignums()
    with pytest.raises(ValueError, match=message):
        erpimage(data, None, times, **options)
    assert plt.get_fignums() == before


def test_ignored_inset_and_tick_inputs_are_logged(caplog) -> None:
    data, times = _trials()
    chanlocs = [{"labels": "C3", "theta": -90.0, "radius": 0.25}, {"labels": "C4", "theta": 90.0, "radius": 0.25}]
    with caplog.at_level(logging.WARNING, logger=erpimage_module.logger.name):
        erpimage(
            data,
            np.linspace(100, 300, 30),
            times,
            topo=(1, chanlocs, {}),
            img_trialax_label="RT (ms)",
            img_trialax_ticks=[150, 900],
        )
    assert "the third 'topo' element (EEG.chaninfo) is not used" in caplog.text
    assert "'img_trialax_ticks' exceed smoothed sorting variable values" in caplog.text
