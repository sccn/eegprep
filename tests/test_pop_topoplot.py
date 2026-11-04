import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eegprep.functions.popfunc.pop_topoplot import (
    _latency_positions,
    _parse_items_text,
    plot_channel_locations,
    pop_topoplot,
    pop_topoplot_dialog_spec,
)
from eegprep.functions.sigprocfunc.topoplot import topoplot
from tests.eeglab_tests import assert_matlab_equal, eeglab_test
from tests.fixtures import create_test_eeg_with_ica


_COLORBAR_SOURCE = "regression_tests/t_pop_topoplot_colorbar.m"


@pytest.fixture
def reference_colorbar_eeg(eeglab_backend, eeglab_suite_root):
    return eeglab_backend(
        "pop_loadset",
        "filename",
        "eeglab_data_epochs_ica.set",
        "filepath",
        str(eeglab_suite_root / "eeglab/sample_data"),
    )


@pytest.fixture
def reference_colorbar_figures(eeglab_backend, request):
    if request.config.getoption("--eeglab-backend") == "matlab":
        eeglab_backend("eegprep_test_colorbar_graphics", "setup", nargout=0)
        try:
            yield
        finally:
            eeglab_backend("eegprep_test_colorbar_graphics", "teardown", nargout=0)
    else:
        previous = set(plt.get_fignums())
        plt.figure()
        try:
            yield
        finally:
            for number in set(plt.get_fignums()) - previous:
                plt.close(number)


def _reference_colorbar_properties(eeglab_backend, request):
    if request.config.getoption("--eeglab-backend") == "matlab":
        count, ticks, labels, limits = eeglab_backend("eegprep_test_colorbar_graphics", "colorbar", nargout=4)
        assert count == 1
        return ticks, labels, limits
    # Matplotlib identifies actual colorbar axes with this label, while
    # EEGLAB uses axes Tag='cbar'. Query the drawn object, not its inputs.
    bars = [axis for axis in plt.gcf().axes if axis.get_label() == "<colorbar>"]
    assert len(bars) == 1
    bar = bars[0]
    return (
        np.asarray(bar.get_yticks())[None, :],
        np.array([[label.get_text()] for label in bar.get_yticklabels()], dtype=object),
        np.asarray(bar.get_ylim())[None, :],
    )


def _reference_check_scale(eeglab_backend, request, eeg, limits, components, signed):
    limits = np.array([limits], dtype=float)
    eeglab_backend(
        "pop_topoplot", eeg, 0.0, components, "Component", np.empty((0, 0)), 0.0, "maplimits", limits, nargout=0
    )
    ticks, labels, axis_range = _reference_colorbar_properties(eeglab_backend, request)
    assert np.all(np.isfinite(ticks)) and np.all(np.diff(ticks) > 0)
    if signed:
        assert_matlab_equal(labels, np.array([["-"], ["0"], ["+"]], dtype=object))
        mapped_zero = (
            limits[0, 0] + (ticks[0, 1] - axis_range[0, 0]) / np.diff(axis_range)[0, 0] * np.diff(limits)[0, 0]
        )
        assert abs(mapped_zero) <= 1e-12
    else:
        # str2double returns NaN for a nonnumeric tick label.
        values = np.array([[_reference_str2double(label)] for label in labels[:, 0]])
        assert np.all(np.isfinite(values))
        np.testing.assert_allclose(values[[0, -1]], limits.T, atol=1e-12, rtol=0)


def _reference_str2double(label):
    try:
        return float(label.replace("−", "-"))
    except ValueError:
        return np.nan


@pytest.mark.gui
@eeglab_test(_COLORBAR_SOURCE, "testPositiveLimits")
def test_reference_colorbar_positive(eeglab_backend, request, reference_colorbar_eeg, reference_colorbar_figures):
    _reference_check_scale(eeglab_backend, request, reference_colorbar_eeg, [1, 2], 2.0, False)


@pytest.mark.gui
@eeglab_test(_COLORBAR_SOURCE, "testNegativeLimits")
def test_reference_colorbar_negative(eeglab_backend, request, reference_colorbar_eeg, reference_colorbar_figures):
    _reference_check_scale(eeglab_backend, request, reference_colorbar_eeg, [-2, -1], 2.0, False)


@pytest.mark.gui
@eeglab_test(_COLORBAR_SOURCE, "testSymmetricLimits")
def test_reference_colorbar_symmetric(eeglab_backend, request, reference_colorbar_eeg, reference_colorbar_figures):
    _reference_check_scale(eeglab_backend, request, reference_colorbar_eeg, [-2, 2], 2.0, True)


@pytest.mark.gui
@eeglab_test(_COLORBAR_SOURCE, "testAsymmetricLimits")
def test_reference_colorbar_asymmetric(eeglab_backend, request, reference_colorbar_eeg, reference_colorbar_figures):
    _reference_check_scale(eeglab_backend, request, reference_colorbar_eeg, [-1, 3], 2.0, True)


@pytest.mark.gui
@eeglab_test(_COLORBAR_SOURCE, "testZeroLowerEndpoint")
def test_reference_colorbar_zero_lower(eeglab_backend, request, reference_colorbar_eeg, reference_colorbar_figures):
    _reference_check_scale(eeglab_backend, request, reference_colorbar_eeg, [0, 2], 2.0, False)


@pytest.mark.gui
@eeglab_test(_COLORBAR_SOURCE, "testZeroUpperEndpoint")
def test_reference_colorbar_zero_upper(eeglab_backend, request, reference_colorbar_eeg, reference_colorbar_figures):
    _reference_check_scale(eeglab_backend, request, reference_colorbar_eeg, [-2, 0], 2.0, False)


@pytest.mark.gui
@eeglab_test(_COLORBAR_SOURCE, "testMultiplePositiveMaps")
def test_reference_colorbar_multiple_positive(
    eeglab_backend, request, reference_colorbar_eeg, reference_colorbar_figures
):
    _reference_check_scale(eeglab_backend, request, reference_colorbar_eeg, [1, 2], np.array([[1.0, 2.0]]), False)


@pytest.mark.gui
@eeglab_test(_COLORBAR_SOURCE, "testMultipleSymmetricMaps")
def test_reference_colorbar_multiple_symmetric(
    eeglab_backend, request, reference_colorbar_eeg, reference_colorbar_figures
):
    _reference_check_scale(eeglab_backend, request, reference_colorbar_eeg, [-2, 2], np.array([[1.0, 2.0]]), True)


@pytest.mark.gui
@eeglab_test(_COLORBAR_SOURCE, "testDefaultLimits")
def test_reference_colorbar_default(eeglab_backend, request, reference_colorbar_eeg, reference_colorbar_figures):
    eeglab_backend("pop_topoplot", reference_colorbar_eeg, 0.0, 2.0, "Component", np.empty((0, 0)), 0.0, nargout=0)
    ticks, labels, _ = _reference_colorbar_properties(eeglab_backend, request)
    assert np.all(np.diff(ticks) > 0)
    assert_matlab_equal(labels, np.array([["-"], ["0"], ["+"]], dtype=object))


@pytest.mark.gui
@eeglab_test(_COLORBAR_SOURCE, "testZeroComponent")
def test_reference_colorbar_zero_component(eeglab_backend, request, reference_colorbar_eeg, reference_colorbar_figures):
    reference_colorbar_eeg["icawinv"][:, 1] = 0
    eeglab_backend("pop_topoplot", reference_colorbar_eeg, 0.0, 2.0, "Zero component", np.empty((0, 0)), 0.0, nargout=0)
    ticks, _, _ = _reference_colorbar_properties(eeglab_backend, request)
    assert np.all(np.diff(ticks) > 0)


@pytest.mark.gui
@eeglab_test(_COLORBAR_SOURCE, "testUnrelatedAxesUnchanged")
def test_reference_colorbar_unrelated_axes(eeglab_backend, request, reference_colorbar_eeg, reference_colorbar_figures):
    matlab = request.config.getoption("--eeglab-backend") == "matlab"
    if matlab:
        eeglab_backend("eegprep_test_colorbar_graphics", "unrelated_axes", nargout=0)
    else:
        other = plt.gcf().add_subplot()
        other.set_yticks([10, 20, 30])
        plt.figure()
    _reference_check_scale(eeglab_backend, request, reference_colorbar_eeg, [-2, 2], 2.0, True)
    ticks = (
        eeglab_backend("eegprep_test_colorbar_graphics", "unrelated_ticks")
        if matlab
        else np.asarray(other.get_yticks())[None, :]
    )
    assert_matlab_equal(ticks, np.array([[10.0, 20.0, 30.0]]))


def test_topoplot_blank_channel_locations_by_label_and_number():
    chanlocs = [
        {"labels": "Fz", "theta": 0, "radius": 0.3},
        {"labels": "Cz", "theta": 0, "radius": 0.0},
        {"labels": "Pz", "theta": 180, "radius": 0.3},
    ]

    label_fig, *_ = topoplot([], chanlocs, style="blank", electrodes="labelpoint")
    number_fig, *_ = topoplot([], chanlocs, style="blank", electrodes="numpoint")

    assert [text.get_text() for text in label_fig.axes[0].texts] == ["Fz", "Cz", "Pz"]
    assert [text.get_text() for text in number_fig.axes[0].texts] == ["1", "2", "3"]
    plt.close(label_fig)
    plt.close(number_fig)


def test_plot_channel_locations_returns_valid_console_command():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=20)

    figure, command = plot_channel_locations(eeg, mode="numbers", return_com=True)

    assert command == "topoplot([], EEG['chanlocs'], style='blank', electrodes='numpoint')"
    assert figure.axes[0].get_title() == "Channel locations"
    plt.close(figure)


def test_pop_topoplot_component_pages_scale_each_map_to_own_absmax():
    eeg = create_test_eeg_with_ica(n_channels=6, n_samples=30, n_components=2)
    eeg["icawinv"] = np.column_stack([np.arange(1, 7) * 1.0, np.arange(1, 7) * 10.0])

    figures = pop_topoplot(
        eeg,
        typeplot=0,
        items=[1, 2],
        topotitle="component scale",
        rowcols=[1, 2],
        electrodes="off",
    )

    expected = []
    for index in range(2):
        _, zi, *_ = topoplot(eeg["icawinv"][:, index], eeg["chanlocs"], noplot="on")
        limit = float(np.nanmax(np.abs(zi))) * 1.05  # topoplot widens the color axis by EEGLAB's 5% margin
        expected.append((-limit, limit))
    clims = [axis.images[0].get_clim() for axis in figures[0].axes[:2]]
    np.testing.assert_allclose(clims[0], expected[0], rtol=1e-6)
    np.testing.assert_allclose(clims[1], expected[1], rtol=1e-6)
    assert not np.allclose(clims[0], clims[1])
    assert len(figures[0].axes) == 3
    plt.close(figures[0])


def test_pop_topoplot_item_text_parses_eeglab_colon_ranges():
    assert _parse_items_text("-100:50:0") == [-100.0, -50.0, 0.0]
    assert _parse_items_text("0.5:0.25:1") == [0.5, 0.75, 1.0]
    tenth_steps = _parse_items_text("0:0.1:1")
    assert len(tenth_steps) == 11
    assert tenth_steps[-1] == 1.0
    parsed = _parse_items_text("1:2 NaN 5")
    assert parsed[:2] == [1.0, 2.0]
    assert np.isnan(parsed[2])
    assert parsed[3] == 5.0


def test_pop_topoplot_latency_positions_use_matlab_rounding():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=4)
    eeg["xmin"] = 0.0
    eeg["xmax"] = 3.0
    eeg["pnts"] = 4

    assert _latency_positions(eeg, np.array([500.0, 2500.0])).tolist() == [1, 3]


def test_pop_topoplot_gui_parses_eeglab_style_options():
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=20, n_components=4)

    class Renderer:
        def run(self, spec, initial_values=None):
            assert pop_topoplot_dialog_spec(eeg, typeplot=0).title == spec.title
            return {
                "items": "1:2",
                "topotitle": "components",
                "rowcols": "[1 2]",
                "plotdip": False,
                "options": "'electrodes', 'off', 'colorbar', 'off'",
            }

    figures, command = pop_topoplot(eeg, typeplot=0, renderer=Renderer(), return_com=True)

    assert len(figures) == 1
    assert "typeplot=0" in command
    assert "items=[1, 2]" in command
    assert "colorbar='off'" in command
    plt.close(figures[0])
