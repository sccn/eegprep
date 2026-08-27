"""Parity tests for the channel/component properties plot ``pop_prop``.

These lock EEGPrep's ``pop_prop`` to EEGLAB ``pop_prop.m``: a square figure with
three panels -- a scalp map, an ERP image (reusing ``erpimage``), and an activity
power spectrum. The numeric assertions are closed-form or derived directly from
EEGLAB source (no live MATLAB), so they run in CI:

* the spectrum is computed from the raw channel/component data, per-epoch averaged
  (``spectopo``), not from the epoch-flattened ERP-image trace;
* component spectra are scaled by the component map RMS power ``mapnorm``, which is
  a constant ``5*log10(mean(map**4))`` dB offset (EEGLAB ``spectopo.m``: linear
  power is multiplied by ``sqrt(mean(mapnorm.^4))`` before the ``10*log10``);
* the ERP image is fed the global-offset-subtracted data (EEGLAB ``nan_mean``).
"""

from __future__ import annotations

from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pytest

from eegprep.functions.popfunc.plot_utils import component_activations
from eegprep.functions.popfunc.pop_prop import pop_prop
from eegprep.functions.sigprocfunc.spectopo import compute_spectra
from eegprep.functions.sigprocfunc.topoplot import topo_screen_coords
from tests.fixtures import create_test_eeg, create_test_eeg_with_ica

SPECTRA_ATOL_DB = 1e-6


def _epoched_eeg() -> dict[str, Any]:
    """Small epoched dataset with ICA (>6 trials so EEGLAB uses ei_smooth=3)."""
    eeg = create_test_eeg_with_ica(n_channels=8, n_samples=128, srate=128.0, n_trials=10)
    eeg["xmin"] = -0.2
    eeg["xmax"] = -0.2 + (eeg["pnts"] - 1) / eeg["srate"]
    eeg["times"] = np.linspace(eeg["xmin"], eeg["xmax"], eeg["pnts"])
    return eeg


def _all_axes(fig: Any) -> list[Any]:
    """Every axes on a figure, recursing into subfigures."""
    axes = list(fig.axes)
    for subfig in getattr(fig, "subfigs", []):
        axes.extend(_all_axes(subfig))
    return axes


def _find_axis(fig: Any, predicate: Callable[[Any], bool]) -> Any:
    return next((ax for ax in _all_axes(fig) if predicate(ax)), None)


def _spectrum_axis(fig: Any) -> Any:
    ax = _find_axis(fig, lambda a: a.get_title().strip() == "Activity power spectrum")
    assert ax is not None, "no 'Activity power spectrum' panel found"
    return ax


def _image_axis(fig: Any) -> Any:
    ax = _find_axis(fig, lambda a: bool(a.get_images()) and a.get_ylabel() == "Trials")
    assert ax is not None, "no ERP-image panel (ylabel 'Trials') found"
    return ax


def _topo_axis(fig: Any) -> Any:
    ax = _find_axis(
        fig,
        lambda a: not a.get_images() and any(isinstance(c, PathCollection) for c in a.collections),
    )
    assert ax is not None, "no scalp-map panel with a channel marker found"
    return ax


def test_channel_property_has_three_eeglab_panels() -> None:
    eeg = _epoched_eeg()
    fig = pop_prop(eeg, 1, 3, plot="off")

    image_ax = _image_axis(fig)
    assert image_ax.get_ylabel() == "Trials"
    erp_ax = _find_axis(fig, lambda a: a.get_ylabel() in {"µV", "uV"})
    assert erp_ax is not None and erp_ax.get_xlabel() == "Time (ms)"

    spec_ax = _spectrum_axis(fig)
    assert spec_ax.get_xlabel() == "Frequency (Hz)"
    assert "Power" in spec_ax.get_ylabel()
    plt.close("all")


def test_component_property_titles_match_eeglab() -> None:
    eeg = _epoched_eeg()
    fig = pop_prop(eeg, 0, 2, plot="off")

    # EEGLAB puts "<basename> activity (global offset ...)" on the ERP image.
    assert _find_axis(fig, lambda a: "activity" in a.get_title().lower()) is not None
    assert _spectrum_axis(fig) is not None
    plt.close("all")


@pytest.mark.parity
def test_channel_spectrum_uses_raw_per_epoch_data() -> None:
    """The spectrum comes from spectopo on raw channel data, per-epoch averaged."""
    eeg = _epoched_eeg()
    channel = 3
    fig = pop_prop(eeg, 1, channel, spec_opt=None, plot="off")

    raw = eeg["data"][channel - 1 : channel, :, :]
    expected, _freqs, _ = compute_spectra(raw, eeg["pnts"], float(eeg["srate"]))

    ydata = np.asarray(_spectrum_axis(fig).lines[0].get_ydata(), dtype=float)
    np.testing.assert_allclose(ydata, expected[0], atol=SPECTRA_ATOL_DB)
    plt.close("all")


@pytest.mark.parity
def test_component_spectrum_applies_mapnorm_offset() -> None:
    """Component spectra are offset by 5*log10(mean(map**4)) dB (spectopo mapnorm)."""
    eeg = _epoched_eeg()
    comp = 2
    fig = pop_prop(eeg, 0, comp, spec_opt=None, plot="off")

    acts = component_activations(eeg)[comp - 1 : comp, :, :]
    base, _freqs, _ = compute_spectra(acts, eeg["pnts"], float(eeg["srate"]))
    mapnorm = np.asarray(eeg["icawinv"], dtype=float)[:, comp - 1]
    expected = base[0] + 5.0 * np.log10(np.mean(mapnorm**4))

    ydata = np.asarray(_spectrum_axis(fig).lines[0].get_ydata(), dtype=float)
    np.testing.assert_allclose(ydata, expected, atol=SPECTRA_ATOL_DB)
    plt.close("all")


@pytest.mark.parity
def test_erp_average_uses_global_offset_subtracted_data() -> None:
    """EEGLAB subtracts the global mean (nan_mean) before drawing the ERP image."""
    eeg = _epoched_eeg()
    channel = 4
    fig = pop_prop(eeg, 1, channel, plot="off")

    trace = eeg["data"][channel - 1]  # (pnts, trials)
    offset = np.nanmean(trace)
    expected_erp = np.nanmean(trace - offset, axis=1)

    erp_ax = _find_axis(fig, lambda a: a.get_ylabel() in {"µV", "uV"})
    assert erp_ax is not None
    ydata = np.asarray(erp_ax.lines[0].get_ydata(), dtype=float)
    np.testing.assert_allclose(ydata, expected_erp, atol=1e-9)
    plt.close("all")


def test_channel_topo_marks_only_the_selected_channel() -> None:
    """The scalp map marks exactly the selected channel (EEGLAB emarkersize1chan)."""
    eeg = _epoched_eeg()
    channel = 5
    fig = pop_prop(eeg, 1, channel, plot="off")

    topo_ax = _topo_axis(fig)
    markers = [c for c in topo_ax.collections if isinstance(c, PathCollection)]
    points = np.vstack([np.asarray(m.get_offsets(), dtype=float) for m in markers])
    assert points.shape[0] == 1, "exactly one channel should be marked"

    loc = eeg["chanlocs"][channel - 1]
    screen_x, screen_y = topo_screen_coords(float(loc["theta"]), float(loc["radius"]))
    np.testing.assert_allclose(points[0], [screen_x, screen_y], atol=1e-6)
    plt.close("all")


@pytest.mark.parity
def test_spectrum_y_axis_hugs_the_visible_frequency_band() -> None:
    """The spectrum y-limits track the data within the plotted band, not out-of-view freqs."""
    eeg = _epoched_eeg()
    fig = pop_prop(eeg, 0, 2, spec_opt="'freqrange', [2, 50]", plot="off")

    spec_ax = _spectrum_axis(fig)
    freqs = np.asarray(spec_ax.lines[0].get_xdata(), dtype=float)
    spectra = np.asarray(spec_ax.lines[0].get_ydata(), dtype=float)
    band = (freqs >= 2.0) & (freqs <= 50.0)
    band_lo, band_hi = float(spectra[band].min()), float(spectra[band].max())
    margin = (band_hi - band_lo) / 7.0

    y_lo, y_hi = spec_ax.get_ylim()
    np.testing.assert_allclose([y_lo, y_hi], [band_lo - margin, band_hi + margin], rtol=1e-9)
    plt.close("all")


def test_component_map_shows_electrode_markers() -> None:
    """The component scalp map keeps electrode dots on, like EEGLAB pop_prop."""
    eeg = _epoched_eeg()
    fig = pop_prop(eeg, 0, 2, plot="off")

    topo_ax = _find_axis(fig, lambda a: bool(a.get_images()) and a.get_title().startswith("IC"))
    assert topo_ax is not None, "no component scalp-map panel found"
    dots = [c for c in topo_ax.collections if isinstance(c, PathCollection)]
    marked = sum(len(np.asarray(c.get_offsets())) for c in dots)
    assert marked == int(eeg["nbchan"]), "component map should mark every electrode"
    plt.close("all")


def test_component_map_color_axis_is_symmetric_about_zero() -> None:
    """The component map uses maplimits='absmax' so 0 maps to the colormap center."""
    eeg = _epoched_eeg()
    fig = pop_prop(eeg, 0, 2, plot="off")

    topo_ax = _find_axis(fig, lambda a: bool(a.get_images()) and a.get_title().startswith("IC"))
    assert topo_ax is not None, "no component scalp-map panel found"
    vmin, vmax = topo_ax.get_images()[0].get_clim()
    assert vmax > 0
    np.testing.assert_allclose(vmin, -vmax, rtol=1e-9)
    plt.close("all")


def test_continuous_data_still_builds_an_erpimage() -> None:
    """Continuous (single-trial) data yields an ERP image, not a bare line plot."""
    eeg = create_test_eeg(n_channels=8, n_samples=4096, srate=128.0, n_trials=1)
    fig = pop_prop(eeg, 1, 2, plot="off")

    image_ax = _image_axis(fig)
    assert "continu" in image_ax.get_title().lower()
    assert _spectrum_axis(fig) is not None
    plt.close("all")


def test_return_com_history_command() -> None:
    eeg = _epoched_eeg()
    _fig, com = pop_prop(eeg, 1, 5, plot="off", return_com=True)
    assert com.startswith("pop_prop(EEG, 1, [5]")
    plt.close("all")
