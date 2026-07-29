from __future__ import annotations

import ast
from copy import deepcopy
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.io

from eegprep.functions.guifunc.qt import QtDialogRenderer
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.popfunc.pop_comperp import pop_comperp, pop_comperp_dialog_spec
from eegprep.functions.popfunc.pop_envtopo import pop_envtopo
from eegprep.functions.popfunc.pop_erpimage import pop_erpimage, pop_erpimage_dialog_spec
from eegprep.functions.popfunc.pop_headplot import (
    pop_headplot,
    pop_headplot_dialog_spec,
    _current_spline_file,
    _shared_maplimits,
)
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_epoch import pop_epoch
from eegprep.functions.popfunc.plot_utils import component_activations, parse_plot_options_text
from eegprep.functions.popfunc.pop_plotdata import pop_plotdata
from eegprep.functions.popfunc.pop_plottopo import pop_plottopo, pop_plottopo_dialog_spec
from eegprep.functions.popfunc.pop_prop import pop_prop, pop_prop_dialog_spec
from eegprep.functions.popfunc.pop_signalstat import pop_signalstat
from eegprep.functions.popfunc.pop_spectopo import pop_spectopo
from eegprep.functions.popfunc.pop_timtopo import pop_timtopo
from eegprep.functions.popfunc.pop_topoplot import pop_topoplot
from eegprep.functions.studyfunc.pop_chanplot import pop_chanplot, pop_chanplot_dialog_spec
from eegprep.functions.sigprocfunc.coregister import (
    ElectrodeSet,
    apply_coregistration_transform,
    coregister,
    estimate_coregistration_transform,
    load_coregistration_electrodes,
    match_electrodes,
    read_electrode_file,
    traditional_transform_matrix,
)
from eegprep.plugins.ICLabel.pop_viewprops import pop_viewprops
from eegprep.functions.sigprocfunc.headplot import (
    MAPLIMIT_PADDING,
    default_headplot_mesh_transform,
    headplot,
    headplot_setup,
    load_headplot_spline,
    packaged_headplot_path,
    _interpolate_values,
)
from tests.fixtures import SAMPLE_DATASET_PATH, create_test_eeg_with_ica


@pytest.fixture(scope="module")
def sample_eeg():
    return pop_loadset(SAMPLE_DATASET_PATH)


@pytest.fixture(scope="module")
def sample_epoch(sample_eeg):
    epoched, _command = pop_epoch(deepcopy(sample_eeg), ["square"], [-0.1, 0.2], return_com=True)
    return epoched


@pytest.fixture
def ica_epoch():
    return create_test_eeg_with_ica(n_channels=6, n_samples=40, n_trials=4, n_components=4)


def test_pop_spectopo_plots_sample_data_headlessly(sample_eeg):
    result, command = pop_spectopo(sample_eeg, dataflag=1, freqs=[6, 10], return_com=True)

    assert result["spectra"].shape[0] == sample_eeg["nbchan"]
    assert result["freqs"].ndim == 1
    assert np.isfinite(result["spectra"]).all()
    assert result["figure"] is not None
    assert "pop_spectopo(EEG" in command
    _assert_python_command(command)
    plt.close(result["figure"])


def test_pop_spectopo_component_default_controls_succeed(ica_epoch):
    result, command = pop_spectopo(
        ica_epoch, dataflag=0, freqs=[10], plotchan=0, icamode=True, icacomps=[1, 2], nicamaps=2, return_com=True
    )

    assert result["spectra"].shape[0] == 2
    assert "pop_spectopo(EEG" in command
    plt.close(result["figure"])


def test_pop_spectopo_channel_figure_structure(sample_eeg):
    freqs = [6, 10, 22]
    fig = pop_spectopo(sample_eeg, dataflag=1, freqs=freqs, gui=False)["figure"]

    spec_ax = next(ax for ax in fig.axes if "Frequency" in ax.get_xlabel())
    map_axes = [ax for ax in fig.axes if ax.images]
    assert len(map_axes) == len(freqs)
    assert any(ax.get_position().x0 > 0.9 for ax in fig.axes)

    marker_x = sorted(
        float(line.get_xdata()[0])
        for line in spec_ax.get_lines()
        if len({round(float(value), 6) for value in line.get_xdata()}) == 1
    )
    assert marker_x == pytest.approx([float(freq) for freq in freqs])
    assert fig.get_suptitle() == ""
    plt.close(fig)


def test_pop_spectopo_component_figure_marks_analysis_frequency(ica_epoch):
    """Component spectra draw a vertical marker at the analysis frequency, as EEGLAB does."""
    fig = pop_spectopo(
        ica_epoch, dataflag=0, freqs=[10], plotchan=0, icamode=True, icacomps=[1, 2], nicamaps=2, gui=False
    )["figure"]

    spec_ax = next(ax for ax in fig.axes if "Frequency" in ax.get_xlabel())
    verticals = sorted(
        float(line.get_xdata()[0])
        for line in spec_ax.get_lines()
        if len({round(float(value), 6) for value in line.get_xdata()}) == 1
    )
    assert verticals == pytest.approx([10.0])
    plt.close(fig)


def test_pop_spectopo_component_maps_labeled_by_index_with_composite(ica_epoch):
    """Component spectra show the composite power-at-frequency map plus the top-N
    component maps labeled by component index (EEGLAB), not a fixed IC 1..N row."""
    res = pop_spectopo(
        ica_epoch, dataflag=0, freqs=[10], plotchan=0, icamode=True, icacomps=[1, 2, 3, 4], nicamaps=2, gui=False
    )
    titles = [ax.get_title() for ax in res["figure"].axes if ax.get_title().strip()]
    assert sum("Hz" in title for title in titles) == 1
    comp_labels = [title for title in titles if "Hz" not in title]
    assert len(comp_labels) == 2
    assert all(title.isdigit() for title in comp_labels)  # component index labels, not the old "IC N"
    plt.close(res["figure"])


def test_pop_spectopo_component_maps_match_eeglab_selection_and_order():
    """On eeglab_data_epochs_ica.set the component-spectra maps reproduce EEGLAB: the
    top-nicamaps selection by projection-scaled power at 10 Hz, and the closestplot
    left-to-right order with the composite centered over the marker."""
    EEG = pop_loadset(str(SAMPLE_DATASET_PATH.parent / "eeglab_data_epochs_ica.set"))
    fig = pop_spectopo(
        EEG, dataflag=0, freqs=[10], freqrange=[2, 25], plotchan=0, percent=100, nicamaps=5, electrodes="off", gui=False
    )["figure"]
    map_titles = [
        ax.get_title()
        for ax in sorted((a for a in fig.axes if a.get_title().strip()), key=lambda a: a.get_position().x0)
    ]
    # one composite power-at-frequency map plus the five selected component maps
    assert map_titles.count("10.0 Hz") == 1
    assert sorted(int(title) for title in map_titles if "Hz" not in title) == [1, 4, 5, 6, 10]
    # closestplot arrangement: composite centered, components ordered around the marker
    assert map_titles == ["4", "6", "10.0 Hz", "10", "5", "1"]
    plt.close(fig)


def test_pop_spectopo_component_traces_report_index_on_click(ica_epoch, capsys):
    """Clicking a component trace prints its component index, like EEGLAB spectopo's
    per-trace ButtonDownFcn."""
    fig = pop_spectopo(
        ica_epoch, dataflag=0, freqs=[10], plotchan=0, icamode=True, icacomps=[1, 2, 3], nicamaps=2, gui=False
    )["figure"]
    spec_ax = next(ax for ax in fig.axes if "Frequency" in ax.get_xlabel())
    pickable = [line for line in spec_ax.get_lines() if line.get_picker()]
    for line in pickable:
        fig.canvas.callbacks.process("pick_event", SimpleNamespace(artist=line))
    printed = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("Component ")]
    # each component is drawn once, so clicking every trace reports each index exactly once
    # (a selected component drawn both thin and bold would print twice)
    assert sorted(int(ln.split()[1]) for ln in printed) == [1, 2, 3]
    plt.close(fig)


def test_pop_spectopo_channel_traces_report_index_on_click(sample_eeg, capsys):
    """Clicking a channel trace prints its channel index, like EEGLAB spectopo's
    per-trace ButtonDownFcn."""
    fig = pop_spectopo(sample_eeg, dataflag=1, freqs=[10], gui=False)["figure"]
    spec_ax = next(ax for ax in fig.axes if "Frequency" in ax.get_xlabel())
    pickable = [line for line in spec_ax.get_lines() if line.get_picker()]
    for line in pickable:
        fig.canvas.callbacks.process("pick_event", SimpleNamespace(artist=line))
    printed = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("Channel ")]
    assert sorted({int(ln.split()[1]) for ln in printed}) == list(range(1, int(sample_eeg["nbchan"]) + 1))
    plt.close(fig)


def test_pop_spectopo_epoched_data_averages_per_trial_pwelch():
    """Epoched (nchan, pnts, trials) input must equal per-trial welch averaged
    in linear power then converted to dB, matching EEGLAB spectopo. Prior code
    reshaped the Fortran-contiguous EEG array with numpy default C-order, which
    interleaved trials and produced tens of dB of error on real datasets.
    """
    from scipy.signal import get_window, welch as scipy_welch

    from eegprep.functions.popfunc.pop_loadset import pop_loadset as _pop_loadset
    from eegprep.functions.sigprocfunc.spectopo import spectopo

    EEG = _pop_loadset(str(Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data_epochs_ica.set"))
    py_spectra, py_freqs = spectopo(EEG["data"], EEG["pnts"], float(EEG["srate"]), plot="off")[:2]

    nperseg = min(round(float(EEG["srate"])), int(EEG["pnts"]))
    window = get_window("hamming", nperseg, fftbins=False)
    power_sum = None
    for trial in range(int(EEG["trials"])):
        ref_freqs, power = scipy_welch(
            EEG["data"][:, :, trial].astype(float),
            fs=float(EEG["srate"]),
            window=window,
            nperseg=nperseg,
            noverlap=0,
            nfft=None,
            axis=1,
            detrend=False,
            scaling="density",
        )
        power_sum = power if power_sum is None else power_sum + power
    ref_spectra = 10.0 * np.log10(power_sum / int(EEG["trials"]))

    np.testing.assert_allclose(py_freqs, ref_freqs, rtol=0, atol=1e-9)
    np.testing.assert_allclose(py_spectra, ref_spectra, rtol=0, atol=1e-10)


def test_pop_spectopo_rejects_nondefault_plotchan(ica_epoch):
    with pytest.raises(ValueError, match="whole-scalp component spectra"):
        pop_spectopo(ica_epoch, dataflag=0, freqs=[10], plotchan=3, icacomps=[1, 2])


def test_pop_spectopo_rejects_max_power_plotchan(ica_epoch):
    with pytest.raises(ValueError, match="whole-scalp component spectra"):
        pop_spectopo(ica_epoch, dataflag=0, freqs=[10], plotchan=[], icacomps=[1, 2])


def test_pop_spectopo_rejects_datacomp_icamode(ica_epoch):
    with pytest.raises(ValueError, match="component spectra"):
        pop_spectopo(ica_epoch, dataflag=0, freqs=[10], icamode=False, icacomps=[1, 2])


@pytest.mark.parametrize(
    "text, expected",
    [
        ("'electrodes', 'off'", {"electrodes": "off"}),
        ("electrodes='off'", {"electrodes": "off"}),
        ("electrodes='off', style='blank'", {"electrodes": "off", "style": "blank"}),
        ("'electrodes', 'off', 'style', 'blank'", {"electrodes": "off", "style": "blank"}),
        ("electrodes='off', 'style', 'blank'", {"electrodes": "off", "style": "blank"}),
        ("gridscale=32", {"gridscale": 32}),
        ("maplimits=[-5, 5]", {"maplimits": [-5.0, 5.0]}),
    ],
)
def test_parse_plot_options_text_accepts_python_and_matlab_styles(text, expected):
    assert parse_plot_options_text(text) == expected


def test_pop_spectopo_gui_options_field_accepts_python_style(sample_eeg):
    class Renderer:
        def run(self, spec, initial_values=None):
            return {
                "timerange": "-1000 1992.19",
                "percent": "100",
                "freqs": "6 10 22",
                "process": "EEG",
                "freqrange": "2 25",
                "options": "electrodes='off'",
            }

    result, command = pop_spectopo(deepcopy(sample_eeg), gui=True, renderer=Renderer(), return_com=True)

    assert result["figure"] is not None
    assert "electrodes='off'" in command
    _assert_python_command(command)
    plt.close(result["figure"])


def test_pop_prop_plots_sample_channel_properties(sample_eeg):
    figure, command = pop_prop(sample_eeg, typecomp=1, chanorcomp=1, return_com=True)

    assert len(figure.axes) >= 3
    assert "pop_prop(EEG" in command
    _assert_python_command(command)
    plt.close(figure)


def test_pop_headplot_plots_sample_latency_map_with_spline_setup(sample_eeg, tmp_path):
    eeg = deepcopy(sample_eeg)
    splinefile = tmp_path / "sample.spl"
    setup = {"splinefile": str(splinefile), "transform": [0, -10, 0, -0.1, 0, -1.6, 1100, 1100, 1100]}

    figures, command = pop_headplot(eeg, typeplot=1, items=[0], setup=setup, return_com=True)

    assert len(figures) == 1
    assert figures[0].axes[0].name == "3d"
    assert splinefile.exists()
    assert _current_spline_file(eeg, 1) != str(splinefile)  # plot must not mutate the caller's EEG
    assert "setup={" in command
    _assert_python_command(command)
    plt.close(figures[0])


def test_pop_headplot_does_not_mutate_caller_eeg(sample_eeg, tmp_path):
    eeg = deepcopy(sample_eeg)
    before = deepcopy(eeg)
    setup = {
        "splinefile": str(tmp_path / "nomutate.spl"),
        "transform": [0, -10, 0, -0.1, 0, -1.6, 1100, 1100, 1100],
    }

    figures = pop_headplot(eeg, typeplot=1, items=[0], setup=setup)

    assert set(eeg.keys()) == set(before.keys())  # no new spline/mesh keys added
    assert np.array_equal(np.asarray(eeg["splinefile"]), np.asarray(before["splinefile"]))
    assert "headplotmeshfile" not in eeg
    assert np.array_equal(np.asarray(eeg["data"]), np.asarray(before["data"]))
    for fig in figures:
        plt.close(fig)


def test_pop_headplot_single_map_has_eeglab_like_title_and_surface(sample_eeg, tmp_path):
    eeg = deepcopy(sample_eeg)
    title = "ERP scalp maps of dataset: eeglab_data"
    splinefile = tmp_path / "single_map.spl"

    figures, _command = pop_headplot(
        eeg,
        typeplot=1,
        items=[0],
        topotitle=title,
        setup={"splinefile": str(splinefile), "transform": [0, -10, 0, -0.1, 0, -1.6, 1100, 1100, 1100]},
        colorbar="off",
        return_com=True,
    )

    figure = figures[0]
    assert figure._suptitle is not None
    assert figure._suptitle.get_text() == title
    assert figure.axes[0].get_title() == ""
    width, height = figure.get_size_inches()
    assert width >= 8.0
    assert height >= 6.0
    assert figure.axes[0].get_position().width > 0.6
    facecolors = np.asarray(figure.axes[0].collections[0].get_facecolors())
    assert np.isfinite(facecolors).all()
    assert np.all(facecolors[:, 3] == 1)
    plt.close(figure)


def test_headplot_setup_file_can_be_reused_for_sample_data(sample_eeg, tmp_path):
    splinefile = tmp_path / "reuse.spl"
    transform = [0, -10, 0, -0.1, 0, -1.6, 1100, 1100, 1100]

    created = headplot_setup(sample_eeg["chanlocs"], splinefile, chaninfo=sample_eeg["chaninfo"], transform=transform)
    spline = load_headplot_spline(created)
    figure = headplot(np.nanmean(np.asarray(sample_eeg["data"], dtype=float), axis=1), created, title="Sample")

    assert spline.g.shape == (sample_eeg["nbchan"], sample_eeg["nbchan"])
    assert spline.gx.shape[1] == sample_eeg["nbchan"]
    assert len(figure.axes) >= 1
    plt.close(figure)


def test_headplot_setup_plotmeshonly_and_orilocs_options(sample_eeg, tmp_path):
    transform = [0, -10, 0, -0.1, 0, -1.6, 1100, 1100, 1100]
    preview_file = tmp_path / "preview.spl"

    preview = headplot_setup(
        sample_eeg["chanlocs"],
        preview_file,
        chaninfo=sample_eeg["chaninfo"],
        transform=transform,
        plotmeshonly="sphere",
    )

    assert preview.axes[0].name == "3d"
    assert not preview_file.exists()
    plt.close(preview)

    created = headplot_setup(
        sample_eeg["chanlocs"],
        tmp_path / "orilocs.spl",
        chaninfo=sample_eeg["chaninfo"],
        transform=transform,
        orilocs="on",
    )
    spline = load_headplot_spline(created)
    np.testing.assert_allclose(spline.new_electrodes, np.column_stack([spline.xe, spline.ye, spline.ze]))


def test_pop_headplot_setup_reuses_existing_spline_file(sample_eeg, tmp_path):
    eeg = deepcopy(sample_eeg)
    splinefile = tmp_path / "existing.spl"
    original_transform = [0, -10, 0, -0.1, 0, -1.6, 1100, 1100, 1100]
    replay_transform = [1, 2, 3, 0.1, 0.2, 0.3, 4, 5, 6]
    headplot_setup(sample_eeg["chanlocs"], splinefile, chaninfo=sample_eeg["chaninfo"], transform=original_transform)

    figures, command = pop_headplot(
        eeg,
        typeplot=1,
        items=[0],
        setup={"splinefile": str(splinefile), "transform": replay_transform},
        return_com=True,
    )

    np.testing.assert_allclose(load_headplot_spline(splinefile).transform, original_transform)
    assert "setup={" in command
    plt.close(figures[0])


def test_headplot_setup_falls_back_when_xyz_fields_are_empty_arrays(sample_eeg, tmp_path):
    eeg = deepcopy(sample_eeg)
    eeg["chanlocs"][0]["X"] = np.asarray([])
    eeg["chanlocs"][0]["Y"] = np.asarray([])
    eeg["chanlocs"][0]["Z"] = np.asarray([])

    created = headplot_setup(
        eeg["chanlocs"],
        tmp_path / "empty_xyz.spl",
        chaninfo=eeg["chaninfo"],
        transform=[0, -10, 0, -0.1, 0, -1.6, 1100, 1100, 1100],
    )

    assert created.exists()


def test_coregister_reads_packaged_reference_and_matches_sample_labels(sample_eeg):
    source = load_coregistration_electrodes(sample_eeg["chanlocs"], chaninfo=sample_eeg["chaninfo"])
    target = read_electrode_file(packaged_headplot_path("mheadnew.xyz"))

    source_indices, target_indices, labels = match_electrodes(source, target)
    transform = estimate_coregistration_transform(source, target, method="traditional")
    transformed = apply_coregistration_transform(source.points[source_indices], transform)

    assert len(labels) >= 20
    assert source_indices.shape == target_indices.shape
    assert transform.shape == (9,)
    assert np.isfinite(transformed).all()


def test_coregister_sample_warp_reports_positive_resize_fields(sample_eeg):
    source = load_coregistration_electrodes(sample_eeg["chanlocs"], chaninfo=sample_eeg["chaninfo"])
    target = read_electrode_file(packaged_headplot_path("mheadnew.xyz"))

    transform = estimate_coregistration_transform(source, target, method="traditional")

    assert np.all(transform[6:9] > 0)
    assert np.all(np.abs(transform[3:6]) <= np.pi)


def test_coregister_fits_traditional_and_shared_scale_transforms():
    source = ElectrodeSet(
        ["Nz", "LPA", "RPA", "Cz", "Pz"],
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, -0.2],
                [1.0, 0.0, -0.2],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=float,
        ),
    )
    traditional = np.asarray([5.0, -3.0, 2.0, 0.05, -0.04, 0.03, 2.0, 1.5, 1.2])
    target = ElectrodeSet(source.labels, apply_coregistration_transform(source.points, traditional))

    fitted = estimate_coregistration_transform(source, target, method="traditional")
    shared = estimate_coregistration_transform(
        source,
        ElectrodeSet(
            source.labels, apply_coregistration_transform(source.points, [1, 2, 3, 0.02, 0.01, -0.02, 2, 2, 2])
        ),
        method="globalrescale",
    )

    np.testing.assert_allclose(apply_coregistration_transform(source.points, fitted), target.points, atol=1e-6)
    np.testing.assert_allclose(shared[6], shared[7])
    np.testing.assert_allclose(shared[7], shared[8])

    with pytest.raises(ValueError, match="traditional.*globalrescale"):
        estimate_coregistration_transform(source, target, method="nonlin")


def test_coregister_noninteractive_path_returns_transformed_electrodes(sample_eeg):
    result = coregister(
        sample_eeg["chanlocs"],
        packaged_headplot_path("mheadnew.xyz"),
        chaninfo1=sample_eeg["chaninfo"],
        warp="auto",
    )

    assert result.electrodes.points.shape[1] == 3
    assert result.transform.shape == (9,)
    assert np.isfinite(result.electrodes.points).all()


def test_pop_headplot_component_path_creates_ica_spline(ica_epoch, tmp_path):
    eeg = deepcopy(ica_epoch)
    for index, chanloc in enumerate(eeg["chanlocs"]):
        chanloc.setdefault("X", float(index - 2))
        chanloc.setdefault("Y", float(index % 3))
        chanloc.setdefault("Z", float(30 + index))
    splinefile = tmp_path / "component.spl"

    figures, command = pop_headplot(
        eeg,
        typeplot=0,
        items=[1, -2],
        setup={"splinefile": str(splinefile), "transform": [0, -10, 0, -0.1, 0, -1.6, 1100, 1100, 1100]},
        return_com=True,
    )

    assert len(figures) == 1
    assert _current_spline_file(eeg, 0) != str(splinefile)  # plot must not mutate the caller's EEG
    assert "IC 2" in figures[0].axes[1].get_title()
    _assert_python_command(command)
    plt.close(figures[0])


def test_pop_headplot_component_path_skips_sample_channels_without_locations(tmp_path):
    eeg = pop_loadset(SAMPLE_DATASET_PATH.parent / "eeglab_data_with_ica_tmp.set")
    splinefile = tmp_path / "sample_component.spl"

    figures, command = pop_headplot(
        eeg,
        typeplot=0,
        items=[1],
        setup={"splinefile": str(splinefile), "transform": [0, -10, 0, -0.1, 0, -1.6, 1100, 1100, 1100]},
        return_com=True,
    )
    spline = load_headplot_spline(splinefile)

    assert len(figures) == 1
    assert spline.xe.size < eeg["nbchan"]
    assert np.max(spline.indices) < np.asarray(eeg["icawinv"]).shape[0]
    _assert_python_command(command)
    plt.close(figures[0])


def test_pop_headplot_shared_string_maplimits_are_replayable(ica_epoch, tmp_path):
    eeg = deepcopy(ica_epoch)
    for index, chanloc in enumerate(eeg["chanlocs"]):
        chanloc.setdefault("X", float(index - 2))
        chanloc.setdefault("Y", float(index % 3))
        chanloc.setdefault("Z", float(30 + index))
    splinefile = tmp_path / "maplimits.spl"

    figures, command = pop_headplot(
        eeg,
        typeplot=0,
        items=[1, 2],
        setup={"splinefile": str(splinefile), "transform": [0, -10, 0, -0.1, 0, -1.6, 1100, 1100, 1100]},
        maplimits="absmax",
        return_com=True,
    )

    assert len(figures) == 1
    assert "maplimits='absmax'" in command
    _assert_python_command(command)
    plt.close(figures[0])


def test_headplot_maplimits_use_eeglab_padding(ica_epoch):
    values = np.asarray(ica_epoch["icawinv"], dtype=float)[:, 0]
    expected = float(np.nanmax(np.abs(values)) * MAPLIMIT_PADDING)
    np.testing.assert_allclose(_shared_maplimits([values], "absmax"), [-expected, expected])


def test_pop_headplot_gui_setup_result_is_replayable(sample_eeg, tmp_path):
    class Renderer:
        def __init__(self):
            self.spec = None

        def run(self, spec, initial_values=None):
            self.spec = spec
            return {
                "loadcb": False,
                "compcb": True,
                "setup_file": str(tmp_path / "gui.spl"),
                "meshfile": 1,
                "meshchanfile": 1,
                "transform": "0 -10 0 -0.1 0 -1.6 1100 1100 1100",
                "items": "0",
                "topotitle": "GUI setup",
                "rowcols": "",
                "options": "'electrodes', 'off'",
            }

    eeg = deepcopy(sample_eeg)
    renderer = Renderer()
    figures, command = pop_headplot(eeg, typeplot=1, gui=True, renderer=renderer, return_com=True)

    assert renderer.spec.title == "ERP head plot(s) -- pop_headplot()"
    assert (tmp_path / "gui.spl").exists()
    assert _current_spline_file(eeg, 1) == ""  # plot must not mutate the caller's EEG
    assert "setup={" in command
    assert "electrodes='off'" in command
    _assert_python_command(command)
    replay_namespace = {"EEG": deepcopy(sample_eeg), "pop_headplot": pop_headplot}
    exec(command, replay_namespace)
    plt.close(figures[0])
    plt.close("all")


@pytest.mark.matlab
def test_headplot_setup_spline_metadata_matches_eeglab(sample_eeg, tmp_path):
    if os.environ.get("EEGPREP_SKIP_MATLAB") == "1":
        pytest.skip("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
    try:
        matlab_engine = importlib.import_module("matlab.engine")
    except ImportError as exc:
        pytest.skip(f"MATLAB not available: {exc}")
    eeglab_root = _eeglab_reference_root()
    if not eeglab_root.exists():
        pytest.skip("EEGLAB reference checkout not available")

    transform = [0, -10, 0, -0.1, 0, -1.6, 1100, 1100, 1100]
    python_spline = tmp_path / "python.spl"
    matlab_spline = tmp_path / "matlab.spl"
    matlab_output = tmp_path / "headplot_setup.mat"
    headplot_setup(sample_eeg["chanlocs"], python_spline, chaninfo=sample_eeg["chaninfo"], transform=transform)

    engine = matlab_engine.start_matlab()
    try:
        for relative in (
            "functions/guifunc",
            "functions/popfunc",
            "functions/adminfunc",
            "functions/sigprocfunc",
            "functions/miscfunc",
            "functions/supportfiles",
            "plugins/dipfit",
        ):
            engine.addpath(str(eeglab_root / relative), nargout=0)
        engine.eval(
            f"""
            EEG = pop_loadset('{_matlab_string(SAMPLE_DATASET_PATH)}');
            headplot('setup', EEG.chanlocs, '{_matlab_string(matlab_spline)}', ...
                'chaninfo', EEG.chaninfo, 'meshfile', 'mheadnew.mat', ...
                'transform', [{_matlab_vector(transform)}]);
            S = load('{_matlab_string(matlab_spline)}', '-mat');
            G = S.G; gx = S.gx; Xe = S.Xe; Ye = S.Ye; Ze = S.Ze; newElect = S.newElect;
            indices = S.indices; transform = S.transform;
            save('{_matlab_string(matlab_output)}', 'G', 'gx', 'Xe', 'Ye', 'Ze', 'newElect', 'indices', 'transform');
            """,
            nargout=0,
        )
    finally:
        engine.quit()

    py_spline = load_headplot_spline(python_spline)
    ml = scipy.io.loadmat(matlab_output, squeeze_me=True)
    np.testing.assert_allclose(py_spline.g, ml["G"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(py_spline.gx, ml["gx"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(py_spline.xe, ml["Xe"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(py_spline.ye, ml["Ye"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(py_spline.ze, ml["Ze"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(py_spline.new_electrodes, ml["newElect"], rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(py_spline.indices + 1, np.asarray(ml["indices"], dtype=int).ravel())
    np.testing.assert_allclose(py_spline.transform, np.asarray(ml["transform"], dtype=float).ravel())


@pytest.mark.matlab
def test_headplot_interpolated_values_match_eeglab(sample_eeg, tmp_path):
    if os.environ.get("EEGPREP_SKIP_MATLAB") == "1":
        pytest.skip("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
    try:
        matlab_engine = importlib.import_module("matlab.engine")
    except ImportError as exc:
        pytest.skip(f"MATLAB not available: {exc}")
    eeglab_root = _eeglab_reference_root()
    if not eeglab_root.exists():
        pytest.skip("EEGLAB reference checkout not available")

    transform = [0, -10, 0, -0.1, 0, -1.6, 1100, 1100, 1100]
    values = np.linspace(-4.0, 6.0, int(sample_eeg["nbchan"]))
    python_spline = tmp_path / "python_interp.spl"
    matlab_spline = tmp_path / "matlab_interp.spl"
    matlab_output = tmp_path / "headplot_interp.mat"
    headplot_setup(sample_eeg["chanlocs"], python_spline, chaninfo=sample_eeg["chaninfo"], transform=transform)

    engine = matlab_engine.start_matlab()
    try:
        for relative in (
            "functions/guifunc",
            "functions/popfunc",
            "functions/adminfunc",
            "functions/sigprocfunc",
            "functions/miscfunc",
            "functions/supportfiles",
            "plugins/dipfit",
        ):
            engine.addpath(str(eeglab_root / relative), nargout=0)
        engine.eval(
            f"""
            EEG = pop_loadset('{_matlab_string(SAMPLE_DATASET_PATH)}');
            headplot('setup', EEG.chanlocs, '{_matlab_string(matlab_spline)}', ...
                'chaninfo', EEG.chaninfo, 'meshfile', 'mheadnew.mat', ...
                'transform', [{_matlab_vector(transform)}]);
            S = load('{_matlab_string(matlab_spline)}', '-mat');
            values = [{_matlab_vector(values)}]';
            meanval = mean(values);
            centered = values - meanval;
            enum = length(values);
            lamd = 0.1;
            C = pinv([(S.G + lamd); ones(1, enum)]) * [centered(:); 0];
            P = S.gx * C + meanval;
            save('{_matlab_string(matlab_output)}', 'P');
            """,
            nargout=0,
        )
    finally:
        engine.quit()

    py_values = _interpolate_values(values, load_headplot_spline(python_spline))
    ml = scipy.io.loadmat(matlab_output, squeeze_me=True)
    np.testing.assert_allclose(py_values, np.asarray(ml["P"], dtype=float).ravel(), rtol=1e-10, atol=1e-10)


@pytest.mark.matlab
def test_headplot_setup_ica_metadata_matches_eeglab(tmp_path):
    if os.environ.get("EEGPREP_SKIP_MATLAB") == "1":
        pytest.skip("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
    try:
        matlab_engine = importlib.import_module("matlab.engine")
    except ImportError as exc:
        pytest.skip(f"MATLAB not available: {exc}")
    eeglab_root = _eeglab_reference_root()
    if not eeglab_root.exists():
        pytest.skip("EEGLAB reference checkout not available")

    sample_ica = SAMPLE_DATASET_PATH.parent / "eeglab_data_with_ica_tmp.set"
    transform = [0, -10, 0, -0.1, 0, -1.6, 1100, 1100, 1100]
    eeg = pop_loadset(sample_ica)
    python_spline = tmp_path / "python_ica.spl"
    matlab_spline = tmp_path / "matlab_ica.spl"
    matlab_output = tmp_path / "headplot_ica.mat"
    headplot_setup(eeg["chanlocs"], python_spline, chaninfo=eeg["chaninfo"], transform=transform, ica="on")

    engine = matlab_engine.start_matlab()
    try:
        for relative in (
            "functions/guifunc",
            "functions/popfunc",
            "functions/adminfunc",
            "functions/sigprocfunc",
            "functions/miscfunc",
            "functions/supportfiles",
            "plugins/dipfit",
        ):
            engine.addpath(str(eeglab_root / relative), nargout=0)
        engine.eval(
            f"""
            EEG = pop_loadset('{_matlab_string(sample_ica)}');
            headplot('setup', EEG.chanlocs, '{_matlab_string(matlab_spline)}', ...
                'chaninfo', EEG.chaninfo, 'ica', 'on', 'meshfile', 'mheadnew.mat', ...
                'transform', [{_matlab_vector(transform)}]);
            S = load('{_matlab_string(matlab_spline)}', '-mat');
            G = S.G; gx = S.gx; Xe = S.Xe; indices = S.indices; transform = S.transform;
            save('{_matlab_string(matlab_output)}', 'G', 'gx', 'Xe', 'indices', 'transform');
            """,
            nargout=0,
        )
    finally:
        engine.quit()

    py_spline = load_headplot_spline(python_spline)
    ml = scipy.io.loadmat(matlab_output, squeeze_me=True)
    assert py_spline.g.shape == np.asarray(ml["G"]).shape
    assert py_spline.gx.shape == np.asarray(ml["gx"]).shape
    np.testing.assert_allclose(py_spline.xe, np.asarray(ml["Xe"], dtype=float).ravel(), rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(py_spline.indices + 1, np.asarray(ml["indices"], dtype=int).ravel())
    np.testing.assert_allclose(py_spline.transform, np.asarray(ml["transform"], dtype=float).ravel())


@pytest.mark.matlab
def test_traditional_transform_matrix_matches_eeglab(tmp_path):
    if os.environ.get("EEGPREP_SKIP_MATLAB") == "1":
        pytest.skip("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
    try:
        matlab_engine = importlib.import_module("matlab.engine")
    except ImportError as exc:
        pytest.skip(f"MATLAB not available: {exc}")
    eeglab_root = _eeglab_reference_root()
    if not eeglab_root.exists():
        pytest.skip("EEGLAB reference checkout not available")

    transform = [5, -3, 2, 0.05, -0.04, 0.03, 2, 1.5, 1.2]
    matlab_output = tmp_path / "traditionaldipfit.mat"
    engine = matlab_engine.start_matlab()
    try:
        engine.addpath(str(eeglab_root / "plugins" / "dipfit"), nargout=0)
        engine.eval(
            f"""
            H = traditionaldipfit([{_matlab_vector(transform)}]);
            save('{_matlab_string(matlab_output)}', 'H');
            """,
            nargout=0,
        )
    finally:
        engine.quit()

    ml = scipy.io.loadmat(matlab_output, squeeze_me=True)
    np.testing.assert_allclose(traditional_transform_matrix(transform), ml["H"], rtol=1e-12, atol=1e-12)


def test_channel_erp_plot_wrappers_work_on_sample_epochs(sample_epoch):
    timtopo_fig, timtopo_command = pop_timtopo(sample_epoch, plottimes=[0], return_com=True)
    plottopo_fig, plottopo_command = pop_plottopo(sample_epoch, chans=[1, 2], return_com=True)
    erpimage_result, erpimage_command = pop_erpimage(sample_epoch, typeplot=1, index=1, return_com=True)

    assert len(timtopo_fig.axes) >= 2
    assert len(plottopo_fig.axes) >= 2
    assert erpimage_result["image"].shape[0] == sample_epoch["trials"]
    for command in (timtopo_command, plottopo_command, erpimage_command):
        _assert_python_command(command)
    plt.close(timtopo_fig)
    plt.close(plottopo_fig)
    plt.close(erpimage_result["figure"])


def test_component_plot_wrappers_work_when_ica_fields_exist(ica_epoch):
    spectopo_result, spectopo_command = pop_spectopo(ica_epoch, dataflag=0, freqs=[10], return_com=True)
    plotdata_fig, plotdata_command = pop_plotdata(ica_epoch, components=[1, 2], return_com=True)
    envtopo_fig, envtopo_command = pop_envtopo(ica_epoch, components=[1, 2], return_com=True)
    erpimage_result, erpimage_command = pop_erpimage(ica_epoch, typeplot=0, index=1, return_com=True)

    assert spectopo_result["spectra"].shape[0] == 4
    assert len(plotdata_fig.axes) >= 2
    assert len(envtopo_fig.axes) >= 2
    assert erpimage_result["image"].shape[0] == 4
    for command in (spectopo_command, plotdata_command, envtopo_command, erpimage_command):
        _assert_python_command(command)
    plt.close(spectopo_result["figure"])
    plt.close(plotdata_fig)
    plt.close(envtopo_fig)
    plt.close(erpimage_result["figure"])


def test_pop_prop_attaches_component_activity_browser_model(ica_epoch):
    figure, command = pop_prop(ica_epoch, typecomp=0, chanorcomp=1, return_com=True)

    activity = figure.eegprep_activity_view

    assert activity.data.mode == "epoched"
    assert activity.data.n_channels == 1
    assert activity.state.title == "Scrolling IC1 Activity -- eegplot()"
    assert "pop_prop(EEG" in command
    plt.close(figure)


def test_pop_viewprops_attaches_channel_activity_browser_models(ica_epoch):
    eeg = deepcopy(ica_epoch)
    eeg["event"] = [{"type": "stim", "latency": 2}]

    figures, command = pop_viewprops(eeg, typecomp=1, chanorcomp=[1], scroll_event=0, return_com=True)
    activity = figures[0].eegprep_activity_views[0]

    assert activity.data.mode == "epoched"
    assert activity.data.channel_labels == ("Ch1",)
    assert activity.state.events == []
    assert "pop_viewprops(EEG" in command
    plt.close(figures[0])


def test_pop_erpimage_projects_components_to_selected_channel(ica_epoch):
    result, command = pop_erpimage(ica_epoch, typeplot=0, index=1, projchan=[2], return_com=True)

    expected = component_activations(ica_epoch)[0].T * np.asarray(ica_epoch["icawinv"], dtype=float)[1, 0]
    np.testing.assert_allclose(result["image"], expected)
    assert "projchan=[2]" in command
    _assert_python_command(command)
    plt.close(result["figure"])


def test_phase4_dialog_specs_match_eeglab_selector_layouts(sample_eeg, ica_epoch):
    prop_controls = controls_by_tag(pop_prop_dialog_spec(sample_eeg, typecomp=1))
    assert prop_controls["chanorcomp_button"].callback is not None
    assert prop_controls["chanorcomp_button"].callback.params["return_indices"] is True

    erpimage_spec = pop_erpimage_dialog_spec(ica_epoch, typeplot=0)
    assert erpimage_spec.size == (1113, 831)
    assert erpimage_spec.row_spacing == 4
    assert erpimage_spec.geometry[0] == (1, 1, 0.1, 0.8, 2.1)
    assert erpimage_spec.geometry[1] == (1, 1, 0.4, 0.5, 2.1)
    assert [control.tag for control in erpimage_spec.controls[:10]] == [
        None,
        "index",
        None,
        None,
        None,
        None,
        "projchan",
        "projchan_button",
        None,
        "title",
    ]

    comperp_spec = pop_comperp_dialog_spec([sample_eeg], flag=1)
    assert comperp_spec.row_spacing == 8
    assert [control.string for control in comperp_spec.controls[2:5]] == ["avg.", "std.", "all ERPs"]
    assert comperp_spec.geometry[0] == comperp_spec.geometry[1]
    assert comperp_spec.geometry[-1] == (1.48, 1.03, 1)

    plottopo_controls = controls_by_tag(pop_plottopo_dialog_spec(sample_eeg))
    assert plottopo_controls["rect"].value is False
    assert plottopo_controls["options"].value == "'ydir', -1"

    chanplot_controls = controls_by_tag(pop_chanplot_dialog_spec({"name": "demo study"}, [sample_eeg]))
    assert chanplot_controls["chan_list"].string.startswith("All channels|")
    assert chanplot_controls["chan_onechan"].string == "All subjects"
    assert chanplot_controls["plot_chan_erp"].callback is not None
    assert chanplot_controls["plot_chan_erp"].callback.params["value"] == "erp"
    assert chanplot_controls["plot_chan_erpimage"].enabled is False

    headplot_controls = controls_by_tag(pop_headplot_dialog_spec(sample_eeg, typeplot=1))
    headplot_spec = pop_headplot_dialog_spec(sample_eeg, typeplot=1)
    assert headplot_spec.size == (1290, 547)
    assert headplot_spec.content_margins == (42, 35, 42, 13)
    assert headplot_controls["loadcb"].callback.name == "headplot_setup_mode"
    assert headplot_controls["compcb"].callback.name == "headplot_setup_mode"
    assert headplot_controls["setup_browse"].callback.name == "select_file"
    np.testing.assert_allclose(
        default_headplot_mesh_transform("mheadnew.mat"), [0, -10, 0, -0.1, 0, -1.6, 1100, 1100, 1100]
    )
    assert headplot_controls["meshfile"].callback.params["transform_choices"] == (
        "0 -10 0 -0.1 0 -1.6 1100 1100 1100",
        "0 -15 -15 0.05 0 -1.57 100 88 110",
    )
    assert headplot_controls["mesh_browse"].callback.params["custom_transform"] == "0 -10 0 -0.1 0 -1.6 1100 1100 1100"
    assert headplot_controls["manual_coreg"].callback.name == "headplot_manual_coreg"
    assert headplot_controls["manual_coreg"].callback.params["mesh_source"] == "meshfile"
    assert headplot_controls["transform"].enabled is True


def test_headplot_setup_mode_checkboxes_behave_like_radio_buttons():
    class Toggle:
        def __init__(self, checked):
            self.checked = checked
            self.signals_blocked = False

        def isChecked(self):
            return self.checked

        def setChecked(self, value):
            self.checked = bool(value)

        def blockSignals(self, value):
            self.signals_blocked = bool(value)

        def setEnabled(self, _enabled):
            pass

    load = Toggle(False)
    comp = Toggle(True)
    widgets = {
        "loadcb": load,
        "compcb": comp,
        "load": Toggle(False),
        "setup_file": Toggle(True),
    }

    QtDialogRenderer._set_headplot_setup_mode(
        widgets,
        {
            "source": "compcb",
            "mode": "compute",
            "peer": "loadcb",
            "load_targets": ("load",),
            "setup_targets": ("setup_file",),
        },
        False,
    )

    assert comp.isChecked() is True
    assert load.isChecked() is False


def test_headplot_mesh_choice_restores_each_template_transform(sample_eeg):
    class Text:
        def __init__(self):
            self.value = ""

        def setText(self, value):
            self.value = value

    class Combo:
        def __init__(self):
            self.index = 0

        def setCurrentIndex(self, index):
            self.index = index

        def count(self):
            return 2

    spec = pop_headplot_dialog_spec(sample_eeg, typeplot=1)
    params = controls_by_tag(spec)["meshfile"].callback.params
    transform = Text()
    reference = Combo()

    QtDialogRenderer._set_headplot_mesh_choice(
        {"meshchanfile": reference, "transform": transform},
        params,
        1,
    )
    assert transform.value == "0 -15 -15 0.05 0 -1.57 100 88 110"
    assert reference.index == 1

    QtDialogRenderer._set_headplot_mesh_choice(
        {"meshchanfile": reference, "transform": transform},
        params,
        0,
    )
    assert transform.value == "0 -10 0 -0.1 0 -1.6 1100 1100 1100"
    assert reference.index == 0


def test_headplot_mesh_browse_resets_to_generic_transform(monkeypatch, sample_eeg, tmp_path):
    class FileDialog:
        @staticmethod
        def getOpenFileName(*_args):
            return str(tmp_path / "custom.mat"), ""

    class QtWidgets:
        QFileDialog = FileDialog

    class Combo:
        def __init__(self):
            self.value = None
            self.text = ""

        def setEditable(self, _editable):
            pass

        def setEditText(self, value):
            self.text = value

        def setProperty(self, _name, value):
            self.value = value

    class Text:
        def __init__(self):
            self.value = ""

        def setText(self, value):
            self.value = value

    monkeypatch.setattr("eegprep.functions.guifunc.qt._require_qt", lambda: (None, QtWidgets))
    params = controls_by_tag(pop_headplot_dialog_spec(sample_eeg, typeplot=1))["mesh_browse"].callback.params
    target = Combo()
    transform = Text()

    QtDialogRenderer._select_file(object(), target, params, {"transform": transform})

    assert target.value == str(tmp_path / "custom.mat")
    assert transform.value == "0 -10 0 -0.1 0 -1.6 1100 1100 1100"


def test_headplot_manual_coreg_callback_writes_transform(monkeypatch, sample_eeg):
    class Text:
        def __init__(self, value=""):
            self.value = value

        def text(self):
            return self.value

        def setText(self, value):
            self.value = value

    class Combo:
        def __init__(self, index=0):
            self.index = index

        def currentIndex(self):
            return self.index

        def property(self, _name):
            return None

    calls = []

    def fake_coregister(*args, **kwargs):
        calls.append((args, kwargs))
        return [1, 2, 3, 0.1, 0.2, 0.3, 4, 5, 6]

    monkeypatch.setattr("eegprep.functions.guifunc.coregister.run_coregister_dialog", fake_coregister)
    params = controls_by_tag(pop_headplot_dialog_spec(sample_eeg, typeplot=1))["manual_coreg"].callback.params
    transform = Text("0 -10 0 -0.1 0 -1.6 1100 1100 1100")

    QtDialogRenderer._run_headplot_manual_coreg(
        object(),
        {"meshfile": Combo(0), "meshchanfile": Combo(0), "transform": transform},
        params,
    )

    assert transform.value == "1 2 3 0.1 0.2 0.3 4 5 6"
    assert calls[0][1]["meshfile"] == "mheadnew.mat"
    assert calls[0][1]["transform"] == "0 -10 0 -0.1 0 -1.6 1100 1100 1100"


def test_pop_plottopo_rect_option_switches_layout(sample_epoch):
    eeg = deepcopy(sample_epoch)
    eeg["data"] = eeg["data"][:4]
    eeg["nbchan"] = 4
    eeg["chanlocs"] = [
        {"labels": "Fz", "theta": 0, "radius": 0.5},
        {"labels": "C4", "theta": 90, "radius": 0.5},
        {"labels": "Pz", "theta": 180, "radius": 0.5},
        {"labels": "C3", "theta": -90, "radius": 0.5},
    ]

    topo_fig, topo_command = pop_plottopo(eeg, chans=[1, 2, 3, 4], return_com=True)
    rect_fig, rect_command = pop_plottopo(eeg, chans=[1, 2, 3, 4], rect=True, return_com=True)

    topo_positions = [axis.get_position().bounds for axis in topo_fig.axes]
    rect_positions = [axis.get_position().bounds for axis in rect_fig.axes]
    assert topo_positions != rect_positions
    assert "rect=True" in rect_command
    _assert_python_command(topo_command)
    _assert_python_command(rect_command)
    plt.close(topo_fig)
    plt.close(rect_fig)


def test_component_activations_use_icachansind_subset(ica_epoch):
    eeg = deepcopy(ica_epoch)
    eeg["icaact"] = None
    eeg["icachansind"] = np.array([1, 3])
    eeg["icaweights"] = np.eye(2)
    eeg["icasphere"] = np.eye(2)
    eeg["icawinv"] = np.eye(2)

    activations = component_activations(eeg)

    np.testing.assert_allclose(activations[0], eeg["data"][1])
    np.testing.assert_allclose(activations[1], eeg["data"][3])


def test_component_map_plots_use_icachansind_subset(ica_epoch):
    eeg = deepcopy(ica_epoch)
    eeg["icaact"] = None
    eeg["icachansind"] = np.array([1, 3])
    eeg["icaweights"] = np.eye(2)
    eeg["icasphere"] = np.eye(2)
    eeg["icawinv"] = np.eye(2)

    figures, topoplot_command = pop_topoplot(eeg, typeplot=0, items=[1], colorbar="off", return_com=True)
    prop_figure, prop_command = pop_prop(eeg, typecomp=0, chanorcomp=1, return_com=True)
    stat_result, stat_command = pop_signalstat(eeg, typeproc=0, cnum=1, return_com=True)

    assert len(figures) == 1
    assert prop_figure is not None
    assert stat_result.figure is not None
    _assert_python_command(topoplot_command)
    _assert_python_command(prop_command)
    _assert_python_command(stat_command)
    plt.close(figures[0])
    plt.close(prop_figure)
    plt.close(stat_result.figure)


def test_pop_envtopo_uses_icachansind_subset_and_rejects_multiple(ica_epoch):
    eeg = deepcopy(ica_epoch)
    eeg["icaact"] = None
    eeg["icachansind"] = np.array([1, 3])
    eeg["icaweights"] = np.eye(2)
    eeg["icasphere"] = np.eye(2)
    eeg["icawinv"] = np.eye(2)

    figure, command = pop_envtopo(eeg, components=[1], return_com=True)

    assert len(figure.axes) >= 2
    _assert_python_command(command)
    plt.close(figure)
    with pytest.raises(ValueError, match="one dataset"):
        pop_envtopo([ica_epoch, deepcopy(ica_epoch)], components=[1])


def test_pop_comperp_and_chanplot_work_on_epoched_dataset_lists(sample_epoch):
    second = deepcopy(sample_epoch)
    second["setname"] = "second"

    comperp_result, comperp_command = pop_comperp([sample_epoch, second], flag=1, datadd=[1, 2], return_com=True)
    study, chanplot_command, chanplot_fig = pop_chanplot(
        {"name": "demo study"}, [sample_epoch, second], channels=[1], return_com=True
    )

    assert comperp_result["erp1"].shape[1] == sample_epoch["pnts"]
    assert study["etc"]["last_chanplot"]["channels"] == [1]
    _assert_python_command(comperp_command)
    _assert_python_command(chanplot_command)
    plt.close(comperp_result["figure"])
    plt.close(chanplot_fig)


def test_pop_chanplot_gui_filters_channels(sample_epoch):
    class Renderer:
        def __init__(self):
            self.spec = None

        def run(self, spec, initial_values=None):
            self.spec = spec
            return {"channels": "1 2", "measure": 1}

    second = deepcopy(sample_epoch)
    renderer = Renderer()
    study, command, figure = pop_chanplot(
        {"name": "demo study"}, [sample_epoch, second], gui=True, renderer=renderer, return_com=True
    )

    assert renderer.spec is not None
    assert study["etc"]["last_chanplot"]["channels"] == [1, 2]
    assert "channels=[1, 2]" in command
    assert "measure='erp'" in command
    _assert_python_command(command)
    plt.close(figure)


def test_pop_comperp_rms_mode_and_grid_validation(sample_epoch):
    first = deepcopy(sample_epoch)
    second = deepcopy(sample_epoch)
    first["data"] = np.ones_like(first["data"])
    second["data"] = -np.ones_like(second["data"])

    ave, _command = pop_comperp([first, second], flag=1, datadd=[1, 2], mode="ave", return_com=True)
    rms, _command = pop_comperp([first, second], flag=1, datadd=[1, 2], mode="rms", return_com=True)

    np.testing.assert_allclose(ave["erp1"], 0)
    np.testing.assert_allclose(rms["erp1"], 1)
    plt.close(ave["figure"])
    plt.close(rms["figure"])
    second["xmax"] = float(second["xmax"]) + 0.1
    with pytest.raises(ValueError, match="time grid"):
        pop_comperp([first, second], flag=1, datadd=[1, 2])


def test_pop_comperp_supports_display_options_and_significance(sample_epoch):
    datasets = [deepcopy(sample_epoch) for _ in range(4)]
    scales = [1.0, 1.12, 0.78, 1.31]
    for offset, dataset in enumerate(datasets):
        dataset["data"] = np.asarray(dataset["data"], dtype=float) * scales[offset] + offset * 0.05
        dataset["setname"] = f"dataset {offset + 1}"

    result, command = pop_comperp(
        datasets,
        flag=1,
        datadd=[3, 4],
        datsub=[1, 2],
        chans=[1, 2],
        alpha=0.05,
        addstd="on",
        substd="on",
        diffstd="on",
        addall="on",
        suball="on",
        diffall="on",
        tlim=[-50, 100],
        ylim=[-20, 20],
        title="Compared ERPs",
        return_com=True,
    )

    assert result["erp1"].shape == (2, sample_epoch["pnts"])
    assert result["erp2"].shape == (2, sample_epoch["pnts"])
    assert result["erpsub"].shape == (2, sample_epoch["pnts"])
    assert result["pvalues"].shape == (2, sample_epoch["pnts"])
    assert np.isfinite(result["pvalues"]).any()
    assert "addstd='on'" in command
    _assert_python_command(command)
    plt.close(result["figure"])

    with pytest.raises(ValueError, match="unsupported option"):
        pop_comperp(datasets, flag=1, datadd=[1, 2], unsupported="on")


def test_pop_comperp_significance_shading_marks_known_effect(sample_epoch):
    datasets = []
    for amplitude in (1.0, 1.1, 0.9):
        dataset = deepcopy(sample_epoch)
        dataset["data"] = np.zeros_like(np.asarray(dataset["data"], dtype=float)) + amplitude
        datasets.append(dataset)
    for _index in range(3):
        dataset = deepcopy(sample_epoch)
        dataset["data"] = np.zeros_like(np.asarray(dataset["data"], dtype=float))
        datasets.append(dataset)

    result = pop_comperp(datasets, flag=1, datadd=[1, 2, 3], datsub=[4, 5, 6], chans=[1, 2], alpha=0.01)

    significant_patches = [
        patch for patch in result["figure"].axes[0].patches if patch.get_alpha() == pytest.approx(0.18)
    ]
    assert significant_patches
    plt.close(result["figure"])


def test_pop_chanplot_validates_time_grid(sample_epoch):
    second = deepcopy(sample_epoch)
    second["xmax"] = float(second["xmax"]) + 0.1

    with pytest.raises(ValueError, match="time grid"):
        pop_chanplot({"name": "demo study"}, [sample_epoch, second], channels=[1])


def test_pop_erpimage_applies_time_limits_and_decimation(sample_epoch):
    result, command = pop_erpimage(
        sample_epoch,
        typeplot=1,
        index=1,
        limits=[-50, 100],
        decimate=2,
        caxis=[-1, 1],
        cbar=False,
        return_com=True,
    )

    expected_samples = np.count_nonzero((sample_epoch["times"] >= -50) & (sample_epoch["times"] <= 100))
    assert result["image"].shape[1] == expected_samples
    assert result["image"].shape[0] == int(np.ceil(sample_epoch["trials"] / 2))
    assert "limits=[-50, 100]" in command
    _assert_python_command(command)
    plt.close(result["figure"])


def test_pop_erpimage_sorts_by_epoch_event_field_and_limits(sample_epoch):
    eeg = deepcopy(sample_epoch)
    eeg["data"] = np.asarray(
        [
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ]
        ]
    )
    eeg["nbchan"] = 1
    eeg["pnts"] = 4
    eeg["trials"] = 3
    eeg["srate"] = 100.0
    eeg["xmin"] = 0.0
    eeg["xmax"] = 0.03
    eeg["times"] = np.asarray([0.0, 10.0, 20.0, 30.0])
    eeg["event"] = [
        {"type": "rt", "latency": 2, "epoch": 1, "rt": 30},
        {"type": "rt", "latency": 6, "epoch": 2, "rt": 10},
        {"type": "rt", "latency": 10, "epoch": 3, "rt": 20},
    ]

    result, command = pop_erpimage(
        eeg,
        typeplot=1,
        index=1,
        sortingeventfield="rt",
        sortingtype=["rt"],
        sortingwin=[0, 20],
        return_com=True,
    )
    unsorted, _command = pop_erpimage(eeg, typeplot=1, index=1, sort_values=[30, 10, 20], nosort=True, return_com=True)

    np.testing.assert_allclose(result["image"][:, 0], [2, 3, 1])
    np.testing.assert_allclose(unsorted["image"][:, 0], [1, 2, 3])
    assert "sortingeventfield='rt'" in command
    _assert_python_command(command)
    plt.close(result["figure"])
    plt.close(unsorted["figure"])

    with pytest.raises(ValueError, match="standalone ERP image"):
        pop_erpimage(eeg, typeplot=1, index=1, align=[0])


def test_plot_history_preserves_effective_options(sample_epoch, ica_epoch):
    timtopo_fig, timtopo_command = pop_timtopo(
        sample_epoch,
        plottimes=[0],
        timerange=[-50, 100],
        winsize=[10],
        title="custom timtopo",
        return_com=True,
    )
    plottopo_fig, plottopo_command = pop_plottopo(
        sample_epoch,
        chans=[1],
        timerange=[-50, 100],
        title="custom plottopo",
        return_com=True,
    )
    envtopo_fig, envtopo_command = pop_envtopo(
        ica_epoch,
        timerange=[0, 100],
        components=[1],
        title="custom envtopo",
        return_com=True,
    )

    assert "timerange=[-50, 100]" in timtopo_command
    assert "winsize=[10]" in timtopo_command
    assert "title='custom timtopo'" in timtopo_command
    assert "timerange=[-50, 100]" in plottopo_command
    assert "title='custom plottopo'" in plottopo_command
    assert "components=[1]" in envtopo_command
    assert "title='custom envtopo'" in envtopo_command
    for command in (timtopo_command, plottopo_command, envtopo_command):
        _assert_python_command(command)
    plt.close(timtopo_fig)
    plt.close(plottopo_fig)
    plt.close(envtopo_fig)


def test_pop_spectopo_component_path_plots_component_maps(ica_epoch):
    result, command = pop_spectopo(
        ica_epoch,
        dataflag=0,
        freqs=[10],
        icacomps=[1, 2],
        icamaps=[1],
        return_com=True,
    )

    assert result["figure"] is not None
    assert len(result["figure"].axes) >= 2
    assert result["specstd"] is None
    assert "icamaps=[1]" in command
    _assert_python_command(command)
    plt.close(result["figure"])


def test_plot_wrappers_fail_clearly_when_required_fields_are_missing(sample_epoch, tmp_path):
    with pytest.raises(ValueError, match="ICA"):
        pop_plotdata(sample_epoch, components=[1])
    continuous = deepcopy(sample_epoch)
    continuous["trials"] = 1
    continuous["data"] = continuous["data"][:, :, 0]
    with pytest.raises(ValueError, match="epoched"):
        pop_erpimage(continuous, typeplot=1, index=1)
    splinefile = tmp_path / "epoch.spl"
    headplot_setup(
        sample_epoch["chanlocs"],
        splinefile,
        chaninfo=sample_epoch["chaninfo"],
        transform=[0, -10, 0, -0.1, 0, -1.6, 1100, 1100, 1100],
    )
    epoched = deepcopy(sample_epoch)
    epoched["splinefile"] = str(splinefile)
    with pytest.raises(ValueError, match="outside the epoch time range"):
        pop_headplot(epoched, typeplot=1, items=[1e6])
    with pytest.raises(ValueError, match="cannot find spline file"):
        pop_headplot(sample_epoch, typeplot=1, items=[0])
    no_locs = deepcopy(sample_epoch)
    no_locs["chanlocs"] = []
    with pytest.raises(ValueError, match="does not contain channel locations"):
        pop_headplot(no_locs, typeplot=1, items=[0])


def _assert_python_command(command: str) -> None:
    ast.parse(command)


def test_component_activations_dedup_contract():
    """Lock the K4 dedup: rejection delegates recompute to the canonical helper.

    The rejection ``component_activations`` (``_rejection``) and the canonical
    plotting helper (``plot_utils``) must agree when recomputing from weights,
    and rejection must ignore a stored ``icaact`` while plotting trusts it.
    """
    from eegprep.functions.popfunc._rejection import component_activations as rejection_activations

    rng = np.random.default_rng(7)
    nbchan, pnts, trials = 5, 16, 4
    data = rng.standard_normal((nbchan, pnts, trials))
    weights = rng.standard_normal((nbchan, nbchan))
    sphere = rng.standard_normal((nbchan, nbchan))
    recompute = (weights @ sphere) @ data.reshape(nbchan, -1, order="F")
    stored = -recompute.reshape(nbchan, pnts, trials, order="F")
    eeg = {
        "data": data,
        "icaweights": weights,
        "icasphere": sphere,
        "icachansind": np.arange(nbchan),
        "nbchan": nbchan,
        "pnts": pnts,
        "trials": trials,
        "icaact": stored,
    }

    plot_recompute = component_activations(eeg, use_stored=False)
    assert np.allclose(rejection_activations(eeg), plot_recompute)
    # Rejection ignores the stored icaact; the default plotting path trusts it.
    assert not np.allclose(rejection_activations(eeg), stored)
    assert np.allclose(component_activations(eeg), stored)


def _matlab_string(path: Any) -> str:
    return str(path).replace("'", "''")


def _matlab_vector(values: list[float]) -> str:
    return " ".join(str(value) for value in values)


def _eeglab_reference_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    package_reference = repo_root / "src" / "eegprep" / "eeglab"
    if (package_reference / "functions" / "popfunc" / "pop_headplot.m").exists():
        return package_reference
    sibling_reference = repo_root.parent / "eeglab"
    if (sibling_reference / "functions" / "popfunc" / "pop_headplot.m").exists():
        return sibling_reference
    return package_reference
