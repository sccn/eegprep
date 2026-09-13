"""Meaningful headless ports of current EEGLAB visual pop-function tests."""

from __future__ import annotations

from copy import deepcopy

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal import get_window, welch

from eegprep.functions.popfunc.eeg_multieegplot import eeg_multieegplot
from eegprep.functions.popfunc.pop_chansel import pop_chansel, pop_chansel_display_values
from eegprep.functions.popfunc.pop_compareerps import pop_compareerps
from eegprep.functions.popfunc.pop_comperp import pop_comperp
from eegprep.functions.popfunc.pop_crossf import pop_crossf
from eegprep.functions.popfunc.pop_eegplot import pop_eegplot
from eegprep.functions.popfunc.pop_envtopo import pop_envtopo
from eegprep.functions.popfunc.pop_erpimage import pop_erpimage
from eegprep.functions.popfunc.pop_headplot import pop_headplot
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_newcrossf import pop_newcrossf
from eegprep.functions.popfunc.pop_plotdata import pop_plotdata, pop_plotdata_dialog_spec
from eegprep.functions.popfunc.pop_plottopo import pop_plottopo
from eegprep.functions.popfunc.pop_prop import pop_prop
from eegprep.functions.popfunc.pop_selectcomps import pop_selectcomps
from eegprep.functions.popfunc.pop_spectopo import pop_spectopo
from eegprep.functions.popfunc.pop_timef import pop_timef
from eegprep.functions.popfunc.pop_timtopo import pop_timtopo
from eegprep.functions.popfunc.pop_topoplot import pop_topoplot
from eegprep.functions.sigprocfunc.eegplot import winrej_to_array
from tests.eeglab_tests import eeglab_test
from tests.fixtures import SAMPLE_DATASET_PATH


def _source(name: str) -> str:
    return f"unittesting_popfunc/{name}/popfunc_{name}_wrapperTest.m"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def epoched_eeg() -> dict:
    rng = np.random.default_rng(9173)
    channels, points, trials, components = 6, 64, 8, 4
    srate = 64.0
    xmin = -0.25
    seconds = xmin + np.arange(points) / srate
    data = np.empty((channels, points, trials), dtype=float)
    for channel in range(channels):
        for trial in range(trials):
            data[channel, :, trial] = (
                (channel + 1) * 0.2
                + np.sin(2 * np.pi * (5 + channel) * seconds + trial * 0.11)
                + rng.normal(scale=0.01, size=points)
            )

    raw_weights = rng.normal(size=(components, channels))
    weights = np.linalg.qr(raw_weights.T)[0].T[:components]
    maps = np.linalg.pinv(weights)
    activations = np.einsum("kc,cpt->kpt", weights, data)
    chanlocs = []
    for index in range(channels):
        theta = 360.0 * index / channels
        radius = 0.32
        chanlocs.append(
            {
                "labels": f"Ch{index + 1}",
                "theta": theta,
                "radius": radius,
                "X": radius * np.cos(np.deg2rad(theta)),
                "Y": radius * np.sin(np.deg2rad(theta)),
                "Z": 0.15 + 0.02 * (index % 2),
            }
        )
    events = [
        {
            "type": "rt",
            "latency": trial * points + 17,
            "epoch": trial + 1,
            "rt": float(trials - trial),
        }
        for trial in range(trials)
    ]
    return {
        "data": data,
        "nbchan": channels,
        "pnts": points,
        "trials": trials,
        "srate": srate,
        "xmin": xmin,
        "xmax": float(seconds[-1]),
        "times": seconds * 1000.0,
        "setname": "synthetic epochs",
        "chanlocs": chanlocs,
        "chaninfo": {},
        "event": events,
        "epoch": [{"event": [index]} for index in range(trials)],
        "icaweights": weights,
        "icasphere": np.eye(channels),
        "icawinv": maps,
        "icaact": activations,
        "icachansind": np.arange(channels),
        "reject": {},
        "splinefile": "",
        "icasplinefile": "",
    }


@pytest.fixture
def continuous_eeg(epoched_eeg: dict) -> dict:
    eeg = deepcopy(epoched_eeg)
    eeg["data"] = epoched_eeg["data"].transpose(0, 2, 1).reshape(epoched_eeg["nbchan"], -1)
    eeg["icaact"] = epoched_eeg["icaact"].transpose(0, 2, 1).reshape(epoched_eeg["icaact"].shape[0], -1)
    eeg["pnts"] = eeg["data"].shape[1]
    eeg["trials"] = 1
    eeg["xmin"] = 0.0
    eeg["xmax"] = (eeg["pnts"] - 1) / eeg["srate"]
    eeg["times"] = np.arange(eeg["pnts"]) * 1000.0 / eeg["srate"]
    eeg["event"] = []
    eeg["epoch"] = []
    eeg["setname"] = "synthetic continuous"
    return eeg


@eeglab_test(_source("eeg_multieegplot"), "test_pass_continuous")
def test_eeg_multieegplot_continuous_preserves_channel_major_samples() -> None:
    data = np.asarray([[1, 1, 1, 2, 2], [2, 1, 2, 1, 1]], dtype=float)

    model = eeg_multieegplot(data, np.zeros((0, 4)), np.zeros(2), show=False)

    np.testing.assert_array_equal(model.data.data, data[:, :, np.newaxis])
    assert model.data.mode == "continuous"
    assert model.state.winlength == 5
    assert model.state.xgrid is False


@eeglab_test(_source("eeg_multieegplot"), "test_pass_continuous_reject")
def test_eeg_multieegplot_continuous_translates_new_rejection_regions() -> None:
    model = eeg_multieegplot(np.zeros((2, 20)), [[0, 0, 3, 7]], np.ones((2, 20)), show=False)

    rows = winrej_to_array(model.state.winrej, 2)
    np.testing.assert_array_equal(rows[:, :2], [[3, 7]])
    np.testing.assert_allclose(rows[:, 2:5], [[0.8, 0.8, 1.0]])
    np.testing.assert_array_equal(rows[:, 5:], [[0, 0]])


@eeglab_test(_source("eeg_multieegplot"), "test_pass_epochs")
def test_eeg_multieegplot_epochs_translate_trial_and_electrode_marks() -> None:
    data = np.zeros((2, 3, 3), dtype=float)
    trial_rejection = np.asarray([0, 1, 0])
    electrode_rejection = np.asarray([[0, 1, 0], [0, 0, 0]])

    model = eeg_multieegplot(data, trial_rejection, electrode_rejection, show=False)

    rows = winrej_to_array(model.state.winrej, 2)
    np.testing.assert_array_equal(rows[:, :2], [[3, 5]])
    np.testing.assert_array_equal(rows[:, 5:], [[1, 0]])
    assert model.data.mode == "epoched"


@eeglab_test(_source("pop_chansel"), "test_test_pop_chansel")
def test_pop_chansel_returns_selected_indices_labels_and_text(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def choose(**kwargs):
        seen.update(kwargs)
        return [1, 3], True, ""

    monkeypatch.setattr("eegprep.functions.popfunc.pop_chansel.listdlg2", choose)

    indices, text, labels = pop_chansel(["Fz", "C z", "Pz"], withindex="on", select=[1, 3], selectionmode="multiple")

    assert indices == [1, 3]
    assert labels == ["Fz", "Pz"]
    assert text == "Fz Pz"
    assert seen["liststring"] == ["1  -  Fz", "2  -  C z", "3  -  Pz"]
    assert seen["initialvalue"] == [1, 3]
    assert pop_chansel_display_values(["Fz", "Cz"], withindex="off") == ["Fz", "Cz"]


@eeglab_test(_source("pop_compareerps"), "test_i_pass_general")
def test_pop_compareerps_gui_path_averages_selected_datasets(epoched_eeg: dict) -> None:
    datasets = [deepcopy(epoched_eeg), deepcopy(epoched_eeg)]
    datasets[1]["data"] = datasets[1]["data"] + 2.0

    class Renderer:
        def run(self, _spec, initial_values=None):
            return {"datadd": "1 2", "datsub": "", "chans": "1", "addavg": True}

    result, command = pop_compareerps(datasets, gui=True, renderer=Renderer(), return_com=True)

    expected = np.mean([dataset["data"][0].mean(axis=1) for dataset in datasets], axis=0)
    np.testing.assert_allclose(result["erp1"][0], expected)
    assert command == "pop_compareerps(ALLEEG);"


@eeglab_test(_source("pop_compareerps"), "test_i_pass_sets_chans_title")
def test_pop_compareerps_honors_dataset_channel_subset_and_title(epoched_eeg: dict) -> None:
    datasets = [deepcopy(epoched_eeg) for _ in range(3)]
    for index, dataset in enumerate(datasets, start=1):
        dataset["data"] = dataset["data"] * index

    result, command = pop_compareerps(
        datasets, setlist=[1, 3], chansubset=[2, 4], plottitle="Comparing datasets", gui=False, return_com=True
    )

    expected = np.mean([datasets[index]["data"][[1, 3]].mean(axis=2) for index in (0, 2)], axis=0)
    np.testing.assert_allclose(result["erp1"], expected)
    assert result["figure"].axes[0].get_title() == "Comparing datasets"
    assert "[1 3], [2 4], 'Comparing datasets'" in command


@eeglab_test(_source("pop_comperp"), "test_test_pop_comperp")
def test_pop_comperp_computes_channel_and_component_differences(epoched_eeg: dict) -> None:
    first = deepcopy(epoched_eeg)
    second = deepcopy(epoched_eeg)
    second["data"] = second["data"] + 0.75
    second["icaact"] = second["icaact"] - 0.5

    channels = pop_comperp([first, second], 1, [1], [2], chans=[1, 2], std="on", allerps="on")
    components = pop_comperp([first, second], 0, [1], [2], chans=[1, 3], std="on", allerps="on")

    np.testing.assert_allclose(channels["erpsub"], -0.75)
    np.testing.assert_allclose(components["erpsub"], 0.5)
    assert channels["figure"].axes[0].lines
    assert components["figure"].axes[0].lines


@eeglab_test(_source("pop_crossf"), "test_pass_tlimits_empty")
def test_pop_crossf_derives_time_limits_when_empty(epoched_eeg: dict) -> None:
    result, command = pop_crossf(epoched_eeg, 1, 1, 2, None, [0], timesout=8, return_com=True)

    assert result.coherence.shape == result.phase.shape
    assert result.times.size <= 8
    assert np.isfinite(result.coherence).all()
    assert command.startswith("pop_crossf(EEG, 1, 1, 2")


@eeglab_test(_source("pop_crossf"), "test_test_pop_crossf")
def test_pop_crossf_channel_and_component_paths_use_requested_signals(epoched_eeg: dict) -> None:
    limits = [epoched_eeg["times"][0], epoched_eeg["times"][-1]]
    channels = pop_crossf(epoched_eeg, 1, 2, 4, limits, [0], timesout=8, type="phasecoher")
    components = pop_crossf(epoched_eeg, 0, 1, 3, limits, [0], timesout=8, type="phasecoher")

    assert channels.alltf_x.shape[-1] == epoched_eeg["trials"]
    assert components.alltf_x.shape[-1] == epoched_eeg["trials"]
    assert not np.allclose(channels.coherence, components.coherence)


@eeglab_test(_source("pop_eegplot"), "test_test_pop_eegplot")
def test_pop_eegplot_builds_continuous_channel_and_epoched_component_models(
    continuous_eeg: dict, epoched_eeg: dict
) -> None:
    continuous = pop_eegplot(continuous_eeg, 1, 0, 0, show=False)
    components = pop_eegplot(epoched_eeg, 0, 0, 0, show=False)

    assert continuous.data.mode == "continuous"
    assert continuous.data.data.shape == (*continuous_eeg["data"].shape, 1)
    assert components.data.mode == "component"
    np.testing.assert_array_equal(components.data.data, epoched_eeg["icaact"])


@eeglab_test(_source("pop_envtopo"), "test_test_pop_envtopo")
def test_pop_envtopo_supports_legacy_negative_component_count_and_contribution_window(epoched_eeg: dict) -> None:
    limits = [epoched_eeg["times"][0], epoched_eeg["times"][-1]]

    figure, command = pop_envtopo(
        epoched_eeg,
        limits,
        limcontrib=[0, 300],
        compnums=-3,
        electrodes="off",
        return_com=True,
    )

    map_axes = [axis for axis in figure.axes if axis.images]
    assert len(map_axes) == 3
    assert all(axis.get_title().startswith("IC ") for axis in map_axes)
    assert "compnums=-3" in command
    assert "limcontrib=[0, 300]" in command


@eeglab_test(_source("pop_erpimage"), "test_test_pop_erpimage")
def test_pop_erpimage_sorts_channel_trials_and_projects_components(epoched_eeg: dict) -> None:
    channels = pop_erpimage(
        epoched_eeg,
        1,
        2,
        sortingeventfield="rt",
        sortingtype=["rt"],
        renorm="yes",
        smooth=1,
        cbar=False,
    )
    components = pop_erpimage(epoched_eeg, 0, 2, projchan=[1, 3], smooth=1, cbar=False)

    channel_values = epoched_eeg["data"][1]
    np.testing.assert_allclose(channels["image"], channel_values.T[::-1])
    projection = epoched_eeg["icawinv"][[0, 2], 1].mean() * epoched_eeg["icaact"][1]
    np.testing.assert_allclose(components["image"], projection.T)
    with pytest.raises(ValueError, match="phase2"):
        pop_erpimage(epoched_eeg, 1, 2, phase2=0.1)


@eeglab_test(_source("pop_headplot"), "test_test_pop_headplot")
def test_pop_headplot_creates_reusable_spline_and_finite_3d_maps(tmp_path) -> None:
    eeg = pop_loadset(SAMPLE_DATASET_PATH)
    spline = tmp_path / "current_suite.spl"
    setup = {"splinefile": str(spline), "transform": [0, -10, 0, -0.1, 0, -1.6, 1100, 1100, 1100]}

    figures, command = pop_headplot(eeg, 1, [0, 100], "ERP scalp maps", [1, 2], setup=setup, return_com=True)

    assert spline.exists()
    assert len(figures) == 1
    assert all(axis.name == "3d" for axis in figures[0].axes[:2])
    facecolors = np.concatenate(
        [collection.get_facecolors() for axis in figures[0].axes[:2] for collection in axis.collections]
    )
    assert np.isfinite(facecolors).all()
    assert "setup={" in command


@eeglab_test(_source("pop_newcrossf"), "test_test_pop_newcrossf")
def test_pop_newcrossf_channel_and_component_coherence_are_bounded(epoched_eeg: dict) -> None:
    limits = [epoched_eeg["times"][0], epoched_eeg["times"][-1]]
    channels = pop_newcrossf(epoched_eeg, 1, 1, 2, limits, [0], timesout=8, type="phasecoher")
    components = pop_newcrossf(epoched_eeg, 0, 1, 2, limits, [0], timesout=8, type="phasecoher")

    for result in (channels, components):
        assert result.coherence.shape == result.phase.shape
        assert np.all(result.coherence >= 0)
        assert np.all(result.coherence <= 1 + 1e-12)
        assert result.alltf_x.shape[-1] == epoched_eeg["trials"]


@eeglab_test(_source("pop_plotdata"), "test_test_pop_plotdata")
def test_pop_plotdata_selects_modes_trials_averages_and_single_trial_overlays(epoched_eeg: dict) -> None:
    channel_figure, command = pop_plotdata(
        epoched_eeg,
        1,
        [2, 4],
        [2, 5, 7],
        "selected channels",
        0,
        -1,
        [-2, 2],
        return_com=True,
    )
    component_figure = pop_plotdata(epoched_eeg, 0, [1, 3], [1, 4], "components", 1, 1, [0, 0])

    channel_axes = {axis.get_title(): axis for axis in channel_figure.axes}
    expected = epoched_eeg["data"][1, :, [1, 4, 6]].mean(axis=0)
    np.testing.assert_allclose(channel_axes["Ch2"].lines[0].get_ydata(), expected)
    assert channel_axes["Ch2"].yaxis_inverted()
    component_axes = {axis.get_title(): axis for axis in component_figure.axes}
    assert len(component_axes["1"].lines) == 3  # two trials plus the zero reference
    assert not component_axes["1"].yaxis_inverted()
    assert "pop_plotdata(EEG, 1, [2, 4], [2, 5, 7]" in command
    assert pop_plotdata_dialog_spec(epoched_eeg, typeplot=1).title.startswith("Channel ERPs")
    assert pop_plotdata_dialog_spec(epoched_eeg, typeplot=0).title.startswith("Component ERPs")


@eeglab_test(_source("pop_plottopo"), "test_test_pop_plottopo")
def test_pop_plottopo_draws_selected_channel_trial_averages(epoched_eeg: dict) -> None:
    figure, command = pop_plottopo(epoched_eeg, [1, 3, 5], "selected channels", 0, return_com=True)
    single_trials = pop_plottopo(epoched_eeg, [1, 3], "single trials", 1, rect=True)

    axes = {axis.get_title(): axis for axis in figure.axes}
    np.testing.assert_allclose(axes["Ch3"].lines[0].get_ydata(), epoched_eeg["data"][2].mean(axis=1))
    assert len(axes) == 3
    assert axes["Ch3"].yaxis_inverted()
    assert command.startswith("pop_plottopo(EEG, [1, 3, 5], 'selected channels', 0)")
    assert len(single_trials.axes[0].lines) == epoched_eeg["trials"] + 1


@eeglab_test(_source("pop_prop"), "test_test_pop_prop")
def test_pop_prop_builds_channel_and_component_property_panels(epoched_eeg: dict, continuous_eeg: dict) -> None:
    channel = pop_prop(epoched_eeg, 1, 2, 0, {"freqrange": [2, 25]}, plot="off")
    components = pop_prop(epoched_eeg, 0, [1, 3], 0, {"freqrange": [2, 25]}, plot="off")
    continuous = pop_prop(continuous_eeg, 1, 1, 0, {"freqrange": [2, 25]}, plot="off")

    assert any(axis.get_title() == "Channel 2" for axis in channel.axes)
    assert [next(axis for axis in figure.axes if axis.get_xlabel() == "Frequency (Hz)") for figure in components]
    assert any("continu" in axis.get_title().lower() for axis in continuous.axes)


@eeglab_test(_source("pop_selectcomps"), "test_test_pop_selectcomps")
def test_pop_selectcomps_marks_only_requested_components_without_mutating_input(epoched_eeg: dict) -> None:
    before = set(plt.get_fignums())
    selected, command = pop_selectcomps(epoched_eeg, [1, 2, 3, 4], reject=[2, 4], plot=True, return_com=True)

    assert "gcompreject" not in epoched_eeg["reject"]
    np.testing.assert_array_equal(selected["reject"]["gcompreject"], [0, 1, 0, 1])
    assert command == "EEG = pop_selectcomps(EEG, [1 2 3 4], reject=[2 4]);"
    created = set(plt.get_fignums()) - before
    assert len(created) == 1
    assert [axis.get_title() for axis in plt.figure(created.pop()).axes[:4]] == ["IC 1", "IC 2", "IC 3", "IC 4"]


@eeglab_test(_source("pop_spectopo"), "test_test_pop_spectopo")
def test_pop_spectopo_blackman_harris_matches_welch_and_component_mode(epoched_eeg: dict) -> None:
    channel = pop_spectopo(
        epoched_eeg,
        1,
        [epoched_eeg["times"][0], epoched_eeg["times"][-1]],
        "EEG",
        percent=100,
        freq=[8, 10],
        freqrange=[2, 25],
        wintype="blackmanharris",
        blckhn=2,
    )
    component = pop_spectopo(epoched_eeg, 0, None, "EEG", freq=[10], plotchan=0, icacomps=[1, 2], nicamaps=2)

    window = get_window("blackmanharris", 32, fftbins=False)
    powers = []
    for trial in range(epoched_eeg["trials"]):
        frequencies, power = welch(
            epoched_eeg["data"][:, :, trial],
            fs=64.0,
            window=window,
            nperseg=32,
            noverlap=0,
            nfft=32,
            detrend=False,
            axis=1,
            scaling="density",
        )
        powers.append(power)
    expected = 10 * np.log10(np.mean(powers, axis=0))
    np.testing.assert_allclose(channel["freqs"], frequencies)
    np.testing.assert_allclose(channel["spectra"], expected, rtol=1e-12, atol=1e-12)
    assert component["spectra"].shape[0] == 2
    assert np.isfinite(component["spectra"]).all()
    with pytest.raises(ValueError, match="whole-scalp component spectra"):
        pop_spectopo(epoched_eeg, 0, None, "EEG", freq=[10], plotchan=3, icacomps=[1, 2])
    with pytest.raises(ValueError, match="data-comp"):
        pop_spectopo(epoched_eeg, 0, None, "EEG", freq=[10], plotchan=0, icamode="sub", icacomps=[1, 2])


@eeglab_test(_source("pop_timef"), "test_test_pop_timef")
def test_pop_timef_channel_and_component_results_have_consistent_tf_arrays(epoched_eeg: dict) -> None:
    limits = [epoched_eeg["times"][0], epoched_eeg["times"][-1]]
    channel, channel_command = pop_timef(
        epoched_eeg, 1, 2, limits, [0], freqs=[4, 20], timesout=8, plotphase="off", return_com=True
    )
    component = pop_timef(epoched_eeg, 0, 1, limits, [0], freqs=[4, 20], timesout=8, plotphase="off")

    for result in (channel, component):
        assert result.ersp.shape == result.itc.shape
        assert result.tfdata.shape[:2] == result.ersp.shape
        assert result.tfdata.shape[-1] == epoched_eeg["trials"]
        assert np.isfinite(result.ersp).all()
    assert channel_command.startswith("pop_timef(EEG, 1, 2")


@eeglab_test(_source("pop_timtopo"), "test_test_pop_timtopo")
def test_pop_timtopo_nan_latency_selects_global_power_peak(epoched_eeg: dict) -> None:
    erp = epoched_eeg["data"].mean(axis=2)
    expected_index = int(np.argmax(np.sum(erp**2, axis=0)))
    expected_latency = float(epoched_eeg["times"][expected_index])

    figure, command = pop_timtopo(epoched_eeg, [np.nan], title="ERP maps", return_com=True)

    map_titles = [axis.get_title() for axis in figure.axes if axis.images]
    assert map_titles == [f"{expected_latency:.0f}"]
    assert figure.texts[0].get_text() == "ERP maps"
    assert "float('nan')" in command


@eeglab_test(_source("pop_topoplot"), "test_test_pop_topoplot")
def test_pop_topoplot_channel_and_component_maps_use_requested_layout_and_polarity(epoched_eeg: dict) -> None:
    channels = pop_topoplot(epoched_eeg, 1, [0, 100], "ERP maps", [1, 2], 0, electrodes="off", colorbar="off")
    components = pop_topoplot(epoched_eeg, 0, [1, -2], "Component maps", [1, 2], 0, electrodes="off", colorbar="off")
    grid_eeg = deepcopy(epoched_eeg)
    grid_eeg["chanlocs"] = []
    grid_eeg["chanmatrix"] = np.asarray([[1, 2, 0], [-3, 4, 5]])
    grid = pop_topoplot(grid_eeg, 1, [0], "Grid map", [1, 1], 0, colorbar="off")

    assert [axis.get_title() for axis in channels[0].axes] == ["0 ms", "100 ms"]
    assert [axis.get_title() for axis in components[0].axes] == ["IC 1", "IC -2"]
    positive = components[0].axes[0].images[0].get_clim()
    negative = components[0].axes[1].images[0].get_clim()
    assert positive[0] == pytest.approx(-positive[1])
    assert negative[0] == pytest.approx(-negative[1])
    frame = int(np.argmin(np.abs(epoched_eeg["times"])))
    values = epoched_eeg["data"][:, frame].mean(axis=1)
    expected_grid = np.asarray([[values[0], values[1], np.nan], [-values[2], values[3], values[4]]])
    np.testing.assert_allclose(grid[0].axes[0].images[0].get_array(), expected_grid)
