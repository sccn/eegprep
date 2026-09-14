"""Generated-data ports of current EEGLAB tutorial workflows.

Upstream suite: sccn/eeglab_tests@ff605546f3f70868916fb8d49c007472b3257b50
EEGLAB tree: sccn/eeglab@8ac485f654d6bbb1a6acb8dc9ef3f2eaf3d409ba
Tutorial scripts: sccn/eeglab-tutorial-scripts@58bf12dd53e894dd3ee1285946563cd94999db16
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import pytest

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.adminfunc.eeg_store import eeg_store
from eegprep.functions.miscfunc.eegmovie import eegmovie
from eegprep.functions.popfunc.pop_comments import pop_comments
from eegprep.functions.popfunc.pop_eegfilt import pop_eegfilt
from eegprep.functions.popfunc.pop_epoch import pop_epoch
from eegprep.functions.popfunc.pop_newset import pop_newset
from eegprep.functions.popfunc.pop_newtimef import pop_newtimef
from eegprep.functions.popfunc.pop_resample import pop_resample
from eegprep.functions.popfunc.pop_reref import pop_reref
from eegprep.functions.popfunc.pop_rmbase import pop_rmbase
from eegprep.functions.popfunc.pop_runica import pop_runica
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.functions.popfunc.pop_select import pop_select
from eegprep.functions.popfunc.pop_subcomp import pop_subcomp
from eegprep.functions.popfunc.pop_topoplot import pop_topoplot
from eegprep.functions.sigprocfunc.plotcurve import plotcurve
from eegprep.functions.sigprocfunc.cart2topo import cart2topo
from eegprep.functions.sigprocfunc.topoplot import topoplot
from eegprep.functions.studyfunc.pop_clust import pop_clust
from eegprep.functions.studyfunc.pop_erpparams import pop_erpparams
from eegprep.functions.studyfunc.pop_statparams import pop_statparams
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.functions.studyfunc.std_editset import std_editset
from eegprep.functions.studyfunc.std_erpplot import std_erpplot
from eegprep.functions.studyfunc.std_makedesign import std_makedesign
from eegprep.functions.studyfunc.std_maketrialinfo import std_maketrialinfo
from eegprep.functions.studyfunc.std_preclust import std_preclust
from eegprep.functions.studyfunc.std_precomp import std_precomp
from eegprep.plugins.clean_rawdata.pop_clean_rawdata import pop_clean_rawdata
from eegprep.plugins.dipfit._fitting import leadfield_matrix
from eegprep.plugins.dipfit._utils import DIPFITUnavailableError
from eegprep.plugins.dipfit.pop_dipfit_loreta import pop_dipfit_loreta
from eegprep.plugins.dipfit.pop_dipfit_settings import pop_dipfit_settings
from eegprep.plugins.dipfit.pop_dipplot import pop_dipplot
from eegprep.plugins.dipfit.pop_leadfield import pop_leadfield
from eegprep.plugins.dipfit.pop_multifit import pop_multifit
from eegprep.plugins.EEG_BIDS.pop_exportbids import pop_exportbids
from eegprep.plugins.EEG_BIDS.pop_importbids import pop_importbids
from eegprep.plugins.ICLabel.pop_icflag import pop_icflag
from tests.eeglab_tests import eeglab_test
from tests.fixtures import create_test_eeg, create_test_eeg_with_ica


TUTORIAL_WRAPPER = "unittesting_tutorial/tutorial_wrapperTest.m"
TUTORIAL2_WRAPPER = "unittesting_tutorial/tutorial2_wrapperTest.m"
CONDITIONS = ("standard", "oddball_with_reponse")
ICLABEL_THRESHOLDS = np.asarray(
    [
        [np.nan, np.nan],
        [0.9, 1.0],
        [0.9, 1.0],
        [np.nan, np.nan],
        [np.nan, np.nan],
        [np.nan, np.nan],
        [np.nan, np.nan],
    ]
)


def _continuous_tutorial_eeg() -> dict:
    eeg = create_test_eeg(n_channels=4, n_samples=1024, srate=128.0)
    seconds = np.arange(eeg["pnts"], dtype=float) / eeg["srate"]
    eeg["data"] = np.stack(
        [
            np.sin(2 * np.pi * 6 * seconds) + 0.15 * np.sin(2 * np.pi * 0.2 * seconds),
            np.sin(2 * np.pi * 8 * seconds + 0.2),
            np.sin(2 * np.pi * 10 * seconds + 0.4),
            np.sin(2 * np.pi * 12 * seconds + 0.6),
        ]
    )
    eeg["times"] = seconds * 1000.0
    eeg["event"] = [
        {"type": "square", "latency": 257.0, "duration": 0.0, "urevent": 1},
        {"type": "rt", "latency": 289.0, "duration": 0.0, "urevent": 2},
        {"type": "square", "latency": 641.0, "duration": 0.0, "urevent": 3},
        {"type": "rt", "latency": 681.0, "duration": 0.0, "urevent": 4},
    ]
    eeg["urevent"] = [dict(event) for event in eeg["event"]]
    eeg["setname"] = "Generated continuous tutorial EEG"
    return eeg


def _epoched_tutorial_eeg(subject: str, condition: str, *, subject_index: int) -> dict:
    srate = 64.0
    pnts = 80
    trials = 4
    channel_count = 6
    component_count = 3
    seconds = np.arange(pnts, dtype=float) / srate - 0.25
    target_scale = 2.0 if condition == "target" else 0.45
    subject_scale = 1.0 + 0.12 * subject_index
    response = np.exp(-0.5 * ((seconds - 0.3) / 0.055) ** 2)
    data = np.empty((channel_count, pnts, trials), dtype=float)
    activations = np.empty((component_count, pnts, trials), dtype=float)
    for trial in range(trials):
        phase = trial * np.pi / 10
        for channel in range(channel_count):
            oscillation = 0.2 * np.sin(2 * np.pi * (6 + channel) * seconds + phase)
            data[channel, :, trial] = oscillation + subject_scale * target_scale * (1 - 0.08 * channel) * response
        for component in range(component_count):
            activations[component, :, trial] = (
                np.sin(2 * np.pi * (7 + 2 * component) * seconds + phase) + 0.15 * target_scale * response
            )
    mixing = np.asarray(
        [
            [1.0, 0.2, -0.1],
            [0.7, -0.3, 0.4],
            [0.5, 0.6, -0.2],
            [0.2, 0.8, 0.3],
            [-0.2, 0.4, 0.9],
            [0.1, -0.5, 0.7],
        ]
    )
    eeg = create_test_eeg(
        n_channels=channel_count,
        n_samples=pnts,
        n_trials=trials,
        srate=srate,
    )
    zero_sample = int(round(-seconds[0] * srate)) + 1
    events = [
        {
            "type": condition,
            "latency": float(trial * pnts + zero_sample),
            "epoch": trial + 1,
            "urevent": trial + 1,
        }
        for trial in range(trials)
    ]
    eeg.update(
        {
            "setname": f"{subject}_{condition}",
            "subject": subject,
            "condition": condition,
            "group": "control",
            "session": 1,
            "run": 1,
            "data": data,
            "xmin": float(seconds[0]),
            "xmax": float(seconds[-1]),
            "times": seconds * 1000.0,
            "event": events,
            "urevent": [{key: value for key, value in event.items() if key != "epoch"} for event in events],
            "epoch": [{"event": [trial], "eventtype": [condition]} for trial in range(trials)],
            "icaact": activations,
            "icawinv": mixing,
            "icaweights": np.linalg.pinv(mixing),
            "icasphere": np.eye(channel_count),
            "icachansind": list(range(channel_count)),
        }
    )
    labels = ("Fz", "Cz", "Pz", "Oz", "F3", "F4")
    for channel, label in zip(eeg["chanlocs"], labels):
        channel["labels"] = label
        channel["Z"] = 0.5
    return eeg


def _tutorial_study() -> tuple[dict, list[dict]]:
    datasets = [
        _epoched_tutorial_eeg(f"S{subject:02d}", condition, subject_index=subject)
        for subject in range(1, 4)
        for condition in ("standard", "target")
    ]
    study, datasets = pop_study(None, datasets, name="Generated N400 study")
    study = std_makedesign(
        study,
        datasets,
        1,
        name="Condition",
        variable1="condition",
        values1=["standard", "target"],
        subjselect=["S01", "S02", "S03"],
    )
    return study, datasets


def _known_source_eeg() -> tuple[dict, np.ndarray]:
    eeg = create_test_eeg_with_ica(n_channels=18, n_samples=80, n_components=1, n_trials=4, srate=64.0)
    polar = np.linspace(0.35, np.pi - 0.35, 3)
    azimuth = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    positions = np.asarray(
        [
            [
                85.0 * np.sin(phi) * np.cos(theta),
                85.0 * np.sin(phi) * np.sin(theta),
                85.0 * np.cos(phi),
            ]
            for phi in polar
            for theta in azimuth
        ]
    )
    theta, radius, _x, _y, _z = cart2topo(positions)
    eeg["chanlocs"] = [
        {
            "labels": f"E{index + 1}",
            "type": "EEG",
            "X": float(position[0]),
            "Y": float(position[1]),
            "Z": float(position[2]),
            "theta": float(theta[index]),
            "radius": float(radius[index]),
        }
        for index, position in enumerate(positions)
    ]
    source = np.asarray([10.0, -20.0, 40.0])
    moment = np.asarray([0.6, -0.2, 0.8])
    topography = leadfield_matrix(positions, source)[0] @ moment
    eeg["icawinv"] = topography[:, np.newaxis]
    eeg["icaweights"] = np.linalg.pinv(eeg["icawinv"])
    eeg["icasphere"] = np.eye(eeg["nbchan"])
    eeg["icachansind"] = np.arange(eeg["nbchan"])
    times = np.arange(eeg["pnts"], dtype=float) / eeg["srate"] - 0.25
    response = np.exp(-0.5 * ((times - 0.1) / 0.025) ** 2)
    eeg["data"] = topography[:, np.newaxis, np.newaxis] * response[np.newaxis, :, np.newaxis]
    eeg["data"] = np.repeat(eeg["data"], eeg["trials"], axis=2)
    eeg["times"] = times * 1000.0
    eeg["xmin"] = float(times[0])
    eeg["xmax"] = float(times[-1])
    eeg, _command = pop_dipfit_settings(eeg, model="standardBESA", return_com=True)
    return eeg, source


def _bids_continuous_eeg(subject_index: int) -> dict:
    srate = 64.0
    pnts = 768
    eeg = create_test_eeg(n_channels=8, n_samples=pnts, srate=srate)
    seconds = np.arange(pnts, dtype=float) / srate
    rng = np.random.default_rng(100 + subject_index)
    sources = np.stack(
        [
            np.sin(2 * np.pi * 6 * seconds),
            np.sin(2 * np.pi * 10 * seconds + 0.1),
            0.2 * np.sin(2 * np.pi * 0.2 * seconds),
            rng.normal(scale=0.1, size=pnts),
            np.sin(2 * np.pi * 14 * seconds),
            np.cos(2 * np.pi * 8 * seconds),
        ]
    )
    mixing = rng.normal(size=(8, 6))
    data = np.sum(mixing[:, :, np.newaxis] * sources[np.newaxis, :, :], axis=1)
    event_types = (CONDITIONS * 3)[:6]
    events = []
    for index, event_type in enumerate(event_types):
        latency = 97 + index * 105
        events.append({"type": event_type, "latency": float(latency), "duration": 0.0, "urevent": index + 1})
        if event_type == "oddball_with_reponse":
            center = latency - 1 + round(0.3 * srate)
            response = np.exp(-0.5 * ((np.arange(pnts) - center) / (0.06 * srate)) ** 2)
            data[0] += 4.0 * response
    eeg.update(
        {
            "data": data,
            "subject": f"S{subject_index:02d}",
            "setname": f"S{subject_index:02d}_P300",
            "event": events,
            "urevent": [dict(event) for event in events],
            "dipfit": np.asarray([]),
            "eventdescription": np.asarray([]),
            "epochdescription": np.asarray([]),
            "etc": {"generated_fixture": True},
        }
    )
    labels = ("Fz", "Cz", "Pz", "Oz", "F3", "F4", "EOG1", "EOG2")
    for channel, label in zip(eeg["chanlocs"], labels):
        channel["labels"] = label
    return eeg


@eeglab_test(TUTORIAL_WRAPPER, "test_eeglab_history")
def test_eeglab_history_tutorial_runs_as_replayable_generated_data_pipeline():
    eeg = _continuous_tutorial_eeg()
    alleeg, eeg, currentset = eeg_store([], eeg)

    eeg, filter_command = pop_eegfilt(eeg, 1, 0, 128, return_com=True)
    alleeg, eeg, currentset, newset_command = pop_newset(
        alleeg,
        eeg,
        currentset,
        setname="filtered Continuous EEG Data",
    )
    eeg, reref_command = pop_reref(eeg, [], return_com=True)
    eeg, comment_command = pop_comments(
        eeg,
        "",
        "Dataset was highpass filtered at 1 Hz and rereferenced.",
        True,
        return_com=True,
    )
    eeg, epoch_command = pop_epoch(
        eeg,
        ["square"],
        [-1, 2],
        newname="Continuous EEG Data epochs",
        return_com=True,
    )
    alleeg, eeg, currentset, overwrite_command = pop_newset(
        alleeg,
        eeg,
        currentset,
        setname="Continuous EEG Data epochs",
        overwrite="on",
    )
    eeg, baseline_command = pop_rmbase(eeg, [-1000, 0], return_com=True)
    alleeg, eeg, currentset = eeg_store(alleeg, eeg, currentset)
    resampled, resample_command = pop_resample(eeg, 64, return_com=True)
    figures, topoplot_command = pop_topoplot(
        eeg,
        1,
        [0, 200, 400],
        "Topographic plot",
        [1, 3],
        0,
        electrodes="on",
        plot="off",
        return_com=True,
    )

    assert eeg["trials"] == 2
    assert eeg["data"].shape == (4, eeg["pnts"], 2)
    baseline = np.asarray(eeg["times"]) <= 0
    np.testing.assert_allclose(np.mean(eeg["data"][:, baseline, :], axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.mean(eeg["data"], axis=0), 0.0, atol=1e-10)
    assert resampled["srate"] == 64.0
    assert resampled["pnts"] < eeg["pnts"]
    assert currentset == 2
    assert len(alleeg) == 2
    assert len(figures) == 1
    assert "highpass filtered" in eeg["comments"]
    for command in (
        filter_command,
        newset_command,
        reref_command,
        comment_command,
        epoch_command,
        overwrite_command,
        baseline_command,
        resample_command,
        topoplot_command,
    ):
        assert command
    plt.close(figures[0])


@eeglab_test(TUTORIAL_WRAPPER, "test_event_processing_single_dataset")
def test_event_processing_single_dataset_shifts_events_and_adds_time_locked_cues():
    eeg = _continuous_tutorial_eeg()
    original_count = len(eeg["event"])
    original_square_latencies = [event["latency"] for event in eeg["event"] if event["type"] == "square"]

    for event in eeg["event"]:
        event["latency"] += 10
    for event in list(eeg["event"]):
        if event["type"] != "square":
            continue
        cue = dict(event)
        cue["latency"] = event["latency"] - 0.1 * eeg["srate"]
        cue["type"] = "cue"
        eeg["event"].append(cue)
    eeg = eeg_checkset(eeg, "eventconsistency")

    events = list(eeg["event"])
    shifted_squares = [event["latency"] for event in events if event["type"] == "square"]
    cue_latencies = [event["latency"] for event in events if event["type"] == "cue"]
    assert len(events) == original_count + 2
    np.testing.assert_allclose(shifted_squares, np.asarray(original_square_latencies) + 10)
    np.testing.assert_allclose(cue_latencies, np.asarray(shifted_squares) - 0.1 * eeg["srate"])
    assert [event["latency"] for event in events] == sorted(event["latency"] for event in events)


@eeglab_test(TUTORIAL_WRAPPER, "test_event_processing_study")
def test_event_processing_study_exposes_derived_reaction_time_as_a_design_variable(tmp_path: Path):
    eeg = create_test_eeg(n_channels=4, n_samples=64, n_trials=2, srate=128.0)
    eeg["setname"] = "generated_rtevents"
    eeg["subject"] = "S01"
    eeg["epoch"] = []
    eeg["event"] = [
        {"type": "square", "latency": 17.0, "epoch": 1},
        {"type": "rt", "latency": 37.0, "epoch": 1},
        {"type": "square", "latency": 81.0, "epoch": 2},
        {"type": "rt", "latency": 113.0, "epoch": 2},
    ]
    for current, following in zip(eeg["event"], eeg["event"][1:]):
        if current["type"] == "square" and following["type"] == "rt" and current["epoch"] == following["epoch"]:
            current["rt"] = (following["latency"] - current["latency"]) / eeg["srate"] * 1000.0
    dataset_path = tmp_path / "generated_rtevents.set"
    pop_saveset(eeg, dataset_path)

    study, alleeg = std_editset(
        None,
        None,
        commands=[["index", 1, "load", dataset_path, "subject", "S01"]],
        updatedat="off",
    )
    study, trialinfo = std_maketrialinfo(study, alleeg)
    rt_values = [row["rt"] for row in trialinfo[0]]
    study = std_makedesign(
        study,
        alleeg,
        1,
        name="Reaction time",
        variable1="rt",
        values1=rt_values,
        vartype1="continuous",
    )

    np.testing.assert_allclose(rt_values, [156.25, 250.0])
    assert study["design"][0]["variable"][0]["label"] == "rt"
    assert study["design"][0]["variable"][0]["vartype"] == "continuous"
    assert study["datasetinfo"][0]["trialinfo"] == trialinfo[0]


@eeglab_test(TUTORIAL_WRAPPER, "test_make_eeg_movie")
def test_make_eeg_movie_smooths_an_erp_and_renders_2d_and_3d_frames():
    eeg = _epoched_tutorial_eeg("S01", "target", subject_index=1)
    window = (np.asarray(eeg["times"]) >= -100) & (np.asarray(eeg["times"]) <= 600)
    scalp_erp = np.mean(np.asarray(eeg["data"])[:, window, :], axis=2)
    smoothed = np.vstack([np.convolve(channel, np.ones(5) / 5, mode="same") for channel in scalp_erp])
    frames = [1, smoothed.shape[1] // 2, smoothed.shape[1]]

    movie_2d, colormap_2d = eegmovie(
        smoothed,
        eeg["srate"],
        eeg["chanlocs"],
        movieframes=frames,
        framenum="off",
        startsec=-0.1,
        vert=[0],
        topoplotopt={"numcontour": 0},
        plot="off",
    )
    movie_3d, colormap_3d = eegmovie(
        smoothed,
        eeg["srate"],
        eeg["chanlocs"],
        movieframes=[frames[1]],
        mode="3d",
        timecourse="off",
        framenum="off",
        startsec=-0.1,
        headplotopt={"lighting": "off"},
        plot="off",
    )

    assert np.var(np.diff(smoothed, axis=1)) < np.var(np.diff(scalp_erp, axis=1))
    assert movie_2d.shape[0] == 3 and movie_2d.shape[-1] == 3
    assert movie_3d.shape[0] == 1 and movie_3d.shape[-1] == 3
    assert movie_2d.dtype == movie_3d.dtype == np.uint8
    assert not np.array_equal(movie_2d[0], movie_2d[-1])
    np.testing.assert_allclose(colormap_2d, colormap_3d)
    plt.close("all")


@eeglab_test(TUTORIAL_WRAPPER, "test_plot_study_erp")
def test_plot_study_erp_precomputes_grouped_conditions_statistics_and_topographies():
    study, alleeg = _tutorial_study()
    study = pop_erpparams(study, timerange=[-200, 800], plotconditions="together")
    study, alleeg = std_precomp(
        study,
        alleeg,
        "channels",
        erp="on",
        recompute="on",
        erpparams={"rmbase": [-250, 0]},
    )
    study = pop_statparams(study, condstats="on", method="param", mcorrect="fdr", alpha=np.nan)

    result = std_erpplot(study, alleeg, channels=["Pz"], return_stats=True, plotstderr="on")
    _study, cells, times, _pgroup, pcond, _pinter, figure = result
    standard, target = cells
    response_window = (times >= 250) & (times <= 350)
    difference = np.mean(target[response_window] - standard[response_window], axis=0)

    assert standard.shape == target.shape == (times.size, 3)
    assert np.all(difference > 1.0)
    assert np.asarray(pcond[0]).shape == times.shape
    assert np.isfinite(np.asarray(pcond[0])[response_window]).all()
    assert len(figure.axes) == 1

    stderr = np.std(target, axis=1, ddof=1) / np.sqrt(target.shape[1])
    curve_figure, curve_axis = plt.subplots()
    plotcurve(
        times,
        target.T,
        target=curve_axis,
        plotmean="on",
        plotindiv="off",
        plotstderr=stderr,
        title="Generated target ERP",
    )
    assert len(curve_axis.lines) == 1
    assert curve_axis.collections

    _study, topo_cells, topo_times, topo_figure = std_erpplot(
        study,
        alleeg,
        channels="channels",
        condstats="off",
        topotime=[250, 350],
    )
    assert topo_cells[0].shape == (topo_times.size, 6, 3)
    assert any(axis.collections for axis in topo_figure.axes)
    plt.close(figure)
    plt.close(curve_figure)
    plt.close(topo_figure)


@eeglab_test(TUTORIAL_WRAPPER, "test_source_reconstruction_eeg")
def test_source_reconstruction_eeg_localizes_an_erp_topography_and_plots_the_result():
    eeg, true_source = _known_source_eeg()
    latency_index = int(np.argmin(np.abs(np.asarray(eeg["times"]) - 100.0)))
    erp_topography = np.mean(np.asarray(eeg["data"])[:, latency_index, :], axis=1)
    correlation = np.corrcoef(erp_topography, np.asarray(eeg["icawinv"])[:, 0])[0, 1]

    fitted, command = pop_multifit(eeg, [1], threshold=100, return_com=True)
    model = fitted["dipfit"]["model"][0]
    dipole_figures = pop_dipplot(fitted, [1], normlen="on", plot=True)
    topo_figures = pop_topoplot(fitted, 0, [1], "ERP 100 ms single-dipole fit", [1, 1], 0, plot="off")

    assert correlation > 0.999999
    assert np.isfinite(model["rv"]) and model["rv"] < 0.05
    assert np.linalg.norm(np.asarray(model["posxyz"])[0] - true_source) < 25.0
    assert np.linalg.norm(np.asarray(model["posxyz"])[0]) < 85.0
    assert len(dipole_figures) == len(topo_figures) == 1
    assert "pop_multifit" in command
    plt.close("all")


@eeglab_test(TUTORIAL_WRAPPER, "test_source_reconstruction_advanced")
def test_source_reconstruction_advanced_builds_a_forward_model_and_exposes_fieldtrip_boundary():
    eeg, _true_source = _known_source_eeg()
    eeg = pop_dipfit_settings(eeg, model="standardBEM")
    source_points = [[-25, 0, 35], [10, -20, 40], [20, 20, 30]]
    with_leadfield, command = pop_leadfield(eeg, sourcemodel={"pos": source_points}, return_com=True)
    leadfields = [np.asarray(value) for value in with_leadfield["dipfit"]["sourcemodel"]["leadfield"]]
    moment = np.asarray([0.5, -0.25, 0.75])
    observation = leadfields[1] @ moment
    residuals = []
    for leadfield in leadfields:
        fitted_moment = np.linalg.lstsq(leadfield, observation, rcond=None)[0]
        residuals.append(np.linalg.norm(observation - leadfield @ fitted_moment))

    assert all(leadfield.shape == (18, 3) for leadfield in leadfields)
    np.testing.assert_allclose([np.mean(leadfield, axis=0) for leadfield in leadfields], 0.0, atol=1e-12)
    assert int(np.argmin(residuals)) == 1
    assert "pop_leadfield" in command
    with pytest.raises(DIPFITUnavailableError, match="FieldTrip"):
        pop_dipfit_loreta(with_leadfield, [1], gui=False)


@eeglab_test(TUTORIAL_WRAPPER, "test_study_script")
def test_study_script_runs_n400_measure_statistics_and_component_clustering_workflow():
    study, alleeg = _tutorial_study()
    study, alleeg, channel_command = std_precomp(
        study,
        alleeg,
        "channels",
        erp="on",
        spec="on",
        recompute="on",
        erpparams={"rmbase": [-250, 0]},
        return_com=True,
    )
    study, alleeg, component_command = std_precomp(
        study,
        alleeg,
        "components",
        erp="on",
        spec="on",
        scalp="on",
        recompute="on",
        erpparams={"rmbase": [-250, 0]},
        return_com=True,
    )
    study, alleeg, precluster_command = std_preclust(
        study,
        alleeg,
        1,
        ["spec", "npca", 2, "weight", 1, "freqrange", [3, 25]],
        ["erp", "npca", 2, "weight", 1, "timewindow", [100, 600]],
        ["scalp", "npca", 2, "weight", 1],
        return_com=True,
    )
    study, cluster_command = pop_clust(study, alleeg, clus_num=2, random_state=17, return_com=True)
    study = pop_statparams(study, condstats="on", method="param", mcorrect="fdr", alpha=np.nan)
    _study, erpdata, times, _pgroup, pcond, _pinter, figure = std_erpplot(
        study,
        alleeg,
        channels=["Pz"],
        timerange=[-200, 800],
        return_stats=True,
    )

    assert study["changrp"][0]["measureinfo"]["computed"] == ["erp", "spec"]
    assert study["cluster"][0]["measureinfo"]["computed"] == ["erp", "spec"]
    assert np.asarray(study["etc"]["preclust"]["preclustdata"]).shape[0] == 18
    assert len(study["cluster"][1:]) == 2
    assert sum(len(cluster["comps"]) for cluster in study["cluster"][1:]) == 18
    assert erpdata[0].shape == erpdata[1].shape == (times.size, 3)
    assert np.asarray(pcond[0]).shape == times.shape
    for command in (channel_command, component_command, precluster_command, cluster_command):
        assert command
    plt.close(figure)


@eeglab_test(TUTORIAL_WRAPPER, "test_time_freq_all_elec")
def test_time_freq_all_electrodes_preserves_trial_power_and_spatial_axes():
    eeg = _epoched_tutorial_eeg("S01", "target", subject_index=1)
    results = [
        pop_newtimef(
            eeg,
            1,
            channel,
            [eeg["xmin"] * 1000, eeg["xmax"] * 1000],
            [0],
            freqs=[4, 20],
            nfreqs=9,
            timesout=8,
            baseline=np.nan,
            scale="abs",
            alpha=0.2,
            naccu=20,
            rng=300 + channel,
            plotphase="off",
            plotersp="off",
            plotitc="off",
            plot="off",
        )
        for channel in range(1, eeg["nbchan"] + 1)
    ]
    all_ersp = np.stack([result.ersp for result in results], axis=-1)
    all_itc = np.stack([result.itc for result in results], axis=-1)
    all_powbase = np.stack([result.powbase for result in results], axis=-1)
    all_erspboot = np.stack([result.erspboot for result in results], axis=-1)
    all_itcboot = np.stack([result.itcboot for result in results], axis=-1)

    for result in results:
        np.testing.assert_allclose(result.ersp, np.mean(np.abs(result.tfdata) ** 2, axis=2), rtol=1e-12)
        assert np.max(np.abs(result.itc)) <= 1.0 + 1e-12
        np.testing.assert_allclose(result.times, results[0].times)
        np.testing.assert_allclose(result.freqs, results[0].freqs)
    assert all_ersp.shape == all_itc.shape == (9, 8, eeg["nbchan"])
    assert all_powbase.shape == (9, eeg["nbchan"])
    assert all_erspboot.shape == (9, 2, eeg["nbchan"])
    assert all_itcboot.shape == (9, eeg["nbchan"])
    assert np.std(all_ersp, axis=-1).max() > 0

    figure, axis = plt.subplots()
    topoplot(
        all_ersp[3, 4],
        eeg["chanlocs"],
        axes=axis,
        electrodes="on",
        maplimits="absmax",
    )
    assert axis.collections
    plt.close(figure)


@eeglab_test(TUTORIAL2_WRAPPER, "test_bids_p300")
def test_bids_p300_runs_generated_import_clean_ica_epoch_and_study_pipeline(tmp_path: Path):
    bids_root = tmp_path / "generated_p300"
    first_root, first_export_command = pop_exportbids(
        _bids_continuous_eeg(1),
        bids_root,
        subject="01",
        task="P300",
        return_com=True,
    )
    second_root, second_export_command = pop_exportbids(
        _bids_continuous_eeg(2),
        bids_root,
        subject="02",
        task="P300",
        return_com=True,
    )
    imported, import_command = pop_importbids(bids_root, return_com=True)
    assert isinstance(imported, list)
    selected, select_command = pop_select(imported, nochannel=["EOG1", "EOG2"], return_com=True)
    referenced, reference_command = pop_reref(selected, [], return_com=True)
    original_low_frequency = []
    for eeg in referenced:
        frequencies = np.fft.rfftfreq(eeg["pnts"], d=1 / eeg["srate"])
        low_frequency_index = int(np.argmin(np.abs(frequencies - 0.2)))
        original_low_frequency.append(np.abs(np.fft.rfft(np.asarray(eeg["data"])[0]))[low_frequency_index])
    cleaned, clean_command = pop_clean_rawdata(
        referenced,
        FlatlineCriterion="off",
        ChannelCriterion="off",
        LineNoiseCriterion="off",
        Highpass=[1, 2],
        BurstCriterion="off",
        WindowCriterion="off",
        gui=False,
        return_com=True,
    )
    decomposed, ica_command = pop_runica(
        cleaned,
        icatype="picard",
        concatcond="on",
        options={"pca": -1, "maxiter": 100, "verbose": False},
        gui=False,
        return_com=True,
    )

    for eeg in decomposed:
        component_count = np.asarray(eeg["icaweights"]).shape[0]
        classifications = np.zeros((component_count, 7))
        classifications[:, 0] = 0.99
        classifications[-1] = [0, 0.95, 0, 0, 0, 0, 0.05]
        eeg.setdefault("etc", {}).setdefault("ic_classification", {})["ICLabel"] = {
            "classes": ["Brain", "Muscle", "Eye", "Heart", "Line Noise", "Channel Noise", "Other"],
            "classifications": classifications,
        }
    flagged, flag_command = pop_icflag(decomposed, ICLABEL_THRESHOLDS, gui=False, return_com=True)
    before_component_removal = [np.asarray(eeg["data"]).copy() for eeg in flagged]
    pruned, prune_command = pop_subcomp(flagged, [], 0, 0, gui=False, return_com=True)

    epochs = []
    epoch_commands = []
    baseline_commands = []
    for eeg in pruned:
        for condition in CONDITIONS:
            epoched, epoch_command = pop_epoch(eeg, [condition], [-0.25, 0.75], return_com=True)
            epoched, baseline_command = pop_rmbase(epoched, [-250, 0], return_com=True)
            epoched["condition"] = condition
            epochs.append(epoched)
            epoch_commands.append(epoch_command)
            baseline_commands.append(baseline_command)

    study, epochs, study_command = pop_study(None, epochs, name="Generated Oddball", return_com=True)
    study, design_command = std_makedesign(
        study,
        epochs,
        1,
        name="P300 condition",
        variable1="condition",
        values1=list(CONDITIONS),
        subjselect=["01", "02"],
        return_com=True,
    )
    study, epochs, precompute_command = std_precomp(
        study,
        epochs,
        "channels",
        erp="on",
        savetrials="on",
        recompute="on",
        erpparams={"rmbase": [-250, 0]},
        return_com=True,
    )
    _study, erpdata, times, figure = std_erpplot(study, epochs, channels=["Fz"], design=1)

    assert first_root == second_root == str(bids_root)
    assert len(imported) == len(selected) == len(cleaned) == len(decomposed) == len(pruned) == 2
    assert [eeg["subject"] for eeg in imported] == ["01", "02"]
    assert all(eeg["nbchan"] == 6 for eeg in selected)
    for eeg in referenced:
        np.testing.assert_allclose(np.mean(np.asarray(eeg["data"]), axis=0), 0.0, atol=1e-6)
    for index, eeg in enumerate(cleaned):
        frequencies = np.fft.rfftfreq(eeg["pnts"], d=1 / eeg["srate"])
        low_frequency_index = int(np.argmin(np.abs(frequencies - 0.2)))
        filtered_amplitude = np.abs(np.fft.rfft(np.asarray(eeg["data"])[0]))[low_frequency_index]
        assert filtered_amplitude < 0.6 * original_low_frequency[index]
    assert all(np.asarray(eeg["icaweights"]).shape == (5, 6) for eeg in decomposed)
    assert all(np.count_nonzero(eeg["reject"]["gcompreject"]) == 1 for eeg in flagged)
    assert all(np.asarray(eeg["icaweights"]).shape == (4, 6) for eeg in pruned)
    assert all(not np.allclose(before, eeg["data"]) for before, eeg in zip(before_component_removal, pruned))
    assert len(epochs) == 4 and all(eeg["trials"] == 3 for eeg in epochs)
    assert all(np.asarray(eeg["data"]).shape == (6, 64, 3) for eeg in epochs)
    assert study["design"][0]["variable"][0]["value"] == list(CONDITIONS)
    assert erpdata[0].shape == erpdata[1].shape == (times.size, 2)
    p300_window = (times >= 150) & (times <= 500)
    assert np.max(np.mean(erpdata[1][p300_window] - erpdata[0][p300_window], axis=1)) > 0.25
    for command in (
        first_export_command,
        second_export_command,
        import_command,
        select_command,
        reference_command,
        clean_command,
        ica_command,
        flag_command,
        prune_command,
        *epoch_commands,
        *baseline_commands,
        study_command,
        design_command,
        precompute_command,
    ):
        assert command
    plt.close(figure)
