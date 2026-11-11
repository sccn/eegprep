"""Regression tests for release-readiness stream 7 plot findings."""

from __future__ import annotations

import ast
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eegprep.functions.adminfunc.console import _console_python_command
from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.adminfunc.pop_editoptions import pop_editoptions
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.popfunc.pop_comperp import pop_comperp
from eegprep.functions.popfunc.pop_envtopo import pop_envtopo, pop_envtopo_dialog_spec
from eegprep.functions.popfunc.pop_eventstat import event_values
from eegprep.functions.popfunc.pop_headplot import pop_headplot, pop_headplot_dialog_spec
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_plotdata import pop_plotdata
from eegprep.functions.popfunc.pop_plottopo import pop_plottopo
from eegprep.functions.popfunc.pop_spectopo import pop_spectopo, pop_spectopo_dialog_spec
from tests.fixtures import SAMPLE_DATASET_PATH
from eegprep.functions.popfunc.pop_topoplot import plot_channel_locations
from eegprep.functions.sigprocfunc.plottopo import plottopo
from eegprep.functions.sigprocfunc.signalstat import signalstat
from eegprep.functions.sigprocfunc.spectopo import spectopo
from eegprep.functions.sigprocfunc.topoplot import _toggle_electrode_label, topoplot


def _epoch(*, amplitude=1.0, trials=4, setname="Demo", theta=True):
    times = np.linspace(-100, 100, 21)
    data = np.zeros((3, times.size, trials))
    data[0] = amplitude
    data[1] = amplitude / 2
    data[2] = -amplitude
    chanlocs = [
        {"labels": "Fz", "theta": 0.0, "radius": 0.5},
        {"labels": "Pz", "theta": 180.0, "radius": 0.5},
        {"labels": "EOG", "theta": np.array([]), "radius": np.array([])},
    ]
    if not theta:
        for loc in chanlocs:
            loc["theta"] = np.array([])
            loc["radius"] = np.array([])
    return {
        "data": data,
        "nbchan": 3,
        "pnts": int(times.size),
        "trials": trials,
        "srate": 100.0,
        "xmin": -0.1,
        "xmax": 0.1,
        "times": times,
        "setname": setname,
        "chanlocs": chanlocs,
        "event": [
            {"type": "square", "latency": 6.0, "epoch": 1},
            {"type": "square", "latency": 26.0, "epoch": 2},
        ],
        "icaweights": np.eye(2),
        "icasphere": np.eye(2),
        "icawinv": np.eye(2),
        "icachansind": np.array([0, 1]),
    }


def _channel_axes(figure):
    return [axis for axis in figure.axes if axis.get_title()]


def _replay(command: str, **namespace):
    tree = ast.parse(_console_python_command(command), mode="eval")
    call = tree.body
    assert isinstance(call, ast.Call)
    call.keywords.append(ast.keyword(arg="plot", value=ast.Constant(value="off")))
    ast.fix_missing_locations(tree)
    return eval(compile(tree, "<history>", "eval"), namespace)


def test_f71_missing_coordinates_skip_maps_and_disable_frequency_field():
    eeg = _epoch(theta=False)
    result = pop_spectopo(eeg, 1, [-100, 100], "EEG", freqs=[10], plot="off")
    figure = result["figure"]
    assert figure.axes
    assert not any(axis.images for axis in figure.axes)
    plt.close(figure)
    spec = pop_spectopo_dialog_spec(eeg, dataflag=1)
    freqs = next(control for control in spec.controls if control.tag == "freqs")
    assert freqs.enabled is False
    with pytest.raises(ValueError, match="topoplot: no channel has a finite theta and radius"):
        topoplot(np.ones(3), eeg["chanlocs"])
    usable = [
        {"labels": "Fz", "theta": 0.0, "radius": 0.4},
        {"labels": "EOG", "theta": np.array([]), "radius": np.array([])},
        {"labels": "Pz", "theta": 180.0, "radius": 0.4},
    ]
    values = np.array([1.0, 5.0, -1.0])
    _, mixed, *_rest = topoplot(values, usable, noplot="on")
    _, kept, *_rest = topoplot(values[[0, 2]], [usable[0], usable[2]], noplot="on")
    np.testing.assert_allclose(mixed, kept, equal_nan=True)


def test_f72_empty_channels_stay_beside_a_shared_scale():
    locs = [
        {"labels": "Fz", "theta": 0.0, "radius": 0.4},
        {"labels": "C3", "theta": -90.0, "radius": 0.4},
        {"labels": "EOG", "theta": np.array([]), "radius": np.array([])},
    ]
    data = np.vstack([np.linspace(-1, 1, 11), np.linspace(-4, 2, 11), np.linspace(0, 3, 11)])
    figure = plottopo(data, times=np.arange(11), chanlocs=locs, channels=[1, 2, 3], rect=False)
    axes = {axis.get_title(): axis for axis in _channel_axes(figure)}
    assert axes["EOG"].get_position().x0 > axes["C3"].get_position().x0
    limits = [axis.get_ylim() for axis in axes.values()]
    assert limits[0] == pytest.approx(limits[1])
    assert limits[0] == pytest.approx(limits[2])
    assert any("+" in text.get_text() for text in figure.axes[-1].texts)
    plt.close(figure)


def test_f73_rectangular_arrays_share_one_scale():
    eeg = _epoch(amplitude=5)
    eeg["data"][1] = 1.0
    figure = pop_plotdata(eeg, 1, [1, 2], [1], ylimits=[0, 0], plot="off")
    axes = _channel_axes(figure)
    assert len(axes) == 2
    assert axes[0].get_ylim() == pytest.approx(axes[1].get_ylim())
    assert max(abs(value) for value in axes[0].get_ylim()) == pytest.approx(5)
    plt.close(figure)


def test_f74_compare_needs_two_datasets_and_subtracts_means():
    session = EEGPrepSession()
    session.store_current(_epoch(), new=True)
    with pytest.raises(ValueError, match="pop_envtopo: Need at least 2 datasets"):
        MenuActionDispatcher(session).dispatch("pop_envtopo:compare", None)

    first = _epoch(amplitude=4, setname="Left")
    second = _epoch(amplitude=1, setname="Right")
    spec = pop_envtopo_dialog_spec([first, second], compare=True)
    subtract = next(control for control in spec.controls if control.tag == "subindices")
    assert subtract.value == "2 1"
    figure, command = pop_envtopo([first, second], [-100, 100], compnums=[1], plot="off", return_com=True)
    assert "EEG([1 2])" in command
    ast.parse(_console_python_command(command))
    plt.close(figure)
    identical, _command = pop_envtopo([first, deepcopy(first)], [-100, 100], compnums=[1], plot="off", return_com=True)
    envelope = [line for line in identical.axes[0].lines if line.get_color() == "k" and line.get_linewidth() == 2]
    assert envelope
    plotted = np.concatenate([line.get_ydata() for line in envelope])
    assert np.nanmax(np.abs(plotted)) < 1e-8
    plt.close(identical)
    with pytest.raises(ValueError, match="Cannot process more than two datasets"):
        pop_envtopo([first, second, deepcopy(first)], [-100, 100], compnums=[1], plot="off")

    class Renderer:
        def run(self, spec, initial_values=None):
            return {
                "subindices": "3 1",
                "timerange": "-100 100",
                "limcontrib": "-100 100",
                "compsplot": "7",
                "components": "1",
                "subcomps": "",
                "title": "picked",
                "options": "",
            }

    third = _epoch(amplitude=9, setname="Third")
    picked, command = pop_envtopo([first, second, third], gui=True, renderer=Renderer(), plot="off", return_com=True)
    assert "EEG([3 1])" in command
    direct = pop_envtopo([third, first], [-100, 100], compnums=[1], title="picked", plot="off")
    picked_line = [line.get_ydata() for line in picked.axes[0].lines if line.get_linewidth() == 2]
    direct_line = [line.get_ydata() for line in direct.axes[0].lines if line.get_linewidth() == 2]
    np.testing.assert_allclose(np.concatenate(picked_line), np.concatenate(direct_line))
    plt.close(picked)
    plt.close(direct)


def test_f76_default_envelope_title_includes_the_dataset_name():
    eeg = _epoch(setname="Demo")
    figure = pop_envtopo(eeg, [-100, 100], compnums=[1], plot="off")
    texts = [axis.get_title() for axis in figure.axes]
    if figure._suptitle is not None:
        texts.append(figure._suptitle.get_text())
    assert any("Largest ERP components of Demo" in text for text in texts)
    plt.close(figure)
    blank = _epoch(setname="")
    untitled = pop_envtopo_dialog_spec(blank)
    title = next(control for control in untitled.controls if control.tag == "title")
    assert title.value == "Largest ERP components"


def test_f77_component_headplot_dialog_coregisters_and_documents_jet(tmp_path):
    eeg = _epoch()
    labels = " ".join(control.string or "" for control in pop_headplot_dialog_spec(eeg, typeplot=0).controls)
    assert "Co-register" in labels
    assert "Talairach" in labels
    assert "Manual coreg" in labels
    recorded = pop_loadset(SAMPLE_DATASET_PATH.parent / "eeglab_data_with_ica_tmp.set")
    figures = pop_headplot(
        recorded,
        typeplot=0,
        items=[1],
        setup={
            "splinefile": str(tmp_path / "component.spl"),
            "transform": [0, -10, 0, -0.1, 0, -1.6, 1100, 1100, 1100],
        },
        plot="off",
    )
    mesh = next(collection for collection in figures[0].axes[1].collections if collection.get_cmap().name == "jet")
    assert mesh.norm.vmin == pytest.approx(-mesh.norm.vmax)
    assert mesh.norm.vmin != pytest.approx(-1)
    plt.close(figures[0])


def test_f78_channel_location_footer_and_name_toggle():
    eeg = _epoch()
    figure = plot_channel_locations(eeg, mode="labels")
    footer = [text.get_text() for text in figure.axes[0].texts]
    assert "2 of 3 electrode locations shown" in footer
    assert "Click on electrodes to toggle name/number" in footer
    named = next(text for text in figure.axes[0].texts if text.get_text() == "Fz")
    _toggle_electrode_label(type("Event", (), {"artist": named})())
    assert named.get_text() == "1"
    _toggle_electrode_label(type("Event", (), {"artist": named})())
    assert named.get_text() == "Fz"
    assert figure.axes[0].get_ylim()[0] < -0.7
    plt.close(figure)


def test_f79_reversed_frequency_range_names_spectopo():
    data = np.sin(np.linspace(0, 8 * np.pi, 64))[None, :]
    with pytest.raises(ValueError, match=r"spectopo: freqrange must be \[lo hi\].*lower frequency first"):
        spectopo(data, 64, 100, freqrange=[25, 2], plot="off")


def test_f82_plot_history_is_matlab_and_replays():
    eeg = _epoch()
    result, command = pop_spectopo(
        eeg,
        1,
        [-100, 100],
        "EEG",
        percent=100,
        freq=[6, 10, 22],
        electrodes="off",
        plot="off",
        return_com=True,
    )
    assert command.startswith("figure; pop_spectopo(EEG, 1, [-100 100], 'EEG'")
    assert "'freq', [6 10 22]" in command
    assert command.endswith(";")
    replayed = _console_python_command(command)
    assert "figure" not in replayed
    ast.parse(replayed)
    plt.close(result["figure"])
    translated = _console_python_command("figure; pop_envtopo(EEG([2 1]), [0 100]);")
    assert "ALLEEG[1]" in translated and "ALLEEG[0]" in translated
    assert translated.index("ALLEEG[1]") < translated.index("ALLEEG[0]")
    ast.parse(translated)


def test_f83_comparison_legend_colors_match_the_traces():
    first = _epoch(amplitude=2)
    second = _epoch(amplitude=-1)
    result = pop_comperp([first, second], 1, [1], [2], chans=[1], plot="off")
    channel = _channel_axes(result["figure"])[0]
    legend = result["figure"].legends[0]
    assert legend.get_texts()
    for handle in legend.legend_handles:
        assert any(line.get_color() == handle.get_color() for line in channel.lines)
    plt.close(result["figure"])


def test_f85_event_latency_is_epoch_relative_milliseconds():
    eeg = _epoch()
    values = event_values(eeg, "latency", type=["square"])
    np.testing.assert_allclose(values, [-50.0, -60.0])


def test_f87_statistics_figure_shows_trimmed_values_and_a_channel_marker():
    eeg = _epoch()
    values = np.concatenate([np.zeros(20), np.array([8.0, -8.0])])
    locs = [loc for loc in eeg["chanlocs"] if np.size(loc["theta"])]
    result = signalstat(values, 1, "Potential", 5, "stats", map=[1], chan_locs=locs)
    text = "\n".join(item.get_text() for axis in result.figure.axes for item in axis.texts)
    assert "Trimmed mean" in text
    assert "quantile" in text
    assert "Kolmogorov-Smirnov test" in text
    assert result.figure.axes[3].collections
    plt.close(result.figure)


def test_positional_plot_history_replays_with_the_values_that_were_requested():
    eeg = _epoch(amplitude=5)
    spectra, spectopo_command = pop_spectopo(eeg, 1, [-100, 100], "EEG", freqrange=[4, 12], plot="off", return_com=True)
    replayed_spectra = _replay(spectopo_command, EEG=eeg, pop_spectopo=pop_spectopo)
    assert any(axis.get_xlim() == pytest.approx((4, 12)) for axis in replayed_spectra["figure"].axes)
    plt.close(spectra["figure"])
    plt.close(replayed_spectra["figure"])

    array, plottopo_command = pop_plottopo(eeg, [1], "only Fz", 0, timerange=[-50, 50], plot="off", return_com=True)
    replayed_array = _replay(plottopo_command, EEG=eeg, pop_plottopo=pop_plottopo)
    assert replayed_array._suptitle.get_text() == "only Fz"
    titled = _channel_axes(replayed_array)
    assert [axis.get_title() for axis in titled] == ["Fz"]
    assert titled[0].get_xlim() == pytest.approx((-50, 50))
    plt.close(array)
    plt.close(replayed_array)

    rectangular, plotdata_command = pop_plotdata(
        eeg, 1, [2], [1], "second channel", 0, 1, [-2, 2], plot="off", return_com=True
    )
    replayed_rectangular = _replay(plotdata_command, EEG=eeg, pop_plotdata=pop_plotdata)
    assert replayed_rectangular._suptitle.get_text() == "second channel"
    replayed_channels = _channel_axes(replayed_rectangular)
    assert len(replayed_channels) == 1
    assert replayed_channels[0].get_ylim() == pytest.approx((-2, 2))
    plt.close(rectangular)
    plt.close(replayed_rectangular)

    other = _epoch(amplitude=1)
    envelope, envtopo_command = pop_envtopo(
        [eeg, other], [-100, 100], compnums=[1], title="Difference", plot="off", return_com=True
    )
    replayed_envelope = _replay(envtopo_command, ALLEEG=[eeg, other], pop_envtopo=pop_envtopo)
    assert replayed_envelope._suptitle.get_text() == "Difference"
    direct_line = [line.get_ydata() for line in envelope.axes[0].lines if line.get_linewidth() == 2]
    replayed_line = [line.get_ydata() for line in replayed_envelope.axes[0].lines if line.get_linewidth() == 2]
    np.testing.assert_allclose(np.concatenate(replayed_line), np.concatenate(direct_line))
    plt.close(envelope)
    plt.close(replayed_envelope)


def test_f99_preferences_history_keeps_changed_options():
    previous = dict(EEG_OPTIONS)
    try:
        command = pop_editoptions(option_allmenus=0, option_cachesize=128, persist=False)
    finally:
        EEG_OPTIONS.clear()
        EEG_OPTIONS.update(previous)
    assert command == "pop_editoptions('option_cachesize', 128);"
