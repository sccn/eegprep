from __future__ import annotations

import ast
from copy import deepcopy

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eegprep.functions.popfunc.pop_comperp import pop_comperp
from eegprep.functions.popfunc.pop_envtopo import pop_envtopo
from eegprep.functions.popfunc.pop_erpimage import pop_erpimage
from eegprep.functions.popfunc.pop_headplot import pop_headplot
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_epoch import pop_epoch
from eegprep.functions.popfunc.pop_plotdata import pop_plotdata
from eegprep.functions.popfunc.pop_plottopo import pop_plottopo
from eegprep.functions.popfunc.pop_prop import pop_prop
from eegprep.functions.popfunc.pop_spectopo import pop_spectopo
from eegprep.functions.popfunc.pop_timtopo import pop_timtopo
from eegprep.functions.studyfunc.pop_chanplot import pop_chanplot
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


def test_pop_prop_plots_sample_channel_properties(sample_eeg):
    figure, command = pop_prop(sample_eeg, typecomp=1, chanorcomp=1, return_com=True)

    assert len(figure.axes) >= 3
    assert "pop_prop(EEG" in command
    _assert_python_command(command)
    plt.close(figure)


def test_pop_headplot_plots_sample_latency_map(sample_eeg):
    figures, command = pop_headplot(sample_eeg, typeplot=1, items=[0], return_com=True)

    assert len(figures) == 1
    assert figures[0].axes[0].name == "3d"
    _assert_python_command(command)
    plt.close(figures[0])


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


def test_plot_wrappers_fail_clearly_when_required_fields_are_missing(sample_epoch):
    with pytest.raises(ValueError, match="ICA"):
        pop_plotdata(sample_epoch, components=[1])
    continuous = deepcopy(sample_epoch)
    continuous["trials"] = 1
    continuous["data"] = continuous["data"][:, :, 0]
    with pytest.raises(ValueError, match="epoched"):
        pop_erpimage(continuous, typeplot=1, index=1)


def _assert_python_command(command: str) -> None:
    ast.parse(command)
