"""Original graphical source contracts and separate Python supplements.

pop_chansel and both interactive pop_compareerps bodies are entirely inactive
in the pinned suite; the supplemental tests do not claim their provenance.
"""

from __future__ import annotations

from copy import deepcopy

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal import get_window, welch

from eegprep.functions.popfunc.pop_compareerps import pop_compareerps
from eegprep.functions.popfunc.pop_comperp import pop_comperp
from eegprep.functions.popfunc.pop_envtopo import pop_envtopo
from eegprep.functions.popfunc.pop_erpimage import pop_erpimage
from eegprep.functions.popfunc.pop_plotdata import pop_plotdata, pop_plotdata_dialog_spec
from eegprep.functions.popfunc.pop_plottopo import pop_plottopo
from eegprep.functions.popfunc.pop_spectopo import pop_spectopo
from eegprep.functions.popfunc.pop_topoplot import pop_topoplot
from tests.eeglab_tests import eeglab_test
from tests.eeglab_tests.gui import close_reference_gui


def _source(name: str) -> str:
    return f"unittesting_popfunc/{name}/popfunc_{name}_wrapperTest.m"


def _reference_sample(backend, suite_root, filename="eeglab_data_epochs_ica.set"):
    return backend("pop_loadset", str(suite_root / "eeglab/sample_data" / filename))


def _reference_cell(*values):
    result = np.empty((1, len(values)), dtype=object)
    result[0] = values
    return result


def _reference_figure(backend, request):
    if request.config.getoption("--eeglab-backend") == "matlab":
        backend("figure", nargout=0)
    else:
        plt.figure()


def _reference_plot(backend, request, function, *arguments, new_figure=False):
    if new_figure:
        _reference_figure(backend, request)
    backend(function, *arguments, nargout=0)
    close_reference_gui(backend, request)


@pytest.mark.gui
@eeglab_test(_source("eeg_multieegplot"), "test_pass_continuous")
@eeglab_test(_source("eeg_multieegplot"), "test_pass_continuous_reject")
@eeglab_test(_source("eeg_multieegplot"), "test_pass_epochs")
def test_reference_multieegplot_original_rejection_arrays(eeglab_backend, request, subtests):
    for case in ("continuous", "continuous_reject", "epochs"):
        with subtests.test(source=case):
            eeg = eeglab_backend("eeg_emptyset")
            eeg.update(nbchan=2.0, srate=1.0, xmin=0.0)
            if case == "epochs":
                eeg.update(pnts=3.0, trials=3.0, xmax=2.0)
                eeg["data"] = np.array([[[1, 1, 2]] * 3, [[2, 2, 2], [1, 1, 1], [1, 1, 1]]], dtype=float)
                trialrej, elecrej = np.zeros((1, 3)), np.zeros((1, 3))
            else:
                eeg.update(pnts=9.0, trials=1.0, xmax=8.0)
                eeg["data"] = np.array([[1, 1, 1, 1, 1, 1, 2, 2, 2], [2, 1, 1, 2, 1, 1, 2, 1, 1]], dtype=float)
                trialrej, elecrej = np.zeros((1, 9)), np.zeros((1, 2))
                if case == "continuous_reject":
                    trialrej[0, [1, 6]] = 1.0
                    elecrej = np.ones((2, 9))
            matlab = request.config.getoption("--eeglab-backend") == "matlab"
            # Retain only the Python window for closing; MATLAB requested no output.
            window = eeglab_backend("eeg_multieegplot", eeg["data"], trialrej, elecrej, nargout=0 if matlab else 1)
            close_reference_gui(eeglab_backend, request, window=window)


@pytest.mark.gui
@eeglab_test(_source("pop_comperp"), "test_test_pop_comperp")
def test_reference_comperp_original_dataset_selections(eeglab_backend, eeglab_suite_root, request):
    selected = []
    for position in (1.0, 2.0):
        eeg = _reference_sample(eeglab_backend, eeglab_suite_root)
        selected.append(
            eeglab_backend("pop_selectevent", eeg, "position", position, "deleteevents", "off", "deleteepochs", "on")
        )
    fields = list(selected[0])
    datasets = np.array(
        [[tuple(eeg[field] for field in fields) for eeg in selected]], dtype=[(field, object) for field in fields]
    )
    for mode in (1.0, 0.0):
        _reference_plot(
            eeglab_backend,
            request,
            "pop_comperp",
            datasets,
            mode,
            np.array([[1.0, 2.0]]),
            np.empty((0, 0)),
            "alpha",
            0.05,
            "std",
            "on",
            "allerps",
            "on",
        )


@pytest.mark.gui
@eeglab_test(_source("pop_crossf"), "test_pass_tlimits_empty")
def test_reference_crossf_original_empty_limits(eeglab_backend, eeglab_suite_root, request):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "unittesting_popfunc/pop_crossf/test.set"), "")
    eeglab_backend("pop_crossf", eeg, 1.0, 1.0, 2.0, np.empty((0, 0)), np.array([[3.0, 0.5]]))
    close_reference_gui(eeglab_backend, request)


@pytest.mark.gui
@eeglab_test(_source("pop_crossf"), "test_test_pop_crossf")
def test_reference_crossf_original_channel_component_calls(eeglab_backend, eeglab_suite_root, request):
    eeg = _reference_sample(eeglab_backend, eeglab_suite_root)
    options = (
        "topovec",
        eeg["icawinv"][:, [3, 8]].T,
        "elocs",
        eeg["chanlocs"],
        "chaninfo",
        eeg["chaninfo"],
        "title",
        "Component 4-9 Phase Coherence",
        "alpha",
        0.01,
        "padratio",
        4.0,
    )
    for mode, extra in ((0.0, options), (1.0, ())):
        _reference_plot(
            eeglab_backend,
            request,
            "pop_crossf",
            eeg,
            mode,
            4.0,
            9.0,
            np.array([[-1000.0, 2000.0]]),
            np.array([[3.0, 0.5]]),
            "type",
            "phasecoher",
            *extra,
            new_figure=True,
        )


@pytest.mark.gui
@eeglab_test(_source("pop_eegplot"), "test_test_pop_eegplot")
def test_reference_pop_eegplot_original_recordings(eeglab_backend, eeglab_suite_root, request):
    for filename, mode in (("eeglab_data.set", 1.0), ("eeglab_data_epochs_ica.set", 0.0)):
        eeg = _reference_sample(eeglab_backend, eeglab_suite_root, filename)
        matlab = request.config.getoption("--eeglab-backend") == "matlab"
        window = eeglab_backend("pop_eegplot", eeg, mode, 0.0, 0.0, nargout=0 if matlab else 1)
        close_reference_gui(eeglab_backend, request, window=window)


@pytest.mark.gui
@eeglab_test(_source("pop_envtopo"), "test_test_pop_envtopo")
def test_reference_envtopo_original_contribution_windows(eeglab_backend, eeglab_suite_root, request):
    eeg = _reference_sample(eeglab_backend, eeglab_suite_root)
    for limits in ((-1000.0, 1999.7949), (200.0, 500.0)):
        _reference_plot(
            eeglab_backend,
            request,
            "pop_envtopo",
            eeg,
            np.array([[-1000.0, 1999.7949]]),
            "limcontrib",
            np.array([limits]),
            "compnums",
            -7.0,
            "title",
            'Largest ERP components of Epoched from "ee114 continuous" dataset',
            "electrodes",
            "off",
            new_figure=True,
        )


@pytest.mark.gui
@eeglab_test(_source("pop_headplot"), "test_test_pop_headplot")
def test_reference_headplot_original_setup(eeglab_backend, eeglab_suite_root, eeglab_working_directory, request):
    eeg = _reference_sample(eeglab_backend, eeglab_suite_root)
    setup = np.empty((1, 5), dtype=object)
    setup[0] = [
        "eeglab_data_epochs_ica.spl",
        "meshfile",
        "mheadnew.mat",
        "transform",
        np.array([[-0.31937, -5.9693, 13.1812, 0.050931, 0.017213, -1.5501, 1.0822, 1.0004, 0.92352]]),
    ]
    eeglab_backend(
        "pop_headplot",
        eeg,
        1.0,
        np.arange(0.0, 501.0, 100)[None, :],
        "ERP scalp maps of dataset:EEG Data epochs",
        np.array([[2.0, 3.0]]),
        "setup",
        setup,
    )
    close_reference_gui(eeglab_backend, request)
    (eeglab_working_directory / "eeglab_data_epochs_ica.spl").unlink()


@pytest.mark.gui
@eeglab_test(_source("pop_newcrossf"), "test_test_pop_newcrossf")
def test_reference_newcrossf_original_recordings(eeglab_backend, eeglab_suite_root, request):
    eeg = _reference_sample(eeglab_backend, eeglab_suite_root)
    for mode, topography, title in (
        (1.0, np.array([[1.0, 2.0]]), "Channel FPz-EOG1 Phase Coherence"),
        (0.0, eeg["icawinv"][:, [0, 1]].T, "Component 1-2 Phase Coherence"),
    ):
        _reference_plot(
            eeglab_backend,
            request,
            "pop_newcrossf",
            eeg,
            mode,
            1.0,
            2.0,
            np.array([[-1000.0, 1992.0]]),
            np.array([[3.0, 0.5]]),
            "type",
            "phasecoher",
            "topovec",
            topography,
            "elocs",
            eeg["chanlocs"],
            "chaninfo",
            eeg["chaninfo"],
            "title",
            title,
            "padratio",
            1.0,
            new_figure=True,
        )
    eeg = _reference_sample(eeglab_backend, eeglab_suite_root, "eeglab_data.set")
    _reference_plot(
        eeglab_backend,
        request,
        "pop_newcrossf",
        eeg,
        1.0,
        1.0,
        2.0,
        np.array([[0.0, 238305.0]]),
        np.array([[3.0, 0.5]]),
        "freqs",
        np.arange(1.0, 51.0)[None, :],
        "type",
        "phasecoher",
        "title",
        "Channel 1-2 Phase Coherence",
        "padratio",
        1.0,
        new_figure=True,
    )


@pytest.mark.gui
@eeglab_test(_source("pop_plotdata"), "test_test_pop_plotdata")
def test_reference_plotdata_original_eleven_calls(eeglab_backend, eeglab_suite_root, request):
    eeg = _reference_sample(eeglab_backend, eeglab_suite_root)
    all_trials = np.arange(1.0, 81.0)[None, :]
    for mode, first, last, trials, title, single, polarity, limits in (
        (1, 1, 32, all_trials, "", 0, 1, [0, 0]),
        (0, 1, 30, all_trials, "", 0, 1, [0, 0]),
        (1, 1, 1, np.arange(55.0, 68.0)[None, :], "", 0, 1, [0, 0]),
        (0, 2, 7, all_trials, "", 0, 1, [0, 0]),
        (0, 1, 30, all_trials, "test", 0, 1, [0, 0]),
        (1, 1, 32, np.array([[2.0, 5.0, 7.0]]), "", 1, 1, [0, 0]),
        (0, 1, 30, np.array([[2.0, 5.0, 7.0]]), "", 1, 1, [0, 0]),
        (1, 1, 32, all_trials, "", 0, -1, [0, 0]),
        (0, 1, 30, all_trials, "", 0, -1, [0, 0]),
        (1, 1, 32, all_trials, "", 0, 1, [-250, 350]),
        (0, 1, 30, all_trials, "", 0, 1, [-250, 350]),
    ):
        eeglab_backend(
            "pop_plotdata",
            eeg,
            float(mode),
            np.arange(float(first), last + 1)[None, :],
            trials,
            title,
            float(single),
            float(polarity),
            np.array([limits], dtype=float),
        )
    close_reference_gui(eeglab_backend, request, all_figures=True)


@pytest.mark.gui
@eeglab_test(_source("pop_plottopo"), "test_test_pop_plottopo")
def test_reference_plottopo_original_channels(eeglab_backend, eeglab_suite_root, request):
    eeg = _reference_sample(eeglab_backend, eeglab_suite_root)
    _reference_plot(
        eeglab_backend,
        request,
        "pop_plottopo",
        eeg,
        np.arange(1.0, 33.0)[None, :],
        "ee114 continuous (h.p. 1Hz) epochs",
        0.0,
        new_figure=True,
    )


@pytest.mark.gui
@eeglab_test(_source("pop_selectcomps"), "test_test_pop_selectcomps")
def test_reference_selectcomps_original_components(eeglab_backend, eeglab_suite_root, request):
    eeg = _reference_sample(eeglab_backend, eeglab_suite_root)
    _reference_plot(eeglab_backend, request, "pop_selectcomps", eeg, np.arange(1.0, 31.0)[None, :])


@pytest.mark.gui
@eeglab_test(_source("pop_timtopo"), "test_test_pop_timtopo")
def test_reference_timtopo_original_default_latency(eeglab_backend, eeglab_suite_root, request):
    eeg = _reference_sample(eeglab_backend, eeglab_suite_root)
    _reference_plot(
        eeglab_backend,
        request,
        "pop_timtopo",
        eeg,
        np.array([[-999.9316, 1992.0513]]),
        np.array([[np.nan]]),
        "ERP data and scalp maps of ee114 continuous (h.p. 1Hz) epochs",
        new_figure=True,
    )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.gui
@eeglab_test(_source("pop_spectopo"), "test_test_pop_spectopo")
def test_reference_spectopo_original_nine_calls(eeglab_backend, eeglab_suite_root, request):
    eeg = _reference_sample(eeglab_backend, eeglab_suite_root, "eeglab_data.set")
    continuous = ("percent", 15.0, "freq", np.empty((0, 0)), "freqrange", np.array([[2.0, 25.0]]), "electrodes", "off")
    for window in ((), ("wintype", "blackmanharris"), ("wintype", "blackmanharris", "blckhn", 3.0)):
        _reference_plot(
            eeglab_backend,
            request,
            "pop_spectopo",
            eeg,
            1.0,
            np.array([[0.0, 238288.3983]]),
            "EEG",
            *continuous,
            *window,
            new_figure=True,
        )
    eeg = _reference_sample(eeglab_backend, eeglab_suite_root)
    components = np.arange(1.0, eeg["icaweights"].shape[0] + 1)[None, :]
    for window in ((), ("wintype", "blackmanharris")):
        _reference_plot(
            eeglab_backend,
            request,
            "pop_spectopo",
            eeg,
            1.0,
            np.array([[0.0, 238288.3983]]),
            "EEG",
            "percent",
            50.0,
            "freq",
            np.array([[8.0, 10.0, 12.0]]),
            "freqrange",
            np.array([[2.0, 25.0]]),
            "electrodes",
            "off",
            *window,
            new_figure=True,
        )
        _reference_plot(
            eeglab_backend,
            request,
            "pop_spectopo",
            eeg,
            0.0,
            np.array([[-1000.0, 1999.7949]]),
            "EEG",
            "freq",
            10.0,
            "plotchan",
            0.0,
            "icacomps",
            components,
            "nicamaps",
            5.0,
            "freqrange",
            np.array([[2.0, 25.0]]),
            "electrodes",
            "off",
            *window,
            new_figure=True,
        )
        _reference_plot(
            eeglab_backend,
            request,
            "pop_spectopo",
            eeg,
            0.0,
            np.array([[-1000.0, 1999.7949]]),
            "EEG",
            "freq",
            10.0,
            "plotchan",
            27.0,
            "icacomps",
            components,
            "nicamaps",
            6.0,
            "icamode",
            "sub",
            "freqrange",
            np.array([[2.0, 30.0]]),
            "electrodes",
            "off",
            *window,
            new_figure=True,
        )


@pytest.mark.gui
@eeglab_test(_source("pop_timef"), "test_test_pop_timef")
def test_reference_timef_original_channel_component_calls(eeglab_backend, eeglab_suite_root, request):
    eeg = _reference_sample(eeglab_backend, eeglab_suite_root)
    _reference_plot(
        eeglab_backend,
        request,
        "pop_timef",
        eeg,
        1.0,
        14.0,
        np.array([[-1000.0, 2000.0]]),
        np.array([[3.0, 0.5]]),
        "type",
        "phasecoher",
        "topovec",
        14.0,
        "elocs",
        eeg["chanlocs"],
        "title",
        'Channel Cz power and inter-trial phase coherence (Epoched from "ee114 continuous" dataset)',
        "alpha",
        0.01,
        "padratio",
        4.0,
        "plotphase",
        "off",
        new_figure=True,
    )
    _reference_plot(
        eeglab_backend,
        request,
        "pop_timef",
        eeg,
        0.0,
        1.0,
        np.array([[-1000.0, 2000.0]]),
        np.array([[0.0]]),
        "type",
        "phasecoher",
        "topovec",
        eeg["icawinv"][:, :1],
        "elocs",
        eeg["chanlocs"],
        "chaninfo",
        eeg["chaninfo"],
        "title",
        'Component 1 power and inter-trial phase coherence (Epoched from "ee114 continuous" dataset)',
        "padratio",
        4.0,
        "plotphase",
        "off",
        "maxfreq",
        30.0,
        new_figure=True,
    )


@pytest.mark.gui
@eeglab_test(_source("pop_topoplot"), "test_test_pop_topoplot")
def test_reference_pop_topoplot_original_montage_variants(eeglab_backend, eeglab_suite_root, request):
    eeg = _reference_sample(eeglab_backend, eeglab_suite_root)
    times = np.arange(0.0, 501.0, 100)[None, :]
    title = "ee114 continuous (h.p. 1Hz) epochs ERP"
    _reference_plot(
        eeglab_backend, request, "pop_topoplot", eeg, 1.0, times, title, np.array([[2.0, 3.0]]), "electrodes", "off"
    )
    _reference_plot(
        eeglab_backend,
        request,
        "topoplot",
        np.empty((0, 0)),
        eeg["chanlocs"],
        "style",
        "blank",
        "electrodes",
        "labelpoint",
        new_figure=True,
    )
    _reference_plot(
        eeglab_backend,
        request,
        "pop_topoplot",
        eeg,
        0.0,
        np.arange(1.0, 13.0)[None, :],
        "Continuous EEG Data epochs ERP",
        np.array([[3.0, 4.0]]),
        "electrodes",
        "off",
    )
    eeglab_backend("eeg_getversion", nargout=2)
    matrix = np.array([[11.0, 12.0, 0.0, 1.0], [13.0, 14.0, 15.0, -2.0]])
    changed = deepcopy(eeg)
    changed["chanmatrix"] = matrix
    _reference_plot(
        eeglab_backend,
        request,
        "pop_topoplot",
        changed,
        1.0,
        np.array([[100.0, 200.0, 300.0, 400.0]]),
        "EEG Data epochs",
        np.array([[2.0, 2.0]]),
        0.0,
    )
    for channel_matrix in (matrix, np.empty((0, 0))):
        changed = deepcopy(eeg)
        changed["chanlocs"] = np.empty((0, 0))
        changed["chanmatrix"] = channel_matrix
        _reference_plot(
            eeglab_backend,
            request,
            "pop_topoplot",
            changed,
            1.0,
            times,
            title,
            np.array([[2.0, 3.0]]),
            "electrodes",
            "off",
        )


@pytest.mark.gui
@eeglab_test(_source("pop_prop"), "test_test_pop_prop")
def test_reference_prop_original_multiwindow_calls(eeglab_backend, eeglab_suite_root, request):
    matlab = request.config.getoption("--eeglab-backend") == "matlab"
    legacy_handles = False
    if matlab:
        constants = eeglab_backend("eegprep_test_run_script", "icadefs", _reference_cell("VERS"))
        legacy_handles = float(np.asarray(constants["VERS"]).item()) < 8.04
    options = _reference_cell("freqrange", np.array([[2.0, 50.0]]))
    for filename, modes in (("eeglab_data.set", (1.0,)), ("eeglab_data_epochs_ica.set", (1.0, 0.0))):
        eeg = _reference_sample(eeglab_backend, eeglab_suite_root, filename)
        for mode in modes:
            for selection in (1.0, np.array([[2.0, 7.0]])):
                eeglab_backend("pop_prop", eeg, mode, selection, 0.0, options, nargout=0)
                close_reference_gui(eeglab_backend, request)
                if not np.isscalar(selection):
                    close_reference_gui(eeglab_backend, request)
        if legacy_handles:
            for mode in modes:
                _reference_plot(eeglab_backend, request, "pop_prop", eeg, mode, 1.0, 1.0, options)
        else:
            if matlab:
                eeglab_backend("eegprep_test_prop_parent", "create", nargout=0)
            else:
                figure = plt.figure()
            for mode in modes:
                if matlab:
                    # pop_prop uses parent.Tag on modern MATLAB. A double
                    # handle transported through MAT files changes that type.
                    eeglab_backend("eegprep_test_prop_parent", "call", eeg, mode, 1.0, options, nargout=0)
                    close_reference_gui(eeglab_backend, request)
                else:
                    _reference_plot(eeglab_backend, request, "pop_prop", eeg, mode, 1.0, figure, options)
            if matlab:
                eeglab_backend("eegprep_test_prop_parent", "close", nargout=0)
            else:
                plt.close(figure)


@pytest.mark.gui
@eeglab_test(_source("pop_erpimage"), "test_test_pop_erpimage")
def test_reference_pop_erpimage_complete_original_workflow(eeglab_backend, eeglab_suite_root, request):
    eeg = _reference_sample(eeglab_backend, eeglab_suite_root)
    empty = np.empty((0, 0))
    no_events = np.empty((0, 0), dtype=object)
    rt = _reference_cell("rt")
    topo = _reference_cell(27.0, eeg["chanlocs"])
    for smooth, event, field, extra in (
        (0.0, no_events, "", ()),
        (10.0, no_events, "", ()),
        (10.0, rt, "latency", ()),
        (10.0, no_events, "position", ("renorm", "yes")),
        (10.0, rt, "position", ("renorm", "yes")),
        (10.0, rt, "position", ("renorm", "100*x")),
        (10.0, no_events, "", ("phasesort", np.array([[0.0, 50.0, 10.0]]))),
    ):
        _reference_plot(
            eeglab_backend,
            request,
            "pop_erpimage",
            eeg,
            1.0,
            27.0,
            empty,
            "POz",
            smooth,
            1.0,
            event,
            empty,
            field,
            "topo",
            topo,
            "erp",
            "cbar",
            *extra,
            new_figure=True,
        )
    _reference_plot(
        eeglab_backend,
        request,
        "pop_erpimage",
        eeg,
        1.0,
        27.0,
        empty,
        "POz",
        10.0,
        1.0,
        no_events,
        empty,
        "",
        "topo",
        topo,
        "erp",
        "limits",
        np.array([[-200.0, 1000.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]),
        "cbar",
        "phasesort",
        np.array([[0.0, 50.0, 9.0, 11.0]]),
        "coher",
        np.array([[9.0, 11.0, 0.01]]),
        new_figure=True,
    )
    _reference_plot(
        eeglab_backend,
        request,
        "pop_erpimage",
        eeg,
        1.0,
        27.0,
        empty,
        "POz",
        10.0,
        1.0,
        rt,
        empty,
        "latency",
        "topo",
        topo,
        "erp",
        "limits",
        np.array([[-500.0, 1500.0, np.nan, np.nan, -3.0, 3.0, np.nan, np.nan]]),
        "cbar",
        "phasesort",
        np.array([[0.0, 50.0, 9.0, 11.0]]),
        "plotamps",
        "coher",
        np.array([[9.0, 11.0, 0.01]]),
        "spec",
        np.array([[2.0, 50.0]]),
        "vert",
        500.0,
        new_figure=True,
    )
    for projection, title in ((empty, "Comp. 6"), (27.0, "Comp. 6 -> POz")):
        _reference_plot(
            eeglab_backend,
            request,
            "pop_erpimage",
            eeg,
            0.0,
            6.0,
            projection,
            title,
            10.0,
            1.0,
            rt,
            empty,
            "latency",
            "yerplabel",
            "",
            "topo",
            _reference_cell(eeg["icawinv"][:, 5:6], eeg["chanlocs"]),
            "erp",
            "limits",
            np.array([[-300.0, 500.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]),
            "cbar",
            "phasesort",
            np.array([[-150.0, 0.0, 9.0, 11.0]]),
            "coher",
            np.array([[9.0, 11.0, 0.01]]),
            new_figure=True,
        )
    rng = np.random.default_rng()
    times = (-250.0 + np.arange(200) * (1000 / 256))[None, :]
    _reference_figure(eeglab_backend, request)
    for position, title, extra, coherence in (
        (1.0, "test phasesort", (), np.array([[10.0]])),
        (2.0, "test phasesort, allamps", ("plotamps",), np.array([[10.0, 10.0, 0.05]])),
    ):
        eeglab_backend("sbplot", 1.0, 2.0, position, nargout=0)
        eeglab_backend(
            "erpimage",
            rng.standard_normal((200, 1000)),
            rng.standard_normal((1, 1000)),
            times,
            title,
            20.0,
            5.0,
            "phasesort",
            np.array([[200.0, 0.0, 10.0]]),
            *extra,
            "erp",
            "cbar",
            "coher",
            coherence,
            nargout=0,
        )
    close_reference_gui(eeglab_backend, request)
    _reference_plot(
        eeglab_backend,
        request,
        "erpimage",
        rng.standard_normal((100, 100)),
        np.arange(-100.0, 0.0)[None, :],
        np.arange(-200.0, 296.0, 5)[None, :],
        "",
        1.0,
        1.0,
        "plotamps",
        "coher",
        np.array([[10.0, 10.0, 0.05]]),
        new_figure=True,
    )
    trials = int(np.asarray(eeg["trials"]).item())
    trial_numbers = np.arange(1.0, trials + 1)[None, :]
    auxvar = trial_numbers.T - np.array([[200.0, 400.0, 600.0]])
    times = np.linspace(
        float(np.asarray(eeg["xmin"]).item()) * 1000,
        float(np.asarray(eeg["xmax"]).item()) * 1000,
        int(np.asarray(eeg["pnts"]).item()),
    )[None, :]
    _reference_figure(eeglab_backend, request)
    for position, title, extra in (
        (1.0, "test auxvar", ()),
        (2.0, "test auxvar (amp sort)", ("ampsort", np.array([[100.0, 10.0, 10.0, 13.0]]))),
        (3.0, "test auxvar (phase sort)", ("phasesort", np.array([[-100.0, 20.0, 10.0, 11.0, 30.0]]))),
        (
            4.0,
            "test auxvar (amp sort)",
            ("ampsort", np.array([[100.0, 10.0, 10.0, 13.0]]), "coher", np.array([[10.0, 11.0, 0.01]]), "plotamps"),
        ),
    ):
        eeglab_backend("sbplot", 2.0, 3.0, position, nargout=0)
        eeglab_backend(
            "erpimage",
            eeg["data"][0].reshape(1, -1, order="F"),
            np.ones((1, trials)) * eeg["xmax"] * 1000,
            times,
            title,
            10.0,
            1.0,
            "topo",
            _reference_cell(1.0, eeg["chanlocs"]),
            "erp",
            "cbar",
            "auxvar",
            auxvar,
            *extra,
            nargout=0,
        )
    close_reference_gui(eeglab_backend, request)
    _reference_figure(eeglab_backend, request)
    for position, title, extra in (
        (1.0, "ERP alone", ("erp",)),
        (2.0, "ERP + plotamps", ("erp", "plotamps", "coher", np.array([[8.0, 12.0, 0.01]]))),
        (3.0, "ERP + erpalpha + plotamps", ("erpalpha", 0.01, "plotamps", "coher", np.array([[8.0, 12.0, 0.01]]))),
    ):
        eeglab_backend("sbplot", 1.0, 3.0, position, nargout=0)
        eeglab_backend("erpimage", eeg["data"][12], trial_numbers, eeg["times"], title, 1.0, 1.0, *extra, nargout=0)
        if position > 1:
            if request.config.getoption("--eeglab-backend") == "matlab":
                eeglab_backend("drawnow", nargout=0)
            else:
                plt.draw()
    close_reference_gui(eeglab_backend, request)
    data = rng.random((1000, 100))
    st1 = np.arange(1.0, 101.0)[None, :] + 100
    st2 = np.arange(1.0, 201.0, 2)[None, :] + 200
    for extra in ((), ("align", np.inf)):
        _reference_plot(
            eeglab_backend,
            request,
            "erpimage",
            data,
            st1,
            np.linspace(-300.0, 700.0, 1000)[None, :],
            "test",
            10.0,
            0.0,
            "auxvar",
            st2,
            *extra,
            new_figure=True,
        )


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


def test_pop_plottopo_draws_selected_channel_trial_averages(epoched_eeg: dict) -> None:
    figure, command = pop_plottopo(epoched_eeg, [1, 3, 5], "selected channels", 0, return_com=True)
    single_trials = pop_plottopo(epoched_eeg, [1, 3], "single trials", 1, rect=True)

    axes = {axis.get_title(): axis for axis in figure.axes}
    np.testing.assert_allclose(axes["Ch3"].lines[0].get_ydata(), epoched_eeg["data"][2].mean(axis=1))
    assert len(axes) == 3
    assert axes["Ch3"].yaxis_inverted()
    assert command.startswith("pop_plottopo(EEG, [1, 3, 5], 'selected channels', 0)")
    assert len(single_trials.axes[0].lines) == epoched_eeg["trials"] + 1


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
