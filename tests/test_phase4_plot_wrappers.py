from __future__ import annotations

import ast
from copy import deepcopy
import importlib
import os
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
import numpy as np
import pytest
import scipy.io

from eegprep.functions.guifunc.qt import QtDialogRenderer
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.popfunc.pop_comperp import pop_comperp
from eegprep.functions.popfunc.pop_envtopo import pop_envtopo
from eegprep.functions.popfunc.pop_erpimage import pop_erpimage
from eegprep.functions.popfunc.pop_headplot import (
    pop_headplot,
    pop_headplot_dialog_spec,
    _current_spline_file,
)
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_epoch import pop_epoch
from eegprep.functions.popfunc.plot_utils import (
    component_activations,
    data_time_slice,
    parse_plot_options_text,
)
from eegprep.functions.popfunc.pop_plotdata import pop_plotdata
from eegprep.functions.popfunc.pop_plottopo import pop_plottopo
from eegprep.functions.popfunc.pop_prop import pop_prop
from eegprep.functions.popfunc.pop_signalstat import pop_signalstat
from eegprep.functions.popfunc.pop_spectopo import pop_spectopo
from eegprep.functions.popfunc.pop_timtopo import pop_timtopo
from eegprep.functions.popfunc.pop_topoplot import pop_topoplot
from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.sigprocfunc.axcopy import axcopy
from eegprep.functions.sigprocfunc.topoplot import plot_channel_location, topoplot
from eegprep.functions.sigprocfunc.coregister import (
    ElectrodeSet,
    apply_coregistration_transform,
    estimate_coregistration_transform,
    traditional_transform_matrix,
)
from eegprep.functions.sigprocfunc.headplot import (
    headplot_setup,
    load_headplot_spline,
    _interpolate_values,
)
from tests.fixtures import SAMPLE_DATASET_PATH, create_test_eeg_with_ica
from tests.eeglab_tests import eeglab_test
from tests.eeglab_tests.gui import close_reference_gui


_AXCOPY_SOURCE = "unittesting_sigprocfunc/axcopy/sigprocfunc_axcopy_wrapperTest.m"
_HEADPLOT_SOURCE = "unittesting_sigprocfunc/headplot/sigprocfunc_headplot_wrapperTest.m"
_TIMTOPO_SOURCE = "unittesting_sigprocfunc/timtopo/sigprocfunc_timtopo_wrapperTest.m"


@pytest.mark.gui
@eeglab_test(_AXCOPY_SOURCE, "test_pass_existing_figure")
@eeglab_test(_AXCOPY_SOURCE, "test_pass_general")
@eeglab_test(_AXCOPY_SOURCE, "test_pass_one_arg")
def test_reference_axcopy_original_calls(eeglab_backend, request, subtests):
    for existing, args in ((True, ()), (False, ()), (True, ("noticks",))):
        with subtests.test(existing=existing, args=args):
            if existing:
                if request.config.getoption("--eeglab-backend") == "matlab":
                    eeglab_backend("plot", np.array([[1.0, 2.0, 3.0]]), np.array([[1.0, 2.0, 3.0]]), nargout=0)
                else:
                    plt.plot([1, 2, 3], [1, 2, 3])
            eeglab_backend("axcopy", *args, nargout=0)
            close_reference_gui(eeglab_backend, request)


@pytest.mark.gui
@eeglab_test(_HEADPLOT_SOURCE, "test_pass_general")
@eeglab_test(_HEADPLOT_SOURCE, "test_pass_options")
@eeglab_test(_HEADPLOT_SOURCE, "test_pass_wireframe")
def test_reference_headplot_original_spline_workflows(
    eeglab_backend,
    request,
    eeglab_suite_root,
    eeglab_working_directory,
    subtests,
):
    shutil.copy2(
        eeglab_suite_root / "unittesting_sigprocfunc/headplot/test.locs", eeglab_working_directory / "test.locs"
    )
    spline_file = eeglab_working_directory / "test.spline"
    for options in (
        {},
        {
            "electrodes": "off",
            "title": "testcase",
            "labels": 2.0,
            "cbar": 0.0,
            "view": np.array([[45.0, 30.0]]),
            "maplimits": "absmax",
            "verbose": "off",
        },
        {"lighting": "off"},
    ):
        with subtests.test(options=options):
            if spline_file.exists():
                spline_file.unlink()
            if request.config.getoption("--eeglab-backend") == "matlab":
                eeglab_backend("headplot", "setup", "test.locs", "test.spline", nargout=0)
            else:
                eeglab_backend("headplot", "setup", "test.locs", splinefile="test.spline", nargout=0)
            assert spline_file.is_file(), "headplot initialisation error (spline_file does not exist)"
            values = np.arange(1.0, 10.0)[None, :]
            if request.config.getoption("--eeglab-backend") == "matlab":
                handle = eeglab_backend("eegprep_test_gui_handle", "headplot", values, "test.spline", **options)
                # Only the native graphics query crosses eval; Python owns the assertion.
                is_struct = eeglab_backend("eval", f"isstruct(get({float(np.asarray(handle).item())!r}))")
                assert bool(np.asarray(is_struct).item())
            else:
                figure = eeglab_backend("headplot", values, "test.spline", **options)
                assert isinstance(figure.properties(), dict)
            if spline_file.exists():
                spline_file.unlink()
            close_reference_gui(eeglab_backend, request)


@pytest.mark.gui
@eeglab_test(_TIMTOPO_SOURCE, "test_test_timtopo")
def test_reference_timtopo_original_seventeen_cases(eeglab_backend, request, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data_epochs_ica.set"))
    data = np.asarray(eeg["data"]).mean(axis=2)
    limits = [float(np.asarray(eeg["xmin"]).item()) * 1000, float(np.asarray(eeg["xmax"]).item()) * 1000]
    chanlocs_file = str(eeglab_suite_root / "eeglab/sample_data/eeglab_chan32.locs")
    title = "ERP data and scalp maps of EEG Data epochs"
    cases = (
        ("", eeg["chanlocs"], {"limits": limits, "plottimes": [np.nan], "title": title}),
        ("", chanlocs_file, {"limits": [-1000, 1992.1875], "plottimes": [np.nan], "title": title}),
        ("", eeg["chanlocs"], {"limits": [*limits, -10, 10], "plottimes": [np.nan], "title": title}),
        ("", eeg["chanlocs"], {"limits": [*limits, -10, 10], "plottimes": [100, 200, 300, 400], "title": title}),
        ("", eeg["chanlocs"], {"limits": limits, "plottimes": [100, 200, 300, 400], "title": title}),
        ("", eeg["chanlocs"], {"limits": limits, "plottimes": [100, 200, 300, 400], "title": title, "voffsets": 10.0}),
        ("Testcase", eeg["chanlocs"], {"limits": [-1000, 1992.1875], "plottimes": [0, 100, 200, 300], "title": title}),
        ("Testcase 1", eeg["chanlocs"], {"limits": [-1000, 1992.1875], "plottimes": [0, 100, 200, 300]}),
        ("Testcase 2", eeg["chanlocs"], {"limits": [-1000, 1992.1875], "title": title}),
        ("Testcase 3", eeg["chanlocs"], {"plottimes": [0, 100, 200, 300], "title": title}),
        ("Testcase 4", eeg["chanlocs"], {}),
        ("Testcase 5", eeg["chanlocs"], {"limits": [-2000, 3000], "plottimes": [np.nan], "title": title}),
        ("Testcase 6", eeg["chanlocs"], {"limits": [-1000, 1992.1875], "plottimes": [-1001], "title": title}),
        ("Testcase 7", eeg["chanlocs"], {"limits": [-1000, 1992.1875], "plottimes": [2001], "title": title}),
        ("Testcase 8", eeg["chanlocs"], {"limits": [-1000, 1992.1875], "plottimes": [-1001, 0, 100], "title": title}),
        ("Testcase 9", eeg["chanlocs"], {"limits": [0, 1000], "plottimes": [-100], "title": title}),
        (
            "Testcase 10",
            eeg["chanlocs"],
            {"limits": [-1000, 1992.1875], "plottimes": np.arange(-1000.0, 1501.0, 10.0), "title": title},
        ),
    )
    for figure_title, locs, options in cases:
        if request.config.getoption("--eeglab-backend") == "matlab":
            eeglab_backend("figure", nargout=0)
            if figure_title:
                eeglab_backend("title", figure_title, "FontSize", 14.0, nargout=0)
        else:
            plt.figure()
            if figure_title:
                plt.title(figure_title, fontsize=14)
        options = {
            key: np.asarray(value, dtype=float).reshape(1, -1) if key in ("limits", "plottimes") else value
            for key, value in options.items()
        }
        eeglab_backend("timtopo", data, locs, **options, nargout=0)
        close_reference_gui(eeglab_backend, request)


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


@pytest.mark.gui
def test_axcopy_enlarges_a_clicked_existing_axes() -> None:
    figure, axes = plt.subplots()
    axes.plot([1, 2, 3], [1, 2, 3])
    before = set(plt.get_fignums())
    axcopy(figure, {axes: lambda target: target.plot([1, 2, 3], [1, 2, 3])})
    figure.canvas.draw()
    x, y = axes.transData.transform((2, 2))

    figure.canvas.callbacks.process(
        "button_press_event", MouseEvent("button_press_event", figure.canvas, x, y, button=1)
    )

    created = set(plt.get_fignums()) - before
    assert len(created) == 1
    popup = plt.figure(next(iter(created)))
    np.testing.assert_array_equal(popup.axes[0].lines[0].get_ydata(), [1, 2, 3])
    plt.close("all")


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


@pytest.mark.parametrize(
    "text, expected",
    [
        ("'electrodes', 'off', 'style', 'blank'", {"electrodes": "off", "style": "blank"}),
        ("electrodes='off', 'style', 'blank'", {"electrodes": "off", "style": "blank"}),
        ("maplimits=[-5, 5]", {"maplimits": [-5.0, 5.0]}),
    ],
)
def test_parse_plot_options_text_accepts_python_and_matlab_styles(text, expected):
    assert parse_plot_options_text(text) == expected


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


def test_plot_wrappers_display_figures_on_interactive_backend(sample_epoch, ica_epoch, monkeypatch):
    """Every plotting pop_* wrapper pops up its figure on an interactive backend,
    like EEGLAB. The GUI relies on this to make graphs appear; on Agg it stays silent."""
    shown = []
    monkeypatch.setattr(Figure, "show", lambda self: shown.append(self))
    monkeypatch.setattr(plt, "get_backend", lambda: "QtAgg")

    timtopo_fig, _ = pop_timtopo(sample_epoch, plottimes=[0], return_com=True)
    plottopo_fig, _ = pop_plottopo(sample_epoch, chans=[1, 2], return_com=True)
    erpimage_result, _ = pop_erpimage(sample_epoch, typeplot=1, index=1, return_com=True)
    prop_fig, _ = pop_prop(ica_epoch, typecomp=0, chanorcomp=1, return_com=True)
    topoplot_figs, _ = pop_topoplot(ica_epoch, typeplot=0, items=[1], colorbar="off", return_com=True)
    spectopo_result, _ = pop_spectopo(ica_epoch, dataflag=0, freqs=[10], return_com=True)
    plotdata_fig, _ = pop_plotdata(ica_epoch, components=[1, 2], return_com=True)
    envtopo_fig, _ = pop_envtopo(ica_epoch, components=[1, 2], return_com=True)

    expected = [
        timtopo_fig,
        plottopo_fig,
        erpimage_result["figure"],
        prop_fig,
        *topoplot_figs,
        spectopo_result["figure"],
        plotdata_fig,
        envtopo_fig,
    ]
    for figure in expected:
        assert figure in shown
        plt.close(figure)


def test_topoplot_core_displays_only_when_it_owns_the_figure(sample_eeg, monkeypatch):
    """Match EEGLAB: a standalone topoplot() call pops a window, but a call that
    draws into a caller's axes (axes=) adds no window of its own."""
    chanlocs = chanlocs_as_list(sample_eeg["chanlocs"])
    shown = []
    monkeypatch.setattr(Figure, "show", lambda self: shown.append(self))
    monkeypatch.setattr(plt, "get_backend", lambda: "QtAgg")

    standalone, *_ = topoplot([], chanlocs, style="blank")
    assert shown == [standalone]

    fig, ax = plt.subplots()
    topoplot([], chanlocs, style="blank", axes=ax)
    assert shown == [standalone]

    plt.close(standalone)
    plt.close(fig)


def test_plot_off_builds_figure_without_a_window(sample_eeg, sample_epoch, monkeypatch):
    """plot='off' still returns a usable figure but leaves nothing in pyplot's
    registry to auto-display; the default plot='on' shows it."""
    shown = []
    monkeypatch.setattr(Figure, "show", lambda self: shown.append(self))
    monkeypatch.setattr(plt, "get_backend", lambda: "QtAgg")
    plt.close("all")

    spec = pop_spectopo(sample_eeg, dataflag=1, freqs=[10], plot="off")
    timtopo_fig = pop_timtopo(sample_epoch, plottimes=[0], plot="off")
    assert spec["figure"] is not None and timtopo_fig is not None
    assert shown == []
    assert plt.get_fignums() == []

    displayed = pop_spectopo(sample_eeg, dataflag=1, freqs=[10], plot="on")
    assert shown == [displayed["figure"]]
    plt.close("all")


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


def test_pop_plottopo_scalp_array_places_channels_on_correct_side(sample_epoch):
    """plottopo scalp-array sub-plots sit on the same side as their electrodes (no L/R mirror)."""
    eeg = deepcopy(sample_epoch)
    eeg["data"] = eeg["data"][:4]
    eeg["nbchan"] = 4
    eeg["chanlocs"] = [
        {"labels": "Fz", "theta": 0, "radius": 0.5},
        {"labels": "F4", "theta": 45, "radius": 0.5},
        {"labels": "Pz", "theta": 180, "radius": 0.5},
        {"labels": "F3", "theta": -45, "radius": 0.5},
    ]
    fig, _command = pop_plottopo(eeg, chans=[1, 2, 3, 4], return_com=True)
    axes_by_label = {axis.get_title(): axis for axis in fig.axes}
    # F4 (right, +theta) sits right of F3 (left, -theta); Fz (front) above Pz (back).
    assert axes_by_label["F4"].get_position().x0 > axes_by_label["F3"].get_position().x0
    assert axes_by_label["Fz"].get_position().y0 > axes_by_label["Pz"].get_position().y0
    plt.close(fig)


def test_erpimage_scalp_inset_marks_channel_on_correct_side():
    """The erpimage scalp inset marks the plotted channel on its true side (no L/R mirror)."""
    chan_locs = [
        {"labels": "F3", "theta": -39.947, "radius": 0.3446},
        {"labels": "F4", "theta": 39.897, "radius": 0.3445},
    ]
    for channel_index, on_right in [(2, True), (1, False)]:
        fig, ax = plt.subplots()
        plot_channel_location(ax, chan_locs, channel_index)
        marker = next(
            np.asarray(c.get_offsets())
            for c in ax.collections
            if isinstance(c, PathCollection) and len(c.get_offsets()) == 1
        )
        if on_right:
            assert marker[0, 0] > 0  # F4 marked on the right
        else:
            assert marker[0, 0] < 0  # F3 marked on the left
        plt.close(fig)


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


def test_pop_envtopo_blank_gui_subcomps_removes_none(ica_epoch):
    """A blank GUI remove-components field means remove none (subcomps=0), not [] (remove all but compnums)."""

    class Renderer:
        def run(self, spec, initial_values=None):
            return {
                "timerange": "",
                "limcontrib": "",
                "compsplot": "2",
                "components": "1 2",
                "subcomps": "",
                "title": "blank subcomps",
                "options": "",
            }

    figure, command = pop_envtopo(ica_epoch, gui=True, renderer=Renderer(), return_com=True)

    assert isinstance(figure, Figure)
    assert "subcomps=0" in command
    assert "subcomps=[]" not in command
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


def test_pop_erpimage_default_caxis_is_symmetric(sample_epoch):
    """With no caxis, the color axis is symmetric about 0 (EEGLAB erpimage default)."""
    result, _command = pop_erpimage(sample_epoch, typeplot=1, index=1, return_com=True)
    image_ax = next(ax for ax in result["figure"].axes if ax.images)
    vmin, vmax = image_ax.images[0].get_clim()
    assert vmax > 0
    assert vmin == pytest.approx(-vmax)
    assert vmax == pytest.approx(float(np.nanmax(np.abs(result["image"]))))
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


def test_timtopo_auto_latency_uses_peak_global_power(sample_epoch):
    """Default (NaN) latency is the frame of peak global power (sum of squares across
    channels), as EEGLAB timtopo picks it -- not the max mean-removed variance frame."""
    data, _ = data_time_slice(sample_epoch, None)
    erp = np.nanmean(data, axis=2)
    x = np.linspace(float(sample_epoch["xmin"]) * 1000.0, float(sample_epoch["xmax"]) * 1000.0, erp.shape[1])
    global_power_latency = x[int(np.argmax(np.sum(erp**2, axis=0)))]
    variance_latency = x[int(np.argmax(np.nanvar(erp, axis=0)))]
    # Guard: the two metrics must disagree here or the test could not catch the bug.
    assert round(global_power_latency) != round(variance_latency)

    fig, _ = pop_timtopo(sample_epoch, plottimes=[float("nan")], return_com=True)
    map_titles = [ax.get_title().strip() for ax in fig.axes if ax.get_title().strip()]
    assert len(map_titles) == 1
    assert float(map_titles[0]) == pytest.approx(global_power_latency, abs=1)
    plt.close(fig)


def test_timtopo_mixed_nan_plottimes_fills_slot_with_auto_latency(sample_epoch):
    """A NaN entry in plottimes is filled with the peak-power latency (EEGLAB), keeping the
    other requested latencies and the slot count -- not silently dropped."""
    data, _ = data_time_slice(sample_epoch, None)
    erp = np.nanmean(data, axis=2)
    x = np.linspace(float(sample_epoch["xmin"]) * 1000.0, float(sample_epoch["xmax"]) * 1000.0, erp.shape[1])
    auto = x[int(np.argmax(np.sum(erp**2, axis=0)))]
    fig, _ = pop_timtopo(sample_epoch, plottimes=[float("nan"), 100.0], return_com=True)
    latencies = sorted(float(ax.get_title().split()[0]) for ax in fig.axes if ax.get_title().strip())
    assert latencies == [pytest.approx(min(auto, 100.0), abs=1), pytest.approx(max(auto, 100.0), abs=1)]
    plt.close(fig)


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
    configured_reference = os.environ.get("EEGPREP_EEGLAB_ROOT")
    if configured_reference:
        configured_root = Path(configured_reference).expanduser()
        if (configured_root / "functions" / "popfunc" / "pop_headplot.m").exists():
            return configured_root
    package_reference = repo_root / "src" / "eegprep" / "eeglab"
    if (package_reference / "functions" / "popfunc" / "pop_headplot.m").exists():
        return package_reference
    sibling_reference = repo_root.parent / "eeglab"
    if (sibling_reference / "functions" / "popfunc" / "pop_headplot.m").exists():
        return sibling_reference
    return package_reference
