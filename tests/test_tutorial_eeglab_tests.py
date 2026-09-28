"""Original EEGLAB tutorial contracts and supplemental generated-data examples.

Upstream suite: sccn/eeglab_tests@ff605546f3f70868916fb8d49c007472b3257b50
EEGLAB tree: sccn/eeglab@8ac485f654d6bbb1a6acb8dc9ef3f2eaf3d409ba
Tutorial scripts: sccn/eeglab-tutorial-scripts@58bf12dd53e894dd3ee1285946563cd94999db16
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

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
from eegprep.plugins.dipfit._utils import DIPFITUnavailableError, STANDARD_TEMPLATES
from eegprep.plugins.dipfit.pop_dipfit_loreta import pop_dipfit_loreta
from eegprep.plugins.dipfit.pop_dipfit_settings import pop_dipfit_settings
from eegprep.plugins.dipfit.pop_dipplot import pop_dipplot
from eegprep.plugins.dipfit.pop_leadfield import pop_leadfield
from eegprep.plugins.dipfit.pop_multifit import pop_multifit
from eegprep.plugins.EEG_BIDS.pop_exportbids import pop_exportbids
from eegprep.plugins.EEG_BIDS.pop_importbids import pop_importbids
from eegprep.plugins.ICLabel.pop_icflag import pop_icflag
from tests.eeglab_tests import eeglab_test, load_matlab_test_fixture
from tests.eeglab_tests.gui import close_reference_gui
from tests.fixtures import create_test_eeg, create_test_eeg_with_ica
from tests.test_eeg_store import _source_dataset_row
from tests.test_study_grouped_measure_plots_eeglab_tests import _cell_row, _records


TUTORIAL_WRAPPER = "unittesting_tutorial/tutorial_wrapperTest.m"
TUTORIAL2_WRAPPER = "unittesting_tutorial/tutorial2_wrapperTest.m"
CONDITIONS = ("standard", "oddball_with_reponse")
FACE_EVENT_TYPES = ("famous_new", "scrambled_new", "unfamiliar_new")
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


def _tutorial_start(backend, request, *, outputs=0):
    if request.config.getoption("--eeglab-backend") == "matlab":
        state = backend("eeglab", nargout=outputs)
        return (None, *state) if outputs else None
    # Python's actual launcher returns its window/session rather than workspace
    # variables. Retain this specific window, never the user's active window.
    window = backend("eeglab")
    request.addfinalizer(window.window.close)
    if outputs:
        session = window.session
        return window, session.ALLEEG, session.EEG, session.current_set_value(), session.ALLCOM
    return window


def _tutorial_redraw(backend, request, window, **state):
    if request.config.getoption("--eeglab-backend") == "matlab":
        backend("eegprep_test_tutorial_redraw", state, nargout=0)
    else:
        window.session.apply_workspace_state(**{key.lower(): value for key, value in state.items()})
        window.refresh()


def _tutorial_fieldtrip(request):
    if request.config.getoption("--eeglab-backend") != "matlab":
        return
    engine = request.getfixturevalue("eeglab_matlab_engine")
    implementation = engine.which("ft_freqstatistics")
    assert implementation, "The full FieldTrip plugin is required for this tutorial"
    previous = engine.path()
    request.addfinalizer(lambda: engine.path(previous, nargout=0))
    # Fileio bundles a partial FieldTrip whose ft_defaults shadows the full
    # plugin and resolves required preproc/statfun directories incorrectly.
    engine.addpath(str(Path(implementation).parent), "-begin", nargout=0)
    engine.clear("ft_defaults", nargout=0)
    engine.ft_defaults(nargout=0)


def _tutorial_figure(backend, request, **kwargs):
    if request.config.getoption("--eeglab-backend") == "matlab":
        backend("eegprep_test_tutorial_figure", nargout=0, **kwargs)
    else:
        if "color" in kwargs:
            kwargs["facecolor"] = kwargs.pop("color")
        figure = plt.figure(**kwargs)
        request.addfinalizer(lambda: plt.close(figure))


def _tutorial_title(backend, request, title):
    if request.config.getoption("--eeglab-backend") == "matlab":
        backend("title", title, nargout=0)
    else:
        plt.title(title)


def _tutorial_timerange(eeg):
    return np.asarray([eeg["xmin"], eeg["xmax"]]).reshape(1, 2)


def _tutorial_cd(backend, request, monkeypatch, directory):
    monkeypatch.chdir(directory)
    if request.config.getoption("--eeglab-backend") == "matlab":
        backend("cd", str(directory), nargout=0)


def _tutorial_load(backend, request, filename):
    if request.config.getoption("--eeglab-backend") == "matlab":
        return backend("load", "-mat", filename)
    return load_matlab_test_fixture(filename)


def _tutorial_hold(backend, request):
    if request.config.getoption("--eeglab-backend") == "matlab":
        backend("hold", "on", nargout=0)
    # Matplotlib already retains artists on the current axes.


def _tutorial_plot(backend, request, x, y):
    if request.config.getoption("--eeglab-backend") == "matlab":
        backend("plot", x, y, nargout=0)
    else:
        plt.plot(np.asarray(x).ravel(), y)


def _tutorial_cells(cells):
    return list(cells.ravel(order="F")) if isinstance(cells, np.ndarray) else cells


def _tutorial_squeeze(data):
    # MATLAB leaves 2-D rows/columns unchanged and retains at least two axes.
    if data.ndim <= 2:
        return data
    squeezed = np.squeeze(data)
    return squeezed.reshape(-1, 1) if squeezed.ndim < 2 else squeezed


def _tutorial_newtimef(backend, request, *args):
    if request.config.getoption("--eeglab-backend") == "matlab":
        return backend("pop_newtimef", *args, nargout=7)
    result = backend("pop_newtimef", *args)
    # The actual Python API names its outputs in a dataclass and uses 1-D
    # vectors; these are the original MATLAB output orientations, not an oracle.
    return (
        result.ersp,
        result.itc,
        result.powbase.reshape(1, -1),
        result.times.reshape(1, -1),
        result.freqs.reshape(1, -1),
        result.erspboot,
        result.itcboot.reshape(-1, 1),
    )


def _tutorial_custom_precomp(backend, request, kind, study, alleeg, eeg):
    if request.config.getoption("--eeglab-backend") == "matlab":
        backend("eegprep_test_tutorial_custom_precomp", kind, study, alleeg, eeg, nargout=0)
        return
    if kind == "baseline":

        def callback(data):
            return data - np.mean(data[:, :410, :], axis=1, keepdims=True)
    else:

        def callback(data):
            first = _records(eeg)[0]
            filtered = backend(
                "eegfilt",
                data.reshape(data.shape[0], -1, order="F"),
                first["srate"],
                0.0,
                10.0,
                first["pnts"],
                60.0,
                0.0,
                "fir1",
            )
            return filtered.reshape(data.shape, order="F")

    backend("std_precomp", study, alleeg, "channels", "customfunc", callback, "interp", "on", nargout=0)


class _TutorialVideo:
    """Test-side VideoWriter equivalent; never resolve it as an EEGPrep API."""

    def __init__(self, backend, request, filename):
        self.backend = backend
        self.native = request.config.getoption("--eeglab-backend") == "matlab"
        self.avi = sys.platform != "darwin" and sys.platform != "win32"
        self.filename = filename + (".avi" if self.avi else ".mp4")
        self.process = None
        self.closed = False
        if self.native:
            backend(
                "eegprep_test_tutorial_video",
                "open",
                self.filename,
                "Uncompressed AVI" if self.avi else "MPEG-4",
                nargout=0,
            )
        request.addfinalizer(self.close)

    def write(self, movie):
        if self.native:
            self.backend("eegprep_test_tutorial_video", "write", movie, nargout=0)
            return
        for frame in movie:
            if self.process is None:
                executable = shutil.which("ffmpeg")
                if executable is None:
                    raise RuntimeError("The Python tutorial video-export boundary requires ffmpeg")
                height, width, _channels = frame.shape
                self.process = subprocess.Popen(
                    [
                        executable,
                        "-loglevel",
                        "error",
                        "-y",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        f"{width}x{height}",
                        "-r",
                        "30",
                        "-i",
                        "-",
                        "-an",
                        "-c:v",
                        "rawvideo" if self.avi else "libx264",
                        "-pix_fmt",
                        "bgr24" if self.avi else "yuv420p",
                        self.filename,
                    ],
                    stdin=subprocess.PIPE,
                )
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def capture(self):
        if self.native:
            self.backend("eegprep_test_tutorial_video", "capture", nargout=0)
        else:
            figure = plt.gcf()
            figure.canvas.draw()
            self.write(np.asarray(figure.canvas.buffer_rgba())[None, :, :, :3])

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.native:
            self.backend("eegprep_test_tutorial_video", "close", nargout=0)
        elif self.process is not None:
            self.process.stdin.close()
            if self.process.wait() != 0:
                raise RuntimeError("ffmpeg failed to write the source tutorial movie")


@pytest.mark.gui
@eeglab_test(TUTORIAL_WRAPPER, "test_make_eeg_movie")
def test_reference_tutorial_make_eeg_movie(eeglab_backend, eeglab_suite_root, eeglab_working_directory, request):
    backend = eeglab_backend
    if request.config.getoption("--eeglab-backend") == "matlab":
        engine = request.getfixturevalue("eeglab_matlab_engine")
        visibility = backend("eegprep_test_transport", "figure_visibility")
        request.addfinalizer(lambda: engine.set(0.0, "DefaultFigureVisible", visibility, nargout=0))
        # Native getframe needs the source's visible figures; hidden figures
        # produce inconsistent frame dimensions on R2026a.
        engine.set(0.0, "DefaultFigureVisible", "on", nargout=0)
    window = _tutorial_start(backend, request)
    close_reference_gui(backend, request, window=window.window if window else None)
    eeg = backend("pop_loadset", str(eeglab_suite_root / "eeglab" / "sample_data" / "eeglab_data_epochs_ica.set"))
    pnts1 = backend("eeg_lat2point", -100.0 / 1000.0, 1.0, eeg["srate"], _tutorial_timerange(eeg))
    pnts2 = backend("eeg_lat2point", 600.0 / 1000.0, 1.0, eeg["srate"], _tutorial_timerange(eeg))
    first = int(np.floor(np.asarray(pnts1).item() + 0.5))
    last = int(np.floor(np.asarray(pnts2).item() + 0.5))
    # The source uses TWO subscripts on EEG.data, collapsing later dimensions.
    # Its mean(...,3) therefore leaves this first-epoch slice unchanged.
    scalp_erp = eeg["data"].reshape(int(eeg["nbchan"]), -1, order="F")[:, first - 1 : last].copy()
    for channel in range(scalp_erp.shape[0]):
        scalp_erp[channel, :] = np.convolve(scalp_erp[channel, :], np.ones(5) / 5.0, mode="same")
    _tutorial_figure(backend, request)
    movie, colormap = backend(
        "eegmovie",
        scalp_erp,
        eeg["srate"],
        eeg["chanlocs"],
        "framenum",
        "off",
        "vert",
        0.0,
        "startsec",
        -0.1,
        "topoplotopt",
        _cell_row("numcontour", 0.0),
        nargout=2,
    )
    backend("seemovie", movie, 1.0, colormap, nargout=0)
    video = _TutorialVideo(backend, request, "erpmovie2d")
    video.write(movie)
    video.close()
    headplotparams1 = (
        "meshfile",
        "mheadnew.mat",
        "transform",
        np.array([[0.664455, -3.39403, -14.2521, -0.00241453, 0.015519, -1.55584, 11.0, 10.1455, 12.0]]),
    )
    _headplotparams2 = (
        "meshfile",
        "colin27headmesh.mat",
        "transform",
        np.array([[0.0, -13.0, 0.0, 0.1, 0.0, -1.57, 11.7, 12.5, 12.0]]),
    )
    headplotparams = headplotparams1
    if request.config.getoption("--eeglab-backend") == "matlab":
        backend("headplot", "setup", eeg["chanlocs"], "STUDY_headplot.spl", *headplotparams, nargout=0)
    else:
        backend(
            "headplot",
            "setup",
            eeg["chanlocs"],
            splinefile="STUDY_headplot.spl",
            meshfile=headplotparams[1],
            transform=headplotparams[3],
            nargout=0,
        )
    close_reference_gui(backend, request)
    _tutorial_figure(backend, request)
    if request.config.getoption("--eeglab-backend") == "matlab":
        backend(
            "headplot",
            scalp_erp[:, -51:-50],
            "STUDY_headplot.spl",
            *headplotparams,
            "maplimits",
            "absmax",
            "lighting",
            "on",
            nargout=0,
        )
    else:
        backend(
            "headplot",
            scalp_erp[:, -51:-50],
            "STUDY_headplot.spl",
            meshfile=headplotparams[1],
            transform=headplotparams[3],
            maplimits="absmax",
            lighting="on",
            nargout=0,
        )
    _tutorial_figure(backend, request)
    backend("topoplot", scalp_erp[:, -51:-50], eeg["chanlocs"], nargout=0)
    _tutorial_figure(backend, request, color="w")
    movie, colormap = backend(
        "eegmovie",
        scalp_erp,
        eeg["srate"],
        eeg["chanlocs"],
        "framenum",
        "off",
        "vert",
        0.0,
        "startsec",
        -0.1,
        "mode",
        "3d",
        "headplotopt",
        _cell_row(*headplotparams, "material", "metal"),
        "camerapath",
        np.array([[-127.0, 2.0, 30.0, 0.0]]),
        nargout=2,
    )
    backend("seemovie", movie, 1.0, colormap, nargout=0)
    video = _TutorialVideo(backend, request, "erpmovie3d1")
    video.write(movie)
    video.close()
    video = _TutorialVideo(backend, request, "erpmovietopoplot")
    _counter = 0.0
    for latency in np.arange(-100.0, 601.0, 10.0):
        _tutorial_figure(backend, request)
        backend("pop_topoplot", eeg, 1.0, latency, "My movie", [], "electrodes", "off", nargout=0)
        video.capture()
        close_reference_gui(backend, request)
    video.close()


@pytest.mark.gui
@eeglab_test(TUTORIAL_WRAPPER, "test_study_script")
def test_reference_tutorial_study_script(
    eeglab_backend, eeglab_writable_study, eeglab_working_directory, request, monkeypatch
):
    backend = eeglab_backend
    _tutorial_cd(backend, request, monkeypatch, eeglab_writable_study)
    window = _tutorial_start(backend, request)
    _tutorial_fieldtrip(request)
    if not (eeglab_writable_study / "n400.study").is_file():
        raise RuntimeError(
            "You must change the path to the folder containing the data to run this script\nDownload the data from https://eeglab.org/tutorials/tutorial_data.html (5 subject study)"
        )
    commands = []
    for condition, code in (("synonyms", 253), ("non-synonyms", 254)):
        for subject in (2, 5, 7, 8, 10):
            filename = eeglab_writable_study / f"s{subject:02d}" / f"syn{subject:02d}-s{code}-clean.set"
            commands.append(
                _cell_row(
                    "index",
                    float(len(commands) + 1),
                    "load",
                    str(filename),
                    "subject",
                    f"S{subject:02d}",
                    "condition",
                    condition,
                )
            )
    commands.append(_cell_row("dipselect", 0.15))
    study, alleeg = backend(
        "std_editset",
        [],
        [],
        "name",
        "N400STUDY",
        "task",
        "Auditory task: Synonyms Vs. Non-synonyms, N400",
        "filename",
        "N400empty.study",
        "filepath",
        "./",
        "commands",
        _cell_row(*commands),
        nargout=2,
    )
    eeg = alleeg
    currentset = np.arange(1.0, len(_records(eeg)) + 1).reshape(1, -1)
    _tutorial_redraw(
        backend, request, window, STUDY=study, ALLEEG=alleeg, EEG=eeg, CURRENTSET=currentset, CURRENTSTUDY=1.0
    )
    study, alleeg = backend(
        "std_precomp",
        study,
        alleeg,
        "channels",
        "erp",
        "on",
        "erpparams",
        _cell_row("rmbase", np.array([[-200.0, 0.0]])),
        nargout=2,
    )
    study = backend("std_erpplot", study, alleeg, "channels", _cell_row("Oz"))
    study, erpdata, erptimes = backend(
        "std_erpplot", study, alleeg, "channels", _cell_row("Oz"), "timerange", np.array([[-200.0, 1000.0]]), nargout=3
    )
    backend(
        "std_plotcurve",
        erptimes,
        erpdata,
        "plotconditions",
        "together",
        "plotstderr",
        "on",
        "figure",
        "on",
        "filter",
        30.0,
        nargout=0,
    )
    study = backend("std_erpplot", study, alleeg, "channels", _cell_row("FP1"))
    study, erpdata, erptimes = backend(
        "std_erpplot", study, alleeg, "channels", _cell_row("Oz"), "timerange", np.array([[-200.0, 1000.0]]), nargout=3
    )
    backend(
        "std_plotcurve", erptimes, erpdata, "plotconditions", "together", "plotstderr", "on", "figure", "on", nargout=0
    )
    study = backend("std_erpplot", study, alleeg, "channels", _cell_row("FP1"))
    study, erpdata, erptimes = backend(
        "std_erpplot", study, alleeg, "channels", _cell_row("FP1"), "noplot", "on", nargout=3
    )
    _tutorial_figure(backend, request)
    _tutorial_plot(backend, request, erptimes, _tutorial_cells(erpdata)[1])
    options = (
        "freqscale",
        "linear",
        "freqs",
        np.array([[3.0, 25.0]]),
        "nfreqs",
        20.0,
        "ntimesout",
        60.0,
        "padratio",
        1.0,
        "winsize",
        64.0,
        "baseline",
        0.0,
    )
    tmpeeg = backend("eeg_checkset", _records(alleeg)[0], "loaddata")
    _tutorial_figure(backend, request)
    backend(
        "pop_newtimef",
        tmpeeg,
        1.0,
        1.0,
        _tutorial_timerange(tmpeeg) * 1000.0,
        np.array([[3.0, 0.8]]),
        "topovec",
        1.0,
        "elocs",
        tmpeeg["chanlocs"],
        "chaninfo",
        tmpeeg["chaninfo"],
        "plotphase",
        "off",
        *options,
        "title",
        tmpeeg["setname"],
        "erspmax ",
        6.6,
    )
    study, alleeg = backend(
        "std_precomp",
        study,
        alleeg,
        "channels",
        "recompute",
        "on",
        "ersp",
        "on",
        "erspparams",
        _cell_row("cycles", np.array([[3.0, 0.8]]), "parallel", "on", *options),
        "itc",
        "on",
        nargout=2,
    )
    study = backend(
        "std_erspplot",
        study,
        alleeg,
        "channels",
        _cell_row(_records(tmpeeg["chanlocs"])[0]["labels"]),
        "subject",
        "S02",
        "design",
        1.0,
    )
    study, alleeg = backend(
        "std_precomp",
        study,
        alleeg,
        "components",
        "erp",
        "on",
        "erpparams",
        _cell_row("rmbase", np.array([[-200.0, 0.0]])),
        "scalp",
        "on",
        "spec",
        "on",
        "specparams",
        _cell_row("freqrange", np.array([[3.0, 50.0]]), "specmode", "fft", "logtrials", "off"),
        "ersp",
        "on",
        "erspparams",
        _cell_row("cycles", np.array([[3.0, 0.8]]), "nfreqs", 20.0, "ntimesout", 60.0),
        "itc",
        "on",
        "recompute",
        "on",
        nargout=2,
    )
    study, alleeg = backend(
        "std_preclust",
        study,
        alleeg,
        1.0,
        _cell_row("spec", "npca", 10.0, "weight", 1.0, "freqrange", np.array([[3.0, 25.0]])),
        _cell_row("erp", "npca", 10.0, "weight", 1.0, "timewindow", np.array([[100.0, 600.0]]), "erpfilter", "20"),
        _cell_row("dipoles", "weight", 10.0),
        _cell_row(
            "ersp",
            "npca",
            10.0,
            "freqrange",
            np.array([[3.0, 25.0]]),
            "timewindow",
            np.array([[-1600.0, 1495.0]]),
            "weight",
            1.0,
            "norm",
            1.0,
            "weight",
            1.0,
        ),
        nargout=2,
    )
    study = backend("pop_clust", study, alleeg, "algorithm", "kmeanscluster", "clus_num", 10.0)
    study = backend("pop_clust", study, alleeg, "algorithm", "kmeanscluster", "clus_num", 10.0)
    study = backend("std_topoplot", study, alleeg, "clusters", 2.0, "mode", "together")
    study = backend("std_topoplot", study, alleeg, "clusters", 2.0, "mode", "apart")
    study = backend("std_topoplot", study, alleeg, "clusters", 2.0, "comps", 1.0)
    study = backend("pop_statparams", study, "condstats", "on")
    stats_output = {"return_stats": True} if request.config.getoption("--eeglab-backend") == "python" else {}
    study, erpdata, erptimes, _pgroup, _pcond, _pinter = backend(
        "std_erpplot", study, alleeg, "channels", _cell_row("FP1"), nargout=6, **stats_output
    )
    study, erpdata, erptimes, _pgroup, _pcond, _pinter = backend(
        "std_erpplot", study, alleeg, "clusters", 1.0, nargout=6, **stats_output
    )
    _tutorial_custom_precomp(backend, request, "baseline", study, alleeg, eeg)
    _tutorial_custom_precomp(backend, request, "filter", study, alleeg, eeg)
    first_eeg, first_alleeg = _records(eeg)[0], _records(alleeg)[0]
    channels = _cell_row(*(channel["labels"] for channel in _records(first_alleeg["chanlocs"])))
    _, customdata = backend("std_readdata", study, alleeg, channels=channels, design=1.0, datatype="custom", nargout=2)
    _, erpdata = backend("std_readdata", study, alleeg, channels=channels, design=1.0, datatype="erp", nargout=2)
    backend("std_plotcurve", first_eeg["times"], erpdata, "chanlocs", first_alleeg["chanlocs"], nargout=0)
    if isinstance(customdata, np.ndarray):
        for index in np.ndindex(customdata.shape):
            customdata[index] = _tutorial_squeeze(customdata[index])
    else:
        customdata = [_tutorial_squeeze(data) for data in customdata]
    backend("std_plotcurve", first_eeg["times"], customdata, "chanlocs", first_alleeg["chanlocs"], nargout=0)
    _tutorial_figure(backend, request)
    values = _records(_records(study["design"])[0]["variable"])[0]["value"]
    ncond = len(_tutorial_cells(values))
    for condition in range(ncond):
        data = _tutorial_cells(customdata)[condition]
        # MATLAB mean(...,3) is identity when that trailing dimension is absent.
        mean_trials = data.mean(axis=2) if data.ndim > 2 else data
        rms = np.sqrt(np.mean(mean_trials**2, axis=1, keepdims=True))
        _tutorial_hold(backend, request)
        _tutorial_plot(backend, request, first_eeg["times"], rms)
    if request.config.getoption("--eeglab-backend") == "matlab":
        backend("legend", values, nargout=0)
        backend("eegprep_test_tutorial_setfont", nargout=0)
    else:
        plt.legend(_tutorial_cells(values))
        backend("setfont", plt.gcf(), "fontsize", 16.0, nargout=0)
    _, erpdata = backend(
        "std_readdata",
        study,
        alleeg,
        channels=_cell_row(_records(first_alleeg["chanlocs"])[0]["labels"]),
        design=1.0,
        datatype="erp",
        nargout=2,
    )
    backend("std_stat", erpdata, condstats="on", mcorrect="fdr", method="permutation", nargout=0)
    backend(
        "std_stat",
        erpdata,
        condstats="on",
        fieldtripmcorrect="cluster",
        fieldtripmethod="montecarlo",
        mode="fieldtrip",
        nargout=0,
    )
    res = backend("statcond", erpdata)
    if request.config.getoption("--eeglab-backend") == "python":
        res = res.stat
    np.shape(res)
    res = backend("statcond", erpdata)
    if request.config.getoption("--eeglab-backend") == "python":
        res = res.stat
    np.shape(res)


@pytest.mark.gui
@eeglab_test(TUTORIAL_WRAPPER, "test_plot_study_erp")
def test_reference_tutorial_plot_study_erp(
    eeglab_backend, eeglab_writable_study, eeglab_working_directory, request, monkeypatch
):
    backend = eeglab_backend
    _tutorial_cd(backend, request, monkeypatch, eeglab_writable_study)
    window = _tutorial_start(backend, request)
    close_reference_gui(backend, request, window=window.window if window else None)
    filename = eeglab_writable_study / "n400.study"
    if not filename.is_file():
        backend(
            "warndlg2",
            "Now select the n400.study file. Download the data from\nhttps://eeglab.org/tutorials/tutorial_data.html (5 subject study)",
            nargout=0,
        )
        study, alleeg = backend("pop_loadstudy", nargout=2)
    else:
        _tutorial_cd(backend, request, monkeypatch, filename.parent)
        study, alleeg = backend("pop_loadstudy", str(filename), nargout=2)
    study, alleeg = backend(
        "std_precomp",
        study,
        alleeg,
        np.empty((0, 0), dtype=object),
        "savetrials",
        "on",
        "interp",
        "on",
        "recompute",
        "on",
        "erp",
        "on",
        "erpparams",
        _cell_row("rmbase", np.array([[-300.0, 0.0]])),
        nargout=2,
    )
    study = backend("pop_statparams", study, "condstats", "on", "method", "perm", "mcorrect", "fdr", "alpha", 0.01)
    timerange_min, timerange_max = -300.0, 1320.0
    study = backend("pop_erpparams", study, "plotconditions", "together")
    study, erpdata, erptimes = backend(
        "std_erpplot",
        study,
        alleeg,
        "channels",
        _cell_row("Fz"),
        "design",
        1.0,
        "timerange",
        np.array([[timerange_min, timerange_max]]),
        nargout=3,
    )
    backend(
        "std_plotcurve", erptimes, erpdata, "plotconditions", "together", "plotstderr", "on", "figure", "on", nargout=0
    )
    backend(
        "std_plotcurve",
        erptimes,
        erpdata,
        "plotconditions",
        "together",
        "plotsubjects",
        "on",
        "figure",
        "on",
        nargout=0,
    )
    backend(
        "std_plotcurve",
        erptimes,
        erpdata,
        "plotdiff",
        "on",
        "plotconditions",
        "together",
        "figure",
        "on",
        "legend",
        _cell_row("cond1", "cond2"),
        nargout=0,
    )
    study = backend("pop_erpparams", study, "topotime", np.array([[1200.0, 1500.0]]))
    backend(
        "std_erpplot",
        study,
        alleeg,
        "channels",
        _cell_row(
            "Fp1",
            "Fpz",
            "Fp2",
            "AF3",
            "AF4",
            "F7",
            "F5",
            "F3",
            "F1",
            "Fz",
            "F2",
            "F4",
            "F6",
            "F8",
            "FC5",
            "FC3",
            "FC1",
            "FCz",
            "FC2",
            "FC4",
            "FC6",
            "T7",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "T8",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "CP6",
            "TP8",
            "P7",
            "P5",
            "P3",
            "P1",
            "Pz",
            "P2",
            "P4",
            "P6",
            "P8",
            "PO5",
            "PO3",
            "PO1",
            "POz",
            "PO2",
            "PO4",
            "PO6",
            "O1",
            "Oz",
            "O2",
            "LO1",
            "IO1",
            "SO1",
            "LO2",
            "TP7",
            "CP5",
        ),
        "design",
        1.0,
    )


@pytest.mark.gui
@eeglab_test(TUTORIAL_WRAPPER, "test_source_reconstruction_eeg")
def test_reference_tutorial_source_reconstruction_eeg(eeglab_backend, eeglab_suite_root, request):
    backend = eeglab_backend
    window = _tutorial_start(backend, request)
    close_reference_gui(backend, request, window=window.window if window else None)
    eeg = backend("pop_loadset", str(eeglab_suite_root / "eeglab" / "sample_data" / "eeglab_data_epochs_ica.set"))
    if request.config.getoption("--eeglab-backend") == "matlab":
        model = backend("eegprep_test_tutorial_dipfitdefs", eeg)
    else:
        # The Python package stores dipfitdefs' metadata as real template data.
        template = STANDARD_TEMPLATES[1]
        model = {key: Path(getattr(template, key)).name for key in ("hdmfile", "mrifile", "chanfile")}
        model["coordformat"] = template.coordformat
    eeg = backend(
        "pop_dipfit_settings",
        eeg,
        "hdmfile",
        model["hdmfile"],
        "coordformat",
        model["coordformat"],
        "mrifile",
        model["mrifile"],
        "chanfile",
        model["chanfile"],
        "coord_transform",
        np.array([[0.83215, -15.6287, 2.4114, 0.081214, 0.00093739, -1.5732, 1.1742, 1.0601, 1.1485]]),
        "chansel",
        np.arange(1.0, 33.0).reshape(1, -1),
    )
    latency = 0.100
    pt100 = int(np.floor(np.asarray((latency - eeg["xmin"]) * eeg["srate"]).item() + 0.5))
    erp = np.mean(eeg["data"], axis=2)
    dipole, _model, tmpeeg = backend(
        "dipfit_erpeeg",
        erp[:, pt100 - 1 : pt100],
        eeg["chanlocs"],
        "settings",
        eeg["dipfit"],
        "threshold",
        100.0,
        nargout=3,
    )
    backend("pop_dipplot", tmpeeg, 1.0, "normlen", "on", nargout=0)
    _tutorial_figure(backend, request)
    rv = np.asarray(_records(dipole)[0]["rv"]).item() * 100.0
    backend(
        "pop_topoplot", tmpeeg, 0.0, 1.0, f"ERP 100ms, fit with a single dipole (RV {rv:.2g}%)", 0.0, 1.0, nargout=0
    )


@pytest.mark.gui
@eeglab_test(TUTORIAL_WRAPPER, "test_source_reconstruction_advanced")
def test_reference_tutorial_source_reconstruction_advanced(eeglab_backend, eeglab_suite_root, request):
    backend = eeglab_backend
    window = _tutorial_start(backend, request)
    _tutorial_fieldtrip(request)
    close_reference_gui(backend, request, window=window.window if window else None)
    eeglab_path = eeglab_suite_root / "eeglab"
    bem_path = eeglab_path / "plugins" / "dipfit" / "standard_BEM"
    eeg = backend("pop_loadset", str(eeglab_path / "sample_data" / "eeglab_data_epochs_ica.set"))
    eeg = backend(
        "pop_dipfit_settings",
        eeg,
        "hdmfile",
        str(bem_path / "standard_vol.mat"),
        "coordformat",
        "MNI",
        "mrifile",
        str(bem_path / "standard_mri.mat"),
        "chanfile",
        str(bem_path / "elec" / "standard_1005.elc"),
        "coord_transform",
        np.array([[0.83215, -15.6287, 2.4114, 0.081214, 0.00093739, -1.5732, 1.1742, 1.0601, 1.1485]]),
        "chansel",
        np.arange(1.0, 33.0).reshape(1, -1),
    )
    data_pre = backend("eeglab2fieldtrip", eeg, "preprocessing", "dipfit")
    cfg = {"channel": _cell_row("all", "-EOG1"), "reref": "yes", "refchannel": _cell_row("all", "-EOG1")}
    data_pre = backend("ft_preprocessing", cfg, data_pre)
    vol = _tutorial_load(backend, request, eeg["dipfit"]["hdmfile"])
    cfg = {
        "elec": data_pre["elec"],
        "headmodel": vol["vol"],
        "resolution": 10.0,
        "unit": "mm",
        "channel": _cell_row("all"),
    }
    sourcemodel = backend("ft_prepare_leadfield", cfg)
    cfg = {"covariance": "yes", "covariancewindow": np.array([[np.asarray(eeg["xmin"]).item(), 0.0]])}
    data_avg = backend("ft_timelockanalysis", cfg, data_pre)
    cfg = {"method": "eloreta", "sourcemodel": sourcemodel, "headmodel": vol["vol"]}
    source = backend("ft_sourceanalysis", cfg, data_avg)
    source_proj = backend("ft_sourcedescriptives", {"projectmom": "yes", "flipori": "yes"}, source)
    source_proj = backend("ft_math", {"parameter": "mom", "operation": "abs"}, source_proj)
    _tutorial_figure(backend, request)
    backend("ft_sourceplot", {"method": "ortho", "funparameter": "mom"}, source_proj, nargout=0)
    mri = _tutorial_load(backend, request, eeg["dipfit"]["mrifile"])
    mri = backend("ft_volumereslice", [], mri["mri"])
    cfg = {"downsample": 2.0, "parameter": "pow"}
    source["oridimord"] = "pos"
    source["momdimord"] = "pos"
    source_int = backend("ft_sourceinterpolate", cfg, source, mri)
    backend("ft_sourceplot", {"method": "slice", "funparameter": "pow"}, source_int, nargout=0)
    _ft_ver, ft_path = backend("ft_version", nargout=2)
    sourcemodel = backend("ft_read_headshape", str(Path(ft_path) / "template" / "sourcemodel" / "cortex_8196.surf.gii"))
    leadfield = backend("ft_prepare_leadfield", {"grid": sourcemodel, "headmodel": vol["vol"]}, data_avg)
    cfg = {"method": "mne", "grid": leadfield, "headmodel": vol["vol"], "mne": {"lambda": 3.0, "scalesourcecov": "yes"}}
    source = backend("ft_sourceanalysis", cfg, data_avg)
    cfg = {
        "funparameter": "pow",
        "maskparameter": "pow",
        "method": "surface",
        "latency": 0.4,
        "opacitylim": np.array([[0.0, 200.0]]),
    }
    backend("ft_sourceplot", cfg, source, nargout=0)
    boundaries = _records(_records(vol["vol"])[0]["bnd"])
    for index in (2, 1, 0):
        _tutorial_hold(backend, request)
        backend(
            "ft_plot_mesh", boundaries[index], "facecolor", "red", "facealpha", 0.05, "edgecolor", "none", nargout=0
        )


@pytest.mark.gui
@eeglab_test(TUTORIAL2_WRAPPER, "test_bids_p300")
def test_reference_tutorial_bids_p300(
    eeglab_backend, eeglab_suite_root, eeglab_working_directory, eeglab_options_directory, request, monkeypatch
):
    backend = eeglab_backend
    # Full source tree: pop_importbids and storedisk mode create derivative files.
    filepath = Path(shutil.copytree(eeglab_suite_root / "ds003061", eeglab_working_directory / "ds003061"))
    _tutorial_cd(backend, request, monkeypatch, filepath)
    _tutorial_start(backend, request)
    if not (filepath / "task-P300_events.json").is_file():
        raise RuntimeError(
            "Download the data from https://openneuro.org/datasets/ds003061/ and go to the downloaded folder"
        )
    if request.config.getoption("--eeglab-backend") == "matlab" and not backend("which", "picard"):
        raise RuntimeError(
            "The source PICARD dependency must be installed before this GUI tutorial; tests do not download plugins"
        )
    backend("pop_editoptions", "option_storedisk", 1.0, nargout=0)
    study, alleeg = backend(
        "pop_importbids", str(filepath), "studyName", "Oddball", "subjects", np.array([[1.0, 2.0]]), nargout=2
    )
    alleeg = backend(
        "pop_select",
        alleeg,
        "nochannel",
        _cell_row(
            "EXG1",
            "EXG2",
            "EXG3",
            "EXG4",
            "EXG5",
            "EXG6",
            "EXG7",
            "EXG8",
            "GSR1",
            "GSR2",
            "Erg1",
            "Erg2",
            "Resp",
            "Plet",
            "Temp",
        ),
    )
    alleeg = backend("pop_reref", alleeg, [])
    alleeg = backend(
        "pop_clean_rawdata",
        alleeg,
        "FlatlineCriterion",
        5.0,
        "ChannelCriterion",
        0.87,
        "LineNoiseCriterion",
        4.0,
        "Highpass",
        np.array([[0.25, 0.75]]),
        "BurstCriterion",
        20.0,
        "WindowCriterion",
        0.25,
        "BurstRejection",
        "on",
        "Distance",
        "Euclidian",
        "WindowCriterionTolerances",
        np.array([[-np.inf, 7.0]]),
        "fusechanrej",
        1.0,
    )
    alleeg = backend("pop_reref", alleeg, [], "interpchan", [])
    backend("plugin_askinstall", "picard", "picard", 1.0, nargout=0)
    alleeg = backend("pop_runica", alleeg, "icatype", "picard", "concatcond", "on", "options", _cell_row("pca", -1.0))
    alleeg = backend("pop_iclabel", alleeg, "default")
    alleeg = backend("pop_icflag", alleeg, ICLABEL_THRESHOLDS.copy())
    alleeg = backend(
        "pop_epoch", alleeg, _cell_row("oddball_with_reponse", "standard"), np.array([[-1.0, 2.0]]), "epochinfo", "yes"
    )
    alleeg = backend("eeg_checkset", alleeg)
    alleeg = backend("pop_rmbase", alleeg, np.array([[-1000.0, 0.0]]), [])
    study = backend("std_maketrialinfo", study, alleeg)
    study = backend(
        "std_makedesign",
        study,
        alleeg,
        1.0,
        "name",
        "STUDY.design 1",
        "delfiles",
        "off",
        "defaultdesign",
        "off",
        "variable1",
        "type",
        "values1",
        _cell_row("oddball_with_reponse", "standard"),
        "vartype1",
        "categorical",
        "subjselect",
        study["subject"],
    )
    study, alleeg = backend(
        "std_precomp",
        study,
        alleeg,
        np.empty((0, 0), dtype=object),
        "savetrials",
        "on",
        "rmicacomps",
        "on",
        "interp",
        "on",
        "recompute",
        "on",
        "erp",
        "on",
        nargout=2,
    )
    study = backend("pop_erpparams", study, "topotime", 350.0)
    chanlocs = backend("eeg_mergelocs", *(dataset["chanlocs"] for dataset in _records(alleeg)))
    study = backend(
        "std_erpplot",
        study,
        alleeg,
        "channels",
        _cell_row(*(channel["labels"] for channel in _records(chanlocs))),
        "design",
        1.0,
    )
    backend("pop_editoptions", "option_storedisk", 0.0, nargout=0)


@pytest.mark.gui
@eeglab_test(TUTORIAL_WRAPPER, "test_eeglab_history")
def test_reference_tutorial_history(eeglab_backend, eeglab_suite_root, eeglab_options_directory, request):
    backend = eeglab_backend
    samples = eeglab_suite_root / "eeglab" / "sample_data"
    window, alleeg, eeg, currentset, allcom = _tutorial_start(backend, request, outputs=4)
    backend("pop_editoptions", "option_storedisk", 0.0, nargout=0)
    eeg = backend("pop_loadset", "eeglab_data.set", str(samples))
    eeg["chanlocs"] = backend(
        "pop_chanedit",
        eeg["chanlocs"],
        "load",
        _cell_row(str(samples / "eeglab_chan32.locs"), "filetype", "autodetect"),
    )
    alleeg, eeg, currentset = backend("eeg_store", alleeg, eeg, nargout=3)
    eeg = backend("pop_eegfilt", eeg, 1.0, 0.0, [], np.array([[0.0]]))
    alleeg, eeg, currentset = backend(
        "pop_newset", alleeg, eeg, currentset, "setname", "filtered Continuous EEG Data", nargout=3
    )
    eeg = backend("pop_reref", eeg, [], "refstate", 0.0)
    eeg["comments"] = backend(
        "pop_comments", eeg["comments"], "", "Dataset was highpass filtered at 1 Hz and rereferenced.", 1.0
    )
    eeg = backend(
        "pop_epoch",
        eeg,
        _cell_row("square"),
        np.array([[-1.0, 2.0]]),
        "newname",
        "Continuous EEG Data epochs",
        "epochinfo",
        "yes",
    )
    alleeg, eeg, currentset = backend(
        "pop_newset", alleeg, eeg, currentset, "setname", "Continuous EEG Data epochs", "overwrite", "on", nargout=3
    )
    eeg = backend("pop_rmbase", eeg, np.array([[-1000.0, 0.0]]))
    eeg["comments"] = backend(
        "pop_comments", eeg["comments"], "", "Extracted 'square' epochs [-1 2] sec, and removed baseline.", 1.0
    )
    alleeg, eeg = backend("eeg_store", alleeg, eeg, currentset, nargout=2)
    _tutorial_redraw(backend, request, window, ALLEEG=alleeg, EEG=eeg, CURRENTSET=currentset, ALLCOM=allcom)

    window, alleeg, eeg, currentset, allcom = _tutorial_start(backend, request, outputs=4)
    eeg = backend("pop_loadset", "eeglab_data.set", str(samples))
    eeg["chanlocs"] = backend(
        "pop_chanedit",
        eeg["chanlocs"],
        "load",
        _cell_row(str(samples / "eeglab_chan32.locs"), "filetype", "autodetect"),
    )
    eeg = backend("pop_eegfilt", eeg, 1.0, 0.0, [], np.array([[0.0]]))
    eeg = backend("pop_reref", eeg, [], "refstate", 0.0)
    eeg["comments"] = backend(
        "pop_comments", eeg["comments"], "", "Dataset was highpass filtered at 1 Hz and rereferenced.", 1.0
    )
    eeg = backend(
        "pop_epoch",
        eeg,
        _cell_row("square"),
        np.array([[-1.0, 2.0]]),
        "newname",
        "Continuous EEG Data epochs",
        "epochinfo",
        "yes",
    )
    eeg = backend("pop_rmbase", eeg, np.array([[-1000.0, 0.0]]))
    eeg["comments"] = backend(
        "pop_comments", eeg["comments"], "", "Extracted 'square' epochs [-1 2] sec, and removed baseline.", 1.0
    )
    alleeg, eeg, currentset = backend("eeg_store", alleeg, eeg, 1.0, nargout=3)
    _tutorial_redraw(backend, request, window, ALLEEG=alleeg, EEG=eeg, CURRENTSET=currentset, ALLCOM=allcom)
    eeg = backend("pop_resample", eeg, 128.0)
    alleeg, eeg, currentset = backend(
        "pop_newset", alleeg, eeg, currentset, "setname", "Continuous EEG Data resampled", nargout=3
    )
    eeg = backend("eeg_retrieve", alleeg, 1.0)
    currentset = 1.0
    times = np.arange(0.0, 501.0, 100.0).reshape(1, -1)
    backend(
        "pop_topoplot", eeg, 1.0, times, "Topographic plot", np.array([[2.0, 3.0]]), 0.0, "electrodes", "on", nargout=0
    )
    pos = backend("eeg_lat2point", times / 1000.0, 1.0, eeg["srate"], _tutorial_timerange(eeg))
    indices = np.floor(np.asarray(pos).ravel() + 0.5).astype(int) - 1
    mean_data = np.mean(eeg["data"][:, indices, :], axis=2)
    maxlim, minlim = np.max(mean_data), np.min(mean_data)
    limits = np.array([[-max(maxlim, -minlim), max(maxlim, -minlim)]])
    _tutorial_figure(backend, request)
    for index in range(6):
        backend("sbplot", 2.0, 3.0, float(index + 1), nargout=0)
        backend(
            "topoplot",
            mean_data[:, index : index + 1],
            eeg["chanlocs"],
            "maplimits",
            limits,
            "electrodes",
            "on",
            "style",
            "both",
            nargout=0,
        )
        _tutorial_title(backend, request, f"{times[0, index]:g} ms")
    backend("cbar", nargout=0)


@pytest.mark.gui
@eeglab_test(TUTORIAL_WRAPPER, "test_event_processing_single_dataset")
def test_reference_tutorial_event_processing_single_dataset(
    eeglab_backend, eeglab_suite_root, eeglab_options_directory, request
):
    backend = eeglab_backend
    window, alleeg, eeg, currentset, allcom = _tutorial_start(backend, request, outputs=4)
    backend("pop_editoptions", "option_storedisk", 0.0, nargout=0)
    eeg = backend("pop_loadset", "eeglab_data.set", str(eeglab_suite_root / "eeglab" / "sample_data"))
    events = _records(eeg["event"])
    for event in events:
        event["latency"] += 10.0
    eeg["event"] = _source_dataset_row(*events)
    alleeg, eeg, currentset = backend("eeg_store", alleeg, eeg, currentset, nargout=3)
    events = _records(eeg["event"])
    nevents = len(events)
    for index in range(nevents):
        event = events[index]
        if isinstance(event["type"], str) and event["type"].lower() == "square":
            events.append(dict(event))
            events[-1]["latency"] = event["latency"] - 0.1 * eeg["srate"]
            events[-1]["type"] = "cue"
    eeg["event"] = _source_dataset_row(*events)
    eeg = backend("eeg_checkset", eeg, "eventconsistency")
    alleeg, eeg, currentset = backend("eeg_store", alleeg, eeg, currentset, nargout=3)
    _tutorial_redraw(backend, request, window, ALLEEG=alleeg, EEG=eeg, CURRENTSET=currentset, ALLCOM=allcom)


@pytest.mark.gui
@eeglab_test(TUTORIAL_WRAPPER, "test_event_processing_study")
def test_reference_tutorial_event_processing_study(
    eeglab_backend, eeglab_suite_root, eeglab_working_directory, eeglab_options_directory, request
):
    backend = eeglab_backend
    samples = shutil.copytree(eeglab_suite_root / "eeglab" / "sample_data", eeglab_working_directory / "sample_data")
    window, alleeg, eeg, currentset, allcom = _tutorial_start(backend, request, outputs=4)
    backend("pop_editoptions", "option_storedisk", 0.0, nargout=0)
    eeg = backend("pop_loadset", "eeglab_data_epochs_ica.set", str(samples))
    alleeg, eeg, currentset = backend("eeg_store", alleeg, eeg, nargout=3)
    commands = []
    datasets = _records(alleeg)
    for index, dataset in enumerate(datasets, start=1):
        events = _records(dataset["event"])
        for current, following in zip(events[:-1], events[1:]):
            if (
                current["type"].lower() == "square"
                and following["type"].lower() == "rt"
                and following["epoch"] == current["epoch"]
            ):
                # MATLAB's struct field assignment initializes other rows empty.
                if "rt" not in events[0]:
                    for event in events:
                        event["rt"] = np.empty((0, 0))
                current["rt"] = (following["latency"] - current["latency"]) / dataset["srate"] * 1000.0
        dataset["event"] = _source_dataset_row(*events)
        # eeg_store without CURRENTSET clears filepath; MATLAB fullfile('',
        # name) then saves relative to the isolated current directory.
        filepath = dataset["filepath"] if np.size(dataset["filepath"]) else ""
        filename = str(Path(filepath) / f"{dataset['setname'][:-4]}_rtevents.set")
        dataset["saved"] = "no"
        # The original passes the complete ALLEEG array, not just dataset iDat.
        datasets[index - 1] = backend("pop_saveset", _source_dataset_row(*datasets), filename)
        if np.size(datasets[index - 1]["subject"]) == 0 or datasets[index - 1]["subject"] == "":
            datasets[index - 1]["subject"] = f"S{index:02d}"
        commands.extend(("index", float(index), "load", filename, "subject", datasets[index - 1]["subject"]))
    study, alleeg = backend("std_editset", [], [], "commands", _cell_row(*commands), "updatedat", "off", nargout=2)
    _tutorial_redraw(
        backend,
        request,
        window,
        ALLEEG=alleeg,
        EEG=eeg,
        CURRENTSET=currentset,
        ALLCOM=allcom,
        STUDY=study,
        CURRENTSTUDY=True,
    )


@pytest.mark.gui
@eeglab_test(TUTORIAL_WRAPPER, "test_time_freq_all_elec")
def test_reference_tutorial_time_freq_all_elec(eeglab_backend, eeglab_suite_root, request):
    backend = eeglab_backend
    window = _tutorial_start(backend, request)
    close_reference_gui(backend, request, window=window.window if window else None)
    eeg = backend("pop_loadset", str(eeglab_suite_root / "eeglab" / "sample_data" / "eeglab_data_epochs_ica.set"))
    for electrode in range(1, int(eeg["nbchan"]) + 1):
        results = _tutorial_newtimef(
            backend,
            request,
            eeg,
            1.0,
            float(electrode),
            _tutorial_timerange(eeg) * 1000.0,
            np.array([[3.0, 0.5]]),
            "maxfreq",
            50.0,
            "padratio",
            16.0,
            "plotphase",
            "off",
            "timesout",
            60.0,
            "alpha",
            0.05,
            "plotersp",
            "off",
            "plotitc",
            "off",
        )
        if electrode == 1:
            all_results = [
                np.zeros((*np.shape(result), int(eeg["nbchan"])), dtype=complex if np.iscomplexobj(result) else float)
                for result in results
            ]
        for accumulated, result in zip(all_results, results, strict=True):
            accumulated[:, :, electrode - 1] = result
    allersp, _allitc, _allpowbase, alltimes, allfreqs, allerspboot, _allitcboot = all_results
    _tutorial_figure(backend, request)
    backend(
        "tftopo",
        allersp,
        alltimes[:, :, 0],
        allfreqs[:, :, 0],
        "mode",
        "ave",
        "limits",
        np.array([[np.nan, np.nan, np.nan, 35.0, -1.5, 1.5]]),
        "signifs",
        allerspboot,
        "sigthresh",
        np.array([[6.0]]),
        "timefreqs",
        np.array([[400.0, 8.0], [350.0, 14.0], [500.0, 24.0], [1050.0, 11.0]]),
        "chanlocs",
        eeg["chanlocs"],
        nargout=0,
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


def _bids_face_eeg(subject_index: int) -> dict:
    eeg = _bids_continuous_eeg(subject_index)
    data = np.asarray(eeg["data"]).copy()
    event_types = FACE_EVENT_TYPES * 2
    amplitudes = {"famous": 5.0, "scrambled": 0.5, "unfamiliar": 2.5}
    events = []
    for index, event_type in enumerate(event_types):
        latency = 97 + index * 105
        face_type = event_type.split("_", maxsplit=1)[0]
        events.append({"type": event_type, "latency": float(latency), "duration": 0.0, "urevent": index + 1})
        center = latency - 1 + round(0.17 * eeg["srate"])
        response = np.exp(-0.5 * ((np.arange(eeg["pnts"]) - center) / (0.045 * eeg["srate"])) ** 2)
        data[0] += amplitudes[face_type] * response
    eeg.update(
        {
            "data": data,
            "setname": f"S{subject_index:02d}_FaceRecognition",
            "event": events,
            "urevent": [dict(event) for event in events],
        }
    )
    return eeg


def _add_face_type_to_bids_events(root: Path) -> None:
    for path in root.rglob("*_events.tsv"):
        lines = path.read_text(encoding="utf-8").splitlines()
        header = lines[0].split("\t")
        trial_type_index = header.index("trial_type")
        rows = [line.split("\t") for line in lines[1:]]
        output = ["\t".join([*header, "face_type"])]
        for row in rows:
            face_type = row[trial_type_index].split("_", maxsplit=1)[0]
            output.append("\t".join([*row, face_type]))
        path.write_text("\n".join(output) + "\n", encoding="utf-8")


@pytest.mark.gui
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


@pytest.mark.gui
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


@pytest.mark.gui
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


@pytest.mark.gui
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


@pytest.mark.gui
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


@pytest.mark.gui
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


# The upstream face-experiment wrapper contains comments only, not a workflow.
@pytest.mark.gui
def test_bids_face_experiment_runs_import_ica_epoch_and_trial_factor_study(tmp_path: Path):
    bids_root = tmp_path / "generated_face_recognition"
    export_commands = []
    for subject in (1, 2):
        _root, command = pop_exportbids(
            _bids_face_eeg(subject),
            bids_root,
            subject=f"{subject:02d}",
            task="FaceRecognition",
            return_com=True,
        )
        export_commands.append(command)
    _add_face_type_to_bids_events(bids_root)

    imported, import_command = pop_importbids(
        bids_root,
        eventtype="trial_type",
        bidsevent="replace",
        return_com=True,
    )
    assert isinstance(imported, list)
    selected, select_command = pop_select(imported, nochannel=["EOG1", "EOG2"], return_com=True)
    cleaned, clean_command = pop_clean_rawdata(
        selected,
        FlatlineCriterion="off",
        ChannelCriterion="off",
        LineNoiseCriterion="off",
        Highpass=[1, 2],
        BurstCriterion="off",
        WindowCriterion="off",
        gui=False,
        return_com=True,
    )
    referenced, reference_command = pop_reref(cleaned, [], return_com=True)
    decomposed, ica_command = pop_runica(
        referenced,
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
    pruned, prune_command = pop_subcomp(flagged, [], 0, 0, gui=False, return_com=True)

    epochs, epoch_command = pop_epoch(pruned, list(FACE_EVENT_TYPES), [-0.25, 0.75], return_com=True)
    study, epochs, study_command = pop_study(None, epochs, name="Generated face recognition", return_com=True)
    study, trialinfo = std_maketrialinfo(study, epochs)
    study, design_command = std_makedesign(
        study,
        epochs,
        1,
        name="Face type",
        variable1="face_type",
        values1=["famous", "scrambled", "unfamiliar"],
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

    assert len(imported) == len(selected) == len(cleaned) == len(decomposed) == len(pruned) == 2
    assert all(eeg["nbchan"] == 6 for eeg in selected)
    assert all(
        {event["face_type"] for event in eeg["event"]} == {"famous", "scrambled", "unfamiliar"} for eeg in imported
    )
    assert all(np.asarray(eeg["icaweights"]).shape == (4, 6) for eeg in pruned)
    assert all(eeg["trials"] == 6 for eeg in epochs)
    assert all([row["face_type"] for row in rows] == ["famous", "scrambled", "unfamiliar"] * 2 for rows in trialinfo)
    assert study["design"][0]["variable"][0]["value"] == ["famous", "scrambled", "unfamiliar"]
    assert len(erpdata) == 3 and all(values.shape == (times.size, 2) for values in erpdata)
    cache = next(entry for entry in study["changrp"] if entry["name"] == "Fz")
    for face_type, actual in zip(("famous", "scrambled", "unfamiliar"), erpdata):
        expected_cases = []
        for values, rows in zip(cache["erpdatatrials"], cache["erptrialinfo"]):
            mask = np.asarray([row["face_type"] == face_type for row in rows])
            expected_cases.append(np.mean(np.asarray(values)[..., mask], axis=-1))
        np.testing.assert_allclose(actual, np.stack(expected_cases, axis=-1), atol=1e-12)
    assert not np.allclose(erpdata[0], erpdata[1])
    for command in (
        *export_commands,
        import_command,
        select_command,
        clean_command,
        reference_command,
        ica_command,
        flag_command,
        prune_command,
        epoch_command,
        study_command,
        design_command,
        precompute_command,
    ):
        assert command
    plt.close(figure)


@pytest.mark.gui
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
