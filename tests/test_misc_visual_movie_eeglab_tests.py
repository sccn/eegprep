"""Source contracts plus separate Python visualization/movie supplements.

Inactive source bodies are not executable coverage: eegmovie/fail_no_arg and
pass_general; eegplotgold/fail_no_chanfile and pass_no_chanlocs_large;
headmovie/pass_camera (immediate return); help2html/pass_general and pass_one_arg;
imagescloglog/i_pass_clim_xticks; imagesclogy/i_pass_clim_xticks;
makehtml/pass_general; seemovie/test_seemovie.

Legacy gradplot and headmovie helpers catch errors and return statuses that
their wrappers ignore. Their translated statuses are recorded, not asserted as
successful validation. All graphical contracts still require GUI validation.
"""

from __future__ import annotations

from contextlib import contextmanager
import shutil
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace
from eegprep.functions.guifunc.menu_spec import menu_item, menu_to_inventory
from eegprep.functions.guifunc.qt import _require_qt
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.miscfunc.eegmovie import eegmovie
from eegprep.functions.miscfunc.gradmap import gradmap
from eegprep.functions.miscfunc.gradplot import gradplot
from eegprep.functions.miscfunc.headmovie import headmovie
from eegprep.functions.miscfunc.imagescloglog import imagescloglog
from eegprep.functions.miscfunc.imagesclogy import imagesclogy
from eegprep.functions.miscfunc.seemovie import seemovie
from eegprep.functions.miscfunc.setfont import setfont
from eegprep.functions.miscfunc.show_events import show_events
from eegprep.functions.sigprocfunc.eegplot import eegplot
from eegprep.functions.sigprocfunc.headplot import headplot_setup
from tests.eeglab_tests import assert_matlab_near, eeglab_test
from tests.eeglab_tests.gui import close_reference_gui


EEGPLOTGOLD = "unittesting_miscfunc/eegplotgold/miscfunc_eegplotgold_wrapperTest.m"
EEGPLOTSOLD = "unittesting_miscfunc/eegplotsold/miscfunc_eegplotsold_wrapperTest.m"
GETALLMENUS = "unittesting_miscfunc/getallmenus/miscfunc_getallmenus_wrapperTest.m"
GRADMAP = "unittesting_miscfunc/gradmap/miscfunc_gradmap_wrapperTest.m"
GRADPLOT = "unittesting_miscfunc/gradplot/miscfunc_gradplot_wrapperTest.m"
HEADMOVIE = "unittesting_miscfunc/headmovie/miscfunc_headmovie_wrapperTest.m"
HELPFOREXE = "unittesting_miscfunc/helpforexe/miscfunc_helpforexe_wrapperTest.m"
IMAGESCLOGLOG = "unittesting_miscfunc/imagescloglog/miscfunc_imagescloglog_wrapperTest.m"
IMAGESCLOGY = "unittesting_miscfunc/imagesclogy/miscfunc_imagesclogy_wrapperTest.m"
SETFONT = "unittesting_miscfunc/setfont/miscfunc_setfont_wrapperTest.m"
SHOW_EVENTS = "unittesting_miscfunc/show_events/miscfunc_show_events_wrapperTest.m"
TEXTGUI = "unittesting_miscfunc/textgui/miscfunc_textgui_wrapperTest.m"


@contextmanager
def _source_global_data(request, eeglab_backend, data):
    if request.config.getoption("--eeglab-backend") == "matlab":
        engine = request.getfixturevalue("eeglab_matlab_engine")
        engine.eval("global data;", nargout=0)
        try:
            eeglab_backend("assignin", "base", "data", data, nargout=0)
            yield None
        finally:
            engine.eval("clear global data;", nargout=0)
        return
    workspace = EEGPrepConsoleWorkspace(EEGPrepSession())
    workspace.namespace["data"] = data
    try:
        yield workspace
    finally:
        del workspace.namespace["data"]
        workspace.close()


@pytest.mark.gui
@eeglab_test(EEGPLOTGOLD, "test_pass_all_args")
@eeglab_test(EEGPLOTGOLD, "test_pass_general")
@eeglab_test(EEGPLOTGOLD, "test_pass_no_chanlocs")
@eeglab_test(EEGPLOTGOLD, "test_pass_no_title")
def test_reference_eegplotgold(eeglab_backend, eeglab_suite_root, eeglab_working_directory, request):
    shutil.copyfile(
        eeglab_suite_root / "unittesting_miscfunc/eegplotgold/test.locs", eeglab_working_directory / "test.locs"
    )
    for arguments in (
        (0.0, "test.locs", "eegplotgold.m - Testcase", 0.0, 10.0),
        (1.0, "test.locs"),
        (1.0,),
        (1.0, "test.locs", 0.0),
    ):
        eeg = eeglab_backend("eeg_emptyset")
        eeg.update(nbchan=3.0, pnts=5.0, trials=1.0, srate=1.0, xmin=0.0, xmax=3.0)
        data = np.arange(1.0, 16.0).reshape(3, 5)
        with _source_global_data(request, eeglab_backend, data) as workspace:
            if workspace is None:
                eeglab_backend("eegplotgold", "data", *arguments, nargout=0)
            else:
                workspace.namespace["arguments"] = arguments
                workspace.execute_history_command("eegprep.eegplotgold('data', *arguments)")
        close_reference_gui(eeglab_backend, request)


@pytest.mark.gui
@eeglab_test(EEGPLOTSOLD, "test_pass_all_args")
@eeglab_test(EEGPLOTSOLD, "test_pass_general")
@eeglab_test(EEGPLOTSOLD, "test_pass_one_arg")
def test_reference_eegplotsold(eeglab_backend, eeglab_working_directory, request):
    # tc_emptyset's other fields never leave the source helper. Its source data
    # and srate are the only EEG fields passed to eegplotsold. The source folder
    # supplies no test.locs; retain that literal input without inventing a file.
    data = np.arange(1.0, 16.0).reshape(3, 5)
    for arguments in (
        (1.0, "test.locs", "eegplotsold - Testcase", 300.0, 10.0, "y", 0.0, np.array([[2.0, 3.0]])),
        (1.0,),
        (),
    ):
        eeglab_backend("eegplotsold", data, *arguments, nargout=0)
        close_reference_gui(eeglab_backend, request)


def _source_near(first, second):
    try:
        assert_matlab_near(first, second)
    except AssertionError:
        return False
    return True


@pytest.mark.gui
@eeglab_test(GRADPLOT, "test_fail_no_arg")
@eeglab_test(GRADPLOT, "test_pass_center")
@eeglab_test(GRADPLOT, "test_pass_center_file")
@eeglab_test(GRADPLOT, "test_pass_corner")
def test_reference_gradplot_ignored_statuses(
    eeglab_backend, eeglab_suite_root, eeglab_working_directory, record_property
):
    shutil.copyfile(
        eeglab_suite_root / "unittesting_miscfunc/gradplot/test.locs", eeglab_working_directory / "test.locs"
    )
    center, locations = _center_gradient_input()
    corner = np.array([[3.0], [4.0], [5.0], [2.0], [3.0], [4.0], [1.0], [2.0], [3.0]])
    x = np.array([[1.0, 1, 1, 0, 0, 0, -1, -1, -1]]) / 2
    y = np.array([[-1.0, 0, 1, -1, 0, 1, -1, 0, 1]]) / 2
    for case, arguments in (
        ("fail_no_arg", ()),
        ("pass_center", (center[:, None], locations[None, :], 1.0)),
        ("pass_center_file", (center[:, None], "test.locs", 1.0)),
        ("pass_corner", (corner, x + 1j * y, 1.0)),
    ):
        status = -1  # tc_notpassed; the wrapper never checks this return value.
        try:
            if not arguments:
                eeglab_backend("gradplot", nargout=0)
                status = 1
            else:
                gradient_x, gradient_y = eeglab_backend("gradplot", *arguments, nargout=2)
                valid = _source_near(max(gradient_x.shape), 9) and _source_near(max(gradient_y.shape), 9)
                if case == "pass_corner":
                    valid = valid and np.all(gradient_x >= 0) and np.all(gradient_y >= 0)
                else:
                    # Unlike gradmap, these source bodies compare a 3x1 vector
                    # with scalar zero. near.m rejects that shape mismatch.
                    valid = (
                        valid
                        and np.all(gradient_x[[0, 1, 2]] < 0)
                        and _source_near(gradient_x[[3, 4, 5]], 0)
                        and np.all(gradient_x[[6, 7, 8]] > 0)
                        and np.all(gradient_y[[0, 3, 6]] > 0)
                        and _source_near(gradient_y[[1, 4, 7]], 0)
                        and np.all(gradient_y[[2, 5, 8]] < 0)
                    )
                if valid:
                    status = 1
        except Exception as error:
            record_property(f"gradplot_{case}_caught_error", f"{type(error).__name__}: {error}")
        record_property(f"gradplot_{case}_ignored_source_status", status)


@pytest.mark.gui
@eeglab_test(GETALLMENUS, "test_pass_general")
def test_reference_getallmenus(eeglab_backend, request):
    matlab = request.config.getoption("--eeglab-backend") == "matlab"
    if matlab:
        engine = request.getfixturevalue("eeglab_matlab_engine")
        userdata = np.empty((1, 2), dtype=object)
        userdata[0] = [np.empty((0, 0)), np.empty((0, 0))]
        eeglab_backend(
            "figure",
            "Units",
            "points",
            "PaperPosition",
            np.array([[18.0, 180, 576, 432]]),
            "PaperUnits",
            "points",
            "name",
            "EEGLAB_TEST",
            "numbertitle",
            "off",
            "resize",
            "off",
            "Position",
            np.array([[100.0, 100, 100, 100]]),
            "color",
            np.array([[0.0, 1, 0]]),
            "Tag",
            "EEGLAB_TEST",
            "visible",
            "off",
            "Userdata",
            userdata,
            nargout=0,
        )
        window = engine.double(engine.gcf())
        eeglab_backend(
            "uicontrol",
            "Parent",
            window,
            "Units",
            "points",
            "BackgroundColor",
            np.array([[1.0, 0, 0]]),
            "ListboxTop",
            0.0,
            "HorizontalAlignment",
            "left",
            "Position",
            np.array([[100.0, 200, 300, 400]]),
            "Style",
            "frame",
            "Tag",
            "Frame1",
            nargout=0,
        )
        eeglab_backend("set", window, "MenuBar", "none", nargout=0)
    else:
        qt_core, qt_widgets = _require_qt()
        app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
        window = qt_widgets.QMainWindow()
        window.setWindowTitle("EEGLAB_TEST")
        window.setObjectName("EEGLAB_TEST")
        points = window.logicalDpiX() / 72
        window.setGeometry(*(round(value * points) for value in (100, 100, 100, 100)))
        window.setFixedSize(window.size())
        window.setStyleSheet("background-color: rgb(0, 255, 0)")
        frame = qt_widgets.QFrame(window)
        frame.setObjectName("Frame1")
        frame.setGeometry(*(round(value * points) for value in (100, 200, 300, 400)))
        frame.setStyleSheet("background-color: rgb(255, 0, 0)")
        frame.setFrameShape(qt_widgets.QFrame.Shape.Box)
        window.menuBar().setNativeMenuBar(False)
    try:
        for label in "abcdefg":
            if matlab:
                menu = engine.double(engine.uimenu(window, "Label", label))
            else:
                menu = window.menuBar().addMenu(label)
            for child in ("aa", "ab", "ac", "ad") if label == "a" else ("da",) if label == "d" else ():
                if matlab:
                    engine.uimenu(menu, "Label", child, nargout=0)
                else:
                    menu.addAction(child)
        tree = eeglab_backend("getallmenus", window)
        count = engine.numel(engine.findobj(0.0)) if matlab else len(window.findChildren(qt_core.QObject)) + 2
    finally:
        if matlab:
            engine.close(window, nargout=0)
        else:
            window.close()
            window.deleteLater()
            app.processEvents()
    expected = np.array(
        ["g", "f", "e", "d", "      da", "c", "b", "a", "      ad", "      ac", "      ab", "      aa", ""],
        dtype="U8",
    )
    # MAT transport stores a character matrix as a vector of padded row strings.
    np.testing.assert_array_equal(np.asarray(tree).reshape(-1), expected)
    assert count >= 15


def _source_headmovie(eeglab_backend, eeglab_suite_root, directory, request, case, record_property):
    matlab = request.config.getoption("--eeglab-backend") == "matlab"
    original_path = None
    stage = "initialisation"
    status = -1
    try:
        if matlab:
            engine = request.getfixturevalue("eeglab_matlab_engine")
            original_path = engine.path()
            engine.addpath(str(eeglab_suite_root / "unittesting_common/helpfunc"), nargout=0)
            plugin = eeglab_backend("tc_getPluginDir", "dipfit")
            if np.asarray(plugin).size == 0 or plugin == "":
                status = 4  # tc_nottested: the source returns immediately.
                return
            if plugin not in engine.path():
                engine.addpath(plugin, nargout=0)
        # Plugin path initialization is MATLAB-only. Python uses its own
        # headplot implementation, never MATLAB plugin code as a dependency.
        shutil.copyfile(eeglab_suite_root / "unittesting_miscfunc/headmovie/test.locs", directory / "test.locs")
        spline = directory / "test.spline"
        if spline.exists():
            spline.unlink()
        data = np.zeros((9, 4))
        data[:4] = np.eye(4)
        if matlab:
            eeglab_backend("headplot", "setup", "test.locs", "test.spline", nargout=0)
        else:
            eeglab_backend("headplot", "setup", "test.locs", splinefile="test.spline", nargout=0)
        if not spline.is_file():
            raise FileNotFoundError("headmovie: initialisation error (spline_file does not exist)")
        stage = "validation"
        arguments = () if case == "general" else (1.0, "testcase", np.array([[-127.0, 0, 30, 40]]))
        movie, colormap, _minimum, _maximum = eeglab_backend(
            "headmovie", data, "test.locs", "test.spline", *arguments, nargout=4
        )
        unique_colors, indices = np.unique(colormap, axis=0, return_index=True)
        pixels = np.concatenate(list(movie["cdata"].ravel(order="F")), axis=1) if matlab else movie
        if (
            np.all(colormap <= 1)
            and np.all(colormap >= 0)
            and _source_near(unique_colors, colormap[indices])
            and np.all(pixels <= 255)
            and np.all(pixels >= 0)
        ):
            status = 1 if case == "general" else 2  # elevation is tc_notvalidated.
        stage = "cleanup"
        if spline.exists():
            spline.unlink()
    except Exception as error:
        record_property(f"headmovie_{case}_{stage}_error", f"{type(error).__name__}: {error}")
        if stage != "cleanup":
            status = 4 if stage == "initialisation" else -1
    finally:
        if original_path is not None:
            engine.path(original_path, nargout=0)
        record_property(f"headmovie_{case}_ignored_source_status", status)


@pytest.mark.gui
@eeglab_test(HEADMOVIE, "test_pass_general")
def test_reference_headmovie_general_ignored_status(
    eeglab_backend, eeglab_suite_root, eeglab_working_directory, request, record_property
):
    _source_headmovie(eeglab_backend, eeglab_suite_root, eeglab_working_directory, request, "general", record_property)


@pytest.mark.gui
@eeglab_test(HEADMOVIE, "test_pass_elevation")
def test_reference_headmovie_elevation_ignored_status(
    eeglab_backend, eeglab_suite_root, eeglab_working_directory, request, record_property
):
    _source_headmovie(
        eeglab_backend, eeglab_suite_root, eeglab_working_directory, request, "elevation", record_property
    )


def _assert_source_center_gradient(gradient_x, gradient_y):
    assert_matlab_near(max(gradient_x.shape), 9)
    assert_matlab_near(max(gradient_y.shape), 9)
    assert np.all(gradient_x[[0, 1, 2]] < 0)
    assert_matlab_near(gradient_x[[3, 4, 5]], np.zeros((3, 1)))
    assert np.all(gradient_x[[6, 7, 8]] > 0)
    assert np.all(gradient_y[[0, 3, 6]] > 0)
    assert_matlab_near(gradient_y[[1, 4, 7]], np.zeros((3, 1)))
    assert np.all(gradient_y[[2, 5, 8]] < 0)


@pytest.mark.gui
@eeglab_test(GRADMAP, "test_pass_center")
def test_reference_gradmap_center(eeglab_backend, request):
    values, locations = _center_gradient_input()
    gradients = eeglab_backend("gradmap", values[:, None], locations[None, :], 1.0, nargout=2)
    _assert_source_center_gradient(*gradients)
    close_reference_gui(eeglab_backend, request)


@pytest.mark.gui
@eeglab_test(GRADMAP, "test_pass_center_file")
def test_reference_gradmap_center_file(eeglab_backend, eeglab_suite_root, eeglab_working_directory, request):
    shutil.copyfile(
        eeglab_suite_root / "unittesting_miscfunc/gradmap/test.locs", eeglab_working_directory / "test.locs"
    )
    values, _locations = _center_gradient_input()
    gradients = eeglab_backend("gradmap", values[:, None], "test.locs", 1.0, nargout=2)
    _assert_source_center_gradient(*gradients)
    close_reference_gui(eeglab_backend, request)


@pytest.mark.gui
@eeglab_test(GRADMAP, "test_pass_corner")
def test_reference_gradmap_corner(eeglab_backend, request):
    values = np.array([[3.0], [4.0], [5.0], [2.0], [3.0], [4.0], [1.0], [2.0], [3.0]])
    x = np.array([[1.0, 1, 1, 0, 0, 0, -1, -1, -1]]) / 2
    y = np.array([[-1.0, 0, 1, -1, 0, 1, -1, 0, 1]]) / 2
    gradient_x, gradient_y = eeglab_backend("gradmap", values, x + 1j * y, 1.0, nargout=2)
    assert_matlab_near(max(gradient_x.shape), 9)
    assert_matlab_near(max(gradient_y.shape), 9)
    assert np.all(gradient_x >= 0)
    assert np.all(gradient_y >= 0)
    close_reference_gui(eeglab_backend, request)


def _source_log_images(eeglab_backend, request, function):
    times = np.arange(1.0, 5.0)[None, :]
    frequencies = np.arange(1.0, 5.0)[None, :]
    data = np.arange(1.0, 17.0).reshape(4, 4)
    empty = np.empty((0, 0))
    # Each active source body only calls the plotting function and closes it.
    for arguments in (
        (),
        (np.array([[10.0, 16.0]]),),
        (empty, np.array([[1.0, 2.0, 3.0]])),
        (empty, np.array([[2.0, 3.0, 4.0]]), np.array([[1.0, 2.0]])),
        (empty, np.array([[2.0, 3.0, 4.0]]), np.array([[1.0, 2.0]]), "XGrid", "on"),
    ):
        eeglab_backend(function, times, frequencies, data, *arguments, nargout=0)
        close_reference_gui(eeglab_backend, request)


@pytest.mark.gui
@eeglab_test(IMAGESCLOGY, "test_pass_general")
@eeglab_test(IMAGESCLOGY, "test_pass_clim")
@eeglab_test(IMAGESCLOGY, "test_pass_xticks")
@eeglab_test(IMAGESCLOGY, "test_pass_ticks")
@eeglab_test(IMAGESCLOGY, "test_pass_varargin")
def test_reference_imagesclogy(eeglab_backend, request):
    _source_log_images(eeglab_backend, request, "imagesclogy")


@pytest.mark.gui
@eeglab_test(IMAGESCLOGLOG, "test_pass_general")
@eeglab_test(IMAGESCLOGLOG, "test_pass_clim")
@eeglab_test(IMAGESCLOGLOG, "test_pass_xticks")
@eeglab_test(IMAGESCLOGLOG, "test_pass_ticks")
@eeglab_test(IMAGESCLOGLOG, "test_pass_varargin")
def test_reference_imagescloglog(eeglab_backend, request):
    _source_log_images(eeglab_backend, request, "imagescloglog")


@eeglab_test(HELPFOREXE, "test_test_helpforexe")
def test_reference_helpforexe(eeglab_backend, eeglab_working_directory, request):
    matlab = request.config.getoption("--eeglab-backend") == "matlab"
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        if matlab:
            eeglab_backend("warning", "WarnTests:convertTest", "Start to test helpforexe!", nargout=0)
        else:
            warnings.warn("WarnTests:convertTest: Start to test helpforexe!", stacklevel=1)
        for filename in ("eeglab.m", "helpforexe.m"):
            eeglab_backend("helpforexe", np.array([[filename]], dtype=object), str(eeglab_working_directory), nargout=0)
            generated = f"help_{filename}"
            if matlab:
                eeglab_backend("delete", generated, nargout=0)
                message = eeglab_backend("lastwarn")
            else:
                output = eeglab_working_directory / generated
                if output.exists():
                    output.unlink()
                else:
                    warnings.warn(f"File '{generated}' not found.", stacklevel=1)
                message = str(emitted[-1].message)
            assert message != f"File '{generated}' not found.", "Help file is not correctly generated"


@pytest.mark.gui
@eeglab_test(SETFONT, "test_test_setfont")
def test_reference_setfont(eeglab_backend, request):
    if request.config.getoption("--eeglab-backend") == "matlab":
        eeglab_backend("figure", nargout=0)
        eeglab_backend("plot", np.arange(1.0, 11.0)[None, :], nargout=0)
        for function, text in (("xlabel", "test"), ("ylabel", "test2"), ("title", "test3")):
            eeglab_backend(function, text, nargout=0)
        engine = request.getfixturevalue("eeglab_matlab_engine")
        # A numeric graphics handle crosses the existing MAT-file transport.
        figure = engine.double(engine.gcf())
    else:
        figure = plt.figure()
        plt.plot(np.arange(1.0, 11.0))
        plt.xlabel("test")
        plt.ylabel("test2")
        plt.title("test3")
    eeglab_backend("setfont", figure, "fontsize", 12.0, nargout=0)
    eeglab_backend("setfont", figure, "handletype", "xlabels", "fontsize", 18.0, nargout=0)
    close_reference_gui(eeglab_backend, request)


@pytest.mark.gui
@eeglab_test(SHOW_EVENTS, "test_test_show_events")
def test_reference_show_events(eeglab_backend, eeglab_suite_root, request):
    # readepochsamplefile loads this dataset when called within a test function.
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data_epochs_ica.set"))
    eeglab_backend("show_events", eeg)
    close_reference_gui(eeglab_backend, request)
    time_warp = eeglab_backend(
        "make_timewarp",
        eeg,
        np.array([["square", "rt"]], dtype=object),
        "baselineLatency",
        0.0,
        "maxSTDForAbsolute",
        0.6,
        "maxSTDForRelative",
        0.4,
    )
    eeglab_backend(
        "show_events", eeg, "eventThicknessCoef", 0.5, "eventNames", time_warp["eventSequence"], "timeWarp", time_warp
    )
    close_reference_gui(eeglab_backend, request)


@pytest.mark.gui
@eeglab_test(TEXTGUI, "test_test_textgui")
def test_reference_textgui(eeglab_backend, request):
    labels = np.array([["Test Function Covary", "Test Function Eucl"]], dtype=object)
    callbacks = np.array([["test_covary", "test_eucl"]], dtype=object)
    for arguments in (
        (),
        (
            "title",
            "Test",
            "fontweight",
            np.array([["light", "bold"]], dtype=object),
            "fontsize",
            np.array([[14.0, 16.0]], dtype=object),
            "fontname",
            np.array([["Courier", "Courier"]], dtype=object),
            "lineperpage",
            10.0,
        ),
    ):
        window = None
        if request.config.getoption("--eeglab-backend") == "matlab":
            eeglab_backend("textgui", labels, callbacks, *arguments, nargout=0)
        else:
            window = eeglab_backend("textgui", labels, callbacks, *arguments)
        close_reference_gui(eeglab_backend, request, window=window)


# The remaining tests are supplemental Python behavior, not source ports.


def _polar_locations(count: int = 8) -> list[dict[str, float | str]]:
    angles = np.linspace(0.0, 360.0, count, endpoint=False)
    locations = []
    for index, angle in enumerate(angles):
        radians = np.deg2rad(angle)
        locations.append(
            {
                "labels": f"E{index + 1}",
                "theta": float(angle),
                "radius": 0.4,
                "X": float(np.cos(radians)),
                "Y": float(np.sin(radians)),
                "Z": 0.5,
            }
        )
    return locations


def _center_gradient_input() -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(2.0) / 2.0
    x = np.asarray([scale, 1, scale, 0, 0, 0, -scale, -1, -scale]) / 2.0
    y = np.asarray([-scale, 0, scale, -1, 0, 1, -scale, 0, scale]) / 2.0
    values = np.asarray([1, 1, 1, 1, 2, 1, 1, 1, 1], dtype=float)
    return values, x + 1j * y


def _assert_center_gradient(function) -> None:
    values, locations = _center_gradient_input()
    gradient_x, gradient_y = function(values, locations, True)
    assert gradient_x.shape == gradient_y.shape == (9, 1)
    assert np.all(gradient_x[:3] < 0)
    assert np.allclose(gradient_x[3:6], 0, atol=1e-12)
    assert np.all(gradient_x[6:] > 0)
    assert np.all(gradient_y[[0, 3, 6]] > 0)
    assert np.allclose(gradient_y[[1, 4, 7]], 0, atol=1e-12)
    assert np.all(gradient_y[[2, 5, 8]] < 0)
    assert plt.get_fignums()
    plt.close("all")


def _assert_corner_gradient(function) -> None:
    x = np.asarray([1, 1, 1, 0, 0, 0, -1, -1, -1], dtype=float) / 2.0
    y = np.asarray([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=float) / 2.0
    values = np.asarray([3, 4, 5, 2, 3, 4, 1, 2, 3], dtype=float)
    gradient_x, gradient_y = function(values, x + 1j * y, True)
    assert np.all(gradient_x >= -1e-12)
    assert np.all(gradient_y >= -1e-12)
    assert plt.get_fignums()
    plt.close("all")


def _write_center_locations(path) -> None:
    theta = (-45, 0, 45, -90, 0, 90, -135, 180, 135)
    radius = (0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5)
    rows = [f"{index}\t{angle}\t{rad}\tE{index}" for index, (angle, rad) in enumerate(zip(theta, radius), 1)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_eegmovie_requires_data() -> None:
    with pytest.raises(TypeError):
        eegmovie()  # ty: ignore[missing-argument]


@pytest.mark.gui
def test_eegmovie_returns_replayable_rgb_frames() -> None:
    data = np.arange(32, dtype=float).reshape(8, 4)
    movie, colormap = eegmovie(
        data,
        128,
        _polar_locations(),
        movieframes=[1, 4],
        timecourse="off",
        plot="off",
    )
    assert movie.shape[0] == 2
    assert movie.ndim == 4 and movie.shape[-1] == 3
    assert movie.dtype == np.uint8
    assert not np.array_equal(movie[0], movie[1])
    assert colormap.shape == (65, 3)
    assert np.all((colormap >= 0) & (colormap <= 1))


def test_modern_eegplot_does_not_require_legacy_channel_file() -> None:
    model = eegplot(np.zeros((3, 10)), show=False)
    assert model.data.channel_labels == ("1", "2", "3")


def test_modern_eegplot_normalizes_all_relevant_legacy_display_inputs() -> None:
    model = eegplot(
        np.arange(40, dtype=float).reshape(4, 10),
        "srate",
        20,
        "spacing",
        5,
        "winlength",
        0.4,
        "title",
        "legacy trace",
        "show",
        False,
    )
    assert model.state.srate == 20
    assert model.state.spacing == 5
    assert model.state.winlength == 0.4
    assert model.state.title == "legacy trace"


def test_modern_eegplot_builds_a_channel_major_browser_model() -> None:
    values = np.arange(24, dtype=float).reshape(3, 8)
    model = eegplot(values, show=False)
    assert np.array_equal(model.data.flat_data, values)
    assert model.data.n_channels == 3


def test_modern_eegplot_uses_numeric_labels_without_locations() -> None:
    model = eegplot(np.zeros((4, 12)), show=False)
    assert model.data.channel_labels == ("1", "2", "3", "4")


def test_modern_eegplot_supports_large_location_free_montages() -> None:
    model = eegplot(np.zeros((128, 2)), show=False)
    assert model.data.n_channels == 128
    assert model.data.channel_labels[-1] == "128"


def test_modern_eegplot_has_a_stable_empty_title_default() -> None:
    assert eegplot(np.zeros((3, 4)), show=False).state.title == "Scroll activity -- eegplot()"


def test_modern_eegplot_replaces_the_one_argument_eegplotsold_path() -> None:
    model = eegplot(np.ones((3, 5)), show=False)
    assert model.data.total_samples == 5


def test_modern_eegplot_replaces_the_general_eegplotsold_path() -> None:
    model = eegplot(np.ones((3, 50)), srate=100, show=False)
    assert model.state.srate == 100


def test_modern_eegplot_replaces_eegplotsold_display_options() -> None:
    model = eegplot(np.ones((3, 50)), srate=100, limits=(0.1, 0.3), color=("r",), show=False)
    assert model.state.limits == (0.1, 0.3)
    assert model.state.colors == ("r",)


def test_declarative_menu_inventory_replaces_matlab_handle_introspection() -> None:
    items = (
        menu_item("a", children=(menu_item("aa"), menu_item("ab"))),
        menu_item("b"),
        menu_item("d", children=(menu_item("da"),)),
    )
    inventory = menu_to_inventory(items)
    assert [item["label"] for item in inventory] == ["a", "b", "d"]
    assert [item["label"] for item in inventory[0]["children"]] == ["aa", "ab"]
    assert inventory[2]["children"][0]["label"] == "da"


@pytest.mark.gui
def test_gradmap_center_points_outward() -> None:
    _assert_center_gradient(gradmap)


@pytest.mark.gui
def test_gradmap_reads_eeglab_location_files(tmp_path) -> None:
    location_file = tmp_path / "test.locs"
    _write_center_locations(location_file)
    values, _locations = _center_gradient_input()
    gradient_x, gradient_y = gradmap(values, location_file, True)
    assert np.all(gradient_x[:3] < 0)
    assert np.all(gradient_y[[0, 3, 6]] > 0)
    plt.close("all")


@pytest.mark.gui
def test_gradmap_corner_is_nonnegative() -> None:
    _assert_corner_gradient(gradmap)


def test_gradplot_requires_inputs() -> None:
    with pytest.raises(TypeError):
        gradplot()  # ty: ignore[missing-argument]


@pytest.mark.gui
def test_gradplot_center_points_outward() -> None:
    _assert_center_gradient(gradplot)


@pytest.mark.gui
def test_gradplot_reads_eeglab_location_files(tmp_path) -> None:
    location_file = tmp_path / "test.locs"
    _write_center_locations(location_file)
    values, _locations = _center_gradient_input()
    gradient_x, gradient_y = gradplot(values, location_file, True)
    assert np.all(gradient_x[:3] < 0)
    assert np.all(gradient_y[[0, 3, 6]] > 0)
    plt.close("all")


@pytest.mark.gui
def test_gradplot_corner_is_nonnegative() -> None:
    _assert_corner_gradient(gradplot)


@pytest.fixture(scope="module")
def headmovie_inputs(tmp_path_factory):
    locations = _polar_locations()
    spline = tmp_path_factory.mktemp("headmovie") / "movie.spl"
    headplot_setup(locations, spline)
    data = np.arange(24, dtype=float).reshape(8, 3)
    return data, locations, spline


def _assert_headmovie(result, expected_frames: int) -> np.ndarray:
    movie, colormap, lower, upper = result
    assert movie.shape[0] == expected_frames
    assert movie.shape[-1] == 3 and movie.dtype == np.uint8
    assert colormap.shape == (65, 3)
    assert lower < 0 < upper and lower == -upper
    return movie


@pytest.mark.gui
def test_headmovie_general(headmovie_inputs) -> None:
    data, locations, spline = headmovie_inputs
    _assert_headmovie(headmovie(data, locations, spline, movieframes=[1], plot="off"), 1)


@pytest.mark.gui
def test_headmovie_camera_path_changes_the_view(headmovie_inputs) -> None:
    data, locations, spline = headmovie_inputs
    movie = _assert_headmovie(
        headmovie(data, locations, spline, camerapath=[-127, 30, 30, 0], movieframes=[1, 2], plot="off"),
        2,
    )
    assert not np.array_equal(movie[0], movie[1])


@pytest.mark.gui
def test_headmovie_elevation_path_changes_the_view(headmovie_inputs) -> None:
    data, locations, spline = headmovie_inputs
    movie = _assert_headmovie(
        headmovie(data, locations, spline, camerapath=[-127, 0, 10, 20], movieframes=[1, 2], plot="off"),
        2,
    )
    assert not np.array_equal(movie[0], movie[1])


def _image_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(1, 5, dtype=float)
    frequencies = np.arange(1, 5, dtype=float)
    values = np.arange(1, 17, dtype=float).reshape(4, 4)
    return times, frequencies, values


@pytest.mark.gui
def test_imagesclogy_data_and_color_limits() -> None:
    times, frequencies, values = _image_data()
    figure, axis = plt.subplots()
    mesh = imagesclogy(times, frequencies, values, [10, 16], ax=axis)
    assert axis.get_xscale() == "linear"
    assert axis.get_yscale() == "log"
    assert np.array_equal(mesh.get_array(), values)
    assert mesh.get_clim() == (10.0, 16.0)
    plt.close(figure)


@pytest.mark.gui
def test_imagesclogy_manual_color_check_has_deterministic_assertions() -> None:
    times, frequencies, values = _image_data()
    figure, axis = plt.subplots()
    mesh = imagesclogy(times, frequencies, values, [8, 16], times, ax=axis)
    assert mesh.norm(8) == 0
    assert mesh.norm(16) == 1
    assert np.array_equal(axis.get_xticks(), times)
    plt.close(figure)


@pytest.mark.gui
def test_imagesclogy_custom_ticks() -> None:
    times, frequencies, values = _image_data()
    figure, axis = plt.subplots()
    imagesclogy(times, frequencies, values, None, [2, 3, 4], [1, 2], ax=axis)
    assert np.array_equal(axis.get_xticks(), [2, 3, 4])
    assert np.array_equal(axis.get_yticks(), [1, 2])
    plt.close(figure)


@pytest.mark.gui
def test_imagesclogy_applies_axes_properties() -> None:
    times, frequencies, values = _image_data()
    figure, axis = plt.subplots()
    imagesclogy(times, frequencies, values, None, None, None, "YGrid", "on", ax=axis)
    assert any(line.get_visible() for line in axis.get_ygridlines())
    plt.close(figure)


@pytest.mark.gui
def test_imagescloglog_data_and_color_limits() -> None:
    times, frequencies, values = _image_data()
    figure, axis = plt.subplots()
    mesh = imagescloglog(times, frequencies, values, [10, 16], ax=axis)
    assert axis.get_xscale() == axis.get_yscale() == "log"
    assert np.array_equal(mesh.get_array(), values)
    assert mesh.get_clim() == (10.0, 16.0)
    plt.close(figure)


@pytest.mark.gui
def test_imagescloglog_manual_color_check_has_deterministic_assertions() -> None:
    times, frequencies, values = _image_data()
    figure, axis = plt.subplots()
    mesh = imagescloglog(times, frequencies, values, [8, 16], times, ax=axis)
    assert mesh.norm(8) == 0
    assert mesh.norm(16) == 1
    assert np.array_equal(axis.get_xticks(), times)
    plt.close(figure)


@pytest.mark.gui
def test_imagescloglog_custom_ticks() -> None:
    times, frequencies, values = _image_data()
    figure, axis = plt.subplots()
    imagescloglog(times, frequencies, values, None, [2, 3, 4], [1, 2], ax=axis)
    assert np.array_equal(axis.get_xticks(), [2, 3, 4])
    assert np.array_equal(axis.get_yticks(), [1, 2])
    plt.close(figure)


@pytest.mark.gui
def test_imagescloglog_applies_axes_properties() -> None:
    times, frequencies, values = _image_data()
    figure, axis = plt.subplots()
    imagescloglog(times, frequencies, values, None, None, None, "XGrid", "on", ax=axis)
    assert any(line.get_visible() for line in axis.get_xgridlines())
    plt.close(figure)


@pytest.mark.gui
def test_seemovie_preserves_legacy_forward_backward_sequence() -> None:
    frames = np.zeros((4, 5, 6, 3), dtype=np.uint8)
    frames[:, :, :, 0] = np.arange(4)[:, None, None]
    colormap = plt.get_cmap("turbo")(np.linspace(0, 1, 65))[:, :3]
    animation = seemovie(frames, -1, colormap, fps=20, plot="off")
    assert list(animation.new_frame_seq()) == [0, 1, 2, 3, 2, 1]


@pytest.mark.gui
def test_setfont_updates_all_text_then_selected_xlabels() -> None:
    figure, axis = plt.subplots()
    axis.plot(np.arange(10))
    axis.set_xlabel("test")
    axis.set_ylabel("test2")
    axis.set_title("test3")
    setfont(figure, "fontsize", 12)
    setfont(figure, "handletype", "xlabels", "fontsize", 18)
    assert axis.xaxis.label.get_fontsize() == 18
    assert axis.yaxis.label.get_fontsize() == 12
    assert axis.title.get_fontsize() == 12
    plt.close(figure)


@pytest.mark.gui
def test_show_events_renders_and_dims_timewarp_rejections() -> None:
    eeg = {
        "xmin": -0.2,
        "xmax": 0.8,
        "epoch": [
            {"eventtype": ["square", "rt"], "eventlatency": [0, 300]},
            {"eventtype": ["square", "rt"], "eventlatency": [0, 500]},
            {"eventtype": ["square", "rt"], "eventlatency": [0, 700]},
        ],
    }
    baseline = show_events(eeg, plot="off", image_shape=(30, 100))
    time_warp = {
        "event_sequence": ["square", "rt"],
        "epochs": np.asarray([0, 2]),
        "latencies": np.asarray([[0, 300], [0, 700]]),
    }
    warped = show_events(
        eeg,
        "eventThicknessCoef",
        0.5,
        "eventNames",
        ["square", "rt"],
        "timeWarp",
        time_warp,
        plot="off",
        image_shape=(30, 100),
    )
    assert baseline.shape == warped.shape == (30, 100, 3)
    square_column = 20
    assert np.allclose(warped[15, square_column], baseline[15, square_column] * 0.3)
    assert np.allclose(warped[5, square_column], baseline[5, square_column])
