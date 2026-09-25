"""Source contracts plus separate Python visualization/movie supplements.

Inactive source bodies are not executable coverage: eegmovie/fail_no_arg and
pass_general; eegplotgold/fail_no_chanfile and pass_no_chanlocs_large;
headmovie/pass_camera (immediate return); help2html/pass_general and pass_one_arg;
imagescloglog/i_pass_clim_xticks; imagesclogy/i_pass_clim_xticks;
makehtml/pass_general; seemovie/test_seemovie.

The active legacy eegplotgold, eegplotsold, getallmenus, gradplot and headmovie
contracts remain unported here. Their existing Python supplements below do not
claim source provenance. Graphical contracts require separate GUI validation.
"""

from __future__ import annotations

import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eegprep.functions.guifunc.menu_spec import menu_item, menu_to_inventory
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


GRADMAP = "unittesting_miscfunc/gradmap/miscfunc_gradmap_wrapperTest.m"
HELPFOREXE = "unittesting_miscfunc/helpforexe/miscfunc_helpforexe_wrapperTest.m"
IMAGESCLOGLOG = "unittesting_miscfunc/imagescloglog/miscfunc_imagescloglog_wrapperTest.m"
IMAGESCLOGY = "unittesting_miscfunc/imagesclogy/miscfunc_imagesclogy_wrapperTest.m"
SETFONT = "unittesting_miscfunc/setfont/miscfunc_setfont_wrapperTest.m"
SHOW_EVENTS = "unittesting_miscfunc/show_events/miscfunc_show_events_wrapperTest.m"
TEXTGUI = "unittesting_miscfunc/textgui/miscfunc_textgui_wrapperTest.m"


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
def test_reference_gradmap_center(eeglab_backend):
    values, locations = _center_gradient_input()
    gradients = eeglab_backend("gradmap", values[:, None], locations[None, :], 1.0, nargout=2)
    _assert_source_center_gradient(*gradients)
    eeglab_backend("close", nargout=0)


@pytest.mark.gui
@eeglab_test(GRADMAP, "test_pass_center_file")
def test_reference_gradmap_center_file(eeglab_backend, eeglab_suite_root, eeglab_working_directory):
    shutil.copyfile(
        eeglab_suite_root / "unittesting_miscfunc/gradmap/test.locs", eeglab_working_directory / "test.locs"
    )
    values, _locations = _center_gradient_input()
    gradients = eeglab_backend("gradmap", values[:, None], "test.locs", 1.0, nargout=2)
    _assert_source_center_gradient(*gradients)
    eeglab_backend("close", nargout=0)


@pytest.mark.gui
@eeglab_test(GRADMAP, "test_pass_corner")
def test_reference_gradmap_corner(eeglab_backend):
    values = np.array([[3.0], [4.0], [5.0], [2.0], [3.0], [4.0], [1.0], [2.0], [3.0]])
    x = np.array([[1.0, 1, 1, 0, 0, 0, -1, -1, -1]]) / 2
    y = np.array([[-1.0, 0, 1, -1, 0, 1, -1, 0, 1]]) / 2
    gradient_x, gradient_y = eeglab_backend("gradmap", values, x + 1j * y, 1.0, nargout=2)
    assert_matlab_near(max(gradient_x.shape), 9)
    assert_matlab_near(max(gradient_y.shape), 9)
    assert np.all(gradient_x >= 0)
    assert np.all(gradient_y >= 0)
    eeglab_backend("close", nargout=0)


def _source_log_images(eeglab_backend, function):
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
        eeglab_backend("close", nargout=0)


@pytest.mark.gui
@eeglab_test(IMAGESCLOGY, "test_pass_general")
@eeglab_test(IMAGESCLOGY, "test_pass_clim")
@eeglab_test(IMAGESCLOGY, "test_pass_xticks")
@eeglab_test(IMAGESCLOGY, "test_pass_ticks")
@eeglab_test(IMAGESCLOGY, "test_pass_varargin")
def test_reference_imagesclogy(eeglab_backend):
    _source_log_images(eeglab_backend, "imagesclogy")


@pytest.mark.gui
@eeglab_test(IMAGESCLOGLOG, "test_pass_general")
@eeglab_test(IMAGESCLOGLOG, "test_pass_clim")
@eeglab_test(IMAGESCLOGLOG, "test_pass_xticks")
@eeglab_test(IMAGESCLOGLOG, "test_pass_ticks")
@eeglab_test(IMAGESCLOGLOG, "test_pass_varargin")
def test_reference_imagescloglog(eeglab_backend):
    _source_log_images(eeglab_backend, "imagescloglog")


@eeglab_test(HELPFOREXE, "test_test_helpforexe")
def test_reference_helpforexe(eeglab_backend, eeglab_working_directory):
    eeglab_backend("warning", "WarnTests:convertTest", "Start to test helpforexe!", nargout=0)
    for filename in ("eeglab.m", "helpforexe.m"):
        eeglab_backend("helpforexe", np.array([[filename]], dtype=object), str(eeglab_working_directory), nargout=0)
        generated = f"help_{filename}"
        eeglab_backend("delete", generated, nargout=0)
        assert eeglab_backend("lastwarn") != f"File '{generated}' not found.", "Help file is not correctly generated"


@pytest.mark.gui
@eeglab_test(SETFONT, "test_test_setfont")
def test_reference_setfont(eeglab_backend, request):
    eeglab_backend("figure", nargout=0)
    eeglab_backend("plot", np.arange(1.0, 11.0)[None, :], nargout=0)
    for function, text in (("xlabel", "test"), ("ylabel", "test2"), ("title", "test3")):
        eeglab_backend(function, text, nargout=0)
    if request.config.getoption("--eeglab-backend") == "matlab":
        engine = request.getfixturevalue("eeglab_matlab_engine")
        # A numeric graphics handle crosses the existing MAT-file transport.
        figure = engine.double(engine.gcf())
    else:
        figure = plt.gcf()
    eeglab_backend("setfont", figure, "fontsize", 12.0, nargout=0)
    eeglab_backend("setfont", figure, "handletype", "xlabels", "fontsize", 18.0, nargout=0)
    eeglab_backend("close", nargout=0)


@pytest.mark.gui
@eeglab_test(SHOW_EVENTS, "test_test_show_events")
def test_reference_show_events(eeglab_backend, eeglab_suite_root):
    # readepochsamplefile loads this dataset when called within a test function.
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data_epochs_ica.set"))
    eeglab_backend("show_events", eeg)
    eeglab_backend("close", nargout=0)
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
    eeglab_backend("close", nargout=0)


@pytest.mark.gui
@eeglab_test(TEXTGUI, "test_test_textgui")
def test_reference_textgui(eeglab_backend):
    labels = np.array([["Test Function Covary", "Test Function Eucl"]], dtype=object)
    callbacks = np.array([["test_covary", "test_eucl"]], dtype=object)
    eeglab_backend("textgui", labels, callbacks, nargout=0)
    eeglab_backend("close", nargout=0)
    eeglab_backend(
        "textgui",
        labels,
        callbacks,
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
        nargout=0,
    )
    eeglab_backend("close", nargout=0)


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
