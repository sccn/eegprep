"""Ports of current EEGLAB miscellaneous visualization/movie tests."""

from __future__ import annotations

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
from tests.eeglab_tests import eeglab_test


EEGMOVIE = "unittesting_miscfunc/eegmovie/miscfunc_eegmovie_wrapperTest.m"
EEGPLOTGOLD = "unittesting_miscfunc/eegplotgold/miscfunc_eegplotgold_wrapperTest.m"
EEGPLOTSOLD = "unittesting_miscfunc/eegplotsold/miscfunc_eegplotsold_wrapperTest.m"
GETALLMENUS = "unittesting_miscfunc/getallmenus/miscfunc_getallmenus_wrapperTest.m"
GRADMAP = "unittesting_miscfunc/gradmap/miscfunc_gradmap_wrapperTest.m"
GRADPLOT = "unittesting_miscfunc/gradplot/miscfunc_gradplot_wrapperTest.m"
HEADMOVIE = "unittesting_miscfunc/headmovie/miscfunc_headmovie_wrapperTest.m"
HELP2HTML = "unittesting_miscfunc/help2html/miscfunc_help2html_wrapperTest.m"
HELPFOREXE = "unittesting_miscfunc/helpforexe/miscfunc_helpforexe_wrapperTest.m"
IMAGESCLOGLOG = "unittesting_miscfunc/imagescloglog/miscfunc_imagescloglog_wrapperTest.m"
IMAGESCLOGY = "unittesting_miscfunc/imagesclogy/miscfunc_imagesclogy_wrapperTest.m"
MAKEHTML = "unittesting_miscfunc/makehtml/miscfunc_makehtml_wrapperTest.m"
SEEMOVIE = "unittesting_miscfunc/seemovie/miscfunc_seemovie_wrapperTest.m"
SETFONT = "unittesting_miscfunc/setfont/miscfunc_setfont_wrapperTest.m"
SHOW_EVENTS = "unittesting_miscfunc/show_events/miscfunc_show_events_wrapperTest.m"
TEXTGUI = "unittesting_miscfunc/textgui/miscfunc_textgui_wrapperTest.m"


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


@eeglab_test(EEGMOVIE, "test_fail_no_arg")
def test_eegmovie_requires_data() -> None:
    with pytest.raises(TypeError):
        eegmovie()  # ty: ignore[missing-argument]


@eeglab_test(EEGMOVIE, "test_pass_general")
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


@eeglab_test(EEGPLOTGOLD, "test_fail_no_chanfile")
def test_modern_eegplot_does_not_require_legacy_channel_file() -> None:
    model = eegplot(np.zeros((3, 10)), show=False)
    assert model.data.channel_labels == ("1", "2", "3")


@eeglab_test(EEGPLOTGOLD, "test_pass_all_args")
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


@eeglab_test(EEGPLOTGOLD, "test_pass_general")
def test_modern_eegplot_builds_a_channel_major_browser_model() -> None:
    values = np.arange(24, dtype=float).reshape(3, 8)
    model = eegplot(values, show=False)
    assert np.array_equal(model.data.flat_data, values)
    assert model.data.n_channels == 3


@eeglab_test(EEGPLOTGOLD, "test_pass_no_chanlocs")
def test_modern_eegplot_uses_numeric_labels_without_locations() -> None:
    model = eegplot(np.zeros((4, 12)), show=False)
    assert model.data.channel_labels == ("1", "2", "3", "4")


@eeglab_test(EEGPLOTGOLD, "test_pass_no_chanlocs_large")
def test_modern_eegplot_supports_large_location_free_montages() -> None:
    model = eegplot(np.zeros((128, 2)), show=False)
    assert model.data.n_channels == 128
    assert model.data.channel_labels[-1] == "128"


@eeglab_test(EEGPLOTGOLD, "test_pass_no_title")
def test_modern_eegplot_has_a_stable_empty_title_default() -> None:
    assert eegplot(np.zeros((3, 4)), show=False).state.title == "Scroll activity -- eegplot()"


@eeglab_test(EEGPLOTSOLD, "test_pass_one_arg")
def test_modern_eegplot_replaces_the_one_argument_eegplotsold_path() -> None:
    model = eegplot(np.ones((3, 5)), show=False)
    assert model.data.total_samples == 5


@eeglab_test(EEGPLOTSOLD, "test_pass_general")
def test_modern_eegplot_replaces_the_general_eegplotsold_path() -> None:
    model = eegplot(np.ones((3, 50)), srate=100, show=False)
    assert model.state.srate == 100


@eeglab_test(EEGPLOTSOLD, "test_pass_all_args")
def test_modern_eegplot_replaces_eegplotsold_display_options() -> None:
    model = eegplot(np.ones((3, 50)), srate=100, limits=(0.1, 0.3), color=("r",), show=False)
    assert model.state.limits == (0.1, 0.3)
    assert model.state.colors == ("r",)


@eeglab_test(GETALLMENUS, "test_pass_general")
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


@eeglab_test(GRADMAP, "test_pass_center")
def test_gradmap_center_points_outward() -> None:
    _assert_center_gradient(gradmap)


@eeglab_test(GRADMAP, "test_pass_center_file")
def test_gradmap_reads_eeglab_location_files(tmp_path) -> None:
    location_file = tmp_path / "test.locs"
    _write_center_locations(location_file)
    values, _locations = _center_gradient_input()
    gradient_x, gradient_y = gradmap(values, location_file, True)
    assert np.all(gradient_x[:3] < 0)
    assert np.all(gradient_y[[0, 3, 6]] > 0)
    plt.close("all")


@eeglab_test(GRADMAP, "test_pass_corner")
def test_gradmap_corner_is_nonnegative() -> None:
    _assert_corner_gradient(gradmap)


@eeglab_test(GRADPLOT, "test_fail_no_arg")
def test_gradplot_requires_inputs() -> None:
    with pytest.raises(TypeError):
        gradplot()  # ty: ignore[missing-argument]


@eeglab_test(GRADPLOT, "test_pass_center")
def test_gradplot_center_points_outward() -> None:
    _assert_center_gradient(gradplot)


@eeglab_test(GRADPLOT, "test_pass_center_file")
def test_gradplot_reads_eeglab_location_files(tmp_path) -> None:
    location_file = tmp_path / "test.locs"
    _write_center_locations(location_file)
    values, _locations = _center_gradient_input()
    gradient_x, gradient_y = gradplot(values, location_file, True)
    assert np.all(gradient_x[:3] < 0)
    assert np.all(gradient_y[[0, 3, 6]] > 0)
    plt.close("all")


@eeglab_test(GRADPLOT, "test_pass_corner")
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


@eeglab_test(HEADMOVIE, "test_pass_general")
def test_headmovie_general(headmovie_inputs) -> None:
    data, locations, spline = headmovie_inputs
    _assert_headmovie(headmovie(data, locations, spline, movieframes=[1], plot="off"), 1)


@eeglab_test(HEADMOVIE, "test_pass_camera")
def test_headmovie_camera_path_changes_the_view(headmovie_inputs) -> None:
    data, locations, spline = headmovie_inputs
    movie = _assert_headmovie(
        headmovie(data, locations, spline, camerapath=[-127, 30, 30, 0], movieframes=[1, 2], plot="off"),
        2,
    )
    assert not np.array_equal(movie[0], movie[1])


@eeglab_test(HEADMOVIE, "test_pass_elevation")
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


@eeglab_test(IMAGESCLOGY, "test_pass_general")
@eeglab_test(IMAGESCLOGY, "test_pass_clim")
def test_imagesclogy_data_and_color_limits() -> None:
    times, frequencies, values = _image_data()
    figure, axis = plt.subplots()
    mesh = imagesclogy(times, frequencies, values, [10, 16], ax=axis)
    assert axis.get_xscale() == "linear"
    assert axis.get_yscale() == "log"
    assert np.array_equal(mesh.get_array(), values)
    assert mesh.get_clim() == (10.0, 16.0)
    plt.close(figure)


@eeglab_test(IMAGESCLOGY, "test_i_pass_clim_xticks")
def test_imagesclogy_manual_color_check_has_deterministic_assertions() -> None:
    times, frequencies, values = _image_data()
    figure, axis = plt.subplots()
    mesh = imagesclogy(times, frequencies, values, [8, 16], times, ax=axis)
    assert mesh.norm(8) == 0
    assert mesh.norm(16) == 1
    assert np.array_equal(axis.get_xticks(), times)
    plt.close(figure)


@eeglab_test(IMAGESCLOGY, "test_pass_ticks")
@eeglab_test(IMAGESCLOGY, "test_pass_xticks")
def test_imagesclogy_custom_ticks() -> None:
    times, frequencies, values = _image_data()
    figure, axis = plt.subplots()
    imagesclogy(times, frequencies, values, None, [2, 3, 4], [1, 2], ax=axis)
    assert np.array_equal(axis.get_xticks(), [2, 3, 4])
    assert np.array_equal(axis.get_yticks(), [1, 2])
    plt.close(figure)


@eeglab_test(IMAGESCLOGY, "test_pass_varargin")
def test_imagesclogy_applies_axes_properties() -> None:
    times, frequencies, values = _image_data()
    figure, axis = plt.subplots()
    imagesclogy(times, frequencies, values, None, None, None, "YGrid", "on", ax=axis)
    assert any(line.get_visible() for line in axis.get_ygridlines())
    plt.close(figure)


@eeglab_test(IMAGESCLOGLOG, "test_pass_general")
@eeglab_test(IMAGESCLOGLOG, "test_pass_clim")
def test_imagescloglog_data_and_color_limits() -> None:
    times, frequencies, values = _image_data()
    figure, axis = plt.subplots()
    mesh = imagescloglog(times, frequencies, values, [10, 16], ax=axis)
    assert axis.get_xscale() == axis.get_yscale() == "log"
    assert np.array_equal(mesh.get_array(), values)
    assert mesh.get_clim() == (10.0, 16.0)
    plt.close(figure)


@eeglab_test(IMAGESCLOGLOG, "test_i_pass_clim_xticks")
def test_imagescloglog_manual_color_check_has_deterministic_assertions() -> None:
    times, frequencies, values = _image_data()
    figure, axis = plt.subplots()
    mesh = imagescloglog(times, frequencies, values, [8, 16], times, ax=axis)
    assert mesh.norm(8) == 0
    assert mesh.norm(16) == 1
    assert np.array_equal(axis.get_xticks(), times)
    plt.close(figure)


@eeglab_test(IMAGESCLOGLOG, "test_pass_ticks")
@eeglab_test(IMAGESCLOGLOG, "test_pass_xticks")
def test_imagescloglog_custom_ticks() -> None:
    times, frequencies, values = _image_data()
    figure, axis = plt.subplots()
    imagescloglog(times, frequencies, values, None, [2, 3, 4], [1, 2], ax=axis)
    assert np.array_equal(axis.get_xticks(), [2, 3, 4])
    assert np.array_equal(axis.get_yticks(), [1, 2])
    plt.close(figure)


@eeglab_test(IMAGESCLOGLOG, "test_pass_varargin")
def test_imagescloglog_applies_axes_properties() -> None:
    times, frequencies, values = _image_data()
    figure, axis = plt.subplots()
    imagescloglog(times, frequencies, values, None, None, None, "XGrid", "on", ax=axis)
    assert any(line.get_visible() for line in axis.get_xgridlines())
    plt.close(figure)


@eeglab_test(SEEMOVIE, "test_test_seemovie")
def test_seemovie_preserves_legacy_forward_backward_sequence() -> None:
    frames = np.zeros((4, 5, 6, 3), dtype=np.uint8)
    frames[:, :, :, 0] = np.arange(4)[:, None, None]
    colormap = plt.get_cmap("turbo")(np.linspace(0, 1, 65))[:, :3]
    animation = seemovie(frames, -1, colormap, fps=20, plot="off")
    assert list(animation.new_frame_seq()) == [0, 1, 2, 3, 2, 1]


@eeglab_test(SETFONT, "test_test_setfont")
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


@eeglab_test(SHOW_EVENTS, "test_test_show_events")
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


@eeglab_test(HELP2HTML, "test_pass_general")
@eeglab_test(HELP2HTML, "test_pass_one_arg")
def test_help2html_is_superseded_by_sphinx() -> None:
    pytest.skip("current MATLAB bodies are commented out; EEGPrep publishes help through Sphinx")


@eeglab_test(HELPFOREXE, "test_test_helpforexe")
def test_helpforexe_is_matlab_compiler_specific() -> None:
    pytest.skip("generating MATLAB help_*.m compiler shims is not part of a standalone Python runtime")


@eeglab_test(MAKEHTML, "test_pass_general")
def test_makehtml_is_superseded_by_sphinx() -> None:
    pytest.skip("current MATLAB body is commented out; EEGPrep builds its website with Sphinx")


@eeglab_test(TEXTGUI, "test_test_textgui")
def test_textgui_callback_eval_is_intentionally_excluded() -> None:
    pytest.skip("MATLAB textgui executes callback strings; EEGPrep uses safe declarative dialogs and packaged help")
