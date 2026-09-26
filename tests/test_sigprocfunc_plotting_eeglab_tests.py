"""Original plotting workflows plus supplemental Python behavior tests.

The source plotcurve case 9 is commented out; the other source bodies below
are active. GUI execution of these contracts is a separate validation step.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eegprep.functions.sigprocfunc.cbar import cbar
from eegprep.functions.sigprocfunc.copyaxis import copyaxis
from eegprep.functions.sigprocfunc.forcelocs import forcelocs
from eegprep.functions.sigprocfunc.headplot import headplot
from eegprep.functions.sigprocfunc.plotcurve import plotcurve
from eegprep.functions.sigprocfunc.sbplot import DEFAULT_AXES_POSITION, sbplot
from eegprep.functions.sigprocfunc.slider import slider
from tests.eeglab_tests import eeglab_test
from tests.eeglab_tests.assertions import assert_matlab_struct_near
from tests.eeglab_tests.gui import close_reference_gui


SIGPROC = "unittesting_sigprocfunc"


def _source(name: str) -> str:
    return f"{SIGPROC}/{name}/sigprocfunc_{name}_wrapperTest.m"


def _source_force_locations(rows, *, besa=False):
    fields = ("theta", "radius", "X", "Y", "Z", "sph_theta", "sph_phi", "sph_radius")
    if besa:
        fields += ("sph_theta_besa", "sph_phi_besa")
    result = np.empty((1, 4), dtype=[(field, object) for field in ("labels", *fields, "type")])
    for index, row in enumerate(rows):
        result["labels"][0, index] = "abcd"[index]
        result["type"][0, index] = "EEG"
        for field, value in zip(fields, row, strict=True):
            result[field][0, index] = np.array([[value]], dtype=float)
    return result


@eeglab_test(_source("forcelocs"), "test_pass_x")
@eeglab_test(_source("forcelocs"), "test_pass_y")
def test_reference_forcelocs_complete_original_coordinates(eeglab_backend, subtests):
    root2 = np.sqrt(2)
    angle5 = np.degrees(np.arccos(np.sqrt(1 / 5)))
    angle58 = np.degrees(np.arccos(np.sqrt(5 / 8)))
    angle38 = np.degrees(np.arccos(np.sqrt(3 / 8)))
    angle45 = np.degrees(np.arccos(np.sqrt(4 / 5)))
    angle13 = np.degrees(np.arccos(np.sqrt(1 / 3)))
    angle23 = np.degrees(np.arccos(np.sqrt(2 / 3)))
    original = [
        [-135, 0.5, -root2 / 2, root2 / 2, 0, 135, 0, 1],
        [0, 0.5, 1, 0, 0, 0, 0, 1],
        [90, 0.5, 0, -1, 0, -90, 0, 1],
        [45, 0.5, root2 / 2, -root2 / 2, 0, -45, 0, 1],
    ]
    x_expected = [
        [
            -angle5,
            0.5 + angle58 / 180,
            root2 / 4,
            root2 / 2,
            -np.sqrt(6) / 4,
            angle5,
            -angle58,
            1,
            angle38 - 180,
            -angle45,
        ],
        [-180, 1 / 6, -0.5, 0, np.sqrt(12) / 4, 180, 60, 1, -30, 90],
        [90, 0.5, 0, -1, 0, -90, 0, 1, 90, 0],
        [
            180 - angle5,
            0.5 - angle58 / 180,
            -root2 / 4,
            -root2 / 2,
            np.sqrt(6) / 4,
            angle5 - 180,
            angle58,
            1,
            angle38,
            -angle45,
        ],
    ]
    y_original = [row.copy() for row in original]
    y_original[0] = [-135, 0.25, -0.5, 0.5, root2 / 2, 135, 45, 1]
    y_expected = [
        [-120, 0.5, -0.5, np.sqrt(12) / 4, 0, 120, 0, 1, -90, 30],
        [0, 0.5, 1, 0, 0, 0, 0, 1, 90, 90],
        [90, 0.5 - angle13 / 180, 0, -np.sqrt(1 / 3), np.sqrt(2 / 3), -90, angle13, 1, angle23, 0],
        [30, 0.5 - angle23 / 180, root2 / 2, -np.sqrt(2 / 3) / 2, np.sqrt(1 / 3), -30, angle23, 1, angle13, 60],
    ]
    for axis, rows, constraint, expected in (
        ("x", original, [-0.5, "x", "b"], x_expected),
        ("y", y_original, [1.0, "y", "a"], y_expected),
    ):
        with subtests.test(source=axis):
            actual = eeglab_backend("forcelocs", _source_force_locations(rows), np.array([constraint], dtype=object))
            assert_matlab_struct_near(_source_force_locations(expected, besa=True), actual)


@pytest.mark.gui
@eeglab_test(_source("cbar"), "test_pass_general")
@eeglab_test(_source("cbar"), "test_pass_horiz")
@eeglab_test(_source("cbar"), "test_pass_horiz_color")
@eeglab_test(_source("cbar"), "test_pass_vert")
@eeglab_test(_source("cbar"), "test_pass_vert_color")
def test_reference_cbar_original_calls(eeglab_backend, request, subtests):
    for args in (
        (),
        ("horiz",),
        ("horiz", np.arange(33.0, 65.0)[None, :]),
        ("vert",),
        ("vert", np.arange(33.0, 65.0)[None, :]),
    ):
        with subtests.test(args=args):
            if request.config.getoption("--eeglab-backend") == "matlab":
                handle = eeglab_backend("eegprep_test_gui_handle", "cbar", *args)
                # get(h) is evaluated but its properties are not asserted upstream.
                eeglab_backend("get", handle, nargout=0)
            else:
                handle = eeglab_backend("cbar", *args)
                handle.properties()
            close_reference_gui(eeglab_backend, request)


@pytest.mark.gui
@eeglab_test(_source("copyaxis"), "test_pass_no_arg")
def test_reference_copyaxis_without_arguments(eeglab_backend, request):
    eeglab_backend("copyaxis", nargout=0)
    close_reference_gui(eeglab_backend, request, all_figures=True)


@pytest.mark.gui
@eeglab_test(_source("plotcurve"), "test_test_plotcurve")
def test_reference_plotcurve_original_eight_cases(eeglab_backend, request, eeglab_suite_root):
    x = np.linspace(1.0, 10.0, 901)
    data = np.vstack((x, np.random.default_rng().standard_normal(901), np.exp(x) / np.exp(10), np.cos(x), np.sin(x)))
    times = np.linspace(-4.0, 5.0, 901)[None, :]
    for options in (
        {},
        {},
        {
            "xlabel": "Time point /ms",
            "ylabel": "EP",
            "legend": np.array([["linear", "random", "exponential", "cos", "sin"]], dtype=object),
            "title": "Unit testing",
            "vert": np.array([[-3.0, -1.6, 2.2345]]),
            "linewidth": 1.0,
        },
        {"maskarray": np.array([[-0.001, 0.001]])},
        {"val2mask": 0.5},
        {"plotmean": "on"},
    ):
        eeglab_backend("plotcurve", times, data, **options, nargout=0)
        close_reference_gui(eeglab_backend, request)
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data.set"))
    locs = eeglab_backend("pop_readlocs", str(eeglab_suite_root / "eeglab/sample_data/eeglab_chan32.locs"))
    times = np.linspace(float(np.asarray(eeg["xmin"]).item()), float(np.asarray(eeg["xmax"]).item()), 30504)[None, :]
    eeglab_backend(
        "plotcurve",
        times,
        np.asarray(eeg["data"])[1:2, :],
        maskarray=np.array([[-0.5, 0.5]]),
        highlightmode="background",
        nargout=0,
    )
    close_reference_gui(eeglab_backend, request)
    eeglab_backend("plotcurve", times, eeg["data"], chanlocs=locs, nargout=0)
    close_reference_gui(eeglab_backend, request)


@pytest.mark.gui
@eeglab_test(_source("sbplot"), "test_test_sbplot")
def test_reference_sbplot_original_calls(eeglab_backend, request):
    native = request.config.getoption("--eeglab-backend") == "matlab"
    for rows, columns, indices, color in (
        (3.0, 3.0, 6.0, "g"),
        (3.0, 3.0, np.array([[7.0, 2.0]]), "r"),
        (8.0, 7.0, 47.0, "b"),
    ):
        eeglab_backend("sbplot", rows, columns, indices, nargout=0)
        values = np.random.default_rng().random((1, 10))
        if native:
            eeglab_backend("plot", values, color, nargout=0)
        else:
            plt.plot(np.arange(1, 11), values.ravel(), color)
    close_reference_gui(eeglab_backend, request)
    for nested in (False, True):
        if nested:
            if native:
                parent = eeglab_backend("eval", "double(axes('position',[.1 .1 .8 .6]))")
            else:
                parent = plt.gcf().add_axes([0.1, 0.1, 0.8, 0.6])
            eeglab_backend("sbplot", 3.0, 3.0, 3.0, "ax", parent, nargout=0)
        else:
            eeglab_backend("sbplot", 3.0, 3.0, 3.0, "Color", "r", nargout=0)
        values = np.random.default_rng().random((1, 10))
        if native:
            eeglab_backend("hold", "on", nargout=0)
            eeglab_backend("plot", values, nargout=0)
        else:
            # Matplotlib retains existing axes artists (MATLAB hold on).
            plt.plot(np.arange(1, 11), values.ravel())
        close_reference_gui(eeglab_backend, request)


@pytest.mark.gui
@eeglab_test(_source("slider"), "test_test_slider")
def test_reference_slider_original_calls(eeglab_backend, request):
    for args in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0, 1.2, 1.2, 0.0)):
        values = np.random.default_rng().random((1, 10))
        if request.config.getoption("--eeglab-backend") == "matlab":
            eeglab_backend("plot", values, nargout=0)
            figure = eeglab_backend("eval", "double(gcf)")
        else:
            plt.plot(np.arange(1, 11), values.ravel())
            figure = plt.gcf()
        eeglab_backend("slider", figure, *args, nargout=0)
        close_reference_gui(eeglab_backend, request)


@pytest.mark.gui
@eeglab_test(_source("headplot"), "test_pass_cartesian")
@eeglab_test(_source("headplot"), "test_pass_example")
def test_reference_headplot_example_commands(eeglab_backend, request, subtests):
    for command in ("cartesian", "example"):
        with subtests.test(command=command):
            eeglab_backend("headplot", command, nargout=0)
            close_reference_gui(eeglab_backend, request)


@pytest.mark.gui
def test_cbar_default_is_a_tagged_vertical_full_colormap() -> None:
    figure, _source_axes = plt.subplots()

    axes = cbar()

    assert axes.get_gid() == "cbar"
    assert axes.images[0].get_array().shape == (plt.get_cmap().N, 1, 4)
    assert not axes.get_xticks().size
    assert axes.yaxis.get_ticks_position() == "right"
    plt.close(figure)


@pytest.mark.gui
def test_cbar_horizontal_moves_the_source_axes_and_draws_full_colormap() -> None:
    figure, source = plt.subplots()
    before = source.get_position().bounds

    axes = cbar("horiz")

    assert axes.images[0].get_array().shape == (1, plt.get_cmap().N, 4)
    assert source.get_position().height < before[3]
    assert axes.get_position().width == source.get_position().width
    assert not axes.get_yticks().size
    plt.close(figure)


@pytest.mark.gui
def test_cbar_horizontal_can_show_a_partial_one_based_color_range() -> None:
    figure, _source_axes = plt.subplots()

    axes = cbar("horiz", np.arange(33, 65))

    rgba = np.asarray(axes.images[0].get_array())
    expected = plt.get_cmap()(np.linspace(0.0, 1.0, plt.get_cmap().N))[32:64]
    assert rgba.shape == (1, 32, 4)
    np.testing.assert_allclose(rgba[0], expected)
    plt.close(figure)


@pytest.mark.gui
def test_cbar_vertical_does_not_resize_the_source_axes() -> None:
    figure, source = plt.subplots()
    before = source.get_position().bounds

    axes = cbar("vert")

    np.testing.assert_allclose(source.get_position().bounds, before)
    assert axes.get_position().x0 > source.get_position().x1
    assert axes.images[0].get_array().shape[1] == 1
    plt.close(figure)


@pytest.mark.gui
def test_cbar_vertical_partial_range_preserves_order_and_value_ticks() -> None:
    figure, _source_axes = plt.subplots()

    axes = cbar("vert", np.arange(33, 65), minmax=(-0.25, 0.75), grad=5)

    rgba = np.asarray(axes.images[0].get_array())
    expected = plt.get_cmap()(np.linspace(0.0, 1.0, plt.get_cmap().N))[32:64]
    np.testing.assert_allclose(rgba[:, 0], expected)
    np.testing.assert_allclose(axes.get_yticks(), np.linspace(0.0, 1.0, 5))
    assert [label.get_text() for label in axes.get_yticklabels()] == ["-0.25", "0.0", "0.25", "0.5", "0.75"]
    plt.close(figure)


@pytest.mark.gui
def test_copyaxis_without_arguments_copies_current_scientific_plot() -> None:
    source_figure, source = plt.subplots()
    source.plot([0, 1, 2], [2, 1, 3], "o--", label="Pz")
    source.set(xlabel="Time (s)", ylabel="Amplitude (µV)", title="ERP", xlim=(-1, 3), ylim=(-2, 4))
    source.legend()
    plt.sca(source)

    copied_figure = copyaxis()
    copied = copied_figure.axes[0]

    np.testing.assert_array_equal(copied.lines[0].get_xdata(), source.lines[0].get_xdata())
    np.testing.assert_array_equal(copied.lines[0].get_ydata(), source.lines[0].get_ydata())
    assert copied.lines[0].get_linestyle() == "--"
    assert copied.get_xlabel() == "Time (s)"
    assert copied.get_ylabel() == "Amplitude (µV)"
    assert copied.get_title() == "ERP"
    np.testing.assert_allclose(copied.get_xlim(), (-1, 3))
    assert [text.get_text() for text in copied.get_legend().get_texts()] == ["Pz"]
    plt.close(source_figure)
    plt.close(copied_figure)

    image_figure, image_axes = plt.subplots()
    image_axes.imshow([[1.0, 2.0], [3.0, 4.0]], cmap="turbo", origin="lower")
    copied_image_figure = copyaxis(source=image_axes)
    np.testing.assert_array_equal(copied_image_figure.axes[0].images[0].get_array(), [[1.0, 2.0], [3.0, 4.0]])
    assert copied_image_figure.axes[0].images[0].get_cmap().name == "turbo"
    plt.close(image_figure)
    plt.close(copied_image_figure)


def _x_rotation_locs() -> list[dict[str, float | str]]:
    return [
        {"labels": "a", "X": -np.sqrt(2) / 2, "Y": np.sqrt(2) / 2, "Z": 0.0},
        {"labels": "b", "X": 1.0, "Y": 0.0, "Z": 0.0},
        {"labels": "c", "X": 0.0, "Y": -1.0, "Z": 0.0},
        {"labels": "d", "X": np.sqrt(2) / 2, "Y": -np.sqrt(2) / 2, "Z": 0.0},
    ]


def test_forcelocs_rotates_xz_plane_and_refreshes_all_coordinate_systems() -> None:
    original = _x_rotation_locs()

    result = forcelocs(original, (-0.5, "x", "b"))

    expected = np.asarray(
        [
            [np.sqrt(2) / 4, np.sqrt(2) / 2, -np.sqrt(6) / 4],
            [-0.5, 0.0, np.sqrt(12) / 4],
            [0.0, -1.0, 0.0],
            [-np.sqrt(2) / 4, -np.sqrt(2) / 2, np.sqrt(6) / 4],
        ]
    )
    np.testing.assert_allclose([[loc["X"], loc["Y"], loc["Z"]] for loc in result], expected, atol=1e-12)
    np.testing.assert_allclose([loc["sph_radius"] for loc in result], 1.0, atol=1e-12)
    assert result[1]["theta"] == -180.0
    assert original[1]["X"] == 1.0


def test_forcelocs_rotates_yz_plane_and_matches_eeglab_expected_montage() -> None:
    original = _x_rotation_locs()
    original[0].update({"X": -0.5, "Y": 0.5, "Z": np.sqrt(2) / 2})

    result = forcelocs(original, (1.0, "y", "a"))

    expected = np.asarray(
        [
            [-0.5, np.sqrt(12) / 4, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -np.sqrt(1 / 3), np.sqrt(2 / 3)],
            [np.sqrt(2) / 2, -np.sqrt(2 / 3) / 2, np.sqrt(1 / 3)],
        ]
    )
    np.testing.assert_allclose([[loc["X"], loc["Y"], loc["Z"]] for loc in result], expected, atol=1e-12)
    np.testing.assert_allclose([result[0]["theta"], result[0]["radius"]], [-120.0, 0.5], atol=1e-12)
    np.testing.assert_allclose([result[2]["sph_theta"], result[2]["sph_phi"]], [-90.0, 54.735610317245346])


@pytest.mark.gui
def test_plotcurve_current_wrapper_cases_have_observable_curve_and_mask_behavior() -> None:
    times = np.arange(-4.0, 5.01, 0.01)
    x = np.arange(1.0, 10.01, 0.01)
    data = np.vstack([x, np.sin(x * 1.7), np.exp(x) / np.exp(10), np.cos(x), np.sin(x)])

    figure1, axis1 = plt.subplots()
    plotcurve(times, data, target=axis1)
    assert len(axis1.lines) == 5
    np.testing.assert_allclose(axis1.lines[0].get_ydata(), x)

    figure2, axis2 = plt.subplots()
    plotcurve(times, data.T, target=axis2)
    np.testing.assert_allclose(axis2.lines[4].get_ydata(), np.sin(x))

    figure3, axis3 = plt.subplots()
    plotcurve(
        times,
        data,
        target=axis3,
        xlabel="Time point /ms",
        ylabel="EP",
        legend=["linear", "oscillation", "exponential", "cos", "sin"],
        title="Unit testing",
        vert=[-3, -1.6, 2.2345],
        linewidth=1,
    )
    assert axis3.get_title() == "Unit testing"
    assert axis3.get_xlabel() == "Time point /ms"
    assert axis3.get_ylabel() == "EP"
    assert [text.get_text() for text in axis3.get_legend().get_texts()][0] == "linear"
    magenta = [line for line in axis3.lines if line.get_color() == "m" and np.asarray(line.get_xdata()).size == 2]
    np.testing.assert_allclose([line.get_xdata()[0] for line in magenta], [-3, -1.6, 2.2345])

    figure4, axis4 = plt.subplots()
    plotcurve(times, data, target=axis4, maskarray=[-0.001, 0.001])
    assert axis4.patches
    assert any(patch.get_x() + patch.get_width() > 4.0 for patch in axis4.patches)

    figure5, axis5 = plt.subplots()
    plotcurve(times, data, target=axis5, val2mask=0.5)
    assert not axis5.patches

    figure6, axis6 = plt.subplots()
    plotcurve(times, data, target=axis6, plotmean="on")
    assert len(axis6.lines) == 6
    assert axis6.lines[-1].get_color() == "k"
    assert axis6.lines[-1].get_linewidth() == 2
    np.testing.assert_allclose(axis6.lines[-1].get_ydata(), np.mean(data, axis=0))

    figure7, axis7 = plt.subplots()
    single = 0.8 * np.sin(x)
    plotcurve(times, single, target=axis7, maskarray=[-0.5, 0.5], highlightmode="background")
    assert len(axis7.lines) == 1
    assert len(axis7.patches) >= 2

    figure8, axis8 = plt.subplots()
    chanlocs = [{"labels": name} for name in ("Fz", "Cz", "Pz", "Oz", "Iz")]
    plotcurve(times, data, target=axis8, chanlocs=chanlocs)
    assert len(axis8.lines) == len(chanlocs)
    np.testing.assert_allclose(axis8.get_xlim(), (times[0], times[-1]))
    for figure in (figure1, figure2, figure3, figure4, figure5, figure6, figure7, figure8):
        plt.close(figure)


@pytest.mark.gui
def test_sbplot_current_wrapper_cases_span_grid_and_honor_properties_and_parent() -> None:
    figure = plt.figure()
    sixth = sbplot(3, 3, 6)
    spanning = sbplot(3, 3, [7, 2])
    forty_seventh = sbplot(8, 7, 47)

    assert sixth in figure.axes and spanning in figure.axes and forty_seventh in figure.axes
    assert spanning.get_position().width > sixth.get_position().width
    assert spanning.get_position().height > sixth.get_position().height
    assert forty_seventh.get_position().width < sixth.get_position().width
    np.testing.assert_allclose(DEFAULT_AXES_POSITION, (0.13, 0.11, 0.775, 0.815))
    plt.close(figure)

    color_figure = plt.figure()
    colored = sbplot(3, 3, 3, "Color", "r")
    np.testing.assert_allclose(colored.get_facecolor(), (1.0, 0.0, 0.0, 1.0))
    plt.close(color_figure)

    parent_figure = plt.figure()
    parent = parent_figure.add_axes([0.1, 0.1, 0.8, 0.6])
    nested = sbplot(3, 3, 3, "ax", parent)
    assert nested.figure is parent_figure
    assert nested.get_position().x0 >= parent.get_position().x0
    assert nested.get_position().x1 <= parent.get_position().x1 + 1e-12
    plt.close(parent_figure)


@pytest.mark.gui
def test_slider_current_wrapper_cases_create_controls_and_pan_magnified_axes() -> None:
    figure1, axis1 = plt.subplots()
    original1 = axis1.get_position().bounds
    controls1 = slider(figure1, 0, 0)
    assert controls1.horizontal is None and controls1.vertical is None
    np.testing.assert_allclose(axis1.get_position().bounds, original1)

    figure2, axis2 = plt.subplots()
    controls2 = slider(figure2, 0, 1)
    assert controls2.horizontal is None and controls2.vertical is not None
    controls2.vertical.set_val(0.25)
    np.testing.assert_allclose(axis2.get_position().bounds, controls2.original_positions[0])

    figure3, axis3 = plt.subplots()
    controls3 = slider(figure3, 1, 0)
    assert controls3.horizontal is not None and controls3.vertical is None
    controls3.horizontal.set_val(0.75)
    np.testing.assert_allclose(axis3.get_position().bounds, controls3.original_positions[0])

    figure4, axis4 = plt.subplots()
    controls4 = slider(figure4, 1, 1, 1.2, 1.2, 0)
    original4 = controls4.original_positions[0]
    initial4 = axis4.get_position().bounds
    np.testing.assert_allclose(initial4[2:], np.asarray(original4[2:]) * 1.2)
    assert controls4.dismiss.active is False
    controls4.horizontal.set_val(1.0)
    assert axis4.get_position().x0 < initial4[0]
    controls4.remove()
    np.testing.assert_allclose(axis4.get_position().bounds, original4)
    assert len(figure4.axes) == 1
    for figure in (figure1, figure2, figure3, figure4):
        plt.close(figure)


def test_headplot_cartesian_command_returns_and_prints_parseable_example(capsys) -> None:
    before = set(plt.get_fignums())

    text = headplot("cartesian")

    assert capsys.readouterr().out == text
    assert "chan_num  x        y        z" in text
    fields = text.splitlines()[3].split()
    assert fields[:4] == ["1", "0.4528", "0.8888", "-0.0694"]
    assert set(plt.get_fignums()) == before


def test_headplot_example_command_returns_and_prints_spherical_table(capsys) -> None:
    before = set(plt.get_fignums())

    text = headplot("example")

    assert capsys.readouterr().out == text
    assert "chan_num cor_deg horiz_deg" in text
    rows = [line.split() for line in text.splitlines() if line.strip() and line.strip()[0].isdigit()]
    assert len(rows) == 21
    assert rows[0][:4] == ["1", "-90", "-72", "Fp1."]
    assert rows[-1][:4] == ["21", "45", "-90", "Pz.."]
    assert set(plt.get_fignums()) == before
