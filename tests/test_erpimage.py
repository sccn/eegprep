"""Tests for ``erpimage`` features that ``pop_prop`` composition depends on.

``pop_prop`` reuses ``erpimage`` for its ERP-image panel, which requires two
EEGLAB-parity behaviours:

* a scalar ``caxis`` fraction sets the colour axis to ``+/- f * max(|data|)``
  (EEGLAB ``erpimage.m``: symmetric range scaled by ``caxfraction``);
* ``target`` lets the caller draw the ERP image into an existing figure/subfigure
  instead of spawning a new window, so it can sit alongside the scalp map and
  spectrum in one properties figure.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pytest

from eegprep.functions.sigprocfunc.erpimage import erpimage
from tests.eeglab_tests import eeglab_test
from tests.eeglab_tests.assertions import assert_matlab_struct_near


_ERPIMAGE_SOURCE = "unittesting_sigprocfunc/erpimage/sigprocfunc_erpimage_wrapperTest.m"


def _reference_erpimage_data():
    return np.array(
        [[21, 24, 25, 28, 31, 37], [22, 25, 26, 29, 32, 38], [23, 26, 27, 30, 33, 39], [24, 27, 28, 31, 34, 40]],
        dtype=float,
    )


@pytest.mark.gui
@eeglab_test(_ERPIMAGE_SOURCE, "test_fail_continuous")
def test_reference_erpimage_continuous(eeglab_backend):
    # These legacy "fail" bodies also pass on normal return. Their error
    # branch calls an undefined thrown_by_eeglab, so do not invent raises.
    eeglab_backend("erpimage", np.arange(21.0, 41.0)[None, :], nargout=0)


@pytest.mark.gui
@eeglab_test(_ERPIMAGE_SOURCE, "test_fail_no_arg")
def test_reference_erpimage_no_arg(eeglab_backend):
    eeglab_backend("erpimage", nargout=0)


@pytest.mark.gui
@eeglab_test(_ERPIMAGE_SOURCE, "test_fail_one_trial")
def test_reference_erpimage_one_trial(eeglab_backend):
    eeglab_backend("erpimage", _reference_erpimage_data(), np.array([[1.0]]), nargout=0)


@pytest.mark.gui
@eeglab_test(_ERPIMAGE_SOURCE, "test_fail_trial_div")
def test_reference_erpimage_trial_div(eeglab_backend):
    eeglab_backend("erpimage", _reference_erpimage_data(), np.arange(1.0, 6.0)[None, :], nargout=0)


def _reference_erpimage_outputs(eeglab_backend, request, *args):
    if request.config.getoption("--eeglab-backend") == "matlab":
        outputs = eeglab_backend("eegprep_test_erpimage_outputs", "call", *args)

        def assert_axis():
            assert eeglab_backend("eegprep_test_erpimage_outputs", "axis_type").lower() == "axes"

        return tuple(outputs.ravel(order="F")), assert_axis
    outputs = list(eeglab_backend("erpimage", *args, nargout=15))
    axes = np.asarray(outputs[4], dtype=object)
    outputs[4] = axes[..., 1:]

    def assert_axis():
        assert isinstance(axes.ravel(order="F")[0], Axes)

    return outputs, assert_axis


def _assert_reference_erpimage_outputs(outputs, assert_axis, *, limits, erp, remaining_axes):
    empty = np.empty((0, 0))
    indices = np.arange(1.0, 7.0)[None, :]
    expected = (
        _reference_erpimage_data(),
        indices,
        indices,
        np.array([limits]),
        np.full((1, remaining_axes), np.nan),
        erp,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        indices,
        empty,
    )
    for index in range(4):
        assert_matlab_struct_near(expected[index], outputs[index])
    # The source compares handles 2:end, or 4:end in the three-axis case.
    assert_matlab_struct_near(expected[4], outputs[4][..., 4 - remaining_axes :])
    assert_axis()
    for index in range(5, 15):
        assert_matlab_struct_near(expected[index], outputs[index])


@pytest.mark.gui
@eeglab_test(_ERPIMAGE_SOURCE, "test_pass_general")
def test_reference_erpimage_general(eeglab_backend, request):
    outputs, assert_axis = _reference_erpimage_outputs(eeglab_backend, request, _reference_erpimage_data())
    _assert_reference_erpimage_outputs(
        outputs, assert_axis, limits=[0, 3, *([np.nan] * 8)], erp=np.empty((0, 0)), remaining_axes=4
    )


@pytest.mark.gui
@eeglab_test(_ERPIMAGE_SOURCE, "test_pass_many_args")
def test_reference_erpimage_many_args(eeglab_backend, request):
    outputs, assert_axis = _reference_erpimage_outputs(
        eeglab_backend,
        request,
        _reference_erpimage_data(),
        np.empty((0, 0)),
        np.empty((0, 0)),
        "testcase",
        1.0,
        1.0,
        "erp",
        "cbar",
        "noxlabel",
    )
    _assert_reference_erpimage_outputs(
        outputs,
        assert_axis,
        limits=[1, 4, 36, 39.9, *([np.nan] * 6)],
        erp=np.array([[27 + 2 / 3, 28 + 2 / 3, 29 + 2 / 3, 30 + 2 / 3]]),
        remaining_axes=2,
    )


@pytest.mark.gui
@eeglab_test(_ERPIMAGE_SOURCE, "test_pass_times")
def test_reference_erpimage_times(eeglab_backend, request):
    outputs, assert_axis = _reference_erpimage_outputs(
        eeglab_backend, request, _reference_erpimage_data(), np.empty((0, 0)), np.array([[0.0, 4.0, 1.0]])
    )
    _assert_reference_erpimage_outputs(
        outputs, assert_axis, limits=[0, 3000, *([np.nan] * 8)], erp=np.empty((0, 0)), remaining_axes=4
    )


def _ramp_trials(points: int = 40, trials: int = 12) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((points, trials))


def test_scalar_caxis_sets_symmetric_fraction_of_max() -> None:
    data = _ramp_trials()
    fraction = 2.0 / 3.0
    fig, image = erpimage(data, caxis=fraction, plot_erp=False, cbar=False)

    magnitude = fraction * float(np.nanmax(np.abs(image)))
    image_ax = next(ax for ax in fig.axes if ax.get_images())
    vmin, vmax = image_ax.get_images()[0].get_clim()
    np.testing.assert_allclose([vmin, vmax], [-magnitude, magnitude], rtol=1e-9)
    plt.close("all")


def test_target_draws_into_existing_subfigure() -> None:
    data = _ramp_trials()
    host = plt.figure()
    target = host.subfigures(2, 1)[0]

    before = set(plt.get_fignums())
    result_fig, _image = erpimage(data, target=target, cbar=False)

    # No new top-level pyplot window; the panels live inside the caller's figure.
    assert set(plt.get_fignums()) == before
    assert result_fig is target
    assert len(target.axes) >= 2  # ERP image + average ERP
    plt.close("all")


def test_erpimage_requires_points_by_trials_data() -> None:
    with pytest.raises(ValueError, match="points x trials"):
        erpimage(np.arange(20))
    with pytest.raises(TypeError):
        erpimage()  # ty: ignore[missing-argument]


def test_erpimage_requires_one_sort_value_per_trial() -> None:
    data = np.array(
        [
            [21, 24, 25, 28, 31, 37],
            [22, 25, 26, 29, 32, 38],
            [23, 26, 27, 30, 33, 39],
            [24, 27, 28, 31, 34, 40],
        ]
    )
    for sort_values in ([1], [1, 2, 3, 4, 5]):
        with pytest.raises(ValueError, match="one value per trial"):
            erpimage(data, sort_values=sort_values)


def test_erpimage_preserves_default_order_and_computes_erp() -> None:
    data = np.array(
        [
            [21, 24, 25, 28, 31, 37],
            [22, 25, 26, 29, 32, 38],
            [23, 26, 27, 30, 33, 39],
            [24, 27, 28, 31, 34, 40],
        ]
    )
    fig, image = erpimage(data, title="testcase", smooth=1, decimate=1, cbar=True)

    np.testing.assert_array_equal(image, data.T)
    expected_erp = [27 + 2 / 3, 28 + 2 / 3, 29 + 2 / 3, 30 + 2 / 3]
    np.testing.assert_allclose(fig.axes[-1].lines[0].get_ydata(), expected_erp)
    assert fig.axes[0].get_title() == "testcase"
    plt.close(fig)


def test_erpimage_expands_eeglab_compact_time_specification() -> None:
    data = np.arange(24, dtype=float).reshape(4, 6)
    fig, _image = erpimage(data, times=[0, 4, 1])

    np.testing.assert_allclose(fig.axes[0].get_xlim(), [0.0, 3000.0])
    np.testing.assert_allclose(fig.axes[-1].lines[0].get_xdata(), [0.0, 1000.0, 2000.0, 3000.0])
    plt.close(fig)


def test_erpimage_sorts_trials_when_event_latency_lines_are_drawn() -> None:
    # Supplemental regression for sorting trials while drawing event lines.
    data = np.asarray(
        [
            [10.0, 20.0, 30.0],
            [11.0, 21.0, 31.0],
            [12.0, 22.0, 32.0],
            [13.0, 23.0, 33.0],
        ]
    )

    figure, image = erpimage(
        data,
        times=[-100.0, 0.0, 100.0, 200.0],
        sort_values=[30.0, 10.0, 20.0],
        vert=[0.0],
        cbar=False,
    )

    np.testing.assert_array_equal(image[:, 0], [20.0, 30.0, 10.0])
    image_axis = next(axis for axis in figure.axes if axis.images)
    assert any(np.allclose(line.get_xdata(), [0.0, 0.0]) for line in image_axis.lines)
    plt.close(figure)
