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
import numpy as np
import pytest

from eegprep.functions.sigprocfunc.erpimage import erpimage
from tests.eeglab_tests import eeglab_test


_ERPIMAGE_SOURCE = "unittesting_sigprocfunc/erpimage/sigprocfunc_erpimage_wrapperTest.m"


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


@eeglab_test(_ERPIMAGE_SOURCE, "test_fail_continuous")
@eeglab_test(_ERPIMAGE_SOURCE, "test_fail_no_arg")
def test_erpimage_requires_points_by_trials_data() -> None:
    with pytest.raises(ValueError, match="points x trials"):
        erpimage(np.arange(20))
    with pytest.raises(TypeError):
        erpimage()  # ty: ignore[missing-argument]


@eeglab_test(_ERPIMAGE_SOURCE, "test_fail_one_trial")
@eeglab_test(_ERPIMAGE_SOURCE, "test_fail_trial_div")
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


@eeglab_test(_ERPIMAGE_SOURCE, "test_pass_general")
@eeglab_test(_ERPIMAGE_SOURCE, "test_pass_many_args")
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


@eeglab_test(_ERPIMAGE_SOURCE, "test_pass_times")
def test_erpimage_expands_eeglab_compact_time_specification() -> None:
    data = np.arange(24, dtype=float).reshape(4, 6)
    fig, _image = erpimage(data, times=[0, 4, 1])

    np.testing.assert_allclose(fig.axes[0].get_xlim(), [0.0, 3000.0])
    np.testing.assert_allclose(fig.axes[-1].lines[0].get_xdata(), [0.0, 1000.0, 2000.0, 3000.0])
    plt.close(fig)


@eeglab_test(_ERPIMAGE_SOURCE, "test_todo_bugzilla_326")
def test_erpimage_sorts_trials_when_event_latency_lines_are_drawn() -> None:
    # The upstream TODO reports a silent sorting failure when sort values and
    # event lines are combined; its body never executes. Assert both effects
    # together so the historical regression cannot return silently.
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
