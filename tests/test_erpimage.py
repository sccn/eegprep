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

from eegprep.functions.sigprocfunc.erpimage import erpimage


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
