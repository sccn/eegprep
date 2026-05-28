"""Bundled DIPFIT/source-localization menu surfaces."""

from __future__ import annotations

from eegprep.plugins.dipfit._utils import DIPFITUnavailableError
from eegprep.plugins.dipfit.pop_dipfit_gridsearch import pop_dipfit_gridsearch
from eegprep.plugins.dipfit.pop_dipfit_headmodel import pop_dipfit_headmodel
from eegprep.plugins.dipfit.pop_dipfit_loreta import pop_dipfit_loreta
from eegprep.plugins.dipfit.pop_dipfit_nonlinear import pop_dipfit_nonlinear
from eegprep.plugins.dipfit.pop_dipfit_settings import pop_dipfit_settings
from eegprep.plugins.dipfit.pop_dipplot import pop_dipplot
from eegprep.plugins.dipfit.pop_leadfield import pop_leadfield
from eegprep.plugins.dipfit.pop_multifit import pop_multifit


__all__ = [
    "DIPFITUnavailableError",
    "pop_dipfit_gridsearch",
    "pop_dipfit_headmodel",
    "pop_dipfit_loreta",
    "pop_dipfit_nonlinear",
    "pop_dipfit_settings",
    "pop_dipplot",
    "pop_leadfield",
    "pop_multifit",
]
