"""EEGLAB's deprecated pop_dipfit_batch entry point."""

from eegprep.plugins.dipfit.pop_dipfit_gridsearch import pop_dipfit_gridsearch


def pop_dipfit_batch(*args, **kwargs):
    """Call :func:`pop_dipfit_gridsearch` with the same GUI and history behavior."""
    return pop_dipfit_gridsearch(*args, **kwargs)
