"""EEGLAB's deprecated pop_dipfit_manual entry point."""

from eegprep.plugins.dipfit.pop_dipfit_nonlinear import pop_dipfit_nonlinear


def pop_dipfit_manual(*args, **kwargs):
    """Call :func:`pop_dipfit_nonlinear` with the same GUI and history behavior."""
    return pop_dipfit_nonlinear(*args, **kwargs)
