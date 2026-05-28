"""EEGLAB-style DIPFIT manual nonlinear fitting entry point."""

from __future__ import annotations

from eegprep.plugins.dipfit._fieldtrip_workflows import pop_dipfit_nonlinear, pop_dipfit_nonlinear_dialog_spec


__all__ = ["pop_dipfit_nonlinear", "pop_dipfit_nonlinear_dialog_spec"]
