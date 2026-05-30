"""EEGLAB-style DIPFIT MRI headmodel entry point."""

from __future__ import annotations

from eegprep.plugins.dipfit._fieldtrip_workflows import pop_dipfit_headmodel, pop_dipfit_headmodel_dialog_spec


__all__ = ["pop_dipfit_headmodel", "pop_dipfit_headmodel_dialog_spec"]
