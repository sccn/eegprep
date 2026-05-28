"""EEGLAB-style DIPFIT coarse grid-search entry point."""

from __future__ import annotations

from eegprep.plugins.dipfit._fieldtrip_workflows import pop_dipfit_gridsearch, pop_dipfit_gridsearch_dialog_spec


__all__ = ["pop_dipfit_gridsearch", "pop_dipfit_gridsearch_dialog_spec"]
