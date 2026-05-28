"""EEGLAB-style DIPFIT multi-component autofit entry point."""

from __future__ import annotations

from eegprep.plugins.dipfit._fieldtrip_workflows import pop_multifit, pop_multifit_dialog_spec


__all__ = ["pop_multifit", "pop_multifit_dialog_spec"]
