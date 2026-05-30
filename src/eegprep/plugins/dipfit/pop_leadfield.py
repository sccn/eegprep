"""EEGLAB-style DIPFIT leadfield entry point."""

from __future__ import annotations

from eegprep.plugins.dipfit._fieldtrip_workflows import pop_leadfield, pop_leadfield_dialog_spec


__all__ = ["pop_leadfield", "pop_leadfield_dialog_spec"]
