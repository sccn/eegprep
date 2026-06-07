"""Shared user-facing messages for unsupported LIMO workflows."""

from __future__ import annotations

from typing import Any


LIMO_LIMITATION = (
    "Standalone EEGPrep does not implement EEGLAB's external LIMO toolbox workflow. "
    "Use EEGPrep statistics helpers and std_limodesign for deterministic in-package analyses and "
    "LIMO-compatible design matrices, or run LIMO in EEGLAB/MATLAB for model fitting and result browsing."
)


def raise_limo_limitation(function_name: str, *_args: Any, **_kwargs: Any) -> None:
    """Raise a clear limitation for an external-only LIMO entry point."""
    raise NotImplementedError(f"{function_name}: {LIMO_LIMITATION}")


__all__ = ["LIMO_LIMITATION", "raise_limo_limitation"]
