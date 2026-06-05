"""Shared user-facing messages for unsupported LIMO workflows."""

from __future__ import annotations

from typing import Any


LIMO_LIMITATION = (
    "Standalone EEGPrep does not implement EEGLAB's external LIMO toolbox workflow. "
    "Use EEGPrep statistics helpers for deterministic in-package analyses, or run LIMO in EEGLAB/MATLAB and "
    "import explicit results through your own analysis code."
)


def raise_limo_limitation(function_name: str, *_args: Any, **_kwargs: Any) -> None:
    """Raise a clear limitation for an external-only LIMO entry point."""
    raise NotImplementedError(f"{function_name}: {LIMO_LIMITATION}")


__all__ = ["LIMO_LIMITATION", "raise_limo_limitation"]
