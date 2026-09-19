"""Shared limitations for source-dependent STUDY helpers."""

from __future__ import annotations

from typing import Any


SOURCE_STUDY_LIMITATION = (
    "Standalone EEGPrep does not compute the EEGLAB FieldTrip/DIPFIT STUDY source workflows used by "
    "std_dipoleclusters. Run source localization first; std_dipplot can visualize existing "
    "EEG.dipfit.model results by STUDY cluster."
)


def raise_source_study_limitation(function_name: str, *_args: Any, **_kwargs: Any) -> None:
    """Raise a clear limitation for source-dependent STUDY helpers."""
    raise NotImplementedError(f"{function_name}: {SOURCE_STUDY_LIMITATION}")


__all__ = ["SOURCE_STUDY_LIMITATION", "raise_source_study_limitation"]
