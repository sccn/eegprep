"""Shared limitations for source-dependent STUDY helpers."""

from __future__ import annotations

from typing import Any


SOURCE_STUDY_LIMITATION = (
    "Standalone EEGPrep does not compute EEGLAB FieldTrip/DIPFIT STUDY source workflows from "
    "std_dipplot or std_dipoleclusters. Run the DIPFIT/source-localization workflow first and use "
    "the dedicated EEGPrep DIPFIT plotting helpers for dataset-level dipoles."
)


def raise_source_study_limitation(function_name: str, *_args: Any, **_kwargs: Any) -> None:
    """Raise a clear limitation for source-dependent STUDY helpers."""
    raise NotImplementedError(f"{function_name}: {SOURCE_STUDY_LIMITATION}")


__all__ = ["SOURCE_STUDY_LIMITATION", "raise_source_study_limitation"]
