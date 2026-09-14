"""Persistence helpers for EEGPrep-owned LIMO-compatible results."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np


_METADATA_KEY = "__eegprep_limo_metadata__"
_FORMAT_VERSION = 1


def save_limo_result(result: dict[str, Any], file: str | Path) -> Path:
    """Save a model or result without pickle-backed object arrays."""
    path = Path(file)
    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"format_version": _FORMAT_VERSION}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            metadata[key] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(item, (str, int, float, bool)) for item in value):
            metadata[key] = list(value)
        else:
            raise TypeError(f"LIMO result field {key!r} cannot be stored safely")
    arrays[_METADATA_KEY] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)
    return path


def load_limo_result(file: str | Path) -> dict[str, Any]:
    """Load an EEGPrep-owned ``.npz`` LIMO model or result."""
    path = Path(file)
    if path.suffix.lower() != ".npz":
        raise NotImplementedError(
            "std_readfilelimo reads EEGPrep-owned .npz outputs only; MATLAB LIMO .mat files require "
            "the external LIMO toolbox or an explicit conversion step"
        )
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read EEGPrep LIMO output {path}") from exc
    with archive:
        if _METADATA_KEY not in archive.files:
            raise ValueError(f"{path} is not an EEGPrep LIMO output")
        metadata = json.loads(str(archive[_METADATA_KEY].item()))
        version = metadata.pop("format_version", None)
        if version != _FORMAT_VERSION:
            raise ValueError(f"Unsupported EEGPrep LIMO format version: {version!r}")
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files if key != _METADATA_KEY}
        return {**metadata, **arrays}


__all__ = ["load_limo_result", "save_limo_result"]
