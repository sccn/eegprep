"""Decode BIDS TSV and JSON sidecars."""

from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
from typing import Any

import pandas as pd


def decode_bids_text(payload: bytes, *, suffix: str) -> str:
    """Decode one BIDS sidecar.

    UTF-8 is tried first and a leading UTF-8 BOM is removed. A ``.tsv`` file
    that is not valid UTF-8 is decoded as Latin-1, so each byte keeps its
    Unicode code point. The micro-sign byte ``0xB5`` stays µ, and ``0x80``
    stays U+0080. JSON that is not UTF-8 raises ``UnicodeDecodeError``.
    """
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        if not _is_tsv(suffix):
            raise
        return payload.decode("latin-1")


def read_bids_text(path: str | Path) -> str:
    """Read a BIDS sidecar as text."""
    file_path = Path(path)
    suffix = ".tsv" if _is_tsv(file_path.name) else file_path.suffix
    return decode_bids_text(file_path.read_bytes(), suffix=suffix)


def read_bids_table(path: str | Path) -> pd.DataFrame:
    """Return a BIDS TSV as a table.

    ``n/a`` is missing data. The file is decoded with :func:`decode_bids_text`.
    """
    return pd.read_csv(StringIO(read_bids_text(path)), sep="\t", na_values="n/a")


def read_bids_json(path: str | Path) -> dict[str, Any]:
    """Return one BIDS JSON object."""
    value = json.loads(decode_bids_text(Path(path).read_bytes(), suffix=".json"))
    if not isinstance(value, dict):
        raise ValueError(f"BIDS JSON sidecar must contain an object: {path}")
    return value


def _is_tsv(suffix: str) -> bool:
    return suffix.lower().endswith(".tsv")
