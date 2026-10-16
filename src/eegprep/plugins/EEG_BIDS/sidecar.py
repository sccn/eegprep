"""Decode BIDS TSV and JSON sidecars."""

from __future__ import annotations

from contextlib import contextmanager
from io import StringIO
import json
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import pandas as pd

_PYBIDS_JSON_LOCK = threading.RLock()


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


class _BidsJsonModule:
    """JSON module stand-in whose ``load`` uses :func:`decode_bids_text`."""

    def __init__(self, module: ModuleType) -> None:
        self._module = module

    def load(self, fp: Any, *args: Any, **kwargs: Any) -> Any:
        """Load one JSON file, decoding a fresh UTF-8 stream as BIDS text."""
        buffer = getattr(fp, "buffer", None)
        if buffer is None or buffer.tell() != 0:
            return self._module.load(fp, *args, **kwargs)
        payload = buffer.read()
        return self._module.loads(decode_bids_text(payload, suffix=".json"), *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


@contextmanager
def bids_json_decoding() -> Iterator[None]:
    """Decode indexed BIDS JSON with :func:`decode_bids_text`.

    pybids' layout indexer and root check import the ``json`` module and call
    ``json.load``. This replaces those module bindings for the duration of the
    call. Other importers of ``json`` keep the standard library module.
    """
    import bids.layout.index as index_mod
    import bids.layout.validation as validation_mod

    modules = (index_mod, validation_mod)
    with _PYBIDS_JSON_LOCK:
        already = isinstance(index_mod.json, _BidsJsonModule)
        saved = None if already else [module.json for module in modules]
        if saved is not None:
            for module, original in zip(modules, saved, strict=True):
                module.json = _BidsJsonModule(original)
        try:
            yield
        finally:
            if saved is not None:
                for module, original in zip(modules, saved, strict=True):
                    module.json = original


def _is_tsv(suffix: str) -> bool:
    return suffix.lower().endswith(".tsv")
