"""Read a text file using EEGLAB's line-concatenation convention."""

from __future__ import annotations

from pathlib import Path


def readtxtfile(filename: str | Path) -> str:
    """Return file contents with one leading newline and normalized lines."""
    lines = Path(filename).read_text(encoding="utf-8-sig").splitlines()
    return "".join(f"\n{line}" for line in lines)


__all__ = ["readtxtfile"]
