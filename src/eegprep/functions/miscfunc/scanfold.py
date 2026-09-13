"""Recursive MATLAB-source folder scanning."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def scanfold(folder: str | Path, ignore: Iterable[str] = (), max_depth: int = 100) -> tuple[list[str], str]:
    """Return MATLAB filenames below ``folder`` and EEGLAB's ``-a`` text."""
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(root)
    if max_depth < 0:
        raise ValueError("max_depth must be nonnegative")
    ignored = {str(name).casefold() for name in ignore}
    filenames = _scan(root, ignored, max_depth)
    return filenames, "".join(f" -a {name}" for name in filenames)


def _scan(folder: Path, ignored: set[str], depth: int) -> list[str]:
    if depth == 0:
        return []
    filenames: list[str] = []
    for entry in sorted(folder.iterdir(), key=lambda path: path.name.casefold()):
        if entry.is_dir():
            if entry.name.casefold() not in ignored:
                filenames.extend(_scan(entry, ignored, depth - 1))
        elif entry.suffix.casefold() == ".m":
            filenames.append(entry.name)
    return filenames


__all__ = ["scanfold"]
