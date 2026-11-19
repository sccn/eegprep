"""Split text using EEGLAB's character-delimiter semantics."""

from __future__ import annotations

from typing import Any


DEFAULT_DELIMITERS = " ,\t\"'"


def parsetxt(txt: Any, delims: Any = None) -> list[str]:
    """Split text at any delimiter character and discard empty tokens."""
    text = str(txt)
    delimiters = DEFAULT_DELIMITERS if delims is None else _delimiter_text(delims)
    delimiter_set = set(delimiters)
    tokens: list[str] = []
    current: list[str] = []
    for character in text:
        if character in delimiter_set:
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(character)
    if current:
        tokens.append("".join(current))
    return tokens


def _delimiter_text(delims: Any) -> str:
    if isinstance(delims, str):
        return delims
    return "".join(chr(int(value)) if isinstance(value, (int, float)) else str(value) for value in delims)


__all__ = ["parsetxt"]
