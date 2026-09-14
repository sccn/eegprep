"""Load rectangular delimited text while preserving mixed values."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


def loadtxt(
    filename: str | Path,
    *args: Any,
    convert: str = "on",
    skipline: int = 0,
    verbose: str = "on",
    uniformdelim: str = "off",
    blankcell: str = "on",
    convertmethod: str = "str2double",
    delim: Any = (9, 32),
    nlines: int | float | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Read an EEGLAB-style numeric or mixed-value text table.

    Positive ``skipline`` counts physical lines. Negative values count only
    non-empty lines, which is useful for files whose line endings were expanded
    during transfer. ``convert='force'`` returns a numeric, MATLAB-column-major
    flattened vector; the other conversion modes return a rectangular table.
    """
    options = _options(
        args,
        kwargs,
        convert=convert,
        skipline=skipline,
        verbose=verbose,
        uniformdelim=uniformdelim,
        blankcell=blankcell,
        convertmethod=convertmethod,
        delim=delim,
        nlines=nlines,
    )
    path = Path(filename)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    lines = _skip_lines(lines, int(options["skipline"]))
    limit = options["nlines"]
    row_limit = None if limit is None or np.isinf(float(limit)) else int(limit)
    delimiters = _delimiters(options["delim"])
    keep_blanks = str(options["blankcell"]).lower() == "on"
    uniform = str(options["uniformdelim"]).lower() == "on" or not keep_blanks

    rows: list[list[Any]] = []
    for line in lines:
        if not line:
            continue
        tokens = _split_line(line, delimiters, keep_blanks=keep_blanks, uniform=uniform)
        if not tokens:
            continue
        rows.append([_convert_token(token, str(options["convert"]), str(options["convertmethod"])) for token in tokens])
        if row_limit is not None and len(rows) >= row_limit:
            break
    if str(options["verbose"]).lower() == "on":
        logger.info("Read %d non-empty line(s) from %s", len(rows), path)
    return _as_array(rows, str(options["convert"]))


def _options(args: tuple[Any, ...], kwargs: dict[str, Any], **defaults: Any) -> dict[str, Any]:
    if len(args) % 2:
        raise ValueError("loadtxt optional arguments must be key/value pairs")
    parsed = dict(defaults)
    for index in range(0, len(args), 2):
        parsed[str(args[index]).lower()] = args[index + 1]
    for key, value in kwargs.items():
        parsed[str(key).lower()] = value
    allowed = set(defaults)
    unknown = set(parsed) - allowed
    if unknown:
        raise ValueError(f"Unsupported loadtxt option: {sorted(unknown)[0]}")
    if str(parsed["convert"]).lower() not in {"on", "off", "force"}:
        raise ValueError("convert must be 'on', 'off', or 'force'")
    if str(parsed["verbose"]).lower() not in {"on", "off"}:
        raise ValueError("verbose must be 'on' or 'off'")
    if str(parsed["uniformdelim"]).lower() not in {"on", "off"}:
        raise ValueError("uniformdelim must be 'on' or 'off'")
    if str(parsed["blankcell"]).lower() not in {"on", "off"}:
        raise ValueError("blankcell must be 'on' or 'off'")
    if str(parsed["convertmethod"]).lower() not in {"str2double", "str2num"}:
        raise ValueError("convertmethod must be 'str2double' or 'str2num'")
    skip = float(parsed["skipline"])
    if not skip.is_integer():
        raise ValueError("skipline must be an integer")
    if parsed["nlines"] is not None and not np.isinf(float(parsed["nlines"])):
        line_count = float(parsed["nlines"])
        if not line_count.is_integer() or line_count < 0:
            raise ValueError("nlines must be a non-negative integer")
    return parsed


def _skip_lines(lines: list[str], count: int) -> list[str]:
    if count >= 0:
        return lines[count:]
    remaining = abs(count)
    index = 0
    while index < len(lines) and remaining:
        if lines[index] != "":
            remaining -= 1
        index += 1
    return lines[index:]


def _delimiters(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return chr(int(value))
    return "".join(chr(int(item)) if isinstance(item, (int, float)) else str(item) for item in value)


def _split_line(line: str, delimiters: str, *, keep_blanks: bool, uniform: bool) -> list[str]:
    if not delimiters:
        return [line]
    if uniform:
        return [token for token in re.split(f"[{re.escape(delimiters)}]+", line.strip(delimiters)) if token]

    delimiter_set = set(delimiters)
    hard_delimiters = delimiter_set.intersection({"\t", ","})
    soft_delimiters = delimiter_set - hard_delimiters
    tokens: list[str] = []
    current: list[str] = []
    previous_hard = False
    at_start = True
    for character in line:
        if character not in delimiter_set:
            current.append(character)
            previous_hard = False
            at_start = False
            continue
        if character in hard_delimiters:
            if current:
                tokens.append("".join(current).strip("".join(soft_delimiters)))
                current = []
            elif keep_blanks and (previous_hard or at_start):
                tokens.append("")
            previous_hard = True
            at_start = False
        elif current:
            tokens.append("".join(current))
            current = []
            previous_hard = False
            at_start = False
    if current:
        tokens.append("".join(current))
    elif keep_blanks and previous_hard:
        tokens.append("")
    return tokens


def _convert_token(token: str, convert: str, convertmethod: str) -> Any:
    del convertmethod
    mode = convert.lower()
    if mode == "off":
        return token
    try:
        value = float(token)
    except ValueError:
        return np.nan if mode == "force" else token
    if np.isnan(value) and mode == "on":
        return token
    return value


def _as_array(rows: list[list[Any]], convert: str) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype=float if convert.lower() == "force" else object)
    width = max(len(row) for row in rows)
    fill = np.nan if convert.lower() == "force" else ""
    padded = [row + [fill] * (width - len(row)) for row in rows]
    if convert.lower() == "force":
        return np.asarray(padded, dtype=float).reshape(-1, order="F")
    return np.asarray(padded, dtype=object)


__all__ = ["loadtxt"]
