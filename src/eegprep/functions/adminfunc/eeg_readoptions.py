"""Read EEGLAB ``eeg_options.m`` files without executing MATLAB code."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


_OPTION_LINE = re.compile(r"^\s*(option_[A-Za-z0-9_]+)\s*=\s*(.*?)\s*;\s*(?:%\s*(.*))?$")


def eeg_readoptions(
    filename: str | Path,
    option_backup: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Parse an EEGLAB option script into header text and option records.

    The parser reads assignments as data and never executes the MATLAB file.
    When ``option_backup`` is supplied, matching records receive values from
    the file while the backup's shape and order are preserved.
    """
    lines = Path(filename).read_text(encoding="utf-8").splitlines()
    header_lines: list[str] = []
    options: list[dict[str, Any]] = []
    in_header = True
    for line in lines:
        if in_header and (not line.strip() or line.lstrip().startswith("%")):
            header_lines.append(line)
        else:
            in_header = False
        match = _OPTION_LINE.match(line)
        if match is None:
            continue
        name, raw_value, description = match.groups()
        options.append(
            {
                "varname": name,
                "value": _parse_matlab_value(raw_value),
                "description": description or "",
            }
        )

    if option_backup is not None:
        values = {option["varname"]: option["value"] for option in options}
        backed_up = [dict(option) for option in option_backup]
        for option in backed_up:
            name = str(option.get("varname", ""))
            if name in values:
                option["value"] = values[name]
        options = backed_up
    return "\n".join(header_lines), options


def _parse_matlab_value(value: str) -> Any:
    text = value.strip()
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1].replace("''", "'")
    if text.startswith("[") and text.endswith("]"):
        values = np.fromstring(text[1:-1].replace(",", " "), sep=" ")
        return values.tolist()
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text
