"""Write EEGLAB-style channel-location files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc._pop_utils import is_empty_value, parse_key_value_args, parse_numeric_sequence
from eegprep.functions.sigprocfunc.convertlocs import convertlocs
from eegprep.functions.sigprocfunc.readlocs import CHANNEL_FORMATS


def writelocs(chans: Any, filename: str | Path, *args: Any, **kwargs: Any) -> None:
    """Write channel locations to an EEGLAB-compatible text file."""
    options = parse_key_value_args(args, kwargs, lowercase_keys=True, lowercase_kwargs=True)
    locs = chanlocs_as_list(chans.get("chanlocs") if isinstance(chans, dict) and "chanlocs" in chans else chans)
    locs = [dict(loc) for loc in locs]
    if str(options.get("unicoord", "on")).lower() == "on":
        locs = convertlocs(locs, "auto")
    elecind = options.get("elecind")
    if not is_empty_value(elecind):
        indices = [int(index) - 1 for index in parse_numeric_sequence(elecind, dtype=int)]
        locs = [locs[index] for index in indices]

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    filetype = _resolve_filetype(path, options)
    fields = tuple(options.get("format") or CHANNEL_FORMATS.get(filetype, CHANNEL_FORMATS["loc"])["format"])
    lines = _header_lines(fields, filetype, len(locs), options)
    lines.extend(_format_row(index, loc, fields) for index, loc in enumerate(locs, start=1))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_filetype(path: Path, options: dict[str, Any]) -> str:
    requested = str(options.get("filetype") or "").strip().lower()
    if requested:
        return {"locs": "loc", "eloc": "loc"}.get(requested, requested)
    suffix = path.suffix.lower().lstrip(".")
    return {"locs": "loc", "eloc": "loc", "ced": "chanedit", "eps": "besa"}.get(suffix, suffix or "loc")


def _header_lines(fields: tuple[str, ...], filetype: str, nchan: int, options: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    custom = str(options.get("customheader") or "").strip()
    if custom:
        for line in custom.splitlines():
            lines.append(line if line.startswith("%") else f"% {line}")
    header = str(options.get("header", "off")).lower()
    if header == "on":
        lines.append("\t".join(fields))
        lines.append("\t".join("-" * max(len(field), 3) for field in fields))
    elif filetype == "chanedit":
        lines.append(str(nchan))
    return lines


def _format_row(index: int, loc: dict[str, Any], fields: tuple[str, ...]) -> str:
    return "\t".join(_format_value(index, loc, field) for field in fields)


def _format_value(index: int, loc: dict[str, Any], field: str) -> str:
    canonical, multiplier = _field_and_multiplier(field)
    if canonical == "channum":
        return str(index)
    if canonical not in loc:
        return ""
    value = loc[canonical]
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value) * multiplier
        return "0" if abs(numeric) <= 1e-10 else f"{numeric:.5g}"
    return str(value)


def _field_and_multiplier(field: str) -> tuple[str, int]:
    text = str(field)
    if text.lower() == "channum":
        return "channum", 1
    if text.lower() in {"x", "y", "z"}:
        return text.upper(), 1
    if text.lower() in {"-x", "-y", "-z"}:
        return text[-1].upper(), -1
    return text.lower(), 1


__all__ = ["writelocs"]
