"""Extract documented variables from a MATLAB help header."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path


logger = logging.getLogger(__name__)
_TITLE = re.compile(
    r"^(usage|authors?|notes?|inputs?|outputs?|examples?|see also)\s*:",
    re.IGNORECASE,
)
_VARIABLE = re.compile(r"^'?([A-Za-z][A-Za-z0-9_]*)'?\s+[-=]\s*(.*)$")


def gethelpvar(filename: str | Path, variables: str | Sequence[str] | None = None) -> tuple[list[str], list[str]]:
    """Return variable descriptions and all variable names from an M-file.

    Args:
        filename: MATLAB source file whose leading help comments are parsed.
        variables: Optional name or ordered names whose descriptions should be
            returned. Unknown names produce an empty description.
    """
    lines = Path(filename).read_text(encoding="utf-8").splitlines()
    help_lines = _leading_help_lines(lines)
    names, descriptions = _parse_variables(help_lines)
    if variables is None:
        return descriptions, names
    requested = [variables] if isinstance(variables, str) else list(variables)
    by_name = dict(zip(names, descriptions))
    selected: list[str] = []
    for name in requested:
        if name not in by_name:
            logger.warning("Variable '%s' not found in %s", name, filename)
        selected.append(by_name.get(name, ""))
    return selected, names


def _leading_help_lines(lines: list[str]) -> list[str]:
    if lines and not lines[0].lstrip().startswith("%"):
        lines = lines[1:]
    result: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped.startswith("%"):
            break
        result.append(stripped[1:].strip())
    return result


def _parse_variables(lines: list[str]) -> tuple[list[str], list[str]]:
    names: list[str] = []
    descriptions: list[str] = []
    current: int | None = None
    for line in lines:
        if _TITLE.match(line):
            current = None
            continue
        match = _VARIABLE.match(line)
        if match:
            name, description = match.groups()
            names.append(name)
            descriptions.append(description.rstrip())
            current = len(names) - 1
            continue
        if current is not None and line:
            descriptions[current] = f"{descriptions[current]}\n{line.rstrip()}"
    return names, descriptions
