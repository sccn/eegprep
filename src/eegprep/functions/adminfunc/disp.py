"""Print a value the way a history command displays it."""

from __future__ import annotations

from typing import Any


def disp(value: Any) -> None:
    """Print ``value`` without a variable name.

    History replay uses this for commands such as ``disp(3)``. The text is
    Python's display of the value.
    """
    print(value)
