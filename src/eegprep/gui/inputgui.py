"""EEGLAB-style GUI entrypoint for Python dialog specs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .spec import DialogSpec


def inputgui(
    spec: DialogSpec,
    initial_values: Mapping[str, Any] | None = None,
    renderer: Any | None = None,
) -> dict[str, Any] | None:
    """Render a dialog spec and return tagged values, or ``None`` on cancel."""
    if renderer is None:
        from .qt import QtDialogRenderer

        renderer = QtDialogRenderer()
    return renderer.run(spec, initial_values=initial_values)
