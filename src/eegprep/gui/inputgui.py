"""EEGLAB-style GUI entrypoint for Python dialog specs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .spec import DialogSpec


def inputgui(
    spec: DialogSpec,
    *,
    initial_values: Mapping[str, Any] | None = None,
    renderer: Any | None = None,
) -> dict[str, Any] | None:
    """Render a dialog spec and return tagged values, or ``None`` on cancel.

    Parameters
    ----------
    spec
        Renderer-independent dialog specification.
    initial_values
        Optional tag-to-value overrides used by callers/tests.
    renderer
        Optional renderer object with a ``run(spec, initial_values=...)`` method.
        If omitted, the optional Qt renderer is used.
    """

    if renderer is None:
        from .qt import QtDialogRenderer

        renderer = QtDialogRenderer()
    return renderer.run(spec, initial_values=initial_values)
