"""Renderer-level construction for EEGLAB-like dialog specifications."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .spec import DialogSpec


def supergui(
    spec: DialogSpec,
    initial_values: Mapping[str, Any] | None = None,
    renderer: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build a dialog without entering its modal event loop.

    This is EEGPrep's declarative counterpart to EEGLAB's low-level
    ``supergui`` builder. The returned tuple contains the application, dialog,
    and tagged widgets. Use :func:`inputgui` for the usual modal workflow.
    """
    if not isinstance(spec, DialogSpec):
        raise TypeError("supergui requires a DialogSpec")
    if renderer is None:
        from .qt import QtDialogRenderer

        renderer = QtDialogRenderer()
    build_dialog = getattr(renderer, "build_dialog", None)
    if build_dialog is None:
        raise TypeError("supergui renderer must provide build_dialog()")
    return build_dialog(spec, initial_values=initial_values)
