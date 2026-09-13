"""EEGLAB-style GUI entrypoint for Python dialog specs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, overload

from .spec import DialogSpec
from .supergui import supergui


@overload
def inputgui(
    spec: DialogSpec,
    initial_values: Mapping[str, Any] | None = None,
    renderer: Any | None = None,
    *,
    mode: Literal["normal"] = "normal",
) -> dict[str, Any] | None: ...


@overload
def inputgui(
    spec: DialogSpec,
    initial_values: Mapping[str, Any] | None = None,
    renderer: Any | None = None,
    *,
    mode: Literal["plot"],
) -> tuple[Any, Any, dict[str, Any]]: ...


def inputgui(
    spec: DialogSpec,
    initial_values: Mapping[str, Any] | None = None,
    renderer: Any | None = None,
    *,
    mode: Literal["normal", "plot"] = "normal",
) -> dict[str, Any] | tuple[Any, Any, dict[str, Any]] | None:
    """Render an EEGLAB-like dialog specification.

    ``mode="normal"`` runs the modal dialog and returns tagged values, or
    ``None`` when the user cancels. ``mode="plot"`` displays the dialog
    without blocking and returns ``(app, dialog, tagged_widgets)`` so callers
    can manage the window. The non-blocking mode is useful for previews and
    custom workflows that need to retain the dialog handle.
    """
    if not isinstance(spec, DialogSpec):
        raise TypeError("inputgui requires a DialogSpec")
    if spec.help_text is not None and not isinstance(spec.help_text, str):
        raise TypeError("inputgui help_text must be a string or None")
    if renderer is None:
        from .qt import QtDialogRenderer

        renderer = QtDialogRenderer()
    mode = mode.lower()
    if mode == "normal":
        return renderer.run(spec, initial_values=initial_values)
    if mode != "plot":
        raise ValueError("inputgui mode must be 'normal' or 'plot'")

    app, dialog, widgets = supergui(spec, initial_values=initial_values, renderer=renderer)
    dialog.show()
    app.processEvents()
    return app, dialog, widgets
