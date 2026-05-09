"""Renderer-independent GUI specifications.

These dataclasses intentionally mirror EEGLAB's inputgui/supergui model at the
spec layer while keeping Python callbacks explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CallbackSpec:
    """Declarative callback metadata for a GUI control."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    matlab_callback: str | None = None


@dataclass(frozen=True)
class ControlSpec:
    """Single EEGLAB-like GUI control."""

    style: str
    string: str = ""
    tag: str | None = None
    value: Any = None
    callback: CallbackSpec | None = None
    tooltip: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class DialogSpec:
    """Complete EEGLAB-like dialog specification."""

    title: str
    controls: tuple[ControlSpec, ...]
    geometry: tuple[Any, ...]
    function_name: str
    eeglab_source: str
    size: tuple[int, int] | None = None
    help_text: str | None = None
    known_differences: tuple[str, ...] = ()
    content_margins: tuple[int, int, int, int] = (42, 17, 42, 13)


def controls_by_tag(spec: DialogSpec) -> dict[str, ControlSpec]:
    """Return tagged controls keyed by EEGLAB-style tag."""
    return {control.tag: control for control in spec.controls if control.tag}
