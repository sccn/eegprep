"""firfilt plugin ports."""

from __future__ import annotations

from typing import Any

_LAZY_EXPORTS = {
    "firws": ("eegprep.plugins.firfilt.firws", "firws"),
    "firwsord": ("eegprep.plugins.firfilt.firwsord", "firwsord"),
}

__all__ = ["firws", "firwsord"]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
