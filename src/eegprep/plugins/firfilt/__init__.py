"""firfilt plugin ports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "firws": ("eegprep.plugins.firfilt.firws", "firws"),
    "firwsord": ("eegprep.plugins.firfilt.firwsord", "firwsord"),
    "pop_eegfiltnew": ("eegprep.plugins.firfilt.pop_eegfiltnew", "pop_eegfiltnew"),
    "pop_firma": ("eegprep.plugins.firfilt.pop_firma", "pop_firma"),
    "pop_firpm": ("eegprep.plugins.firfilt.pop_firpm", "pop_firpm"),
    "pop_firws": ("eegprep.plugins.firfilt.pop_firws", "pop_firws"),
}

__all__ = ["firws", "firwsord", "pop_eegfiltnew", "pop_firma", "pop_firpm", "pop_firws"]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
