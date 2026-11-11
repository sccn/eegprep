"""EEGLAB-style signal processing function modules."""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Importing this package must not import Matplotlib. Plotting helpers load on
# first use so a numerical submodule such as ``spectra`` stays headless.
_LAZY_EXPORTS = {
    "cbar": ("eegprep.functions.sigprocfunc.cbar", "cbar"),
    "celltomat": ("eegprep.functions.sigprocfunc.celltomat", "celltomat"),
    "copyaxis": ("eegprep.functions.sigprocfunc.copyaxis", "copyaxis"),
    "eyelike": ("eegprep.functions.sigprocfunc.eyelike", "eyelike"),
    "fastif": ("eegprep.functions.sigprocfunc.fastif", "fastif"),
    "forcelocs": ("eegprep.functions.sigprocfunc.forcelocs", "forcelocs"),
    "headplot": ("eegprep.functions.sigprocfunc.headplot", "headplot"),
    "matsel": ("eegprep.functions.sigprocfunc.matsel", "matsel"),
    "mattocell": ("eegprep.functions.sigprocfunc.mattocell", "mattocell"),
    "nan_mean": ("eegprep.functions.sigprocfunc.nan_mean", "nan_mean"),
    "openbdf": ("eegprep.functions.sigprocfunc.openbdf", "openbdf"),
    "plotcurve": ("eegprep.functions.sigprocfunc.plotcurve", "plotcurve"),
    "quantile": ("eegprep.functions.sigprocfunc.quantile", "quantile"),
    "readbdf": ("eegprep.functions.sigprocfunc.readbdf", "readbdf"),
    "sbplot": ("eegprep.functions.sigprocfunc.sbplot", "sbplot"),
    "shuffle": ("eegprep.functions.sigprocfunc.shuffle", "shuffle"),
    "slider": ("eegprep.functions.sigprocfunc.slider", "slider"),
    "writegdf": ("eegprep.functions.sigprocfunc.writegdf", "writegdf"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(_LAZY_EXPORTS) | set(globals()))
