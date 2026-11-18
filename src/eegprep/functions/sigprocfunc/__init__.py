"""EEGLAB-style signal processing function modules."""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType
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


class _SigProcFuncModule(ModuleType):
    def __getattribute__(self, name: str) -> Any:
        exports = ModuleType.__getattribute__(self, "_LAZY_EXPORTS")
        if name in exports:
            module_name, attr_name = exports[name]
            return getattr(import_module(module_name), attr_name)
        return ModuleType.__getattribute__(self, name)

    def __dir__(self) -> list[str]:
        exports = ModuleType.__getattribute__(self, "_LAZY_EXPORTS")
        return sorted(set(exports) | set(ModuleType.__dir__(self)))


sys.modules[__name__].__class__ = _SigProcFuncModule
