"""EEGLAB-style statistics helper functions."""

from importlib import import_module

# Import same-name thin modules before binding package callables. Without this,
# a later ``import eegprep.functions.statistics.fdr`` can replace
# ``statistics.fdr`` with the submodule object.
_THIN_MODULES = (
    "anova1_cell",
    "anova1rm_cell",
    "anova2_cell",
    "anova2rm_cell",
    "concatdata",
    "corrcoef_cell",
    "fdr",
    "stat_surrogate_ci",
    "stat_surrogate_pvals",
    "statcond",
    "surrogdistrib",
    "teststat",
    "ttest2_cell",
    "ttest_cell",
)
for _module_name in _THIN_MODULES:
    import_module(f"{__name__}.{_module_name}")

_MODULES = {_module_name: import_module(f"{__name__}.{_module_name}") for _module_name in _THIN_MODULES}

ConcatenatedData = _MODULES["concatdata"].ConcatenatedData
FDRResult = _MODULES["fdr"].FDRResult
StatcondResult = _MODULES["statcond"].StatcondResult
SurrogateDistribution = _MODULES["surrogdistrib"].SurrogateDistribution
TwoWayAnovaResult = _MODULES["anova2_cell"].TwoWayAnovaResult
TwoWayEffects = _MODULES["statcond"].TwoWayEffects
anova1_cell = _MODULES["anova1_cell"].anova1_cell
anova1rm_cell = _MODULES["anova1rm_cell"].anova1rm_cell
anova2_cell = _MODULES["anova2_cell"].anova2_cell
anova2rm_cell = _MODULES["anova2rm_cell"].anova2rm_cell
concatdata = _MODULES["concatdata"].concatdata
corrcoef_cell = _MODULES["corrcoef_cell"].corrcoef_cell
fdr = _MODULES["fdr"].fdr
stat_surrogate_ci = _MODULES["stat_surrogate_ci"].stat_surrogate_ci
stat_surrogate_pvals = _MODULES["stat_surrogate_pvals"].stat_surrogate_pvals
statcond = _MODULES["statcond"].statcond
surrogdistrib = _MODULES["surrogdistrib"].surrogdistrib
teststat = _MODULES["teststat"].teststat
ttest2_cell = _MODULES["ttest2_cell"].ttest2_cell
ttest_cell = _MODULES["ttest_cell"].ttest_cell

del import_module, _module_name, _MODULES, _THIN_MODULES

__all__ = [
    "ConcatenatedData",
    "FDRResult",
    "StatcondResult",
    "SurrogateDistribution",
    "TwoWayAnovaResult",
    "TwoWayEffects",
    "anova1_cell",
    "anova1rm_cell",
    "anova2_cell",
    "anova2rm_cell",
    "concatdata",
    "corrcoef_cell",
    "fdr",
    "stat_surrogate_ci",
    "stat_surrogate_pvals",
    "statcond",
    "surrogdistrib",
    "teststat",
    "ttest2_cell",
    "ttest_cell",
]
