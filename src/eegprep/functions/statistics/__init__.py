"""EEGLAB-style statistics helper functions."""

from importlib import import_module

from eegprep.functions.statistics import _core as _statistics_core

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

ConcatenatedData = _statistics_core.ConcatenatedData
FDRResult = _statistics_core.FDRResult
StatcondResult = _statistics_core.StatcondResult
SurrogateDistribution = _statistics_core.SurrogateDistribution
TwoWayAnovaResult = _statistics_core.TwoWayAnovaResult
TwoWayEffects = _statistics_core.TwoWayEffects
anova1_cell = _statistics_core.anova1_cell
anova1rm_cell = _statistics_core.anova1rm_cell
anova2_cell = _statistics_core.anova2_cell
anova2rm_cell = _statistics_core.anova2rm_cell
concatdata = _statistics_core.concatdata
corrcoef_cell = _statistics_core.corrcoef_cell
fdr = _statistics_core.fdr
stat_surrogate_ci = _statistics_core.stat_surrogate_ci
stat_surrogate_pvals = _statistics_core.stat_surrogate_pvals
statcond = _statistics_core.statcond
surrogdistrib = _statistics_core.surrogdistrib
teststat = _statistics_core.teststat
ttest2_cell = _statistics_core.ttest2_cell
ttest_cell = _statistics_core.ttest_cell

del import_module, _module_name, _THIN_MODULES, _statistics_core

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
