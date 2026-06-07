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

from eegprep.functions.statistics._core import (
    ConcatenatedData,
    FDRResult,
    StatcondResult,
    SurrogateDistribution,
    TwoWayAnovaResult,
    TwoWayEffects,
    anova1_cell,
    anova1rm_cell,
    anova2_cell,
    anova2rm_cell,
    concatdata,
    corrcoef_cell,
    fdr,
    stat_surrogate_ci,
    stat_surrogate_pvals,
    statcond,
    surrogdistrib,
    teststat,
    ttest2_cell,
    ttest_cell,
)

del import_module, _module_name, _THIN_MODULES

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
