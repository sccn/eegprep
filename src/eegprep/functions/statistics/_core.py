"""Compatibility re-exports for statistics helpers.

Implementations live in the same-name public modules. This private facade is
kept so older internal imports from ``statistics._core`` continue to resolve
while the large implementation module is retired.
"""

from eegprep.functions.statistics._shared import TwoWayAnovaResult, TwoWayEffects
from eegprep.functions.statistics.anova1_cell import anova1_cell
from eegprep.functions.statistics.anova1rm_cell import anova1rm_cell
from eegprep.functions.statistics.anova2_cell import anova2_cell
from eegprep.functions.statistics.anova2rm_cell import anova2rm_cell
from eegprep.functions.statistics.concatdata import ConcatenatedData, concatdata
from eegprep.functions.statistics.corrcoef_cell import corrcoef_cell
from eegprep.functions.statistics.fdr import FDRResult, fdr
from eegprep.functions.statistics.stat_surrogate_ci import stat_surrogate_ci
from eegprep.functions.statistics.stat_surrogate_pvals import stat_surrogate_pvals
from eegprep.functions.statistics.statcond import StatcondResult, statcond
from eegprep.functions.statistics.surrogdistrib import SurrogateDistribution, surrogdistrib
from eegprep.functions.statistics.teststat import teststat
from eegprep.functions.statistics.ttest2_cell import ttest2_cell
from eegprep.functions.statistics.ttest_cell import ttest_cell

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
