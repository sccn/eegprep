"""Statistics package smoke-test helper."""

from __future__ import annotations

import numpy as np

from eegprep.functions.statistics._shared import TwoWayEffects
from eegprep.functions.statistics.anova1_cell import anova1_cell
from eegprep.functions.statistics.statcond import StatcondResult, statcond
from eegprep.functions.statistics.ttest_cell import ttest_cell


def teststat(seed: int = 0) -> dict[str, float]:
    """Run deterministic smoke checks for the EEGPrep statistics package."""

    rng = np.random.default_rng(seed)
    first = rng.normal(size=(3, 12))
    second = first + rng.normal(loc=0.25, scale=0.2, size=(3, 12))
    paired_result = statcond([first, second], paired="on", method="param")
    if not isinstance(paired_result, StatcondResult):
        raise AssertionError("paired statcond unexpectedly returned surrogate grids")
    t_values, df = ttest_cell(first, second)
    np.testing.assert_allclose(paired_result.stat, t_values)
    if paired_result.df != df:
        raise AssertionError("paired t-test degrees of freedom changed")

    groups = [rng.normal(loc=offset, size=(3, 10)) for offset in (0.0, 0.2, 0.5)]
    one_way = statcond(groups, paired="off", method="param")
    if not isinstance(one_way, StatcondResult):
        raise AssertionError("one-way statcond unexpectedly returned surrogate grids")
    direct_one_way, one_way_df = anova1_cell(groups)
    np.testing.assert_allclose(one_way.stat, direct_one_way)
    if one_way.df != one_way_df:
        raise AssertionError("one-way ANOVA degrees of freedom changed")

    grid = (
        (rng.normal(size=(2, 9)), rng.normal(loc=0.1, size=(2, 9))),
        (rng.normal(loc=0.2, size=(2, 9)), rng.normal(loc=0.4, size=(2, 9))),
    )
    two_way = statcond(grid, paired="on", method="param")
    if not isinstance(two_way, StatcondResult):
        raise AssertionError("two-way statcond unexpectedly returned surrogate grids")
    if not isinstance(two_way.stat, TwoWayEffects):
        raise AssertionError("two-way statcond did not return factor effects")

    return {
        "paired_t_mean": float(np.mean(paired_result.stat)),
        "one_way_f_mean": float(np.mean(one_way.stat)),
        "two_way_interaction_mean": float(np.mean(two_way.stat.interaction)),
    }


__all__ = ["teststat"]
