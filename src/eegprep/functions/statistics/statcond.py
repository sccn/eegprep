"""Condition-level statistical comparison helper."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from eegprep.functions.statistics._shared import (
    TwoWayEffects,
    condition_grid,
    effect_map,
    normalize_method,
    paired_flag,
)
from eegprep.functions.statistics.anova1_cell import anova1_cell
from eegprep.functions.statistics.anova1rm_cell import anova1rm_cell
from eegprep.functions.statistics.anova2_cell import anova2_cell
from eegprep.functions.statistics.anova2rm_cell import anova2rm_cell
from eegprep.functions.statistics.stat_surrogate_ci import stat_surrogate_ci
from eegprep.functions.statistics.stat_surrogate_pvals import stat_surrogate_pvals
from eegprep.functions.statistics.surrogdistrib import SurrogateDistribution, surrogdistrib
from eegprep.functions.statistics.ttest2_cell import ttest2_cell
from eegprep.functions.statistics.ttest_cell import ttest_cell


@dataclass(frozen=True)
class StatcondResult:
    """Result returned by :func:`statcond`."""

    stat: Any
    df: Any
    pvalue: Any
    surrogate: Any
    method: str
    paired: bool
    ci: Any = None
    mask: Any = None

    def __iter__(self) -> Iterator[Any]:
        yield self.stat
        yield self.df
        yield self.pvalue
        yield self.surrogate


def statcond(
    data: Any,
    *,
    paired: str | bool = "auto",
    method: str = "param",
    mode: str | None = None,
    naccu: int = 200,
    variance: str = "homogenous",
    forceanova: bool = False,
    tail: str = "both",
    axis: int = -1,
    rng: np.random.Generator | int | None = None,
    alpha: float | None = None,
    surrog: Any = None,
    stats: Any = None,
    return_resampling_array: bool = False,
) -> StatcondResult | SurrogateDistribution:
    """Compare condition arrays using EEGLAB-style t-tests or ANOVAs.

    Args:
        data: One- or two-dimensional sequence of condition arrays. The case
            dimension is the last axis by default.
        paired: ``"auto"``, ``"on"``/``True``, or ``"off"``/``False``.
        method: ``"param"``, ``"perm"``, or ``"bootstrap"``.
        mode: Legacy alias for ``method``.
        naccu: Number of surrogate samples for nonparametric methods.
        variance: ``"homogenous"`` or ``"inhomogenous"`` for unpaired t-tests.
        forceanova: Use one-way ANOVA instead of a two-condition t-test.
        tail: Empirical-tail mode for supplied or computed surrogates.
        axis: Axis in each condition array that stores cases.
        rng: Optional NumPy generator or seed for deterministic resampling.
        alpha: Optional threshold for confidence intervals and masks; requires
            a nonparametric method or supplied surrogate statistics.
        surrog: Precomputed surrogate statistic array.
        stats: Observed statistic to pair with ``surrog``.
        return_resampling_array: Return surrogate condition grids instead of
            computing statistics.
    """
    if isinstance(data, dict) and "datasetinfo" in data:
        # Group-level STUDY dictionary data structure handling
        from eegprep.functions.studyfunc.std_readdata import std_readdata

        design = data.get("currentdesign", 1)
        _study, study_data, _x, _y = std_readdata(data, [], datatype="erp", design=design)
        data = study_data

    method_name = normalize_method(mode or method)
    grid = condition_grid(data, axis=axis, min_cases=2)
    paired_flag_value = paired_flag(grid, paired)
    if return_resampling_array:
        if method_name == "param":
            raise ValueError("return_resampling_array requires 'perm' or 'bootstrap'")
        return surrogdistrib(
            grid,
            method=method_name,
            pairing="on" if paired_flag_value else "off",
            naccu=naccu,
            rng=rng,
        )

    if surrog is not None:
        if stats is None:
            raise ValueError("stats must be supplied when surrog is supplied")
        observed_stat = stats
        observed_df = None
        surrogate_stat = surrog
        pvalue = _surrogate_pvalues(surrogate_stat, observed_stat, tail)
        ci = None
        mask = None
        if alpha is not None:
            ci = _surrogate_ci(surrogate_stat, alpha, _ci_tail(tail))
            mask = effect_map(pvalue, lambda value: value < alpha)
        return StatcondResult(
            observed_stat, observed_df, pvalue, surrogate_stat, method_name, paired_flag_value, ci=ci, mask=mask
        )

    observed_stat, observed_df, statistic_kind = _compute_statistic(
        grid,
        paired=paired_flag_value,
        variance=variance,
        forceanova=forceanova,
    )
    surrogate_stat = None
    if method_name == "param":
        pvalue = _parametric_pvalues(observed_stat, observed_df, statistic_kind)
    else:
        surrogate_stat = _compute_surrogate_statistics(
            grid,
            paired=paired_flag_value,
            method=method_name,
            naccu=naccu,
            variance=variance,
            forceanova=forceanova,
            rng=rng,
        )
        empirical_tail = "one" if statistic_kind.startswith("f") else tail
        pvalue = _surrogate_pvalues(surrogate_stat, observed_stat, empirical_tail)

    ci = None
    mask = None
    if alpha is not None:
        if surrogate_stat is None:
            raise ValueError("alpha confidence intervals require a nonparametric method or supplied surrogates")
        empirical_tail = "one" if statistic_kind.startswith("f") else tail
        ci = _surrogate_ci(surrogate_stat, alpha, _ci_tail(empirical_tail))
        mask = effect_map(pvalue, lambda value: value < alpha)

    return StatcondResult(
        observed_stat, observed_df, pvalue, surrogate_stat, method_name, paired_flag_value, ci=ci, mask=mask
    )


def _compute_statistic(
    grid: tuple[tuple[np.ndarray, ...], ...],
    *,
    paired: bool,
    variance: str,
    forceanova: bool,
) -> tuple[Any, Any, str]:
    rows = len(grid)
    columns = len(grid[0])
    if rows == 1:
        if columns == 2 and not forceanova:
            if paired:
                stat, df = ttest_cell(grid[0][0], grid[0][1])
            else:
                stat, df = ttest2_cell(grid[0][0], grid[0][1], variance=variance)
            return stat, df, "t"
        if paired:
            stat, df = anova1rm_cell(grid[0])
        else:
            stat, df = anova1_cell(grid[0])
        return stat, df, "f_one_way"

    anova = anova2rm_cell(grid) if paired else anova2_cell(grid)
    return anova.as_effects(), anova.df_effects(), "f_two_way"


def _parametric_pvalues(stat: Any, df: Any, statistic_kind: str) -> Any:
    if statistic_kind == "t":
        return 2 * scipy_stats.t.sf(np.abs(stat), df)
    if isinstance(stat, TwoWayEffects):
        return TwoWayEffects(
            scipy_stats.f.sf(stat.rows, df.rows[0], df.rows[1]),
            scipy_stats.f.sf(stat.columns, df.columns[0], df.columns[1]),
            scipy_stats.f.sf(stat.interaction, df.interaction[0], df.interaction[1]),
        )
    return scipy_stats.f.sf(stat, df[0], df[1])


def _compute_surrogate_statistics(
    grid: tuple[tuple[np.ndarray, ...], ...],
    *,
    paired: bool,
    method: str,
    naccu: int,
    variance: str,
    forceanova: bool,
    rng: np.random.Generator | int | None,
) -> Any:
    distribution = surrogdistrib(
        grid,
        method=method,
        pairing="on" if paired else "off",
        naccu=naccu,
        rng=rng,
    )
    stats = []
    for sample in distribution:
        sample_stat, _sample_df, _kind = _compute_statistic(
            sample,
            paired=paired,
            variance=variance,
            forceanova=forceanova,
        )
        stats.append(sample_stat)
    return _stack_effects(stats)


def _stack_effects(values: Sequence[Any]) -> Any:
    first = values[0]
    if isinstance(first, TwoWayEffects):
        return TwoWayEffects(
            np.stack([value.rows for value in values], axis=-1),
            np.stack([value.columns for value in values], axis=-1),
            np.stack([value.interaction for value in values], axis=-1),
        )
    return np.stack(values, axis=-1)


def _surrogate_pvalues(surrogate: Any, observed: Any, tail: str) -> Any:
    if isinstance(surrogate, TwoWayEffects):
        return TwoWayEffects(
            stat_surrogate_pvals(surrogate.rows, observed.rows, tail),
            stat_surrogate_pvals(surrogate.columns, observed.columns, tail),
            stat_surrogate_pvals(surrogate.interaction, observed.interaction, tail),
        )
    return stat_surrogate_pvals(surrogate, observed, tail)


def _surrogate_ci(surrogate: Any, alpha: float, tail: str) -> Any:
    if isinstance(surrogate, TwoWayEffects):
        return TwoWayEffects(
            stat_surrogate_ci(surrogate.rows, alpha, tail),
            stat_surrogate_ci(surrogate.columns, alpha, tail),
            stat_surrogate_ci(surrogate.interaction, alpha, tail),
        )
    return stat_surrogate_ci(surrogate, alpha, tail)


def _ci_tail(tail: str) -> str:
    tail_name = tail.lower()
    if tail_name == "right":
        return "upper"
    if tail_name == "left":
        return "lower"
    return tail_name


__all__ = ["StatcondResult", "TwoWayEffects", "statcond"]
