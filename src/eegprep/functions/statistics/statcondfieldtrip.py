"""Standalone FieldTrip-style condition statistics boundary."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from operator import index
from typing import Any

import numpy as np

from eegprep.functions.statistics._shared import condition_grid, paired_flag
from eegprep.functions.statistics.fdr import fdr
from eegprep.functions.statistics.statcond import StatcondResult, statcond


@dataclass(frozen=True)
class StatcondFieldtripResult:
    """Statistical result with FieldTrip-style multiple-comparison output."""

    stat: Any
    df: Any
    pvalue: np.ndarray
    mask: np.ndarray
    raw_pvalue: np.ndarray
    surrogate: Any
    method: str
    mcorrect: str
    paired: bool

    def __iter__(self) -> Iterator[Any]:
        yield self.stat
        yield self.df
        yield self.pvalue


def statcondfieldtrip(
    data: Any,
    *,
    paired: str | bool = "auto",
    method: str = "analytic",
    mode: str | None = None,
    naccu: int = 200,
    variance: str = "homogenous",
    mcorrect: str = "none",
    alpha: float = 0.05,
    axis: int = -1,
    rng: np.random.Generator | int | None = None,
    neighbours: Any = None,
) -> StatcondFieldtripResult:
    """Compare conditions using the supported ``statcondfieldtrip`` contract.

    The standalone backend supports paired and unpaired two-condition t-tests
    plus unpaired one-way ANOVA. Condition arrays may have any number of
    feature axes; cases occupy ``axis`` and result arrays preserve the feature
    shape.

    Args:
        data: Sequence of condition arrays.
        paired: Pairing mode. ``"auto"`` pairs conditions only when every case
            count is equal.
        method: ``"analytic"`` or ``"montecarlo"``. EEGLAB aliases
            ``"param"``, ``"parametric"``, ``"perm"``, ``"permutation"``,
            and ``"bootstrap"`` are accepted; bootstrap follows EEGLAB's
            FieldTrip wrapper and selects permutation inference.
        mode: EEGLAB alias for ``method``. A non-empty value takes precedence.
        naccu: Number of permutations for Monte Carlo inference.
        variance: Equal-variance mode for an unpaired t-test. Both spellings of
            ``"homogeneous"`` are accepted.
        mcorrect: Multiple-comparison correction: ``"none"``,
            ``"bonferroni"``, ``"holm"``, ``"fdr"``, or ``"max"``.
            EEGLAB/FieldTrip spellings ``"bonferoni"`` and ``"holms"`` are
            accepted. Max correction requires Monte Carlo inference.
        alpha: Family significance threshold used to form ``mask``.
        axis: Case axis in every condition array.
        rng: Optional NumPy generator or seed for Monte Carlo inference.
        neighbours: Reserved FieldTrip spatial-neighbour input. Non-empty
            values require cluster correction and are not supported.

    Returns:
        A structured result. Iteration yields ``stat``, ``df``, and ``pvalue``
        for compatibility with the MATLAB function's three outputs. Pointwise
        corrections leave ``pvalue`` unadjusted and correct ``mask`` instead;
        max-statistic correction returns family-wise corrected ``pvalue``.

    Raises:
        NotImplementedError: For paired one-way ANOVA, two-way designs,
            cluster correction, or spatial-neighbour statistics.
        ValueError: For invalid methods, corrections, alpha, variance, or
            condition shapes.
    """

    method_name = _normalize_method(mode if mode else method)
    correction_name = _normalize_correction(mcorrect)
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value) or not 0 < alpha_value <= 1:
        raise ValueError("alpha must be greater than 0 and at most 1")
    if _has_values(neighbours):
        raise NotImplementedError("spatial-neighbour statistics require the unsupported cluster backend")
    if correction_name == "cluster":
        raise NotImplementedError("cluster correction requires an explicit adjacency graph and cluster policy")
    if correction_name == "max" and method_name != "montecarlo":
        raise ValueError("max correction requires method='montecarlo'")

    grid = condition_grid(data, axis=axis, min_cases=2)
    if len(grid) != 1:
        raise NotImplementedError("statcondfieldtrip two-way designs are not supported")
    n_conditions = len(grid[0])
    if n_conditions < 2:
        raise ValueError("statcondfieldtrip requires at least two conditions")
    paired_value = paired_flag(grid, paired)
    if n_conditions > 2 and paired_value:
        raise NotImplementedError("statcondfieldtrip paired one-way ANOVA is not supported")

    variance_name = _normalize_variance(variance)
    if n_conditions == 2 and not paired_value and variance_name != "homogenous":
        raise NotImplementedError("FieldTrip parity supports only equal-variance unpaired t-tests")

    try:
        count = index(naccu)
    except TypeError:
        raise ValueError("naccu must be an integer") from None
    if isinstance(naccu, bool):
        raise ValueError("naccu must be an integer")
    if count < 1:
        raise ValueError("naccu must be at least 1")
    statcond_method = "param" if method_name == "analytic" else "perm"
    result = statcond(
        grid[0],
        paired=paired_value,
        method=statcond_method,
        naccu=count,
        variance=variance_name,
        rng=rng,
    )
    if not isinstance(result, StatcondResult):  # pragma: no cover - fixed by call arguments
        raise RuntimeError("statcond returned resampling arrays instead of statistics")

    raw_pvalue = np.asarray(result.pvalue, dtype=float)
    if method_name == "montecarlo":
        raw_pvalue = _montecarlo_pvalues(
            result.stat,
            result.surrogate,
            two_sided=n_conditions == 2,
        )
    if correction_name == "max":
        pvalue = _max_statistic_pvalues(
            result.stat,
            result.surrogate,
            two_sided=n_conditions == 2,
        )
        mask = pvalue <= alpha_value
    else:
        pvalue = raw_pvalue.copy()
        mask = _corrected_mask(raw_pvalue, alpha_value, correction_name)
    return StatcondFieldtripResult(
        stat=result.stat,
        df=result.df,
        pvalue=pvalue,
        mask=mask,
        raw_pvalue=raw_pvalue,
        surrogate=result.surrogate,
        method=method_name,
        mcorrect=correction_name,
        paired=paired_value,
    )


def _normalize_method(method: str) -> str:
    method_name = str(method).lower()
    if method_name in {"analytic", "param", "parametric"}:
        return "analytic"
    if method_name in {"montecarlo", "perm", "permutation", "bootstrap"}:
        return "montecarlo"
    raise ValueError("method must be 'analytic' or 'montecarlo'")


def _normalize_correction(correction: str) -> str:
    correction_name = str(correction).lower()
    aliases = {
        "no": "none",
        "none": "none",
        "bonferoni": "bonferroni",
        "bonferroni": "bonferroni",
        "holm": "holm",
        "holms": "holm",
        "fdr": "fdr",
        "max": "max",
        "cluster": "cluster",
    }
    if correction_name not in aliases:
        raise ValueError("mcorrect must be 'none', 'bonferroni', 'holm', 'fdr', 'max', or 'cluster'")
    return aliases[correction_name]


def _normalize_variance(variance: str) -> str:
    variance_name = str(variance).lower()
    aliases = {
        "homogenous": "homogenous",
        "homogeneous": "homogenous",
        "inhomogenous": "inhomogenous",
        "inhomogeneous": "inhomogenous",
    }
    if variance_name not in aliases:
        raise ValueError("variance must be 'homogenous' or 'inhomogenous'")
    return aliases[variance_name]


def _has_values(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return bool(value)
    return np.asarray(value, dtype=object).size > 0


def _corrected_mask(pvalues: np.ndarray, alpha: float, correction: str) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    if correction == "none":
        return values <= alpha

    flat = values.reshape(-1)
    finite = np.isfinite(flat)
    selected = flat[finite]
    mask = np.zeros(flat.shape, dtype=bool)
    if selected.size == 0:
        return mask.reshape(values.shape)
    if correction == "bonferroni":
        mask[finite] = selected <= alpha / selected.size
        return mask.reshape(values.shape)
    if correction == "fdr":
        return fdr(values, alpha).mask

    order = np.argsort(selected)
    ordered = selected[order]
    count = selected.size
    accepted = np.zeros(count, dtype=bool)
    for rank, pvalue in enumerate(ordered):
        if pvalue > alpha / (count - rank):
            break
        accepted[rank] = True
    selected_mask = np.zeros(count, dtype=bool)
    selected_mask[order] = accepted
    mask[finite] = selected_mask
    return mask.reshape(values.shape)


def _montecarlo_pvalues(statistic: Any, surrogate: Any, *, two_sided: bool) -> np.ndarray:
    if surrogate is None:
        raise ValueError("Monte Carlo inference requires a surrogate statistic distribution")
    observed = np.asarray(statistic, dtype=float)
    distribution = np.asarray(surrogate, dtype=float)
    if distribution.shape[:-1] != observed.shape:
        raise ValueError("surrogate shape must equal statistic shape plus a final permutation axis")
    if two_sided:
        observed = np.abs(observed)
        distribution = np.abs(distribution)
    exceedances = np.sum(distribution >= observed[..., np.newaxis], axis=-1)
    pvalues = (exceedances + 1) / (distribution.shape[-1] + 1)
    return np.where(np.isnan(observed), np.nan, pvalues)


def _max_statistic_pvalues(statistic: Any, surrogate: Any, *, two_sided: bool) -> np.ndarray:
    if surrogate is None:
        raise ValueError("max correction requires a surrogate statistic distribution")
    observed = np.asarray(statistic, dtype=float)
    distribution = np.asarray(surrogate, dtype=float)
    if distribution.shape[:-1] != observed.shape:
        raise ValueError("surrogate shape must equal statistic shape plus a final permutation axis")
    if two_sided:
        observed = np.abs(observed)
        distribution = np.abs(distribution)
    null_maximum = np.max(distribution.reshape(-1, distribution.shape[-1]), axis=0)
    exceedances = np.sum(null_maximum >= observed[..., np.newaxis], axis=-1)
    pvalues = (exceedances + 1) / (distribution.shape[-1] + 1)
    return np.where(np.isnan(observed), np.nan, pvalues)


__all__ = ["StatcondFieldtripResult", "statcondfieldtrip"]
