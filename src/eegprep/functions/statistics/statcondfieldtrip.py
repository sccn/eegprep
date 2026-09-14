"""Standalone FieldTrip-style condition statistics boundary."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations, product
from math import comb
from operator import index
from typing import Any

import numpy as np
from scipy import sparse
from scipy import stats as scipy_stats

from eegprep.functions.statistics._shared import condition_grid, paired_flag
from eegprep.functions.statistics.fdr import fdr
from eegprep.functions.statistics.statcond import StatcondResult, statcond


MAX_EXACT_PERMUTATIONS = 100_000


@dataclass(frozen=True)
class StatcondFieldtripCluster:
    """One observed cluster from cluster-based permutation inference."""

    label: int
    sign: int
    indices: tuple[int, ...]
    mass: float
    pvalue: float


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
    cluster_labels: np.ndarray | None = None
    clusters: tuple[StatcondFieldtripCluster, ...] = ()
    cluster_null: np.ndarray | None = None
    cluster_critical_value: float | None = None
    exact: bool = False

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
    naccu: int | str = 200,
    variance: str = "homogenous",
    mcorrect: str = "none",
    alpha: float = 0.05,
    axis: int = -1,
    rng: np.random.Generator | int | None = None,
    neighbours: Any = None,
    clusteralpha: float = 0.05,
    clusterstatistic: str = "maxsum",
    clusterthreshold: str = "parametric",
    clustercritval: float | None = None,
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
        naccu: Number of permutations for Monte Carlo inference, or ``"all"``
            to enumerate small permutation spaces exactly.
        variance: Equal-variance mode for an unpaired t-test. Both spellings of
            ``"homogeneous"`` are accepted.
        mcorrect: Multiple-comparison correction: ``"none"``,
            ``"bonferroni"``, ``"holm"``, ``"fdr"``, ``"max"``, or
            ``"cluster"``.
            EEGLAB/FieldTrip spellings ``"bonferoni"`` and ``"holms"`` are
            accepted. Max correction requires Monte Carlo inference.
        alpha: Family significance threshold used to form ``mask``.
        axis: Case axis in every condition array.
        rng: Optional NumPy generator or seed for Monte Carlo inference.
        neighbours: Explicit square, symmetric adjacency matrix over features
            flattened in NumPy C order. Dense and SciPy sparse matrices are
            supported. Required for cluster correction and otherwise rejected.
        clusteralpha: Per-feature cluster-forming probability. Two-condition
            tests split it equally between the two tails.
        clusterstatistic: Cluster statistic. Only FieldTrip's default
            ``"maxsum"`` is supported.
        clusterthreshold: Cluster-forming threshold policy. Only
            ``"parametric"`` is supported.
        clustercritval: Optional positive statistic cutoff overriding the
            value derived from ``clusteralpha`` and the reference distribution.

    Returns:
        A structured result. Iteration yields ``stat``, ``df``, and ``pvalue``
        for compatibility with the MATLAB function's three outputs. Pointwise
        corrections leave ``pvalue`` unadjusted and correct ``mask`` instead;
        max-statistic and cluster correction return family-wise corrected
        ``pvalue``. Cluster results also expose the observed labels, cluster
        records, critical statistic, and maximum-cluster null distribution.

    Raises:
        NotImplementedError: For paired one-way ANOVA, two-way designs, or
            unsupported cluster-statistic and threshold policies.
        ValueError: For invalid methods, corrections, alpha, variance, or
            condition shapes.
    """

    method_name = _normalize_method(mode if mode else method)
    correction_name = _normalize_correction(mcorrect)
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value) or not 0 < alpha_value <= 1:
        raise ValueError("alpha must be greater than 0 and at most 1")
    if correction_name != "cluster" and _has_values(neighbours):
        raise ValueError("neighbours requires mcorrect='cluster'")
    if correction_name == "max" and method_name != "montecarlo":
        raise ValueError("max correction requires method='montecarlo'")
    if correction_name == "cluster" and method_name != "montecarlo":
        raise ValueError("cluster correction requires method='montecarlo'")

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

    count = _randomization_count(naccu)
    exact = count is None
    if exact and method_name != "montecarlo":
        raise ValueError("naccu='all' requires method='montecarlo'")

    result = statcond(
        grid[0],
        paired=paired_value,
        method="param" if exact or method_name == "analytic" else "perm",
        naccu=1 if exact else count,
        variance=variance_name,
        rng=rng,
    )
    if not isinstance(result, StatcondResult):  # pragma: no cover - fixed by call arguments
        raise RuntimeError("statcond returned resampling arrays instead of statistics")

    surrogate = result.surrogate
    if exact:
        surrogate = _exact_surrogate_statistics(grid[0], paired=paired_value, variance=variance_name)

    raw_pvalue = np.asarray(result.pvalue, dtype=float)
    if method_name == "montecarlo":
        raw_pvalue = _montecarlo_pvalues(
            result.stat,
            surrogate,
            two_sided=n_conditions == 2,
            plus_one=not exact,
        )
    cluster_labels = None
    clusters: tuple[StatcondFieldtripCluster, ...] = ()
    cluster_null = None
    cluster_critical_value = None
    if correction_name == "max":
        pvalue = _max_statistic_pvalues(
            result.stat,
            surrogate,
            two_sided=n_conditions == 2,
            plus_one=not exact,
        )
        mask = pvalue <= alpha_value
    elif correction_name == "cluster":
        adjacency = _validate_adjacency(neighbours, np.asarray(result.stat).size)
        cluster_critical_value = _cluster_critical_value(
            result.df,
            two_sided=n_conditions == 2,
            clusteralpha=clusteralpha,
            clusterthreshold=clusterthreshold,
            clustercritval=clustercritval,
            clusterstatistic=clusterstatistic,
        )
        pvalue, mask, cluster_labels, clusters, cluster_null = _cluster_correction(
            result.stat,
            surrogate,
            adjacency,
            critical_value=cluster_critical_value,
            alpha=alpha_value,
            two_sided=n_conditions == 2,
            plus_one=not exact,
        )
    else:
        pvalue = raw_pvalue.copy()
        mask = _corrected_mask(raw_pvalue, alpha_value, correction_name)
    return StatcondFieldtripResult(
        stat=result.stat,
        df=result.df,
        pvalue=pvalue,
        mask=mask,
        raw_pvalue=raw_pvalue,
        surrogate=surrogate,
        method=method_name,
        mcorrect=correction_name,
        paired=paired_value,
        cluster_labels=cluster_labels,
        clusters=clusters,
        cluster_null=cluster_null,
        cluster_critical_value=cluster_critical_value,
        exact=exact,
    )


def _randomization_count(naccu: int | str) -> int | None:
    if isinstance(naccu, str):
        if naccu.lower() == "all":
            return None
        raise ValueError("naccu must be an integer or 'all'")
    try:
        count = index(naccu)
    except TypeError:
        raise ValueError("naccu must be an integer or 'all'") from None
    if isinstance(naccu, bool):
        raise ValueError("naccu must be an integer or 'all'")
    if count < 1:
        raise ValueError("naccu must be at least 1")
    return count


def _exact_surrogate_statistics(
    conditions: tuple[np.ndarray, ...],
    *,
    paired: bool,
    variance: str,
) -> np.ndarray:
    total = _exact_permutation_count(conditions, paired=paired)
    if total > MAX_EXACT_PERMUTATIONS:
        raise ValueError(
            f"naccu='all' would require {total} permutations; the standalone limit is {MAX_EXACT_PERMUTATIONS}"
        )
    statistics = []
    for sample in _exact_condition_permutations(conditions, paired=paired):
        sample_result = statcond(sample, paired=paired, method="param", variance=variance)
        if not isinstance(sample_result, StatcondResult):  # pragma: no cover - fixed by call arguments
            raise RuntimeError("statcond returned resampling arrays instead of statistics")
        statistics.append(np.asarray(sample_result.stat, dtype=float))
    return np.stack(statistics, axis=-1)


def _exact_permutation_count(conditions: tuple[np.ndarray, ...], *, paired: bool) -> int:
    if paired:
        return 2 ** conditions[0].shape[-1]
    remaining = sum(condition.shape[-1] for condition in conditions)
    count = 1
    for condition in conditions[:-1]:
        condition_count = condition.shape[-1]
        count *= comb(remaining, condition_count)
        remaining -= condition_count
    return count


def _exact_condition_permutations(
    conditions: tuple[np.ndarray, ...],
    *,
    paired: bool,
) -> Iterator[tuple[np.ndarray, ...]]:
    if paired:
        first, second = conditions
        reshape = (1,) * (first.ndim - 1) + (first.shape[-1],)
        for swapped in product((False, True), repeat=first.shape[-1]):
            mask = np.asarray(swapped, dtype=bool).reshape(reshape)
            yield (np.where(mask, second, first), np.where(mask, first, second))
        return

    pooled = np.concatenate(conditions, axis=-1)
    counts = tuple(condition.shape[-1] for condition in conditions)
    for groups in _index_partitions(tuple(range(pooled.shape[-1])), counts):
        yield tuple(np.take(pooled, group, axis=-1) for group in groups)


def _index_partitions(indices: tuple[int, ...], counts: tuple[int, ...]) -> Iterator[tuple[tuple[int, ...], ...]]:
    if len(counts) == 1:
        yield (indices,)
        return
    for first in combinations(indices, counts[0]):
        selected = set(first)
        remaining = tuple(value for value in indices if value not in selected)
        for rest in _index_partitions(remaining, counts[1:]):
            yield (first, *rest)


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
    if sparse.issparse(value):
        return value.shape != (0, 0)
    if isinstance(value, dict):
        return bool(value)
    return np.asarray(value, dtype=object).size > 0


def _validate_adjacency(neighbours: Any, feature_count: int) -> tuple[np.ndarray, ...]:
    if neighbours is None:
        raise ValueError("cluster correction requires an explicit neighbours adjacency matrix")
    if sparse.issparse(neighbours):
        raw_matrix = sparse.csr_matrix(neighbours)
        if not (np.issubdtype(raw_matrix.dtype, np.number) or np.issubdtype(raw_matrix.dtype, np.bool_)):
            raise TypeError("neighbours must be a numeric or boolean adjacency matrix")
        if np.any(~np.isfinite(raw_matrix.data)):
            raise ValueError("neighbours must contain only finite values")
        if np.iscomplexobj(raw_matrix.data):
            raise TypeError("neighbours must be a real-valued adjacency matrix")
        matrix = raw_matrix.astype(bool)
        if matrix.shape != (feature_count, feature_count):
            raise ValueError(f"neighbours must have shape ({feature_count}, {feature_count}), got {matrix.shape}")
        matrix.setdiag(False)
        matrix.eliminate_zeros()
        if (matrix != matrix.T).nnz:
            raise ValueError("neighbours must be symmetric")
        matrix.sort_indices()
        return tuple(matrix.indices[matrix.indptr[row] : matrix.indptr[row + 1]] for row in range(feature_count))

    matrix = np.asarray(neighbours)
    if matrix.ndim != 2 or matrix.shape != (feature_count, feature_count):
        raise ValueError(f"neighbours must have shape ({feature_count}, {feature_count}), got {matrix.shape}")
    if not (np.issubdtype(matrix.dtype, np.number) or np.issubdtype(matrix.dtype, np.bool_)):
        raise TypeError("neighbours must be a numeric or boolean adjacency matrix")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("neighbours must contain only finite values")
    if np.iscomplexobj(matrix):
        raise TypeError("neighbours must be a real-valued adjacency matrix")
    connected = matrix != 0
    if not np.array_equal(connected, connected.T):
        raise ValueError("neighbours must be symmetric")
    np.fill_diagonal(connected, False)
    return tuple(np.flatnonzero(connected[row]) for row in range(feature_count))


def _cluster_critical_value(
    df: Any,
    *,
    two_sided: bool,
    clusteralpha: float,
    clusterthreshold: str,
    clustercritval: float | None,
    clusterstatistic: str,
) -> float:
    if str(clusterstatistic).lower() != "maxsum":
        raise NotImplementedError("clusterstatistic supports only 'maxsum'")
    if str(clusterthreshold).lower() != "parametric":
        raise NotImplementedError("clusterthreshold supports only 'parametric'")
    if clustercritval is not None:
        critical = float(clustercritval)
        if not np.isfinite(critical) or critical <= 0:
            raise ValueError("clustercritval must be a finite positive statistic")
        return critical

    cluster_alpha = float(clusteralpha)
    if not np.isfinite(cluster_alpha) or not 0 < cluster_alpha < 1:
        raise ValueError("clusteralpha must be between 0 and 1")
    if two_sided:
        critical = scipy_stats.t.isf(cluster_alpha / 2, df)
    else:
        critical = scipy_stats.f.isf(cluster_alpha, df[0], df[1])
    return float(critical)


def _cluster_correction(
    statistic: Any,
    surrogate: Any,
    adjacency: tuple[np.ndarray, ...],
    *,
    critical_value: float,
    alpha: float,
    two_sided: bool,
    plus_one: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[StatcondFieldtripCluster, ...],
    np.ndarray,
]:
    if surrogate is None:
        raise ValueError("cluster correction requires a surrogate statistic distribution")
    observed = np.asarray(statistic, dtype=float)
    distribution = np.asarray(surrogate, dtype=float)
    if distribution.shape[:-1] != observed.shape:
        raise ValueError("surrogate shape must equal statistic shape plus a final permutation axis")

    observed_candidates = _cluster_candidates(
        observed,
        adjacency,
        critical_value=critical_value,
        two_sided=two_sided,
    )
    null_maximum = np.zeros(distribution.shape[-1], dtype=float)
    for permutation in range(distribution.shape[-1]):
        candidates = _cluster_candidates(
            distribution[..., permutation],
            adjacency,
            critical_value=critical_value,
            two_sided=two_sided,
        )
        null_maximum[permutation] = max((mass for _sign, _indices, mass in candidates), default=0.0)

    pvalue = np.ones(observed.shape, dtype=float)
    labels = np.zeros(observed.shape, dtype=int)
    clusters = []
    flat_pvalue = pvalue.reshape(-1)
    flat_labels = labels.reshape(-1)
    denominator = null_maximum.size + int(plus_one)
    for label, (sign, indices, mass) in enumerate(observed_candidates, start=1):
        exceedances = np.count_nonzero(null_maximum >= mass) + int(plus_one)
        cluster_pvalue = float(exceedances / denominator)
        flat_pvalue[list(indices)] = cluster_pvalue
        flat_labels[list(indices)] = label
        clusters.append(StatcondFieldtripCluster(label, sign, indices, mass, cluster_pvalue))
    pvalue[np.isnan(observed)] = np.nan
    return pvalue, pvalue <= alpha, labels, tuple(clusters), null_maximum


def _cluster_candidates(
    statistic: Any,
    adjacency: tuple[np.ndarray, ...],
    *,
    critical_value: float,
    two_sided: bool,
) -> tuple[tuple[int, tuple[int, ...], float], ...]:
    values = np.asarray(statistic, dtype=float).reshape(-1)
    signs = (1, -1) if two_sided else (1,)
    clusters = []
    for sign in signs:
        active = sign * values >= critical_value
        visited = np.zeros(values.size, dtype=bool)
        for start in np.flatnonzero(active):
            if visited[start]:
                continue
            stack = [int(start)]
            visited[start] = True
            indices = []
            while stack:
                current = stack.pop()
                indices.append(current)
                for neighbour in adjacency[current]:
                    neighbour_index = int(neighbour)
                    if active[neighbour_index] and not visited[neighbour_index]:
                        visited[neighbour_index] = True
                        stack.append(neighbour_index)
            ordered = tuple(sorted(indices))
            mass = float(np.sum(sign * values[list(ordered)]))
            clusters.append((sign, ordered, mass))
    return tuple(clusters)


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


def _montecarlo_pvalues(
    statistic: Any,
    surrogate: Any,
    *,
    two_sided: bool,
    plus_one: bool = True,
) -> np.ndarray:
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
    correction = int(plus_one)
    pvalues = (exceedances + correction) / (distribution.shape[-1] + correction)
    return np.where(np.isnan(observed), np.nan, pvalues)


def _max_statistic_pvalues(
    statistic: Any,
    surrogate: Any,
    *,
    two_sided: bool,
    plus_one: bool = True,
) -> np.ndarray:
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
    correction = int(plus_one)
    pvalues = (exceedances + correction) / (distribution.shape[-1] + correction)
    return np.where(np.isnan(observed), np.nan, pvalues)


__all__ = ["StatcondFieldtripCluster", "StatcondFieldtripResult", "statcondfieldtrip"]
