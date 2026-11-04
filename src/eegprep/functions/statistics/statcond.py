"""Condition-level statistical comparison helper."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
from scipy import special
from scipy import stats as scipy_stats

from eegprep.functions.miscfunc.value_parsing import parse_key_value_args
from eegprep.functions.statistics._matlab_surrogate import matlab_resampling_cells
from eegprep.functions.statistics._shared import (
    TwoWayEffects,
    condition_grid,
    effect_map,
    normalize_method,
    paired_flag,
    rng_from_seed,
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

logger = logging.getLogger(__name__)

_STATCOND_OPTIONS = frozenset(
    {
        "method",
        "mode",
        "paired",
        "naccu",
        "verbose",
        "variance",
        "surrog",
        "stats",
        "structoutput",
        "forceanova",
        "arraycomp",
        "cluster",
        "alpha",
        "threshold",
        "tail",
        "returnresamplingarray",
        "return_resampling_array",
        "rng",
        "axis",
    }
)


@dataclass(frozen=True)
class StatcondResult:
    """Result returned by :func:`statcond`.

    For a two-way design computed from condition arrays, the ``stat``, ``df``,
    and ``pvalue`` fields are :class:`TwoWayEffects` in row, column, and
    interaction order. Effect-valued nonparametric outputs follow the same
    order.
    """

    stat: Any
    df: Any
    pvalue: Any
    surrogate: Any
    method: str
    paired: bool
    ci: Any = None
    mask: Any = None
    matlab_stat: Any = None
    matlab_df: Any = None
    matlab_pvalue: Any = None
    matlab_surrogate: Any = None
    matlab_primary: Any = None
    # Field order used when a caller asks for more than one MATLAB output.
    matlab_outputs = ("matlab_stat", "matlab_df", "matlab_pvalue", "matlab_surrogate")

    def __iter__(self) -> Iterator[Any]:
        yield self.stat
        yield self.df
        yield self.pvalue
        yield self.surrogate


def statcond(
    data: Any,
    *args: Any,
    **kwargs: Any,
) -> StatcondResult | SurrogateDistribution:
    """Compare condition arrays using EEGLAB-style t-tests or ANOVAs.

    Python callers pass keywords. Positional name/value pairs are accepted
    through :func:`parse_key_value_args`, including EEGLAB names such as
    ``mode``, ``verbose``, ``returnresamplingarray``, and ``structoutput``.

    Args:
        data: One- or two-dimensional sequence of condition arrays. The case
            dimension is the last axis by default.
        paired: ``"auto"``, ``"on"``/``True``, or ``"off"``/``False``.
        method: ``"param"``, ``"perm"``, or ``"bootstrap"``. ``mode`` is the
            legacy alias and wins when it is non-empty.
        naccu: Number of surrogate samples for nonparametric methods.
        variance: ``"homogenous"`` or ``"inhomogenous"`` for unpaired t-tests.
            The default is Welch's inhomogenous variance, matching EEGLAB
            ``finputcheck`` in ``statcond.m:187``. ``ttest2_cell`` still
            defaults to equal variance, matching ``ttest2_cell.m:79``.
        forceanova: Use one-way ANOVA instead of a two-condition t-test.
        tail: Empirical-tail mode for supplied or computed surrogates.
        axis: Axis in each condition array that stores cases.
        rng: Optional NumPy generator or seed for the Python resampling stream.
        alpha: Optional threshold for confidence intervals and masks. ``threshold``
            is the legacy alias. ``NaN`` means unset.
        surrog: Precomputed surrogate statistic array.
        stats: Observed statistic to pair with ``surrog``.
        return_resampling_array: Return surrogate condition grids instead of
            computing statistics. ``returnresamplingarray`` is the EEGLAB name.
        arraycomp: ``True``/``"on"`` batches all resamples. ``False``/``"off"``
            computes one resample at a time and, with
            ``return_resampling_array=True``, returns one grid as in EEGLAB.
        verbose: ``"on"`` logs the selected design. The default is ``"off"``.
        structoutput: Accepted for EEGLAB calls. The Python return is still a
            :class:`StatcondResult`; a single MATLAB output is the struct.
        cluster: ``"on"`` replaces pointwise permutation p-values with cluster
            extent results for one-way designs. The cluster-forming threshold
            uses the test degrees of freedom. EEGLAB's ``statcond.m:470`` calls
            ``tcdf(-abs(surrogval), size(surrogval,1))`` and ``statcond.m:490``
            calls ``tcdf(-abs(ori_vals), length(ori_vals))``, so both thresholds
            use the feature count. That degrees-of-freedom choice is not
            reproduced.

    Returns:
        A :class:`StatcondResult`, or a :class:`SurrogateDistribution` when
        ``return_resampling_array`` is true. Computed two-way result fields use
        named ``rows``, ``columns``, and ``interaction`` effects in that order
        for both paired and unpaired designs. MATLAB-shaped copies live on
        ``matlab_stat``, ``matlab_df``, ``matlab_pvalue``, and
        ``matlab_surrogate``. Unpaired two-way MATLAB cells are stored in
        column, row, interaction order.
    """

    options = _statcond_options(args, kwargs)
    paired = options["paired"]
    method_name = options["method"]
    naccu = options["naccu"]
    variance = options["variance"]
    forceanova = options["forceanova"]
    tail = options["tail"]
    axis = options["axis"]
    rng = options["rng"]
    alpha = options["alpha"]
    surrog = options["surrog"]
    stats = options["stats"]
    return_resampling_array = options["return_resampling_array"]
    arraycomp_flag_value = options["arraycomp"]
    verbose = options["verbose"]
    structoutput = options["structoutput"]
    cluster = options["cluster"]

    grid = condition_grid(data, axis=axis, min_cases=2)
    paired_flag_value = paired_flag(grid, paired)
    if verbose:
        logger.info(
            "statcond %s x %s, paired=%s, method=%s",
            len(grid),
            len(grid[0]),
            "on" if paired_flag_value else "off",
            method_name,
        )
    if return_resampling_array:
        if method_name == "param":
            raise ValueError("return_resampling_array requires 'perm' or 'bootstrap'")
        distribution = surrogdistrib(
            grid,
            method=method_name,
            pairing="on" if paired_flag_value else "off",
            naccu=naccu if arraycomp_flag_value else 1,
            rng=rng,
        )
        cells = matlab_resampling_cells(
            data,
            method=method_name,
            paired=paired_flag_value,
            naccu=naccu,
            arraycomp=arraycomp_flag_value,
        )
        return SurrogateDistribution(distribution.samples, matlab_primary=cells)

    statistic_kind = None
    pvalue_tail = tail
    if surrog is not None:
        if stats is None:
            raise ValueError("stats must be supplied when surrog is supplied")
        observed_stat = _align_supplied_statistic(stats, surrog)
        observed_df = None
        surrogate_stat = surrog
        pvalue = _surrogate_pvalues(surrogate_stat, observed_stat, tail)
    else:
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
                arraycomp=arraycomp_flag_value,
            )
            pvalue_tail = "one" if statistic_kind.startswith("f") else tail
            pvalue = _surrogate_pvalues(surrogate_stat, observed_stat, pvalue_tail)

    if cluster and method_name != "param" and not isinstance(observed_stat, TwoWayEffects):
        cluster_df = observed_df
        cluster_kind = statistic_kind
        if cluster_df is None:
            _design_stat, cluster_df, cluster_kind = _compute_statistic(
                grid,
                paired=paired_flag_value,
                variance=variance,
                forceanova=forceanova,
            )
        if alpha is None:
            alpha = 0.05
            logger.warning("Alpha value set to 0.05 automatically")
        pvalue = _cluster_pvalues(observed_stat, surrogate_stat, cluster_df, alpha, cluster_kind)

    ci = None
    mask = None
    if alpha is not None:
        if surrogate_stat is None:
            raise ValueError("alpha confidence intervals require a nonparametric method or supplied surrogates")
        ci = _surrogate_ci(surrogate_stat, alpha, _ci_tail(pvalue_tail))
        mask = effect_map(pvalue, lambda value: value < alpha)

    feature_shape = grid[0][0].shape[:-1]
    n_conditions = sum(len(row) for row in grid)
    return _package_result(
        observed_stat,
        observed_df,
        pvalue,
        surrogate_stat,
        method_name,
        paired_flag_value,
        ci=ci,
        mask=mask,
        statistic_kind=statistic_kind,
        feature_shape=feature_shape,
        structoutput=structoutput,
        n_conditions=n_conditions,
        forceanova=forceanova,
        arraycomp=arraycomp_flag_value,
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
            _f_tail_probability(stat.rows, df.rows[0], df.rows[1]),
            _f_tail_probability(stat.columns, df.columns[0], df.columns[1]),
            _f_tail_probability(stat.interaction, df.interaction[0], df.interaction[1]),
        )
    return _f_tail_probability(stat, df[0], df[1])


def _f_tail_probability(f_values: Any, df1: Any, df2: Any) -> np.ndarray:
    """Return the F upper tail using the incomplete-beta choice in ``myfpval``.

    ``myfpval.m`` lines 11-19 use the complement ``betainc(..., 'upper')`` when
    ``F < df2/df1``. Otherwise they use the upper tail in its symmetric form,
    ``betainc(df2/(df2+df1*F), df2/2, df1/2)``. The branch matches that choice.
    """

    f_values = np.asarray(f_values, dtype=np.float64)
    df1 = np.asarray(df1, dtype=np.float64)
    df2 = np.asarray(df2, dtype=np.float64)
    probability = np.ones(np.broadcast_shapes(f_values.shape, df1.shape, df2.shape), dtype=np.float64)
    f_values = np.broadcast_to(f_values, probability.shape)
    df1 = np.broadcast_to(df1, probability.shape)
    df2 = np.broadcast_to(df2, probability.shape)
    probability[np.isnan(f_values)] = np.nan
    usable = f_values > 0
    if not np.any(usable):
        return probability
    observed = f_values[usable]
    numerator_df = df1[usable]
    denominator_df = df2[usable]
    complement = observed < (denominator_df / numerator_df)
    tail = np.empty(observed.shape, dtype=np.float64)
    symmetric = ~complement
    if np.any(symmetric):
        statistic = observed[symmetric]
        d1 = numerator_df[symmetric]
        d2 = denominator_df[symmetric]
        tail[symmetric] = special.betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * statistic))
    if np.any(complement):
        statistic = observed[complement]
        d1 = numerator_df[complement]
        d2 = denominator_df[complement]
        tail[complement] = special.betaincc(d1 / 2.0, d2 / 2.0, (d1 * statistic) / (d2 + d1 * statistic))
    probability[usable] = tail
    return probability


def _compute_surrogate_statistics(
    grid: tuple[tuple[np.ndarray, ...], ...],
    *,
    paired: bool,
    method: str,
    naccu: int,
    variance: str,
    forceanova: bool,
    rng: np.random.Generator | int | None,
    arraycomp: bool,
) -> Any:
    pairing = "on" if paired else "off"
    if arraycomp:
        distribution: Iterable[Any] = surrogdistrib(
            grid,
            method=method,
            pairing=pairing,
            naccu=naccu,
            rng=rng,
        ).samples
    else:
        generator = rng_from_seed(rng)
        distribution = (
            surrogdistrib(grid, method=method, pairing=pairing, naccu=1, rng=generator).samples[0] for _ in range(naccu)
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


def _shape_without_trailing_ones(shape: tuple[int, ...]) -> tuple[int, ...]:
    end = len(shape)
    while end > 0 and shape[end - 1] == 1:
        end -= 1
    return shape[:end]


def _align_supplied_statistic(stats: Any, surrog: Any) -> Any:
    if isinstance(stats, TwoWayEffects) or isinstance(surrog, TwoWayEffects):
        return stats
    observed = np.asarray(stats)
    distribution = np.asarray(surrog)
    if not np.issubdtype(observed.dtype, np.number) or not np.issubdtype(distribution.dtype, np.number):
        return stats
    if distribution.ndim < 1:
        raise ValueError("resampling array must have a surrogate axis")
    target = distribution.shape[:-1]
    # stat_surrogate_pvals.m:63 sums bsxfun(@ge, distribution, observed) along
    # the surrogate axis. A trailing length-1 dimension is the singleton that
    # axis expands, so an [n 1] column or a [d1 d2 1] slab matches. A transpose
    # or a flattened column does not: reshaping either one would reorder values.
    if _shape_without_trailing_ones(observed.shape) != _shape_without_trailing_ones(target):
        raise ValueError("stats shape must match the resampling array without its last axis")
    if observed.shape == target:
        return observed
    return np.reshape(observed, target, order="F")


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


def _arraycomp_flag(value: str | bool) -> bool:
    if isinstance(value, str):
        normalized = value.lower()
        if normalized not in {"on", "off"}:
            raise ValueError("arraycomp must be 'on', 'off', True, or False")
        return normalized == "on"
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError("arraycomp must be 'on', 'off', True, or False")


def _on_off(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in {"on", "off"}:
            return normalized == "on"
    raise ValueError(f"{name} must be 'on', 'off', True, or False")


def _matlab_round(value: Any) -> int:
    number = float(np.asarray(value).reshape(-1)[0])
    if not np.isfinite(number):
        raise ValueError("naccu must be a finite number")
    sign = 1 if number >= 0 else -1
    return int(np.floor(np.abs(number) + 0.5) * sign)


def _statcond_alpha(options: Mapping[str, Any]) -> float | None:
    if "alpha" in options:
        raw = options["alpha"]
    elif "threshold" in options:
        raw = options["threshold"]
    else:
        return None
    if raw is None:
        return None
    values = np.asarray(raw, dtype=float)
    if values.size == 0 or bool(np.all(np.isnan(values))):
        return None
    return float(values.ravel()[0])


def _statcond_options(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    options = parse_key_value_args(args, dict(kwargs), lowercase_keys=True, lowercase_kwargs=True)
    unknown = sorted(set(options) - _STATCOND_OPTIONS)
    if unknown:
        raise ValueError(f"statcond: unsupported option(s): {', '.join(unknown)}")
    method = options.get("method", "param")
    mode = options.get("mode", "")
    if mode not in (None, ""):
        method = mode
    naccu = _matlab_round(options.get("naccu", 200))
    if naccu < 1:
        raise ValueError("naccu must be at least 1")
    variance = str(options.get("variance", "inhomogenous")).replace("homogeneous", "homogenous")
    if variance not in {"homogenous", "inhomogenous"}:
        raise ValueError("variance must be 'homogenous' or 'inhomogenous'")
    if "returnresamplingarray" in options:
        return_resampling = _on_off(options["returnresamplingarray"], "returnresamplingarray")
    else:
        return_resampling = _on_off(options.get("return_resampling_array", False), "return_resampling_array")
    return {
        "paired": options.get("paired", "auto"),
        "method": normalize_method(str(method)),
        "naccu": naccu,
        "variance": variance,
        "forceanova": _on_off(options.get("forceanova", False), "forceanova"),
        "tail": str(options.get("tail", "both")),
        "axis": options.get("axis", -1),
        "rng": options.get("rng"),
        "alpha": _statcond_alpha(options),
        "surrog": options.get("surrog"),
        "stats": options.get("stats"),
        "return_resampling_array": return_resampling,
        "arraycomp": _arraycomp_flag(options.get("arraycomp", True)),
        "verbose": _on_off(options.get("verbose", False), "verbose"),
        "structoutput": _on_off(options.get("structoutput", False), "structoutput"),
        "cluster": _on_off(options.get("cluster", False), "cluster"),
    }


def _matlab_feature_shape(feature_shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(feature_shape) <= 1:
        width = 1 if len(feature_shape) == 0 else int(feature_shape[0])
        return (width, 1)
    return tuple(int(dim) for dim in feature_shape)


def _matlab_array(value: Any, feature_shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    array = np.asarray(value)
    if dtype is not None:
        array = np.asarray(array, dtype=dtype)
    if array.size == 0:
        return np.empty((0, 0), dtype=array.dtype)
    return np.reshape(array, _matlab_feature_shape(feature_shape), order="F")


def _effect_cells(effects: TwoWayEffects, convert: Any, paired: bool) -> np.ndarray:
    ordered = (
        (effects.rows, effects.columns, effects.interaction)
        if paired
        else (effects.columns, effects.rows, effects.interaction)
    )
    cells = np.empty((1, 3), dtype=object)
    for index, value in enumerate(ordered):
        cells[0, index] = convert(value)
    return cells


def _matlab_df_value(df: Any, paired: bool) -> np.ndarray:
    if df is None:
        return np.empty((0, 0), dtype=np.float64)
    if isinstance(df, TwoWayEffects):
        return _effect_cells(df, lambda item: _matlab_df_value(item, True), paired)
    if isinstance(df, tuple):
        return np.asarray(df, dtype=np.float64).reshape(1, -1)
    return np.asarray([[df]], dtype=np.float64)


def _matlab_surrogate_value(value: Any, feature_shape: tuple[int, ...], paired: bool, dtype: np.dtype) -> np.ndarray:
    if value is None:
        return np.empty((0, 0), dtype=np.float64)
    if isinstance(value, TwoWayEffects):
        return _effect_cells(value, lambda item: _matlab_surrogate_value(item, feature_shape, True, dtype), paired)
    array = np.asarray(value, dtype=dtype)
    if len(feature_shape) <= 1:
        n_features = 1 if len(feature_shape) == 0 else int(feature_shape[0])
        return np.reshape(array, (n_features, array.shape[-1]), order="F")
    return array


def _matlab_statistic_value(value: Any, feature_shape: tuple[int, ...], dtype: Any, paired: bool) -> np.ndarray:
    if isinstance(value, TwoWayEffects):
        return _effect_cells(value, lambda item: _matlab_array(item, feature_shape, dtype), paired)
    return _matlab_array(value, feature_shape, dtype)


def _output_kind(statistic_kind: str | None, n_conditions: int, forceanova: bool) -> str:
    if statistic_kind is not None:
        return statistic_kind
    if n_conditions == 2 and not forceanova:
        return "t"
    return "f_one_way"


def _package_result(
    observed_stat: Any,
    observed_df: Any,
    pvalue: Any,
    surrogate_stat: Any,
    method_name: str,
    paired_flag_value: bool,
    *,
    ci: Any,
    mask: Any,
    statistic_kind: str | None,
    feature_shape: tuple[int, ...],
    structoutput: bool,
    n_conditions: int,
    forceanova: bool,
    arraycomp: bool,
) -> StatcondResult:
    kind = _output_kind(statistic_kind, n_conditions, forceanova)
    stat_dtype = np.float64
    pvalue_dtype = np.float64
    # statcond.m:261 preallocates single. concatdata.m stores the draws as
    # single, and ttest_cell keeps that class (statcond.m:352, :366-367).
    # ANOVA arraycomp on replaces the array with double anova*_cell output
    # (statcond.m:384, :421); arraycomp off assigns into the single array
    # (statcond.m:398-399, :435-436). A supplied array keeps its own class
    # (statcond.m:444-445). stat_surrogate_ci preserves that class (:512).
    f_statistic = statistic_kind is not None and statistic_kind.startswith("f")
    surrogate_dtype = np.dtype(np.float64 if f_statistic and arraycomp else np.float32)
    if statistic_kind is None:
        if not isinstance(observed_stat, TwoWayEffects):
            stat_dtype = np.asarray(observed_stat).dtype
        if not isinstance(surrogate_stat, TwoWayEffects):
            surrogate_dtype = np.dtype(np.asarray(surrogate_stat).dtype)
    ci_dtype = surrogate_dtype
    matlab_stat = _matlab_statistic_value(observed_stat, feature_shape, stat_dtype, paired_flag_value)
    matlab_df = _matlab_df_value(observed_df, paired_flag_value)
    matlab_pvalue = _matlab_statistic_value(pvalue, feature_shape, pvalue_dtype, paired_flag_value)
    matlab_surrogate = _matlab_surrogate_value(surrogate_stat, feature_shape, paired_flag_value, surrogate_dtype)
    matlab_primary = None
    if structoutput:
        payload: dict[str, Any] = {
            "method": method_name,
            "pval": matlab_pvalue,
            "df": matlab_df,
            "surrog": matlab_surrogate,
            "stat": matlab_stat,
        }
        payload["t" if kind == "t" else "f"] = matlab_stat
        if ci is not None:
            payload["ci"] = _matlab_ci(ci, feature_shape, paired_flag_value, ci_dtype)
            payload["mask"] = _matlab_statistic_value(mask, feature_shape, bool, paired_flag_value)
        matlab_stat = payload
        matlab_primary = payload
    return StatcondResult(
        observed_stat,
        observed_df,
        pvalue,
        surrogate_stat,
        method_name,
        paired_flag_value,
        ci=ci,
        mask=mask,
        matlab_stat=matlab_stat,
        matlab_df=matlab_df,
        matlab_pvalue=matlab_pvalue,
        matlab_surrogate=matlab_surrogate,
        matlab_primary=matlab_primary,
    )


def _matlab_ci(ci: Any, feature_shape: tuple[int, ...], paired: bool, dtype: np.dtype) -> np.ndarray:
    if isinstance(ci, TwoWayEffects):
        return _effect_cells(ci, lambda item: _matlab_ci(item, feature_shape, True, dtype), paired)
    array = np.asarray(ci)
    bounds = [_matlab_array(array[index], feature_shape, dtype) for index in range(array.shape[0])]
    return np.stack(bounds, axis=0)


def _longest_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for flag in mask:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _tail_mask(values: np.ndarray, df: Any, alpha: float, kind: str) -> np.ndarray:
    if kind == "t":
        return scipy_stats.t.sf(np.abs(values), df) * 2 < alpha
    return scipy_stats.f.sf(values, df[0], df[1]) < alpha


def _cluster_pvalues(observed: Any, surrogate: Any, df: Any, alpha: float, kind: str) -> np.ndarray:
    observed_array = np.asarray(observed, dtype=float)
    observed_flat = observed_array.ravel(order="F")
    surrogate_array = np.asarray(surrogate, dtype=float)
    naccu = int(surrogate_array.shape[-1])
    flat_replicates = np.stack(
        [surrogate_array[..., index].ravel(order="F") for index in range(naccu)],
        axis=1,
    )
    surrogate_mask = _tail_mask(flat_replicates, df, alpha, kind)
    observed_mask = _tail_mask(observed_flat, df, alpha, kind)
    extents = np.array([_longest_run(surrogate_mask[:, index]) for index in range(naccu)], dtype=int)
    index = _matlab_round(naccu * (1.0 - alpha))
    if not 1 <= index <= naccu:
        raise ValueError("cluster extent threshold is outside the null distribution")
    threshold = int(np.sort(extents)[index - 1])
    pvalues = np.ones(observed_flat.shape, dtype=float)
    start = None
    for position, flag in enumerate(np.concatenate([observed_mask, [False]])):
        if flag and start is None:
            start = position
        elif not flag and start is not None:
            if position - start >= threshold:
                pvalues[start:position] = 0.0
            start = None
    return pvalues.reshape(observed_array.shape, order="F")


__all__ = ["StatcondResult", "TwoWayEffects", "statcond"]
