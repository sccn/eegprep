"""Statistics for condition-by-group STUDY measure cells."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from eegprep.functions.statistics._shared import TwoWayEffects
from eegprep.functions.statistics.statcond import StatcondResult, statcond
from eegprep.functions.studyfunc.pop_statparams import COMMON_DEFAULTS, EEGLAB_DEFAULTS


@dataclass(frozen=True)
class StudyStatistics:
    """P-values and significance masks for a STUDY measure grid."""

    pcond: list[np.ndarray]
    pgroup: list[np.ndarray]
    pinter: list[np.ndarray]
    condmask: list[np.ndarray]
    groupmask: list[np.ndarray]
    intermask: list[np.ndarray]
    alpha: float | tuple[float, ...] | None
    method: str
    mcorrect: str

    def __iter__(self):
        yield from self.output()

    def output(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Return EEGLAB outputs: p-values, or masks when alpha is set."""
        if self.alpha is None:
            return self.pcond, self.pgroup, self.pinter
        return self.condmask, self.groupmask, self.intermask


def std_stat(
    data: Any,
    options: dict[str, Any] | None = None,
    *,
    return_result: bool = False,
    rng: np.random.Generator | int | None = 0,
    **kwargs: Any,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] | StudyStatistics:
    """Compute EEGLAB-style condition, group, and interaction statistics.

    ``data`` is a rectangular ``condition x group`` grid. Each numeric cell
    stores observations on its last axis. Parametric, permutation, and
    bootstrap methods are delegated to :func:`eegprep.statcond`; FDR correction
    is applied across every sample in each returned effect.
    """
    config = _statistics_options(options, kwargs)
    grid = _condition_grid(data)
    if not _enabled(config.get("condstats")) and not _enabled(config.get("groupstats")):
        result = StudyStatistics(
            [],
            [],
            [],
            [],
            [],
            [],
            _alpha(config["eeglab"].get("alpha")),
            str(config["eeglab"].get("method") or "param").lower(),
            str(config["eeglab"].get("mcorrect") or "none").lower(),
        )
        return result if return_result else result.output()
    method = str(config["eeglab"]["method"] or "param").lower()
    naccu = config["eeglab"]["naccu"]
    naccu = 2000 if naccu is None or np.asarray(naccu).size == 0 else int(np.asarray(naccu).reshape(-1)[0])
    pairing = list(config.get("paired") or ["off", "off"])
    while len(pairing) < 2:
        pairing.append("off")

    pcond: list[np.ndarray] = []
    if _enabled(config.get("condstats")) and len(grid) > 1:
        for group_index in range(len(grid[0])):
            values = [row[group_index] for row in grid]
            result = _run_statcond(values, paired=pairing[0], method=method, naccu=naccu, rng=rng)
            pcond.append(np.asarray(result.pvalue, dtype=float))

    pgroup: list[np.ndarray] = []
    if _enabled(config.get("groupstats")) and len(grid[0]) > 1:
        for row in grid:
            result = _run_statcond(row, paired=pairing[1], method=method, naccu=naccu, rng=rng)
            pgroup.append(np.asarray(result.pvalue, dtype=float))

    pinter: list[np.ndarray] = []
    if (pcond or pgroup) and len(grid) > 1 and len(grid[0]) > 1:
        # EEGLAB uses the unpaired factor when a mixed design contains one.
        interaction_pairing = "on" if all(_enabled(value) for value in pairing[:2]) else "off"
        result = _run_statcond(grid, paired=interaction_pairing, method=method, naccu=naccu, rng=rng)
        if not isinstance(result.pvalue, TwoWayEffects):
            raise RuntimeError("two-factor STUDY statistics did not return three effects")
        pinter = [
            np.asarray(result.pvalue.rows, dtype=float),
            np.asarray(result.pvalue.columns, dtype=float),
            np.asarray(result.pvalue.interaction, dtype=float),
        ]

    mcorrect = str(config["eeglab"].get("mcorrect") or "none").lower()
    if mcorrect not in {"none", "fdr"}:
        raise NotImplementedError("STUDY statistics currently support mcorrect='none' or 'fdr'")
    if mcorrect == "fdr":
        pcond = [_fdr_adjust(values) for values in pcond]
        pgroup = [_fdr_adjust(values) for values in pgroup]
        pinter = [_fdr_adjust(values) for values in pinter]

    alpha = _alpha(config["eeglab"].get("alpha"))
    condmask = _masks(pcond, alpha)
    groupmask = _masks(pgroup, alpha)
    intermask = _masks(pinter, alpha)
    result = StudyStatistics(
        pcond,
        pgroup,
        pinter,
        condmask,
        groupmask,
        intermask,
        alpha,
        method,
        mcorrect,
    )
    return result if return_result else result.output()


def _run_statcond(data: Any, *, paired: Any, method: str, naccu: int, rng: Any) -> StatcondResult:
    result = statcond(data, paired=paired, method=method, naccu=naccu, axis=-1, rng=rng)
    if not isinstance(result, StatcondResult):
        raise RuntimeError("STUDY statistics unexpectedly returned a resampling array")
    return result


def _condition_grid(data: Any) -> list[list[np.ndarray]]:
    if not isinstance(data, (list, tuple)) or not data:
        raise ValueError("STUDY statistics require a non-empty condition grid")
    rows = list(data)
    if not isinstance(rows[0], (list, tuple)):
        rows = [[value] for value in rows]
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("STUDY condition/group cells must form a rectangular grid")
    return [[np.asarray(value) for value in row] for row in rows]


def _statistics_options(options: dict[str, Any] | None, overrides: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(options) if isinstance(options, dict) else {}
    if isinstance(config.get("etc"), dict):
        config = deepcopy(config["etc"].get("statistics") or {})
    for key, value in COMMON_DEFAULTS.items():
        config.setdefault(key, deepcopy(value))
    eeglab = config.get("eeglab")
    if not isinstance(eeglab, dict):
        eeglab = {}
    for key, value in EEGLAB_DEFAULTS.items():
        eeglab.setdefault(key, deepcopy(value))
    config["eeglab"] = eeglab
    for key, value in overrides.items():
        target = {"threshold": "alpha", "statistics": "method"}.get(key.lower(), key.lower())
        if target in COMMON_DEFAULTS or target == "paired":
            config[target] = value
        elif target in EEGLAB_DEFAULTS:
            config["eeglab"][target] = value
        else:
            raise ValueError(f"Unknown std_stat option: {key}")
    if str(config.get("mode") or "eeglab").lower() != "eeglab":
        raise NotImplementedError("FieldTrip STUDY statistics require the external FieldTrip backend")
    return config


def _fdr_adjust(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    flat = array.ravel()
    finite_positions = np.flatnonzero(np.isfinite(flat))
    output = np.full_like(flat, np.nan)
    if finite_positions.size == 0:
        return output.reshape(array.shape)
    finite = flat[finite_positions]
    order = np.argsort(finite)
    ranked = finite[order]
    adjusted = ranked * ranked.size / np.arange(1, ranked.size + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    finite_output = np.empty_like(adjusted)
    finite_output[order] = np.minimum(adjusted, 1.0)
    output[finite_positions] = finite_output
    return output.reshape(array.shape)


def _alpha(value: Any) -> float | tuple[float, ...] | None:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0 or np.isnan(array).all():
        return None
    if np.isnan(array).any() or np.any((array <= 0) | (array > 1)):
        raise ValueError("STUDY statistics threshold must be in (0, 1]")
    values = tuple(float(item) for item in np.sort(array))
    return values[0] if len(values) == 1 else values


def _masks(values: list[np.ndarray], alpha: float | tuple[float, ...] | None) -> list[np.ndarray]:
    if alpha is None:
        return []
    thresholds = (alpha,) if isinstance(alpha, float) else alpha
    masks = []
    for value in values:
        mask = np.zeros(np.asarray(value).shape, dtype=float)
        for index, threshold in enumerate(thresholds):
            selected = np.isfinite(value) & (value < threshold) & (mask == 0)
            mask[selected] = len(thresholds) - index
        masks.append(mask)
    return masks


def _enabled(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() == "on")


__all__ = ["StudyStatistics", "std_stat"]
