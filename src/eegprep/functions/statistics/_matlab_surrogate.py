"""EEGLAB ``surrogdistrib`` index streams for MATLAB-shaped ``statcond`` outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.statistics._matlab_rand import MatlabTwister, matlab_rng
from eegprep.functions.statistics._shared import condition_grid


def matlab_resampling_cells(
    data: Any,
    *,
    method: str,
    paired: bool,
    naccu: int,
    arraycomp: bool,
    rng: MatlabTwister | None = None,
) -> np.ndarray:
    """Return one EEGLAB surrogate cell array drawn from the MATLAB Twister.

    ``arraycomp`` selects ``surrogdistrib``'s batched or one-draw index path.
    Both paths consume ``rand`` in the same order as the pinned ``surrogdistrib.m``.
    """
    stream = matlab_rng() if rng is None else rng
    grid = condition_grid(data, axis=-1, min_cases=1)
    rows = len(grid)
    columns = len(grid[0])
    flat = [grid[row][column] for column in range(columns) for row in range(rows)]
    matrices = [_feature_by_case(array) for array in flat]
    lengths = np.cumsum([0, *(matrix.shape[1] for matrix in matrices)])
    pooled = np.concatenate(matrices, axis=1).astype(np.float32, copy=False)
    bootstrap = method == "bootstrap"
    draws = (
        _supersurrogate(pooled, lengths, naccu, bootstrap, paired, stream)
        if arraycomp
        else _surrogate(pooled, lengths, bootstrap, paired, stream)
    )
    cells = np.empty((rows, columns), dtype=object)
    index = 0
    for column in range(columns):
        for row in range(rows):
            cells[row, column] = draws[index]
            index += 1
    return cells


def _feature_by_case(array: np.ndarray) -> np.ndarray:
    features = int(np.prod(array.shape[:-1], dtype=int))
    return np.reshape(array, (features, array.shape[-1]), order="F")


def _supersurrogate(
    pooled: np.ndarray,
    lengths: np.ndarray,
    naccu: int,
    bootstrap: bool,
    paired: bool,
    rng: MatlabTwister,
) -> list[np.ndarray]:
    n_conditions = len(lengths) - 1
    n_cases = int(lengths[1])
    total = int(lengths[-1])
    if bootstrap and paired:
        base = np.tile(np.arange(1, total + 1), (naccu, 1))
        draw = np.ceil(np.asarray(rng.rand(naccu, total)) * (n_conditions + 1))
        indices = np.mod(base + draw * n_cases - 1, total) + 1
    elif bootstrap:
        indices = np.ceil(np.asarray(rng.rand(naccu, total)) * total)
    elif paired:
        draws = np.asarray(rng.rand(naccu, n_cases, n_conditions))
        order = np.argsort(draws, axis=2, kind="stable") + 1
        subjects = np.arange(1, n_cases + 1).reshape(1, n_cases, 1)
        indices = ((order - 1) * n_cases + subjects).reshape(naccu, total, order="F")
    else:
        draws = np.asarray(rng.rand(naccu, total))
        indices = np.argsort(draws, axis=1, kind="stable") + 1
    selected = []
    for index in range(n_conditions):
        columns = indices[:, int(lengths[index]) : int(lengths[index + 1])]
        selected.append(_take_cases(pooled, columns, naccu))
    return selected


def _surrogate(
    pooled: np.ndarray,
    lengths: np.ndarray,
    bootstrap: bool,
    paired: bool,
    rng: MatlabTwister,
) -> list[np.ndarray]:
    n_conditions = len(lengths) - 1
    n_cases = int(lengths[1])
    total = int(lengths[-1])
    if bootstrap and paired:
        base = np.arange(1, total + 1)
        draw = np.ceil(np.asarray(rng.rand(1, total)) * (n_conditions + 1)).ravel()
        indices = np.mod(base + draw * n_cases - 1, total) + 1
    elif bootstrap:
        indices = np.ceil(np.asarray(rng.rand(1, total)) * total).ravel()
    elif paired:
        indices = np.arange(1, total + 1).reshape((n_cases, n_conditions), order="F")
        for row in range(n_cases):
            order = np.argsort(np.asarray(rng.rand(1, n_conditions)).ravel(), kind="stable")
            indices[row, :] = indices[row, order]
        indices = indices.reshape(total, order="F")
    else:
        order = np.argsort(np.asarray(rng.rand(1, total)).ravel(), kind="stable")
        indices = np.arange(1, total + 1)[order]
    selected = []
    for index in range(n_conditions):
        columns = indices[int(lengths[index]) : int(lengths[index + 1])].astype(int) - 1
        selected.append(pooled[:, columns])
    return selected


def _take_cases(pooled: np.ndarray, columns: np.ndarray, naccu: int) -> np.ndarray:
    chosen = pooled[:, columns.astype(int).ravel(order="F") - 1]
    return chosen.reshape((pooled.shape[0], naccu, columns.shape[1]), order="F")
