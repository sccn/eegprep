"""Build spatially distributed channel-location subsets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


def loc_subsets(
    chanlocs: Sequence[dict[str, Any]],
    subset_sizes: Sequence[int],
    plot_optimization: bool = False,
    plot_subsets: bool = False,
    mandatory_channels: Sequence[Sequence[int]] | None = None,
    *,
    random_state: int = 0,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Separate channels into maximally spaced subsets.

    Channel indices are zero-based. When requested subset sizes do not consume
    every channel, a final subset contains the remainder. Mandatory channels
    remain in their requested subset throughout optimization.
    """
    positions = _positions(chanlocs)
    sizes = _subset_sizes(subset_sizes, positions.shape[1])
    mandatory = _mandatory_sets(mandatory_channels, sizes, positions.shape[1])
    fixed = {channel for subset in mandatory for channel in subset}

    rng = np.random.default_rng(random_state)
    remaining = np.asarray([channel for channel in range(positions.shape[1]) if channel not in fixed], dtype=int)
    remaining = rng.permutation(remaining)
    subsets: list[list[int]] = [list(channels) for channels in mandatory]
    cursor = 0
    for subset_index, size in enumerate(sizes):
        needed = size - len(subsets[subset_index])
        subsets[subset_index].extend(remaining[cursor : cursor + needed].tolist())
        cursor += needed
    if cursor < remaining.size:
        subsets.append(remaining[cursor:].tolist())

    distances = cdist(positions.T, positions.T)
    history = [_spacing_objective(subsets, distances)]
    improved = True
    while improved:
        improved = False
        for first in range(len(subsets) - 1):
            for second in range(first + 1, len(subsets)):
                for first_position, first_channel in enumerate(subsets[first]):
                    if first_channel in fixed:
                        continue
                    for second_position, second_channel in enumerate(subsets[second]):
                        if second_channel in fixed:
                            continue
                        before = _subset_objective(subsets[first], distances) + _subset_objective(
                            subsets[second], distances
                        )
                        first_candidate = list(subsets[first])
                        second_candidate = list(subsets[second])
                        first_candidate[first_position] = second_channel
                        second_candidate[second_position] = first_channel
                        after = _subset_objective(first_candidate, distances) + _subset_objective(
                            second_candidate, distances
                        )
                        if after <= before + np.finfo(float).eps * max(1.0, abs(before)):
                            continue
                        subsets[first] = first_candidate
                        subsets[second] = second_candidate
                        history.append(history[-1] + after - before)
                        improved = True

    arrays = [np.asarray(sorted(subset), dtype=int) for subset in subsets]
    memberships = np.empty(positions.shape[1], dtype=int)
    for subset_index, channels in enumerate(arrays):
        memberships[channels] = subset_index
    if plot_optimization:
        _plot_optimization(history)
    if plot_subsets:
        _plot_channel_subsets(positions, memberships)
    return arrays, memberships, positions


def _positions(chanlocs: Sequence[dict[str, Any]]) -> np.ndarray:
    if len(chanlocs) == 0:
        raise ValueError("chanlocs must contain at least one channel")
    rows = []
    for index, location in enumerate(chanlocs):
        try:
            row = [location[axis] if axis in location else location[axis.lower()] for axis in ("X", "Y", "Z")]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"chanlocs[{index}] must define finite X, Y, and Z coordinates") from exc
        rows.append(row)
    positions = np.asarray(rows, dtype=float).T
    if positions.shape != (3, len(chanlocs)) or not np.all(np.isfinite(positions)):
        raise ValueError("chanlocs must define finite X, Y, and Z coordinates")
    return positions


def _subset_sizes(values: Sequence[int], channels: int) -> list[int]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.number):
        raise ValueError("subset_sizes must be a non-empty sequence of integers")
    numeric = raw.astype(float)
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError("subset_sizes must contain integers")
    sizes = numeric.astype(int).tolist()
    if min(sizes) < 2:
        raise ValueError("requested subsets must each contain at least two channels")
    if sum(sizes) > channels:
        raise ValueError("requested subset sizes exceed the number of channels")
    return sizes


def _mandatory_sets(
    values: Sequence[Sequence[int]] | None,
    sizes: Sequence[int],
    channels: int,
) -> list[list[int]]:
    result = [[] for _size in sizes]
    if values is None:
        return result
    if len(values) > len(sizes):
        raise ValueError("mandatory_channels has more entries than requested subsets")
    seen: set[int] = set()
    for subset_index, subset in enumerate(values):
        raw = np.asarray(subset)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.number):
            raise ValueError("mandatory channel sets must be one-dimensional integer sequences")
        numeric = raw.astype(float)
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise ValueError("mandatory channel indices must be integers")
        selected = numeric.astype(int).tolist()
        if len(selected) > sizes[subset_index]:
            raise ValueError("a mandatory channel set exceeds its requested subset size")
        if any(channel < 0 or channel >= channels for channel in selected):
            raise ValueError(f"mandatory channel indices must be within 0..{channels - 1}")
        if seen.intersection(selected) or len(set(selected)) != len(selected):
            raise ValueError("mandatory channel indices must be unique across subsets")
        seen.update(selected)
        result[subset_index] = selected
    return result


def _subset_objective(subset: Sequence[int], distances: np.ndarray) -> float:
    indices = np.asarray(subset, dtype=int)
    return float(np.sum(distances[np.ix_(indices, indices)]) / indices.size)


def _spacing_objective(subsets: Sequence[Sequence[int]], distances: np.ndarray) -> float:
    return sum(_subset_objective(subset, distances) for subset in subsets)


def _plot_optimization(history: Sequence[float]) -> None:
    figure, axis = plt.subplots()
    axis.plot(np.arange(len(history)), history)
    axis.set_xlabel("Number of exchanges")
    axis.set_ylabel("Sum of mean within-subset distances")
    figure.tight_layout()


def _plot_channel_subsets(positions: np.ndarray, memberships: np.ndarray) -> None:
    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(positions[0], positions[1], positions[2], c=memberships, s=50)
    axis.set_title("Channel Subsets")
    axis.set_box_aspect((1, 1, 1))
    figure.tight_layout()


__all__ = ["loc_subsets"]
