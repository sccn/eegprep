"""Low-level EEGLAB-style EEG data re-referencing."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from eegprep.functions.popfunc._chanutils import normalise_reflocs as _normalise_reflocs


def reref(
    data: Any,
    ref: Any = None,
    *,
    exclude: Any = None,
    keepref: str = "off",
    elocs: Any = None,
    refloc: Any = None,
    huber: float | None = None,
):
    """Re-reference channel-major EEG data.

    Numeric channel indices are 0-based, matching existing EEGPrep channel
    selection helpers. String channel labels are resolved by ``pop_reref``
    before calling this low-level helper.

    Args:
        data: EEG data with shape ``(channels, points)`` or
            ``(channels, points, trials)``.
        ref: Reference channel index or indices. Empty/``None`` computes an
            average reference.
        exclude: Channel indices excluded from the re-reference operation.
        keepref: ``"on"`` keeps explicit reference channels in the output;
            ``"off"`` removes them, matching EEGLAB's default.
        elocs: Optional channel-location dictionaries to update.
        refloc: Optional old reference-channel location(s) to reconstruct as
            zero-valued channels before re-referencing.
        huber: Optional Huber threshold for average reference.

    Returns:
        tuple: ``(data, elocs, removed_ref_channels, mean_data)``.
    """
    data_arr = np.asarray(data)
    if data_arr.ndim not in (2, 3):
        raise ValueError("data must have shape (channels, points) or (channels, points, trials)")

    original_dtype = data_arr.dtype
    work = data_arr.astype(np.result_type(data_arr.dtype, np.float64), copy=True)
    original_tail_shape = work.shape[1:]
    work = work.reshape(work.shape[0], -1)

    locs = _copy_locs(elocs)
    ref_indices = _normalise_indices(ref, work.shape[0], "reference")
    exclude_indices = _normalise_indices(exclude, work.shape[0], "exclude")
    keepref_value = str(keepref).lower()
    if keepref_value not in {"on", "off"}:
        raise ValueError("keepref must be 'on' or 'off'")
    if ref_indices and set(ref_indices).intersection(exclude_indices):
        raise ValueError("Reference channels cannot also be excluded")

    if refloc not in (None, [], ()):
        new_locs = _normalise_reflocs(refloc)
        work = np.vstack([work, np.zeros((len(new_locs), work.shape[1]), dtype=work.dtype)])
        if locs is not None:
            locs.extend(new_locs)

    chansin = [idx for idx in range(work.shape[0]) if idx not in set(exclude_indices)]
    if not chansin:
        raise ValueError("No channels available for re-referencing")

    mean_data = None
    chansin_array = np.asarray(chansin)
    if huber is not None and not np.isnan(float(huber)) and not ref_indices:
        work[chansin_array, :] = _huber_average_reference(work[chansin_array, :], float(huber))
    elif ref_indices:
        mean_data = work[np.asarray(ref_indices), :].mean(axis=0)
        work[chansin_array, :] = work[chansin_array, :] - mean_data
    else:
        mean_data = work[chansin_array, :].mean(axis=0)
        work[chansin_array, :] = work[chansin_array, :] - mean_data

    if locs is not None:
        _update_references(locs, chansin, ref_indices)

    removed = []
    if ref_indices and keepref_value == "off":
        if locs is not None:
            removed = [copy.deepcopy(locs[idx]) for idx in ref_indices if idx < len(locs)]
            locs = [loc for idx, loc in enumerate(locs) if idx not in set(ref_indices)]
        keep_mask = np.ones(work.shape[0], dtype=bool)
        keep_mask[ref_indices] = False
        work = work[keep_mask, :]

    out = work.reshape((work.shape[0], *original_tail_shape))
    if np.issubdtype(original_dtype, np.floating):
        out = out.astype(original_dtype, copy=False)
    return out, locs, removed, mean_data


def _copy_locs(elocs: Any) -> list[dict[str, Any]] | None:
    if elocs is None:
        return None
    if isinstance(elocs, np.ndarray):
        elocs = elocs.tolist()
    copied = [copy.deepcopy(dict(loc)) for loc in list(elocs)]
    return copied or None


def _normalise_indices(values: Any, nchan: int, name: str) -> list[int]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, (int, np.integer)):
        values = [int(values)]
    elif isinstance(values, float) and values.is_integer():
        values = [int(values)]
    elif isinstance(values, (str, bytes)):
        raise TypeError(f"{name} indices must already be resolved to integers")
    else:
        values = list(values)

    indices: list[int] = []
    for value in values:
        if isinstance(value, np.integer):
            value = int(value)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int):
            raise TypeError(f"{name} indices must be integers")
        if value < 0 or value >= nchan:
            raise ValueError(f"{name.capitalize()} channel index out of range")
        indices.append(value)
    return sorted(set(indices))


def _update_references(locs: list[dict[str, Any]], chansin: list[int], ref_indices: list[int]) -> None:
    if not ref_indices:
        ref_text = "average"
    else:
        labels = [
            str(locs[idx].get("labels", idx))
            for idx in ref_indices
            if idx < len(locs)
        ]
        ref_text = labels[0] if len(labels) == 1 else " ".join(labels)
    for idx in chansin:
        if idx < len(locs):
            locs[idx]["ref"] = ref_text


def _huber_average_reference(data: np.ndarray, threshold: float) -> np.ndarray:
    mean = data.mean(axis=0)
    for _ in range(100):
        residual = np.mean(data - mean, axis=1, keepdims=True)
        weights = np.ones_like(residual)
        large = np.abs(residual) > threshold
        weights[large] = threshold / np.abs(residual[large])
        new_mean = np.sum(weights * data, axis=0) / np.sum(weights, axis=0)
        if np.all(np.abs(new_mean - mean) < 1e-6):
            mean = new_mean
            break
        mean = new_mean
    return data - mean
