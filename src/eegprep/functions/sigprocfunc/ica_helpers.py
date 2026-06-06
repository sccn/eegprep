"""EEGLAB-compatible ICA projection helper calculations."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc.misc import finite_matmul
from eegprep.functions.popfunc._rejection import component_activations, data_3d


def icaact(data: Any, weights: Any, datamean: Any = 0) -> np.ndarray:
    """Compute ICA activation waveforms from channel data and weights.

    Args:
        data: Channel-major data, shaped ``channels x frames``.
        weights: Unmixing matrix.
        datamean: Optional per-channel mean column(s), one column per epoch.

    Returns:
        Component activations shaped ``components x frames``.
    """
    values = np.asarray(data, dtype=float)
    w = np.asarray(weights, dtype=float)
    if values.ndim != 2 or w.ndim != 2:
        raise ValueError("data and weights must be 2-D arrays")
    means = _datamean(values.shape[0], values.shape[1], datamean)
    frames = values.shape[1] // means.shape[1]
    activations = np.zeros((w.shape[0], values.shape[1]), dtype=float)
    for epoch in range(means.shape[1]):
        start = epoch * frames
        stop = start + frames
        activations[:, start:stop] = finite_matmul(w, values[:, start:stop] - means[:, [epoch]])
    return activations


def icaproj(
    data: Any,
    weights: Any,
    compindex: Any,
    datamean: Any = 0,
    chansout: Any | None = None,
) -> np.ndarray:
    """Back-project selected ICA components into channel space."""
    values = np.asarray(data, dtype=float)
    w = np.asarray(weights, dtype=float)
    if values.ndim != 2 or w.ndim != 2:
        raise ValueError("data and weights must be 2-D arrays")
    comps = _one_based_indices(compindex, w.shape[0], "component") - 1
    chans = (
        np.arange(values.shape[0], dtype=int)
        if chansout is None or np.asarray(chansout).size == 0
        else _one_based_indices(chansout, values.shape[0], "channel") - 1
    )
    means = _datamean(values.shape[0], values.shape[1], datamean)
    centered = values.copy()
    frames = values.shape[1] // means.shape[1]
    for epoch in range(means.shape[1]):
        start = epoch * frames
        stop = start + frames
        centered[:, start:stop] -= means[:, [epoch]]
    inv_weights = np.linalg.pinv(w)
    activations = finite_matmul(w[comps, :], centered)
    return finite_matmul(inv_weights[np.ix_(chans, comps)], activations)


def icavar(data: Any, weights: Any, sphere: Any | None = None, compnums: Any | None = None) -> np.ndarray:
    """Return per-frame scalp variance contributed by selected components."""
    values = np.asarray(data, dtype=float)
    w = np.asarray(weights, dtype=float)
    s = np.eye(w.shape[1]) if sphere is None or np.asarray(sphere).size == 0 else np.asarray(sphere, dtype=float)
    if values.ndim != 2 or w.ndim != 2 or s.ndim != 2:
        raise ValueError("data, weights, and sphere must be 2-D arrays")
    if w.shape[1] != s.shape[0] or s.shape[1] != values.shape[0]:
        raise ValueError("weights, sphere, and data sizes are incompatible")
    activations = finite_matmul(finite_matmul(w, s), values)
    comps = (
        np.arange(activations.shape[0], dtype=int)
        if compnums is None or np.asarray(compnums).size == 0 or np.asarray(compnums).ravel()[0] == 0
        else _one_based_indices(compnums, activations.shape[0], "component") - 1
    )
    projection = np.linalg.pinv(finite_matmul(w, s))
    out = np.zeros((comps.size, values.shape[1]), dtype=float)
    denom = max(values.shape[0] - 1, 1)
    for row, comp in enumerate(comps):
        projected = projection[:, [comp]] * activations[[comp], :]
        out[row, :] = np.sum(projected * projected, axis=0) / denom
    return out


def compvar(data: Any, act: Any, winv: Any, compnums: Any) -> tuple[np.ndarray, float]:
    """Back-project components and compute percent variance accounted for."""
    values = _flatten_channels(data)
    activations = _activations_from_arg(values, act)
    inverse = np.asarray(winv, dtype=float)
    comps = _one_based_indices(compnums, activations.shape[0], "component") - 1
    projected = finite_matmul(inverse[:, comps], activations[comps, :])
    squaredata = float(np.sum(values * values))
    residual = projected - values
    squarecomp = float(np.sum(residual * residual))
    pvaf = 100.0 * (1.0 - squarecomp / squaredata) if squaredata else 0.0
    return projected.reshape(np.asarray(data).shape, order="F"), pvaf


def eeg_getica(EEG: dict[str, Any], comp: Any | None = None) -> np.ndarray:
    """Return stored or recomputed ICA component activations."""
    weights = np.asarray(EEG.get("icaweights", []), dtype=float)
    if weights.ndim != 2 or weights.size == 0:
        raise ValueError("EEG.icaweights is required")
    comps = (
        np.arange(weights.shape[0], dtype=int)
        if comp is None or np.asarray(comp).size == 0
        else _one_based_indices(comp, weights.shape[0], "component") - 1
    )
    stored = np.asarray(EEG.get("icaact", []), dtype=float)
    if stored.size:
        return np.asarray(stored)[comps, ...]
    return component_activations(EEG)[comps, ...]


def eeg_pvaf(
    EEG: dict[str, Any],
    comps: Any | None = None,
    *,
    artcomps: Any | None = None,
    omitchans: Any | None = None,
    chans: Any | None = None,
    fraction: float = 1.0,
    plot: str = "off",
) -> tuple[np.ndarray | float, np.ndarray, np.ndarray]:
    """Compute percent variance accounted for by ICA component projections."""
    del plot
    return _eeg_pv_common(
        EEG,
        comps,
        artcomps=artcomps,
        omitchans=omitchans,
        chans=chans,
        fraction=fraction,
    )


def eeg_pv(
    EEG: dict[str, Any],
    comps: Any | None = None,
    artcomps: Any | None = None,
    omitchans: Any | None = None,
    fraction: float = 1.0,
    plotflag: Any | None = None,
) -> tuple[np.ndarray | float, np.ndarray, np.ndarray]:
    """Compatibility alias for ``eeg_pvaf`` using legacy positional options."""
    del plotflag
    return _eeg_pv_common(EEG, comps, artcomps=artcomps, omitchans=omitchans, chans=None, fraction=fraction)


def _eeg_pv_common(
    EEG: dict[str, Any],
    comps: Any | None,
    *,
    artcomps: Any | None,
    omitchans: Any | None,
    chans: Any | None,
    fraction: float,
) -> tuple[np.ndarray | float, np.ndarray, np.ndarray]:
    raw_data = data_3d(EEG)
    full_data = raw_data.reshape(int(EEG.get("nbchan", 0) or raw_data.shape[0]), -1, order="F")
    acts = eeg_getica(EEG)
    acts_2d = acts.reshape(acts.shape[0], -1, order="F")
    winv = np.asarray(EEG.get("icawinv", []), dtype=float)
    if winv.ndim != 2 or winv.size == 0:
        weights = np.asarray(EEG.get("icaweights", []), dtype=float)
        sphere = np.asarray(EEG.get("icasphere", []), dtype=float)
        winv = np.linalg.pinv(finite_matmul(weights, sphere))
    ica_chans = _ica_channel_indices(EEG, full_data.shape[0])
    data = full_data[ica_chans, :]
    if winv.shape[0] != ica_chans.size:
        raise ValueError("EEG.icawinv rows must match EEG.icachansind")
    selected_chans = _selected_ica_channels(ica_chans, full_data.shape[0], chans=chans, omitchans=omitchans)
    sample_count = int(round(float(fraction) * data.shape[1]))
    if sample_count < 1:
        raise ValueError("fraction of data specified too small")
    sample_count = min(sample_count, data.shape[1])
    samples = np.arange(sample_count)
    data_selected = data[np.ix_(selected_chans, samples)]
    acts_selected = acts_2d[:, samples]
    art = _optional_component_indices(artcomps, acts_selected.shape[0])
    if art.size:
        data_selected = data_selected - finite_matmul(winv[np.ix_(selected_chans, art)], acts_selected[art, :])
    pvall = np.var(data_selected, axis=1)
    pvall[pvall == 0] = np.finfo(float).eps
    progressive = comps is None or np.asarray(comps).size == 0 or np.asarray(comps).ravel()[0] == 0
    if progressive:
        pvaf_values = []
        pvafs_values = []
        for stop in range(1, acts_selected.shape[0] + 1):
            pvaf, pvafs = _pv_for_components(
                data_selected,
                acts_selected,
                winv[selected_chans, :],
                np.arange(stop),
                pvall,
            )
            pvaf_values.append(pvaf)
            pvafs_values.append(pvafs)
        return np.asarray(pvaf_values), np.vstack(pvafs_values), pvall
    comp_indices = _one_based_indices(comps, acts_selected.shape[0], "component") - 1
    if art.size:
        comp_indices = np.asarray([index for index in comp_indices if index not in set(art.tolist())], dtype=int)
    pvaf, pvafs = _pv_for_components(data_selected, acts_selected, winv[selected_chans, :], comp_indices, pvall)
    return pvaf, pvafs, pvall


def _pv_for_components(
    data: np.ndarray,
    activations: np.ndarray,
    winv: np.ndarray,
    comps: np.ndarray,
    pvall: np.ndarray,
) -> tuple[float, np.ndarray]:
    residual = data.copy()
    if comps.size:
        residual = residual - finite_matmul(winv[:, comps], activations[comps, :])
    pvdiff = np.var(residual, axis=1)
    pvafs = 100.0 - 100.0 * pvdiff / pvall
    pvaf = 100.0 - 100.0 * float(np.sum(pvdiff) / np.sum(pvall))
    return pvaf, pvafs


def _datamean(channels: int, total_frames: int, datamean: Any) -> np.ndarray:
    if datamean is None or (np.isscalar(datamean) and float(datamean) == 0.0):
        return np.zeros((channels, 1), dtype=float)
    means = np.asarray(datamean, dtype=float)
    if means.ndim == 1:
        means = means[:, np.newaxis]
    if means.shape[0] != channels:
        raise ValueError("datamean channels must match data")
    if total_frames % means.shape[1] != 0:
        raise ValueError("datamean epochs must divide data frames")
    return means


def _flatten_channels(data: Any) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.ndim == 2:
        return values
    if values.ndim == 3:
        return values.reshape(values.shape[0], -1, order="F")
    raise ValueError("data must be 2-D or 3-D")


def _activations_from_arg(data: np.ndarray, act: Any) -> np.ndarray:
    if isinstance(act, (list, tuple)) and len(act) == 1:
        act = act[0]
    if isinstance(act, (list, tuple)) and len(act) == 2:
        sphere = np.asarray(act[0], dtype=float)
        weights = np.asarray(act[1], dtype=float)
        return finite_matmul(finite_matmul(weights, sphere), data)
    return _flatten_channels(act)


def _one_based_indices(values: Any, limit: int, label: str) -> np.ndarray:
    arr = np.asarray(values if values is not None else [], dtype=int).ravel()
    if arr.size == 0:
        return arr
    if np.any(arr < 1) or np.any(arr > limit):
        raise ValueError(f"{label} indices must be within 1..{limit}")
    return arr


def _optional_component_indices(values: Any | None, limit: int) -> np.ndarray:
    if values is None or np.asarray(values).size == 0:
        return np.asarray([], dtype=int)
    return _one_based_indices(values, limit, "component") - 1


def _ica_channel_indices(EEG: dict[str, Any], channel_count: int) -> np.ndarray:
    values = np.asarray(EEG.get("icachansind", []), dtype=int).ravel()
    if values.size == 0:
        return np.arange(channel_count, dtype=int)
    if np.any(values < 0) or np.any(values >= channel_count):
        raise ValueError(f"icachansind values must be within 0..{channel_count - 1}")
    if np.unique(values).size != values.size:
        raise ValueError("icachansind values must be unique")
    return values


def _selected_ica_channels(
    ica_chans: np.ndarray,
    channel_count: int,
    *,
    chans: Any | None,
    omitchans: Any | None,
) -> np.ndarray:
    if chans is not None and np.asarray(chans).size:
        requested = _one_based_indices(chans, channel_count, "channel") - 1
        positions = {int(channel): position for position, channel in enumerate(ica_chans.tolist())}
        missing = [index + 1 for index in requested if int(index) not in positions]
        if missing:
            raise ValueError("Selected channels must be present in EEG.icachansind")
        return np.asarray([positions[int(index)] for index in requested], dtype=int)
    selected = np.arange(ica_chans.size, dtype=int)
    if omitchans is None or np.asarray(omitchans).size == 0:
        return selected
    omitted = set((_one_based_indices(omitchans, channel_count, "channel") - 1).tolist())
    return np.asarray(
        [position for position, channel in enumerate(ica_chans.tolist()) if channel not in omitted], dtype=int
    )
