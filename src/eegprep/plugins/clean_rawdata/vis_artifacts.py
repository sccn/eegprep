"""Diagnostics and browser display for clean_rawdata artifacts."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from eegprep.functions.sigprocfunc.eegplot import eegplot
from eegprep.plugins.clean_rawdata.private.masks import mask_to_intervals


logger = logging.getLogger(__name__)

CLEAN_RAWDATA_REJECT_COLOR = (1.0, 0.95, 0.25)


def vis_artifacts(clean_eeg: dict[str, Any], original_eeg: dict[str, Any] | None = None, *, show: bool = True) -> Any:
    """Compare a cleaned dataset against its original clean_rawdata source.

    Args:
        clean_eeg: Cleaned EEG dictionary, usually returned by ``clean_artifacts``.
        original_eeg: Original EEG dictionary. Defaults to ``clean_eeg`` when omitted.
        show: Open an ``eegplot`` browser when true; otherwise return diagnostics only.

    Returns:
        Diagnostics when ``show`` is false. When ``show`` is true, returns the
        ``eegplot`` window/model object if it can be opened, otherwise ``None``.
    """
    diagnostics = vis_artifacts_diagnostics(clean_eeg, original_eeg)
    if not show:
        return diagnostics

    source = diagnostics["original_eeg"]
    try:
        return eegplot(
            source,
            winrej=diagnostics["winrej"],
            wincolor=CLEAN_RAWDATA_REJECT_COLOR,
            events=source.get("event", []),
            xgrid="off",
            title=f"Rejected data highlighted -- eegplot() -- {source.get('setname', '')}".rstrip(),
        )
    except RuntimeError as exc:
        logger.info("Could not open clean_rawdata rejected-data browser: %s", exc)
        return None


def vis_artifacts_diagnostics(
    clean_eeg: dict[str, Any],
    original_eeg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return clean_rawdata sample/channel rejection diagnostics."""
    source = original_eeg if original_eeg is not None else clean_eeg
    raw_sample_mask = _mask(clean_eeg, "clean_sample_mask")
    raw_channel_mask = _mask(clean_eeg, "clean_channel_mask")
    original_samples = _pnts(source)
    if original_eeg is None and raw_sample_mask.size:
        original_samples = max(original_samples, raw_sample_mask.size)
    clean_samples = _pnts(clean_eeg)
    original_channels = _nbchan(source)
    if original_eeg is None and raw_channel_mask.size:
        original_channels = max(original_channels, raw_channel_mask.size)
    clean_channels = _nbchan(clean_eeg)
    sample_mask = _sample_mask(raw_sample_mask, original_samples)
    rejected_intervals = mask_to_intervals(sample_mask, value=False)
    rejected_sample_count = int(np.count_nonzero(~sample_mask))
    channel_mask = _channel_mask(raw_channel_mask, original_channels)
    removed_indices = [index + 1 for index, keep in enumerate(channel_mask) if not keep]
    removed_labels = _removed_channel_labels(source, removed_indices)
    return {
        "original_eeg": source,
        "clean_eeg": clean_eeg,
        "original_samples": original_samples,
        "clean_samples": clean_samples,
        "original_channels": original_channels,
        "clean_channels": clean_channels,
        "sample_mask": sample_mask,
        "rejected_intervals": rejected_intervals,
        "rejected_sample_count": rejected_sample_count,
        "rejected_fraction": rejected_sample_count / original_samples if original_samples else 0.0,
        "channel_mask": channel_mask,
        "removed_channel_indices": removed_indices,
        "removed_channel_labels": removed_labels,
        "removed_channel_count": len(removed_indices),
        "winrej": clean_rawdata_winrej(rejected_intervals, original_channels),
    }


def clean_rawdata_winrej(rejected_intervals: np.ndarray, nbchan: int) -> np.ndarray:
    """Convert rejected intervals to an EEGLAB ``eegplot`` ``winrej`` matrix."""
    rejected = np.asarray(rejected_intervals, dtype=float)
    if rejected.size == 0:
        return np.zeros((0, 5 + int(nbchan)), dtype=float)
    rows = np.zeros((rejected.shape[0], 5 + int(nbchan)), dtype=float)
    rows[:, 0:2] = rejected[:, 0:2]
    rows[:, 2:5] = np.asarray(CLEAN_RAWDATA_REJECT_COLOR, dtype=float)
    rows[:, 5:] = 1
    return rows


def _mask(clean_eeg: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray((clean_eeg.get("etc") or {}).get(key, []), dtype=bool).ravel()


def _sample_mask(mask: np.ndarray, original_samples: int) -> np.ndarray:
    if mask.size == original_samples:
        return mask.copy()
    return np.ones(original_samples, dtype=bool)


def _channel_mask(mask: np.ndarray, original_channels: int) -> np.ndarray:
    if mask.size == original_channels:
        return mask.copy()
    return np.ones(original_channels, dtype=bool)


def _removed_channel_labels(source: dict[str, Any], removed_indices: list[int]) -> list[str]:
    chanlocs = _chanlocs(source)
    labels = []
    for index in removed_indices:
        if 1 <= index <= len(chanlocs):
            labels.append(str(chanlocs[index - 1].get("labels", index)))
        else:
            labels.append(str(index))
    return labels


def _pnts(eeg: dict[str, Any]) -> int:
    if eeg.get("pnts") is not None:
        return int(eeg["pnts"])
    data = np.asarray(eeg.get("data"))
    return int(data.shape[1]) if data.ndim >= 2 else 0


def _nbchan(eeg: dict[str, Any]) -> int:
    if eeg.get("nbchan") is not None:
        return int(eeg["nbchan"])
    data = np.asarray(eeg.get("data"))
    return int(data.shape[0]) if data.ndim >= 2 else 0


def _chanlocs(eeg: dict[str, Any]) -> list[dict[str, Any]]:
    chanlocs = eeg.get("chanlocs", [])
    if isinstance(chanlocs, np.ndarray):
        chanlocs = chanlocs.tolist()
    if isinstance(chanlocs, dict):
        chanlocs = [chanlocs]
    return [chan if isinstance(chan, dict) else {} for chan in list(chanlocs or [])]
