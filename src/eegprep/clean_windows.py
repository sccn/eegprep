"""EEG data window cleaning utilities.

This module provides functions for removing periods with abnormally high-power
content from continuous EEG data.
"""

import warnings
import logging
from typing import *

import numpy as np

from .utils.stats import fit_eeg_distribution
from .utils.misc import round_mat

logger = logging.getLogger(__name__)


def clean_windows(
        EEG: Dict[str, Any],
        max_bad_channels: Union[int, float] = 0.2,
        zthresholds: Tuple[float, float] = (-3.5, 5),
        window_len: float = 1.0,
        window_overlap: float = 0.66,
        max_dropout_fraction: float = 0.1,
        min_clean_fraction: float = 0.25,
        truncate_quant: Tuple[float, float] = (0.022, 0.6),
        step_sizes: Tuple[float, float] = (0.01, 0.01),
        shape_range: Union[np.ndarray, Sequence[float]] = np.arange(1.7, 3.6, 0.15),
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Remove periods with abnormally high-power content from continuous data.

    This function cuts segments from the data which contain high-power artifacts.
    Specifically, only windows are retained which have less than a certain
    fraction of *bad* channels, where a channel is bad in a window if its RMS
    power is above or below some *z*-threshold relative to a robust estimate
    of clean EEG power in that channel.

    Args
    ----
    EEG : dict
        Continuous dataset using the EEGLAB dict schema. The data is
        expected to be high-passed appropriately (>1 Hz recommended).
    max_bad_channels : int | float
        The maximum number **or** fraction of channels that may exceed the
        thresholds inside a time-window for the window to be kept. Values in
        (0,1) are interpreted as a fraction; otherwise as an absolute count.
    zthresholds : tuple(float, float)
        Lower and upper *z*-score limits for RMS power ([low, high]).
    window_len : float
        Window length in seconds. Should be at least half a period of the high-
        pass cut-off that was used. Default is 1 s.
    window_overlap : float
        Fractional overlap between consecutive windows (0-1). Higher overlap
        finds more artefacts but is slower. Default is 0.66 (≈⅔ overlap).
    max_dropout_fraction : float
        Maximum fraction of windows that may have arbitrarily low amplitude
        (e.g. sensor unplugged). Default is 0.1.
    min_clean_fraction : float
        Minimum fraction of windows expected to be clean (essentially
        uncontaminated EEG). Default is 0.25.
    truncate_quant : tuple(float, float)
        Quantile range of the truncated Gaussian to fit (default (0.022,0.6)).
    step_sizes : tuple(float, float)
        Grid-search step sizes in quantiles for lower/upper edge.
    shape_range : sequence(float)
        Range for the *beta* shape parameter in the generalised Gaussian used
        for distribution fitting.

    Returns
    -------
    EEG : dict
        The passed-in structure with bad time periods excised.
    sample_mask : np.ndarray[bool]
        Boolean mask (length == original ``pnts``) indicating which samples are
        retained (``True``) or removed (``False``).
    """
    # ------------------------------------------------------------------
    #                           Input handling
    # ------------------------------------------------------------------
    EEG['data'] = np.asarray(EEG['data'], dtype=np.float64)
    C, S = EEG['data'].shape
    Fs = EEG['srate']

    # Convert fractional parameters to absolute where necessary
    if C == 0 or S == 0:
        raise ValueError('Empty data array encountered.')

    if max_bad_channels is not None and 0 < max_bad_channels < 1:
        max_bad_channels = int(round_mat(C * max_bad_channels))
    else:
        max_bad_channels = int(max_bad_channels)

    shape_range = np.asarray(shape_range)

    # ------------------------------------------------------------------
    #                    Prepare window indexing helpers
    # ------------------------------------------------------------------
    N = int(round_mat(window_len * Fs))  # samples per window
    if N <= 0:
        raise ValueError('Window length too small - results in N <= 0.')

    # MATLAB: offsets = round(1:N*(1-window_overlap):S-N)
    step = N * (1.0 - window_overlap)
    if step <= 0:
        # Avoid infinite loop when overlap >= 1
        step = 1.0
    offsets = round_mat(np.arange(0, S - N + 1, step)).astype(int)
    if len(offsets) == 0:
        raise ValueError('Not enough data for even a single window.')

    wnd = np.arange(N, dtype=int)
    W = len(offsets)

    logger.info('Determining time window rejection thresholds...')

    # ------------------------------------------------------------------
    #                      Compute z-score per channel
    # ------------------------------------------------------------------
    wz = np.zeros((C, W), dtype=float)
    for c in reversed(range(C)):
        # compute RMS amplitude for each window
        Xsq = EEG['data'][c, :] ** 2  # power
        # Gather samples for all windows using broadcasting (W, N)
        indices = offsets[:, None] + wnd[None, :]
        # Extract data and compute RMS per window
        rms = np.sqrt(np.sum(Xsq[indices], axis=1) / N)

        # Fit distribution to clean EEG portion
        mu, sig, *_ = fit_eeg_distribution(
            rms,
            min_clean_fraction=min_clean_fraction,
            max_dropout_fraction=max_dropout_fraction,
            quants=truncate_quant,
            step_sizes=step_sizes,
            beta=shape_range,
        )
        if sig == 0 or np.isnan(sig):
            # Fallback to robust MAD if fitting failed
            sig = np.median(np.abs(rms - np.median(rms))) * 1.4826
            mu = np.median(rms)
            if sig == 0:
                sig = 1.0  # avoid division by zero

        # z-score relative to fitted distribution
        wz[c, :] = (rms - mu) / sig
    logger.info('done.')

    # ------------------------------------------------------------------
    #                Identify windows to be removed/kept
    # ------------------------------------------------------------------
    swz = np.sort(wz, axis=0)  # sort each column (ascending)

    remove_mask = np.zeros(W, dtype=bool)
    zmin, zmax = zthresholds
    if zmax > 0:
        # upper threshold – check the (max_bad_channels+1)-th largest value
        idx_hi = C - max_bad_channels - 1  # zero-based index
        idx_hi = max(min(idx_hi, C - 1), 0)
        remove_mask |= swz[idx_hi, :] > zmax
    if zmin < 0:
        # lower threshold – check the (max_bad_channels+1)-th smallest value
        idx_lo = max_bad_channels
        idx_lo = max(min(idx_lo, C - 1), 0)
        remove_mask |= swz[idx_lo, :] < zmin

    removed_windows = np.where(remove_mask)[0]

    # ------------------------------------------------------------------
    #                Convert window removals to sample mask
    # ------------------------------------------------------------------
    sample_mask = np.ones(S, dtype=bool)
    for w in removed_windows:
        start = offsets[w]
        sample_mask[start:start + N] = False

    kept_pct = 100.0 * np.mean(sample_mask)
    kept_seconds = np.count_nonzero(sample_mask) / Fs
    logger.info(f'Keeping {kept_pct:.1f}% ({kept_seconds:.0f} seconds) of the data.')

    # ------------------------------------------------------------------
    #                    Determine retain intervals (inclusive)
    # ------------------------------------------------------------------
    padded = np.concatenate([[False], sample_mask, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    # assuming that pop-select will accept 1-based intervals for point
    retain_intervals = np.stack([starts, ends], axis=1) + 1  # shape (K,2)

    # ------------------------------------------------------------------
    #               Apply selection (pop_select if available)
    # ------------------------------------------------------------------
    try:
        from eegprep import pop_select  # type: ignore
        EEG = pop_select(EEG, point=retain_intervals)
        logger.warning("This call to pop_select() assumes that time intervals use "
                      "1-based indexing; if this has been verified, please remove this warning.")
    except Exception as e:  # noqa: BLE001 – we really want to catch *everything*
        # Fall back to manual trimming and minimal bookkeeping
        if isinstance(e, ImportError):
            logger.error("Apparently you do not have EEGLAB's pop_select() on the path.")
        else:
            logger.error('Could not select time windows using EEGLAB\'s pop_select(); details: %s', str(e))
            logger.debug('Exception traceback:', exc_info=True)

        logger.info('Falling back to a basic substitute and dropping signal meta-data.')
        # pop_select() by default truncates to single precision in EEGLAB, which we're mirroring here
        EEG['data'] = np.asarray(EEG['data'], dtype=np.float32)
        EEG['data'] = EEG['data'][:, sample_mask]
        EEG['pnts'] = EEG['data'].shape[1]
        EEG['xmax'] = EEG['xmin'] + (EEG['pnts'] - 1) / Fs
        # Wipe or reset fields that are now inconsistent
        for fld in ['event', 'urevent', 'epoch', 'icaact', 'reject',
                    'stats', 'specdata', 'specicaact']:
            if fld in EEG:
                EEG[fld] = [] if isinstance(EEG[fld], list) else np.array([])

    # ------------------------------------------------------------------
    #                     Update/insert clean_sample_mask
    # ------------------------------------------------------------------
    if 'etc' not in EEG:
        EEG['etc'] = {}

    etc = EEG['etc']
    if 'clean_sample_mask' in etc:
        prev_mask = np.asarray(etc['clean_sample_mask']).astype(bool)
        one_inds = np.where(prev_mask)[0]
        if len(one_inds) == len(sample_mask):
            prev_mask[one_inds] = sample_mask
            etc['clean_sample_mask'] = prev_mask
        else:
            logger.warning('EEG.etc.clean_sample is present but incompatible; it is being overwritten.')
            etc['clean_sample_mask'] = sample_mask
    else:
        etc['clean_sample_mask'] = sample_mask

    return EEG, sample_mask