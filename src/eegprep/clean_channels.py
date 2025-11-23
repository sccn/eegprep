"""EEG channel cleaning utilities."""

from typing import *
import logging
import traceback

import numpy as np

from .utils.sigproc import design_fir, filtfilt_fast
from .utils.ransac import calc_projector
from .utils.stats import mad
from .utils.misc import round_mat

logger = logging.getLogger(__name__)


def clean_channels(
        EEG: Dict[str, Any],
        corr_threshold: float = 0.8,
        noise_threshold: float = 5.0,
        window_len: float = 5,
        max_broken_time: float = 0.4,
        num_samples: int = 50,
        subset_size: float = 0.25,
) -> Dict[str, Any]:
    """Remove channels with problematic data from a continuous data set.

    This is an automated artifact rejection function which ensures that the data contains no channels
    that record only noise for extended periods of time. If channels with control signals are
    contained in the data these are usually also removed. The criterion is based on correlation: if a
    channel has lower correlation to its robust estimate (based on other channels) than a given threshold
    for a minimum period of time (or percentage of the recording), it will be removed.

    Args:
      EEG: Continuous data set, assumed to be appropriately high-passed
        (e.g. >0.5Hz or with a 0.5Hz - 2.0Hz transition band).
      corr_threshold: Correlation threshold. If a channel is correlated at
        less than this value to its robust estimate (based on other channels),
        it is considered abnormal in the given time window.
      noise_threshold: If a channel has more (high-frequency) noise relative to its signal
        than this value, in standard deviations from the channel population mean,
        it is considered abnormal.
      window_len: Length of the windows (in seconds) for which correlation is computed; ideally
        short enough to reasonably capture periods of global artifacts or intermittent
        sensor dropouts, but not shorter (for statistical reasons).
      max_broken_time: Maximum time (either in seconds or as fraction of the recording)
        during which a channel is allowed to have artifacts. Reasonable range:
        0.1 (very aggressive) to 0.6 very lax).
      num_samples: Number of samples generated for a RANSAC reconstruction. This is the
        number of samples to generate in the random sampling consensus process. The larger
        this value, the more robust but also slower the processing will be.
      subset_size: Subset size. This is the size of the channel subsets to use
        for robust reconstruction,  as a number or fraction of the total number
        of channels.

    Returns:
      EEG: data set with bad channels removed
    """
    EEG['data'] = np.asarray(EEG['data'], dtype=np.float64)
    C, S = EEG['data'].shape
    Fs = EEG['srate']

    # handle fractions or absolute values
    if subset_size >= 1:
        subset_size = int(subset_size)
    else:
        subset_size = int(round_mat(C * subset_size))
    if max_broken_time < 1:
        max_broken_time = S * max_broken_time
    else:
        max_broken_time = round_mat(Fs) * max_broken_time

    window_len = int(window_len * round_mat(Fs))
    wnd = np.arange(int(window_len))
    offsets = np.arange(0, S - window_len, window_len, dtype=int)
    W = len(offsets)

    logger.info('Scanning for bad channels...')

    if Fs > 100:
        # remove signal content above 50Hz
        B = design_fir(100, 2 * np.array([0, 45, 50, Fs/2]) / Fs, [1, 1, 0, 0])
        X = np.zeros((S, C))
        for c in range(C):
            X[:, c] = filtfilt_fast(B, 1, EEG['data'][c, :])
        
        # determine z-scored level of EM noise-to-signal ratio for each channel
        noisiness = mad(EEG['data'].T - X) / mad(X)
        znoise = (noisiness - np.median(noisiness)) / (mad(noisiness) * 1.4826)
        
        # trim channels based on that
        noise_mask = znoise > noise_threshold
    else:
        X = EEG['data'].T
        noise_mask = np.zeros(C, dtype=bool)  # transpose added in MATLAB comment

    # get the matrix of all channel locations [3xN]
    xyz = [
        [ch.get(coord, np.nan) for ch in EEG['chanlocs']]
        for coord in ['X', 'Y', 'Z']]
    xyz = [[x if not (isinstance(x, np.ndarray) and x.size == 0) else np.nan for x in xyz_sub] for xyz_sub in xyz]
    xyz = np.asarray([np.asarray([np.nan if x is None else x for x in row], dtype=float) for row in xyz])
    if np.mean(np.any(np.isnan(xyz), axis=0)) > 0.5:
        raise ValueError(
            'To use this function most of your channels should have X,Y,Z location measurements.')
    usable_channels = np.where(~np.any(np.isnan(xyz), axis=0))[0]
    
    locs = xyz[:, usable_channels].T
    X = np.asarray(X[:, usable_channels])

    # replicate MATLAB's default randseed, for exact compatibility
    stream = np.random.RandomState(5489)
    P = np.asarray(calc_projector(locs, num_samples, subset_size, stream=stream))
    corrs = np.zeros((len(usable_channels), W))

    # calculate each channel's correlation to its RANSAC reconstruction for each window
    time_passed_list = np.zeros(W)
    for o in range(W):
        import time
        start_time = time.time()
        
        XX = X[offsets[o] + wnd, :]
        YY = np.sort(np.reshape((XX @ P).T, (num_samples, -1)), axis=0)
        YY = np.reshape(YY[int(round_mat(num_samples / 2)) - 1, :], (-1, window_len)).T

        # Calculate correlation for each channel
        for c in range(len(usable_channels)):
            numerator = np.sum(XX[:, c] * YY[:, c])
            denominator = np.sqrt(np.sum(XX[:, c]**2)) * np.sqrt(np.sum(YY[:, c]**2))
            corrs[c, o] = numerator / denominator
        
        time_passed_list[o] = time.time() - start_time
        median_time_passed = np.median(time_passed_list[:o+1])
        if o % 50 == 0:
            logger.info(f'{o+1:3d}/{W} blocks, {median_time_passed*(W-o-1)/60:.1f} minutes remaining.')

    flagged = corrs < corr_threshold
    
    # mark all channels for removal which have more flagged samples than the maximum number of
    # ignored samples
    removed_channels = np.zeros(C, dtype=bool)
    removed_channels[usable_channels] = np.sum(flagged, axis=1) * window_len > max_broken_time
    removed_channels = removed_channels | noise_mask
    
    # apply removal
    if np.mean(removed_channels) > 0.75:
        raise ValueError('More than 75% of your channels were removed -- this is probably caused by incorrect channel location measurements (e.g., wrong cap design).')
    elif np.any(removed_channels):
        try:
            from eegprep import pop_select
            EEG = pop_select(EEG, nochannel=list(np.where(removed_channels)[0]))
        except Exception as e:
            if isinstance(e, ImportError):
                logger.error("Apparently you do not have EEGLAB's pop_select() on the path.")
            else:
                logger.error("Could not select channels using EEGLAB's pop_select(); details: %s", str(e))
                logger.debug("Exception traceback:", exc_info=True)
            
            logger.info(f'Removing {np.sum(removed_channels)} channels and dropping signal meta-data.')
            if len(EEG['chanlocs']) == EEG['data'].shape[0]:
                EEG['chanlocs'] = np.asarray([ch for i, ch in enumerate(EEG['chanlocs']) if not removed_channels[i]])
            # pop_select() by default truncates the data to float32, so we need to do the same
            EEG['data'] = np.asarray(EEG['data'], dtype=np.float32)
            EEG['data'] = EEG['data'][~removed_channels, :]
            EEG['nbchan'] = EEG['data'].shape[0]
            
            # Clear other fields
            for field in ['icawinv', 'icasphere', 'icaweights', 'icaact', 'stats', 'specdata', 'specicaact']:
                if field in EEG:
                    EEG[field] = np.array([])
        
        # Update clean_channel_mask
        if 'etc' in EEG and 'clean_channel_mask' in EEG['etc']:
            EEG['etc']['clean_channel_mask'][EEG['etc']['clean_channel_mask']] = ~removed_channels
        else:
            if 'etc' not in EEG:
                EEG['etc'] = {}
            EEG['etc']['clean_channel_mask'] = ~removed_channels
    
    return EEG
