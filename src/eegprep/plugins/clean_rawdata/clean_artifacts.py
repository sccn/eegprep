"""EEG artifact cleaning functions."""

import copy
import logging
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

# Local imports from the eegprep package
from .clean_flatlines import clean_flatlines
from .clean_drifts import clean_drifts
from .clean_channels import clean_channels
from .clean_channels_nolocs import clean_channels_nolocs
from .clean_asr import clean_asr
from .clean_windows import clean_windows
from .private.masks import mask_to_intervals
from ...functions.miscfunc.misc import round_mat
from ...functions.popfunc.eeg_eegrej import eeg_eegrej


logger = logging.getLogger(__name__)
_DISTANCE_MODES = {
    'euclidian': None,
    'euclidean': None,
    'riemannian': 'calib',
}

# -----------------------------------------------------------------------------
#                               Public API
# -----------------------------------------------------------------------------


def clean_artifacts(
    EEG: Dict[str, Any],
    # Core parameters
    ChannelCriterion: Union[float, str, None] = 0.8,
    LineNoiseCriterion: Union[float, str, None] = 4.0,
    BurstCriterion: Union[float, str, None] = 5.0,
    WindowCriterion: Union[float, str, None] = 0.25,
    Highpass: Union[Tuple[float, float], str, None] = (0.25, 0.75),
    # Detail parameters
    ChannelCriterionMaxBadTime: float = 0.5,
    BurstCriterionRefMaxBadChns: Union[float, str, None] = 0.075,
    BurstCriterionRefTolerances: Union[Tuple[float, float], str, None] = (-np.inf, 5.5),
    BurstRejection: bool = False,
    WindowCriterionTolerances: Union[Tuple[float, float], str, None] = (-np.inf, 7),
    FlatlineCriterion: Union[float, str, None] = 5.0,
    NumSamples: int = 50,
    SubsetSize: float = 0.25,
    NoLocsChannelCriterion: float = 0.45,
    NoLocsChannelCriterionExcluded: float = 0.1,
    MaxMem: int = 64,
    Distance: str = 'euclidian',
    # Misc.
    Channels: Optional[Sequence[str]] = None,
    Channels_ignore: Optional[Sequence[str]] = None,
    availableRAM_GB: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], np.ndarray]:
    """All-in-one artifact removal, port of MATLAB clean_artifacts.

    Removes flatline channels, low-frequency drifts, noisy channels, short-time bursts,
    and irrecoverable windows in sequence. Core parameters can be passed as None or 'off'
    to use defaults or disable stages.

    Parameters
    ----------
    EEG : dict
        Raw continuous EEG dataset dict (must include 'data', 'srate', 'chanlocs', etc.).
    ChannelCriterion : float or 'off'
        Minimum channel correlation threshold for channel cleaning; channels below
        this value are considered bad. Pass 'off' to skip channel criterion. Default 0.8.
    LineNoiseCriterion : float or 'off'
        Z-score threshold for line-noise contamination; channels exceeding this are
        considered bad. 'off' disables line-noise check. Default 4.0.
    BurstCriterion : float or 'off'
        ASR standard-deviation cutoff for high-amplitude bursts; values above this
        relative to calibration data are repaired (or removed if BurstRejection='on').
        'off' skips ASR. Default 5.0.
    WindowCriterion : float or 'off'
        Fraction (0-1) or count of channels allowed to be bad per window; windows with
        more bad channels are removed. 'off' disables final window removal. Default 0.25.
    Highpass : tuple(float, float) or 'off'
        Transition band [low, high] in Hz for initial high-pass filtering. 'off' skips
        drift removal. Default (0.25, 0.75).
    ChannelCriterionMaxBadTime : float
        Maximum tolerated time (seconds or fraction of recording) a channel may be flagged
        bad before being removed. Default 0.5.
    BurstCriterionRefMaxBadChns : float or 'off'
        Maximum fraction of bad channels tolerated when selecting calibration data for ASR.
        'off' uses all data for calibration. Default 0.075.
    BurstCriterionRefTolerances : tuple(float, float) or 'off'
        Power Z-score tolerances for selecting calibration windows in ASR. 'off' uses
        all data. Default (-inf, 5.5).
    BurstRejection : bool
        'on' to reject (drop) burst segments instead of reconstructing with ASR,
        'off' to apply ASR repair. Default 'off'.
    WindowCriterionTolerances : tuple(float, float) or 'off'
        Power Z-score bounds for final window removal. 'off' disables this stage.
        Default (-inf, 7).
    FlatlineCriterion : float or 'off'
        Maximum flatline duration in seconds; channels exceeding this are removed.
        'off' disables flatline removal. Default 5.0.
    NumSamples : int
        Number of RANSAC samples for channel cleaning. Default 50.
    SubsetSize : float
        Size of channel subsets for RANSAC, as fraction (0-1) or count. Default 0.25.
    NoLocsChannelCriterion : float
        Correlation threshold for fallback channel cleaning when no channel locations.
        Default 0.45.
    NoLocsChannelCriterionExcluded : float
        Fraction of channels excluded when assessing correlation in nolocs cleaning.
        Default 0.1.
    MaxMem : int
        Maximum memory in MB for ASR processing. Default 64.
    Distance : str
        Distance metric for ASR calibration ('euclidian' or 'riemannian'). The
        Riemannian path uses EEGPrep's calibration-time estimate; full
        Riemannian ASR processing is not ported.
    Channels : sequence of str or None
        List of channel labels to include before cleaning (pop_select). The
        returned dataset contains only these channels; channels outside this list
        are dropped and not re-inserted. Default None.
    Channels_ignore : sequence of str or None
        List of channel labels to exclude before cleaning. The excluded channels
        are dropped from the returned dataset and not re-inserted. Default None.
    availableRAM_GB : float or None
        Available system RAM in GB to adjust MaxMem. Default None.

    Returns
    -------
    EEG : dict
        Final cleaned EEG dataset.
    HP : dict
        EEG dataset after initial high-pass (drift removal).
    BUR : dict
        EEG dataset after ASR burst repair (before final window removal).
    removed_channels : ndarray of bool
        Mask indicating which channels were removed during cleaning.
    """
    # ------------------------------------------------------------------
    #                Basic argument sanity / aliases
    # ------------------------------------------------------------------
    if availableRAM_GB is not None and not np.isnan(availableRAM_GB):
        MaxMem = int(round_mat(availableRAM_GB * 1000))

    if Channels is not None and Channels_ignore is not None and len(Channels) and len(Channels_ignore):
        raise ValueError('"Channels" and "Channels_ignore" are mutually exclusive – supply at most one.')

    distance = str(Distance).strip().lower()
    if distance not in _DISTANCE_MODES:
        raise ValueError("Distance must be 'euclidian', 'euclidean', or 'riemannian'")

    # Ensure some obligatory fields exist in the structure (MATLAB code assumes)
    if 'etc' not in EEG:
        EEG['etc'] = {}

    # ------------------------------------------------------------------
    #             Optional: restrict to / ignore certain channels
    # ------------------------------------------------------------------
    if Channels is not None and len(Channels):
        # Attempt pop_select based on labels; the manual fallback only covers the
        # documented case where pop_select is unavailable (ImportError). Errors
        # raised inside pop_select itself must surface, not be masked.
        try:
            from eegprep import pop_select

            EEG = pop_select(EEG, channel=list(Channels))
        except ImportError:
            lbl_to_idx = {ch['labels']: idx for idx, ch in enumerate(EEG['chanlocs'])}
            keep_idx = [lbl_to_idx[lbl] for lbl in Channels if lbl in lbl_to_idx]
            EEG['data'] = EEG['data'][keep_idx, :]
            EEG['chanlocs'] = [EEG['chanlocs'][i] for i in keep_idx]
            EEG['nbchan'] = len(keep_idx)
    elif Channels_ignore is not None and len(Channels_ignore):
        try:
            from eegprep import pop_select

            EEG = pop_select(EEG, nochannel=list(Channels_ignore))
        except ImportError:
            lbl_to_idx = {ch['labels']: idx for idx, ch in enumerate(EEG['chanlocs'])}
            drop_idx_set = {lbl_to_idx[lbl] for lbl in Channels_ignore if lbl in lbl_to_idx}
            keep_idx = [i for i in range(len(EEG['chanlocs'])) if i not in drop_idx_set]
            EEG['data'] = EEG['data'][keep_idx, :]
            EEG['chanlocs'] = [EEG['chanlocs'][i] for i in keep_idx]
            EEG['nbchan'] = len(keep_idx)

    # ------------------------------------------------------------------
    #                     1) Flat‑line channel removal
    # ------------------------------------------------------------------
    if FlatlineCriterion not in (None, 'off'):
        logger.info('Detecting flat line channels...')
        EEG = clean_flatlines(EEG, max_flatline_duration=float(FlatlineCriterion))

    # ------------------------------------------------------------------
    #                        2) High‑pass filtering
    # ------------------------------------------------------------------
    if Highpass not in (None, 'off'):
        if not isinstance(Highpass, (tuple, list)) or len(Highpass) != 2:
            raise ValueError('Highpass must be a (low, high) tuple or None/"off".')
        logger.info('Applying high‑pass filter...')
        EEG = clean_drifts(EEG, tuple(Highpass))
    # Keep a point-in-time snapshot after HP for optional return. Deep-copy so
    # later stages that mutate EEG['etc'] (channel/sample masks) do not bleed
    # into the returned high-pass dataset.
    HP = copy.deepcopy(EEG)

    # ------------------------------------------------------------------
    #            3) Channel cleaning (noisy / disconnected)
    # ------------------------------------------------------------------
    removed_channels = np.zeros(EEG['nbchan'], dtype=bool)
    if ChannelCriterion not in (None, 'off') or LineNoiseCriterion not in (None, 'off'):
        chancorr_crit = 0.0 if ChannelCriterion in (None, 'off') else float(ChannelCriterion)
        line_crit = 100.0 if LineNoiseCriterion in (None, 'off') else float(LineNoiseCriterion)
        try:
            # Match MATLAB clean_artifacts.m line 254:
            # clean_channels(EEG,chancorr_crit,line_crit,[],channel_crit_maxbad_time, num_samples)
            # MATLAB passes [] for window_len (uses default 5) and doesn't pass subset_size (uses default 0.25)
            # Python: not passing window_len uses default 5, and we pass subset_size explicitly (default 0.25)
            EEG = clean_channels(
                EEG,
                corr_threshold=chancorr_crit,
                noise_threshold=line_crit,
                # window_len not passed - uses default 5 (MATLAB passes [] which also uses default 5)
                max_broken_time=float(ChannelCriterionMaxBadTime),
                num_samples=int(NumSamples),
                subset_size=SubsetSize,  # Default 0.25, matches MATLAB default when not passed
            )
            # clean_channels only writes clean_channel_mask when it removes channels;
            # an absent mask means nothing was removed, so keep the all-False default.
            mask = EEG.get('etc', {}).get('clean_channel_mask')
            if mask is not None:
                removed_channels = ~mask
        except ValueError as e:
            # Only the missing-channel-locations case warrants the no-locs fallback;
            # any other ValueError is a genuine failure and must propagate.
            if 'location' not in str(e).lower():
                raise
            logger.warning(f'clean_channels lacks usable locations ({e}); falling back to clean_channels_nolocs.')
            EEG, removed_channels = clean_channels_nolocs(
                EEG,
                min_corr=float(NoLocsChannelCriterion),
                ignored_quantile=float(NoLocsChannelCriterionExcluded),
                window_len=2.0,
                max_broken_time=float(ChannelCriterionMaxBadTime),
            )

    # ------------------------------------------------------------------
    #                     4) Burst repair via ASR
    # ------------------------------------------------------------------
    BUR = EEG  # default in case ASR is skipped
    if BurstCriterion not in (None, 'off'):
        logger.info('Applying ASR burst repair...')
        # Snapshot the pre-repair data to compare against the ASR-repaired
        # result; clean_asr returns a fresh dataset (BUR) and leaves EEG intact.
        original_data = EEG['data'].copy() if BurstRejection else None
        useriemannian = _DISTANCE_MODES[distance]
        BUR = clean_asr(
            EEG,
            cutoff=float(BurstCriterion),
            ref_maxbadchannels=BurstCriterionRefMaxBadChns,
            ref_tolerances=BurstCriterionRefTolerances,
            use_gpu=False,
            useriemannian=useriemannian,
            maxmem=int(MaxMem),
        )

        if BurstRejection:
            # Determine unchanged samples: compare the pre-repair snapshot with
            # the ASR-repaired data returned in BUR.
            sample_mask = np.sum(np.abs(original_data - BUR['data']), axis=0) < 1e-8
            del original_data
            # Convert retained samples to inclusive zero-based intervals.
            retain_intervals = mask_to_intervals(sample_mask, value=True) - 1

            # Remove very short intervals < 5 samples
            if retain_intervals.size:
                lengths = retain_intervals[:, 1] - retain_intervals[:, 0]
                small = lengths < 5
                for s, e in retain_intervals[small]:
                    sample_mask[s : e + 1] = False
                retain_intervals = retain_intervals[~small]

            # Reject bad periods from the ASR-repaired dataset (BUR).
            EEG = BUR
            rejected_intervals = mask_to_intervals(sample_mask, value=False)
            if rejected_intervals.size:
                EEG = eeg_eegrej(EEG, rejected_intervals)

            # Update mask in EEG.etc
            EEG['etc']['clean_sample_mask'] = sample_mask
        else:
            EEG = BUR

    # ------------------------------------------------------------------
    #                     5) Post‑clean windows stage
    # ------------------------------------------------------------------
    if WindowCriterion not in (None, 'off') and WindowCriterionTolerances not in (None, 'off'):
        logger.info('Final post‑processing – removing irrecoverable windows...')
        EEG, _ = clean_windows(
            EEG,
            max_bad_channels=float(WindowCriterion),
            zthresholds=WindowCriterionTolerances,
        )

    logger.info('Use vis_artifacts to compare the cleaned data to the original.')

    # When Channels/Channels_ignore restrict the dataset, the returned EEG holds
    # only the cleaned subset; excluded channels are not re-inserted (unlike the
    # MATLAB reference). Callers that need the ignored channels back must merge
    # them manually. This is documented on the Channels/Channels_ignore parameters.

    return EEG, HP, BUR, removed_channels
