import os
import contextlib
import multiprocessing
from time import time as now
import logging
from typing import Union, Tuple, Optional, Sequence

import numpy as np

from . import eeg_checkset
from .utils import ExceptionUnlessDebug, num_jobs_from_reservation, is_debug, humanize_seconds, num_cpus_from_reservation


logger = logging.getLogger(__name__)

# list of valid file extensions for raw EEG data files in BIDS format
eeg_extensions = ('.vhdr', '.edf', '.bdf', '.set')


def bids_preproc(
        root: str,
        *,
        # Overall run configuration
        AsDerivative: str = 'eegprep',
        SkipIfPresent: bool = True,
        ReservePerJob: str = '',
        # Overall processing parameters
        SamplingRate: Optional[float] = None,
        WithPicard: bool = False,
        WithICLabel: bool = False,
        # Core cleaning parameters
        ChannelCriterion: Union[float, str] = 0.8,
        LineNoiseCriterion: Union[float, str] = 4.0,
        BurstCriterion: Union[float, str] = 5.0,
        WindowCriterion: Union[float, str] = 0.25,
        Highpass: Union[str, Tuple[float, float]] = (0.25, 0.75),
        # Detail cleaning parameters
        ChannelCriterionMaxBadTime: float = 0.5,
        BurstCriterionRefMaxBadChns: Union[float, str] = 0.075,
        BurstCriterionRefTolerances: Union[Tuple[float, float], str] = (-np.inf, 5.5),
        BurstRejection: str = 'off',
        WindowCriterionTolerances: Union[Tuple[float, float], str] = (-np.inf, 7),
        FlatlineCriterion: Union[float, str] = 5.0,
        NumSamples: int = 50,
        NoLocsChannelCriterion: float = 0.45,
        NoLocsChannelCriterionExcluded: float = 0.1,
        MaxMem: int = 64,
        Distance: str = 'euclidian',
        # Misc cleaning config
        Channels: Optional[Sequence[str]] = None,
        Channels_ignore: Optional[Sequence[str]] = None,
        availableRAM_GB: Optional[float] = None,
        # BIDS loader parameters
        ApplyMetadata: bool = True,
        ApplyChanlocs: bool = True,
        ApplyEvents: bool = False,
        # Derived data parameters
        StageNames: Sequence[str] = ('desc-cleaned', 'desc-picard', 'desc-iclabel'),
        MinimizeDiskUsage: bool = True,
        # Reserved parameters
        _lock: Optional[multiprocessing.Lock] = contextlib.nullcontext(),
        _n_skipped: Optional[multiprocessing.Value] = None,
        _k: int = 0,
        _n_total: int = 1,
        _n_jobs: int = 1,
        _t0: float = now(),
) -> None:
    """
    Apply data cleaning to EEG files in a BIDS dataset.

    Parameters:
    -----------
    root_or_fn : str
        The root directory containing BIDS data or a single EEG file path.

    AsDerivative (str):
      The name of the subdirectory where cleaned files will be saved.
    SkipIfPresent (bool):
      skip processing files that already have a cleaned version present.
    ReservePerJob (str):
      Optionally the resource amount and type to reserve per job, e.g. '4GB' or '2CPU';
      the run will then use as many jobs as possible without exceeding the available resources.
      - Can also contain a total or percentage margin, as in '4GB-10GB', '2CPU-10%'.
      - Can also be specified as a total/maximum, as in '10 total' or '10max'.
      - Can also be a comma-separated list of reservations, e.g. '4GB,2CPU-1CPU,5max'.
      - if not set, will assume a single job. Generally runs serially when in debug mode.
      It is recommended to check in a serial run how much peak RAM a single job takes,
      and then sizing this to 1CPU,<N>GB-5GB or some other margin of your choice.

    SamplingRate (float):
        Desired sampling rate for the preprocessed data. If not specified, will retain
        the original sampling rate.
    WithPicard (bool):
        Whether to apply PICARD ICA decomposition after cleaning.
    WithICLabel (bool):
        Whether to apply ICLabel classification after ICA. Normally requires
        WithPicard=True.

    (below are the parameters for clean_artifacts function)
    ChannelCriterion (float or 'off'):
        Minimum channel correlation threshold for channel cleaning; channels below
        this value are considered bad. Pass 'off' to skip channel criterion. Default 0.8.
    LineNoiseCriterion (float or 'off'):
        Z-score threshold for line-noise contamination; channels exceeding this are
        considered bad. 'off' disables line-noise check. Default 4.0.
    BurstCriterion (float or 'off'):
        ASR standard-deviation cutoff for high-amplitude bursts; values above this
        relative to calibration data are repaired (or removed if BurstRejection='on').
        'off' skips ASR. Default 5.0.
    WindowCriterion (float or 'off'):
        Fraction (0-1) or count of channels allowed to be bad per window; windows with
        more bad channels are removed. 'off' disables final window removal. Default 0.25.
    Highpass (tuple(float, float) or 'off'):
        Transition band [low, high] in Hz for initial high-pass filtering. 'off' skips
        drift removal. Default (0.25, 0.75).
    ChannelCriterionMaxBadTime (float):
        Maximum tolerated time (seconds or fraction of recording) a channel may be flagged
        bad before being removed. Default 0.5.
    BurstCriterionRefMaxBadChns (float or 'off'):
        Maximum fraction of bad channels tolerated when selecting calibration data for ASR.
        'off' uses all data for calibration. Default 0.075.
    BurstCriterionRefTolerances (tuple(float, float) or 'off'):
        Power Z-score tolerances for selecting calibration windows in ASR. 'off' uses
        all data. Default (-inf, 5.5).
    BurstRejection (str):
        'on' to reject (drop) burst segments instead of reconstructing with ASR,
        'off' to apply ASR repair. Default 'off'.
    WindowCriterionTolerances (tuple(float, float) or 'off'):
        Power Z-score bounds for final window removal. 'off' disables this stage.
        Default (-inf, 7).
    FlatlineCriterion (float or 'off'):
        Maximum flatline duration in seconds; channels exceeding this are removed.
        'off' disables flatline removal. Default 5.0.
    NumSamples (int):
        Number of RANSAC samples for channel cleaning. Default 50.
    NoLocsChannelCriterion (float):
        Correlation threshold for fallback channel cleaning when no channel locations.
        Default 0.45.
    NoLocsChannelCriterionExcluded (float):
        Fraction of channels excluded when assessing correlation in nolocs cleaning.
        Default 0.1.
    MaxMem (int):
        Maximum memory in MB for ASR processing. Default 64.
    Distance (str):
        Distance metric for ASR processing ('euclidian'). Default 'euclidian'.
    Channels (Sequence[str] or None):
        List of channel labels to include before cleaning (pop_select). Default None.
    Channels_ignore (Sequence[str] or None):
        List of channel labels to exclude before cleaning. Default None.
    availableRAM_GB (float or None):
        Available system RAM in GB to adjust MaxMem. Default None.

    (parameters specific to the BIDS loading routine)
    ApplyMetadata (bool):
        whether to apply metadata from BIDS sidecar files when loading raw EEG data.
    ApplyEvents (bool):
        whether to apply events from BIDS sidecar files when loading raw EEG data.
    ApplyChanlocs (bool):
        whether to apply channel locations from BIDS sidecar files when loading raw EEG data.
    StageNames Sequence[str]:
        list of file name parts for the preprocessing stages, in the order of cleaning,ica,iclabel;
        these can be adjusted when working with different preprocessed versions (e.g., using
        different parameters for cleaning). It is recommended that these start with 'desc-'.
    MinimizeDiskUsage (bool):
        whether to minimize disk usage by not saving some intermediate files (specifically
        the PICARD output if WithICLabel=False). Default True.

    (note: if you add arguments here, also update the kwargs= clause in the function body)

    Returns:
    --------

    List[str]
        A list of file paths to EEG files in the BIDS dataset.
    """
    from eegprep import (bids_list_eeg_files, clean_artifacts, pop_load_frombids,
                         pop_saveset, eeg_picard, iclabel, pop_loadset, pop_resample)
    from .utils.bids import gen_derived_fpath
    if len(StageNames) != 3:
        raise ValueError("StageNames, if given, must be a list of 3 strings, as in: "
                         "['cleaned', 'picard', 'iclabel'].")
    if WithICLabel and not WithPicard:
        logger.warning("WithICLabel=True implies WithPicard=True; setting WithPicard=True.")
        WithPicard = True

    if not os.path.isdir(root) and root.endswith(eeg_extensions):
        fn = root  # process a single file
        if _n_skipped is None:
            _n_skipped = multiprocessing.Value('i', 0)

        fpath_cln = gen_derived_fpath(fn, toplevel=AsDerivative, keyword=StageNames[0])
        fpath_picard = gen_derived_fpath(fn, toplevel=AsDerivative, keyword=StageNames[1])
        fpath_iclabel = gen_derived_fpath(fn, toplevel=AsDerivative, keyword=StageNames[2])

        if WithICLabel:
            needed_files = [fpath_cln, fpath_iclabel]
        elif WithPicard:
            needed_files = [fpath_cln, fpath_picard]
        else:
            needed_files = [fpath_cln]

        if SkipIfPresent and all(os.path.exists(fn) for fn in needed_files):
            logger.info(f"*** Skipping {fn} as preprocessed file(s) already exists: {','.join(needed_files)} ***")
            with _lock:
                _n_skipped.value += 1
                return

        try:
            # if we get here we need to process at least one stage

            # calc new ETA
            elapsed = now() - _t0
            with _lock:
                n_processed = max(0, _k - _n_skipped.value)
            n_remaining = _n_total - _k
            if n_processed >= _n_jobs:
                time_per_job = elapsed / n_processed
                eta = time_per_job * n_remaining
                ETA = humanize_seconds(eta)
            else:
                ETA = 'estimating...'

            logger.info(f"*** Processing [{_k+1}/{_n_total} | ETA {ETA}] {fn} ***")

            try:
                # noinspection PyUnresolvedReferences
                from threadpoolctl import threadpool_limits
                limit = num_cpus_from_reservation(ReservePerJob, default=4)
                thread_ctx = threadpool_limits(limits=limit, user_api='blas')
            except ImportError:
                logger.warning(
                    "threadpoolctl not installed, using default thread limits.")
                thread_ctx = contextlib.nullcontext()

            with thread_ctx:
                if os.path.exists(fpath_cln) and SkipIfPresent:
                    logger.info(f"Found {fpath_cln}, skipping clean_artifacts stage.")
                    EEG = pop_loadset(fpath_cln)
                else:
                    # load input file
                    EEG = pop_load_frombids(
                        fn,
                        apply_bids_metadata=ApplyMetadata,
                        apply_bids_channels=ApplyChanlocs,
                        apply_bids_events=ApplyEvents)

                    if SamplingRate:
                        EEG = pop_resample(EEG, SamplingRate)

                    # apply processing chain
                    EEG, *_ = clean_artifacts(
                        EEG,
                        ChannelCriterion=ChannelCriterion,
                        LineNoiseCriterion=LineNoiseCriterion,
                        BurstCriterion=BurstCriterion,
                        WindowCriterion=WindowCriterion,
                        Highpass=Highpass,
                        ChannelCriterionMaxBadTime=ChannelCriterionMaxBadTime,
                        BurstCriterionRefMaxBadChns=BurstCriterionRefMaxBadChns,
                        BurstCriterionRefTolerances=BurstCriterionRefTolerances,
                        BurstRejection=BurstRejection,
                        WindowCriterionTolerances=WindowCriterionTolerances,
                        FlatlineCriterion=FlatlineCriterion,
                        NumSamples=NumSamples,
                        NoLocsChannelCriterion=NoLocsChannelCriterion,
                        NoLocsChannelCriterionExcluded=NoLocsChannelCriterionExcluded,
                        MaxMem=MaxMem,
                        Distance=Distance,
                        Channels=Channels,
                        Channels_ignore=Channels_ignore,
                        availableRAM_GB=availableRAM_GB)

                    # we always save out the cleaned EEG data
                    pop_saveset(EEG, fpath_cln)

                if WithICLabel and os.path.exists(fpath_iclabel) and SkipIfPresent and MinimizeDiskUsage:
                    # in this case we have all the necessary data already and we won't try to
                    # recompute PICARD
                    return

                if WithPicard:
                    if os.path.exists(fpath_picard) and SkipIfPresent:
                        logger.info(f"Found {fpath_picard}, skipping PICARD stage.")
                        EEG = pop_loadset(fpath_picard)
                    else:
                        EEG = eeg_picard(EEG)
                        EEG = eeg_checkset(EEG)
                        if not WithICLabel or not MinimizeDiskUsage:
                            # only save the PICARD output if we don't do ICLabel (to save disk space)
                            pop_saveset(EEG, fpath_picard)

                if WithICLabel:
                    if os.path.exists(fpath_iclabel) and SkipIfPresent:
                        logger.info(f"Found {fpath_iclabel}, skipping ICLabel stage.")
                        # we'd only load if we had more processing to do downstream
                    else:
                        EEG = iclabel(EEG)
                        pop_saveset(EEG, fpath_iclabel)

        except ExceptionUnlessDebug as e:
            logger.exception(f"Error processing {fn}: {e}")
            with _lock:
                _n_skipped.value += 1
            return
    elif os.path.isdir(root):
        # process all files under a BIDS root recursively
        all_files = bids_list_eeg_files(root)
        n_jobs = 1 if is_debug() else num_jobs_from_reservation(ReservePerJob)
        n_total = len(all_files)
        t0 = now()

        kwargs = dict(
            AsDerivative=AsDerivative,
            SkipIfPresent=SkipIfPresent,
            ReservePerJob=ReservePerJob,
            WithPicard=WithPicard,
            WithICLabel=WithICLabel,

            ChannelCriterion=ChannelCriterion,
            LineNoiseCriterion=LineNoiseCriterion,
            BurstCriterion=BurstCriterion,
            WindowCriterion=WindowCriterion,
            Highpass=Highpass,
            ChannelCriterionMaxBadTime=ChannelCriterionMaxBadTime,
            BurstCriterionRefMaxBadChns=BurstCriterionRefMaxBadChns,
            BurstCriterionRefTolerances=BurstCriterionRefTolerances,
            BurstRejection=BurstRejection,
            WindowCriterionTolerances=WindowCriterionTolerances,
            FlatlineCriterion=FlatlineCriterion,
            NumSamples=NumSamples,
            NoLocsChannelCriterion=NoLocsChannelCriterion,
            NoLocsChannelCriterionExcluded=NoLocsChannelCriterionExcluded,
            MaxMem=MaxMem,
            Distance=Distance,
            Channels=Channels,
            Channels_ignore=Channels_ignore,
            availableRAM_GB=availableRAM_GB,
            ApplyMetadata=ApplyMetadata,
            ApplyEvents=ApplyEvents,
            ApplyChanlocs=ApplyChanlocs,
            StageNames=StageNames,
        )

        with multiprocessing.Manager() as manager:
            n_skipped = manager.Value('i', 0)
            lock = manager.Lock()

            if n_jobs == 1:
                # run sequentially
                for k, fn in enumerate(all_files):
                    bids_preproc(
                        fn,
                        **kwargs,
                        # reserved parameters
                        _lock=lock, _n_skipped=n_skipped, _k=k, _n_total=n_total, _n_jobs=n_jobs, _t0=t0
                    )
            else:
                # run in parallel
                logger.info(f"Running {n_jobs} parallel jobs to process {n_total} files...")
                from multiprocessing import Pool
                with Pool(n_jobs) as pool:
                    results = [
                        pool.apply_async(
                            bids_preproc,
                            args=(fn,),
                            kwds={
                                **kwargs,
                                '_lock': lock, '_n_skipped': n_skipped, '_k': k, '_n_total': n_total, '_n_jobs': n_jobs, '_t0': t0
                            }
                        )
                        for k, fn in enumerate(all_files)
                    ]
                    # wait for all jobs to finish
                    for result in results:
                        result.get()

            logger.info(f"Processed {n_total - n_skipped.value} files, "
                        f"skipped {n_skipped.value} files; total time: "
                        f"{humanize_seconds(now() - t0)}.")
