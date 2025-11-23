"""Module for BIDS preprocessing of EEG data."""

import os
import hashlib
import json
from copy import deepcopy
import contextlib
import multiprocessing
import time
from time import time as now
import logging
from typing import Union, Tuple, Optional, Sequence, List, Dict, Any

import numpy as np

from .utils import ExceptionUnlessDebug, num_jobs_from_reservation, is_debug, \
    humanize_seconds, num_cpus_from_reservation, ToolError
from .utils.bids import layout_for_fpath
from .utils.coords import chanloc_has_coords
from .utils.git import get_git_commit_id

# whether pop_rmbase accepts the time range in ms (change if needed)
pop_rmbase_in_ms = True

logger = logging.getLogger(__name__)

# list of valid file extensions for raw EEG data files in BIDS format
eeg_extensions = ('.vhdr', '.edf', '.bdf', '.set')

# list of all possible processing stages, in the order in which they apply
all_stages = ['Import', 'ChannelSelection', 'Resample', 'CleanArtifacts', 'ICA', 'ICLabel', 'ChannelInterp', 'Epoching', 'CommonAverageRef']

def _copy_misc_root_files(root: str, dst: str, exclude: List[str]) -> None:
    """Move miscellaneous description files from the study root to the target
    directory."""
    from bids import BIDSLayout
    from bids.layout.models import BIDSJSONFile
    layout: BIDSLayout = layout_for_fpath(root)
    files = layout.get()
    # apply filter rules
    # 1. none of the excluded files
    exclude = set(exclude)
    files = [f for f in files if f.path not in exclude]
    # 2. no files that are in a disallowed path relative to root
    no_files = {os.path.join(root, 'derivatives'), os.path.join(root, 'sourcedata'), os.path.join(root, 'dataset_description.json')}
    files = [f for f in files if f.path not in no_files]
    # 3. no (other) files for the eeg modality except if they're json
    files = [f for f in files if f.entities.get('suffix') != 'eeg' or isinstance(f, BIDSJSONFile)]
    # 4. no events, electrodes or channels files (since we'll rewrite those)
    no_suffixes = {'channels', 'events', 'electrodes'}
    files = [f for f in files if f.entities.get('suffix') not in no_suffixes]
    filepaths = [f.path for f in files]
    # now perform the copy
    for srcpath in filepaths:
        relpath = os.path.relpath(srcpath, root)
        dstpath = os.path.join(dst, relpath)
        from shutil import copyfile
        os.makedirs(os.path.dirname(dstpath), exist_ok=True)
        try:
            if not os.path.exists(dstpath):
                copyfile(srcpath, dstpath)
                logger.info(f"Copied {srcpath} to {dstpath}")
        except OSError as e:
            logger.error(f"Failed to copy {srcpath} to {dstpath}: {e}")


def _legacy_override(new_and_name: Tuple[Any, str], old_and_name: Tuple[Any, str], default: Any):
    """Handle overrides with values from legacy parameters and a default if
    both the new and legacy parameter are None."""
    new, new_name = new_and_name
    old, old_name = old_and_name
    if old is not None:
        assert new is None, f"Only one of {new_name} and f{old_name} can be specified, but neither were set to None."
        new = old
    if new is None:
        new = default
    return new


def bids_preproc(
        root: str,
        *,
        # BIDS loader parameters
        ApplyChanlocs: Optional[bool] = None,
        ApplyEvents: Optional[bool] = None,
        ApplyMetadata: Optional[bool] = None,
        EventColumn: Optional[str] = None,
        Subjects: Sequence[str | int] | str | int | None = None,
        Sessions: Sequence[str | int] | str | int | None = None,
        Runs: Sequence[str | int] | str | int | None = None,
        Tasks: Sequence[str | int] | str | int | None = None,
        # Overall run configuration
        SkipIfPresent: bool = True,
        NumJobs: Optional[int] = None,
        ReservePerJob: str = '',
        UseHashes: bool = False,
        ReturnData: bool = False,
        OutputDir: Optional[str] = None,
        # Overall processing parameters
        SamplingRate: Optional[float] = None,
        OnlyChannelsWithPosition: bool = True,
        OnlyModalities: Sequence[str] = (),
        WithInterp: bool = False,
        WithPicard: bool = False,
        WithICLabel: bool = False,
        WithReport: bool = True,
        CommonAverageReference: bool = True,
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
        # Optional epoching stage
        EpochEvents: Optional[Union[str, Sequence[str]]] = None,
        EpochLimits: Sequence[float] = (-1, 2),
        EpochBaseline: Optional[Sequence[float]] = None,
        # Derived data parameters
        StageNames: Sequence[str] = ('desc-cleaned', 'desc-picard', 'desc-iclabel', 'desc-epoch'),
        MinimizeDiskUsage: bool = True,

        # Legacy parameter names for compatibility with EEGLAB
        bidschanloc: Optional[bool] = None,
        bidsevent: Optional[bool] = None,
        bidsmetadata: Optional[bool] = None,
        eventtype: Optional[str] = None,
        subjects: Sequence[str | int] | str | int | None = None,
        sessions: Sequence[str | int] | str | int | None = None,
        runs: Sequence[str | int] | str | int | None = None,
        tasks: Sequence[str | int] | str | int | None = None,
        outputdir: Optional[str] = None,

        # Reserved parameters
        _lock: Optional[multiprocessing.Lock] = contextlib.nullcontext(),
        _n_skipped: Optional[multiprocessing.Value] = None,
        _k: int = 0,
        _n_total: int = 1,
        _n_jobs: int = 1,
        _t0: float = now(),
) -> Dict[str,Any] | List[Dict[str, Any]] | None:
    """Apply data cleaning to EEG files in a BIDS dataset.

    Parameters
    ----------
    root : str
        The root directory containing BIDS data or a single EEG file path.

    (BIDS import stage parameters)

    ApplyMetadata : bool
        Whether to apply metadata from BIDS sidecar files when loading raw EEG data.
        (default True)
    ApplyEvents : bool
        Whether to apply events from BIDS sidecar files when loading raw EEG data.
        (default False)
    ApplyChanlocs : bool
        Whether to apply channel locations from BIDS sidecar files when loading raw EEG data.
        (default True)
    EventColumn : str
        Optionally the column name in the BIDS events file to use for event types; if not
        set, will be inferred heuristically.
    Subjects : Sequence[str | int], optional
        A sequence of subject identifiers or (zero-based) indices to filter the files by.
        If empty, all subjects are included.
    Sessions : Sequence[str | int], optional
        A sequence of session identifiers or (zero-based) indices to filter the files by.
        If empty, all sessions are included.
    Runs : Sequence[str | int], optional
        A sequence of run numbers or identifiers to filter the files by. If empty, all runs
        are included. Note that zero-based indexing does not apply to runs, unlike
        subjects and sessions since runs are already integers.
    Tasks : Sequence[str] | str, optional
        A sequence of task names or single task to filter the files by. If empty, all
        tasks are included (default is an empty sequence).
    OutputDir : str
        The name of the subdirectory where cleaned files will be saved. This can start
        with the placeholder '{root}' which will be replaced with the root path of
        the BIDS dataset. Defaults to '{root}/derivatives/eegprep' if not specified.    (overall run configuration)

    SkipIfPresent : bool
        skip processing files that already have a cleaned version present.
    NumJobs : int, optional
        The number of jobs to run in parallel. If set to -1, this will default to the
        number of logical cores on the system. If the ReservePerJob clause is also
        specified, this will be treated as a maximum, otherwise as the *total*. If neither
        of the two parameters is specified, a single job will run.
        Note: as usual when running multiple processes in Python, you need to use the
        if __name__ == "__main__": guard pattern in your main processing script.
    ReservePerJob : str
        Optionally the resource amount and type to reserve per job, e.g. '4GB' or '2CPU';
        the run will then use as many jobs as fit within the system resources of the specified type.
        * You can also specify how much of a margin of the total system resources should
        be *withheld* for use by other programs on the computer, by following the amount
        by a : and then the margin, as in '4GB:10GB' (always leave 10GB unused), '2CPU:10%'
        (always leave 10% of the total installed RAM unused). This also works with other metrics.
        * one may also specify a total or maximum number of jobs, as in '10total' or '10max'.
        * Multiple criteria can be spefied in a comma-separated list of reservations, e.g.
        '4GB:20%, 2CPU, 5max'.
        * If neither this nor NumJobs are specified, a single job will run. Note that the
        system will also run in serial when in debug mode and when on a platform that does
        not cleanly support multiprocessing.
        Tip: a good way to size this is to perform a serial run and to monitor how much
        peak RAM a single job takes, and then setting this to <PeakUsage>GB:<YourMargin>GB
        where YourMargin is however much you want to leave to other programs, e.g., 5GB
        (this will depend on what else you expect to be running on the machine).
    UseHashes : bool
        Whether to bake hashes into intermediate file names; if you experiment
        with alternative preprocessing settings, it is recommended to enable this or disable
        the SkipIfPresent option since otherwise the routine may pick up a stale result.
    ReturnData : bool
        Whether to return the final EEG data objects as a list. Note that this can use
        quite a lot of memory for large studies and it may be better to iterate over
        the preprocessed files in downstream analyses.

    (overall processing parameters)
    OnlyChannelsWithPosition : bool
        Whether to retain only channels for which positions were recorded or could be
        inferred. If this is not set, then OnlyModalities should be set so as to retain
        only modalities that should be preprocessed together.
    OnlyModalities : Sequence[str], optional
        If set, retain only channels that have the associated modalities. If enabled, this
        is typically set to ['EEG'] but may also include other ExG modalities such as
        EOG or EMG that have the same unit and scale as EEG. If non-electrophysiological
        modalities are included, some artifact removal steps may not function correctly.
    SamplingRate : float
        Desired sampling rate for the preprocessed data. If not specified, will retain
        the original sampling rate.
    WithInterp : bool
        Whether to reinterpolate dropped channels, thus retaining the same channel
        count as the raw data.
    WithPicard : bool
        Whether to apply PICARD ICA decomposition after cleaning.
    WithICLabel : bool
        Whether to apply ICLabel classification after ICA. Normally requires
        WithPicard=True.
    CommonAverageReference : bool
        Whether to transform the EEG data to a common average referencing scheme;
        recommended for cross-study processing.

    (parameters for artifact removal - same as in clean_artifacts function)

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
    BurstRejection : str
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
    NoLocsChannelCriterion : float
        Correlation threshold for fallback channel cleaning when no channel locations.
        Default 0.45.
    NoLocsChannelCriterionExcluded : float
        Fraction of channels excluded when assessing correlation in nolocs cleaning.
        Default 0.1.
    MaxMem : int
        Maximum memory in MB for ASR processing. Default 64.
    Distance : str
        Distance metric for ASR processing ('euclidian'). Default 'euclidian'.
    Channels : Sequence[str] or None
        List of channel labels to include before cleaning (pop_select). Default None.
    Channels_ignore : Sequence[str] or None
        List of channel labels to exclude before cleaning. Default None.
    availableRAM_GB : float or None
        Available system RAM in GB to adjust MaxMem. Default None.

     (parameters for an optional epoching and baseline removal step)
    EpochEvents : str or Sequence[str] or None
        Optionally a list of event types or regular expression matching event types
        at which to time-lock epochs. If None (default), no epoching is done. If [],
        will time-lock to every event in the data (warning, this can amplify the data
        if epochs overlap!)
    EpochLimits : Sequence[float]
        The time limits in seconds relative to the event markers for epoching. Default (-1, 2).
    EpochBaseline : Sequence[float] or None
        Optionally a time range in seconds relative to the event markers for baseline
        correction. If None (default), no baseline correction is applied. The special
        value None can be used to refer to the respective end of the epoch limits,
        as in (None, 0).

    (misc parameters)

    StageNames : Sequence[str]
        list of file name parts for the preprocessing stages, in the order of cleaning,ica,iclabel;
        these can be adjusted when working with different preprocessed versions (e.g., using
        different parameters for cleaning). It is recommended that these start with 'desc-'.
    MinimizeDiskUsage : bool
        whether to minimize disk usage by not saving some intermediate files (specifically
        the PICARD output if WithICLabel=False). Default True.

    (parameters retained for backwards compatibility with EEGLAB's pop_importbids call signature)

    bidsmetadata : bool
        alias for ApplyMetadata
    bidsevent : bool
        alias for ApplyEvents
    bidschanloc : bool
        alias for ApplyChanlocs
    eventtype : str
        alias for EventColumn
    subjects : Sequence[str | int], optional
        alias for Subjects
    sessions : Sequence[str | int], optional
        alias for Sessions
    runs : Sequence[str | int], optional
        alias for RUns
    tasks : Sequence[str] | str, optional
        alias for Tasks
    outputdir : str
        alias for OutputDir

    Returns
    -------
    result : Dict[str,Any] | List[Dict[str, Any]] | None
        Depending on ReturnData, either a list of EEG objects (if BIDS root folder was
        specified) or a single EEG object (if a single file was specified), otherwise None.
    """
    # get a dictionary of all arguments
    kwargs = {k: v for k, v in locals().items() if not k.startswith('_')}
    del kwargs['root']  # we don't need the root here, only in the function body
    from scipy.io.matlab import loadmat
    from eegprep import (bids_list_eeg_files, clean_artifacts, pop_load_frombids, eeg_checkset,
                         pop_saveset, eeg_picard, iclabel, pop_loadset, pop_resample,
                         eeg_interp, pop_select, eeg_checkset_strict_mode, pop_reref)
    from .utils.bids import gen_derived_fpath

    def hash_suffix(ignore: Optional[set] = None, *, prefix='#') -> str:
        """Get a hash for all options that affect results minus the ones listed
        in ignore.

        Unless UseHashes is False (in which case an empty string is returned).
        """
        if not UseHashes:
            return ''
        # set of options in kwargs that do NOT influence the processing result; all others
        # are used to calc an options hash
        non_proc_options = {
            'root', 'Subjects', 'Sessions', 'Runs', 'Tasks', 'SkipIfPresent', 'NumJobs',
            'ReservePerJob', 'ReturnData', 'OutputDir', 'MinimizeDiskUsage', 'UseHashes',
            'subjects', 'sessions', 'runs', 'tasks', 'outputdir'}
        if ignore is not None:
            non_proc_options = non_proc_options | ignore
        # and collection of options that DO influence results
        proc_options = {k: kwargs[k] for k in sorted(kwargs) if k not in non_proc_options}
        options_str = ','.join([f'{k}:{v!r}' for k, v in proc_options.items()])
        # get an abbreviated options hash (note: this is only used to uniquely identify
        # the final result, but we're not currently using that for preproc options)
        hasher = hashlib.sha256()
        hasher.update(options_str.encode('utf-8'))
        options_hash = hasher.hexdigest()[:8]
        return prefix + options_hash

    # handle support for legacy parameters and defaults
    ApplyChanlocs = _legacy_override((ApplyChanlocs, 'ApplyChanlocs'), (bidschanloc, 'bidschanloc'),
                                     True)
    ApplyEvents = _legacy_override((ApplyEvents, 'ApplyEvents'), (bidsevent, 'bidsevent'),
                                   False)
    ApplyMetadata = _legacy_override((ApplyMetadata, 'ApplyMetadata'), (bidsmetadata, 'bidsmetadata'),
                                     True)
    EventColumn = _legacy_override((EventColumn, 'EventColumn'), (eventtype, 'eventtype'),
                                 '')
    OutputDir = _legacy_override((OutputDir, 'OutputDir'), (outputdir, 'outputdir'),
                                 '{root}/derivatives/eegprep')
    Subjects = _legacy_override((Subjects, 'Subjects'), (subjects, 'subjects'),
                                ())
    Sessions = _legacy_override((Sessions, 'Sessions'), (sessions, 'sessions'),
                                ())
    Runs = _legacy_override((Runs, 'Runs'), (runs, 'runs'),
                            ())
    Tasks = _legacy_override((Tasks, 'Tasks'), (tasks, 'tasks'),
                             ())

    # account for the NumJobs parameter
    if NumJobs == -1:
        NumJobs = os.cpu_count()
    if NumJobs is not None:
        if ReservePerJob:
            ReservePerJob = f"{ReservePerJob},{NumJobs}max"
        else:
            ReservePerJob = f"{NumJobs}total"

    # other sanity checks
    if len(StageNames) != 4:
        raise ValueError("StageNames, if given, must be a list of 4 strings, as in: "
                         "['desc-cleaned', 'desc-picard', 'desc-iclabel', 'desc-epoch'].")
    if WithICLabel and not WithPicard:
        logger.warning("WithICLabel=True implies WithPicard=True; setting WithPicard=True.")
        WithPicard = True

    if not os.path.isdir(root) and root.endswith(eeg_extensions):
        fn = root  # process a single file
        if _n_skipped is None:
            _n_skipped = multiprocessing.Value('i', 0)

        late_opts = {'WithInterp', 'EpochLimits', 'EpochEvents', 'EpochBaseline', 'CommonAverageReference'}
        fpath_cln = gen_derived_fpath(fn, outputdir=OutputDir, keyword=StageNames[0] + hash_suffix(ignore={'WithPicard', 'WithICLabel'} | late_opts))
        fpath_picard = gen_derived_fpath(fn, outputdir=OutputDir, keyword=StageNames[1] + hash_suffix(ignore={'WithICLabel'} | late_opts))
        fpath_iclabel = gen_derived_fpath(fn, outputdir=OutputDir, keyword=StageNames[2] + hash_suffix(ignore=late_opts))
        fpath_epoch = gen_derived_fpath(fn, outputdir=OutputDir, keyword=StageNames[3] + hash_suffix(ignore={'CommonAverageReference'}))
        fpath_final = gen_derived_fpath(fn, outputdir=OutputDir, keyword=f'desc-final' + hash_suffix())
        fpath_report = gen_derived_fpath(fn, outputdir=OutputDir, keyword='desc-report' + hash_suffix(), extension='.json')

        # JSON report file
        if os.path.exists(fpath_report):
            logger.info(f"Found existing report file: {fpath_report}; extending.")
            with open(fpath_report, 'r') as f:
                try:
                    report = json.load(f)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse existing report file {fpath_report}, overwriting.")
                    report = {
                        "Errors": [f"Failed to parse existing report file: {e}. Prior report was overridden/regenerated."],
                    }
        else:
            report = {}
        report["LastUpdated"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if "Errors" not in report:
            report["Errors"] = []

        # stages for reporting purposes
        needed_files = []
        # whether we need to generate/load a _final.set file
        need_final = False

        StagesToGo = ['Import']
        need_final = True

        if OnlyModalities or OnlyChannelsWithPosition:
            StagesToGo += ['ChannelSelection']
            need_final = True
        if SamplingRate:
            StagesToGo += ['Resample']
            need_final = True

        StagesToGo += ['CleanArtifacts']

        if WithICLabel:
            needed_files += [fpath_cln, fpath_iclabel]
            StagesToGo += ['ICA', 'ICLabel']
        elif WithPicard:
            needed_files += [fpath_cln, fpath_picard]
            StagesToGo += ['ICA']
        else:
            needed_files += [fpath_cln]
        need_final = False

        if WithInterp:
            StagesToGo += ['ChannelInterp']
            need_final = True

        if EpochEvents is not None:
            needed_files += [fpath_epoch]
            StagesToGo += ['Epoching']
            need_final = False

        if CommonAverageReference:
            StagesToGo += ['CommonAverageRef']
            need_final = True

        if need_final:
            needed_files += [fpath_final]
        SkippedStages = [s for s in all_stages if s not in StagesToGo]

        if SkipIfPresent and all(os.path.exists(fn) for fn in needed_files):
            logger.info(f"*** Skipping {fn} as preprocessed file(s) already exists: {', '.join(needed_files)} ***")
            # load the final file if requested
            with eeg_checkset_strict_mode(False):
                EEG = pop_loadset(needed_files[-1]) if ReturnData else None
            with _lock:
                _n_skipped.value += 1
                return EEG

        try:
            # if we get here we need to process at least one stage
            EEG = None

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

            old_chanlocs = None
            with thread_ctx:

                def select_channels(EEG, report=None):
                    """Apply channel selection, optionally update the provided
                    report in-place."""
                    if report is None:
                        report = {}
                    keep = np.ones_like(EEG['chanlocs'], dtype=bool)
                    if OnlyChannelsWithPosition:
                        keep &= [chanloc_has_coords(ch) for ch in EEG['chanlocs']]
                    if OnlyModalities:
                        OM = [m.upper() for m in OnlyModalities]
                        keep &= [ch.get('type', 'EEG').upper() in OM for ch in EEG['chanlocs']]
                    retain = [ch['labels'] for ch, kp in zip(EEG['chanlocs'], keep) if kp]
                    if 0 < len(retain) < len(EEG['chanlocs']):
                        EEG = pop_select(EEG, channel=retain)
                        EEG = eeg_checkset(EEG)
                        report["ChannelSelection"] = {
                            "Applied": True,
                            "Retain": retain,
                        }
                    else:
                        detail = 'no' if not retain else 'all'
                        logger.info(f"No channel selection applied: {detail} channels retained")
                        report["ChannelSelection"] = {
                            "Applied": False
                        }
                    return EEG

                if os.path.exists(fpath_cln) and SkipIfPresent:
                    logger.info(f"Found {fpath_cln}, skipping clean_artifacts stage.")
                    try:
                        with eeg_checkset_strict_mode(False):
                            EEG = pop_loadset(fpath_cln)
                    except OSError as e:
                        # this can happen if a previous export was truncated eg due to
                        # file-write error
                        logger.error(f"Failed to load existing cleaned file {fpath_cln}: {e}. Recomputing.")
                if EEG is not None:
                    StagesToGo.remove('Import')
                    StagesToGo.remove('CleanArtifacts')
                else:
                    # load input file
                    EEG, import_report = pop_load_frombids(
                        fn,
                        bidsmetadata=ApplyMetadata,
                        bidschanloc=ApplyChanlocs,
                        bidsevent=ApplyEvents,
                        eventtype=EventColumn,
                        return_report=True)
                    StagesToGo.remove('Import')

                    report["Import"] = {
                        "ApplyMetadata": ApplyMetadata,
                        "ApplyChanlocs": ApplyChanlocs,
                        "ApplyEvents": ApplyEvents,
                        "EventColumn": EventColumn,
                        "InputFile": {
                            "Filename": os.path.basename(fn),
                            "Relpath": os.path.relpath(fn, root),
                            "Filesize": os.path.getsize(fn),
                        },
                        # these are kept for back compat with previously generated reports
                        "bidsmetadata": ApplyMetadata,
                        "bidschanloc": ApplyChanlocs,
                        "bidsevent": ApplyEvents,
                        "eventtype": EventColumn,
                        **import_report,
                    }

                    if OnlyChannelsWithPosition or OnlyModalities:
                        EEG = select_channels(EEG, report)
                        StagesToGo.remove('ChannelSelection')

                    if SamplingRate:
                        EEG = pop_resample(EEG, SamplingRate)
                        report["Resample"] = {
                            "Applied": True,
                            "SamplingRate": SamplingRate,
                        }
                        StagesToGo.remove('Resample')
                    else:
                        logger.info("No resampling requested, keeping original sampling rate.")
                        report["Resample"] = {
                            "Applied": False
                        }

                    old_events = EEG['event']
                    old_chanlocs = EEG['chanlocs']

                    # apply processing chain
                    needs_recalc = True
                    if os.path.exists(fpath_cln) and SkipIfPresent:
                        logger.info(f"Found {fpath_cln}, skipping cleaning stage.")
                        try:
                            EEG = pop_loadset(fpath_cln)
                            needs_recalc = False
                        except OSError as e:
                            logger.warning(f"Encountered read error trying to look up "
                                           f"cached derivative data {fpath_cln}. Recomputing.")
                    if needs_recalc:
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
                        report["CleanArtifacts"] = {
                            "Applied": True,
                            "ChannelCriterion": ChannelCriterion,
                            "LineNoiseCriterion": LineNoiseCriterion,
                            "BurstCriterion": BurstCriterion,
                            "WindowCriterion": WindowCriterion,
                            "Highpass": Highpass,
                            "ChannelCriterionMaxBadTime": ChannelCriterionMaxBadTime,
                            "BurstCriterionRefMaxBadChns": BurstCriterionRefMaxBadChns,
                            "BurstCriterionRefTolerances": BurstCriterionRefTolerances,
                            "BurstRejection": BurstRejection,
                            "WindowCriterionTolerances": WindowCriterionTolerances,
                            "FlatlineCriterion": FlatlineCriterion,
                            "NumSamples": NumSamples,
                            "NoLocsChannelCriterion": NoLocsChannelCriterion,
                            "NoLocsChannelCriterionExcluded": NoLocsChannelCriterionExcluded,
                            "MaxMem": MaxMem,
                            "Distance": Distance,
                        }
                        StagesToGo.remove('CleanArtifacts')

                        # we always save out the cleaned EEG data
                        pop_saveset(EEG, fpath_cln)

                if WithPicard:
                    if os.path.exists(fpath_picard) and SkipIfPresent:
                        logger.info(f"Found {fpath_picard}, skipping PICARD stage.")
                        EEG = pop_loadset(fpath_picard)
                    else:
                        EEG = eeg_picard(EEG, posact=True, sortcomps=True)
                        EEG = eeg_checkset(EEG)
                        if not WithICLabel or not MinimizeDiskUsage:
                            # only save the PICARD output if we don't do ICLabel (to save disk space)
                            pop_saveset(EEG, fpath_picard)
                        report["ICA"] = {
                            "Type": "PICARD",
                            "Applied": True,
                        }
                        StagesToGo.remove('ICA')
                else:
                    report["ICA"] = {
                        "Applied": False,
                    }

                if WithICLabel:
                    if os.path.exists(fpath_iclabel) and SkipIfPresent:
                        logger.info(f"Found {fpath_iclabel}, skipping ICLabel stage.")
                        EEG = pop_loadset(fpath_iclabel)
                    else:
                        EEG = iclabel(EEG)
                        pop_saveset(EEG, fpath_iclabel)
                        report["ICLabel"] = {
                            "Applied": True,
                        }
                        StagesToGo.remove('ICLabel')
                else:
                    report["ICLabel"] = {
                        "Applied": False,
                    }

                # reinterpolate to original channel set if any channels were removed
                if WithInterp:
                    if old_chanlocs is None:
                        # load input file to get original channel locations
                        tmp, import_report = pop_load_frombids(
                            fn,
                            bidsmetadata=ApplyMetadata,
                            bidschanloc=ApplyChanlocs,
                            bidsevent=ApplyEvents,
                            eventtype=EventColumn,
                            return_report=True)
                        # apply channel selection to that also
                        if OnlyChannelsWithPosition or OnlyModalities:
                            tmp = select_channels(tmp)
                        old_chanlocs = tmp['chanlocs']
                        del tmp
                    if nDropped := (len(old_chanlocs) - len(EEG['chanlocs'])):
                        logger.info(F"Reinterpolating {nDropped} dropped channels.")
                        # note: this assumes that no non-ExG channels were dropped by
                        # the above preproc, since those can't really be restored by
                        # interpolation (although in the worst case they will contain
                        # low-amplitude noise afterwards)
                        try:
                            EEG = eeg_interp(EEG, list(old_chanlocs))
                            report["ChannelInterp"] = {
                                "Applied": True,
                                "NumInterpolated": nDropped
                            }
                        except RuntimeError as e:
                            if 'locations required' in str(e):
                                logger.warning("Cannot reinterpolate dropped channels as original "
                                               "channel locations are missing and could not be "
                                               "inferred.")
                                report["ChannelInterp"] = {
                                    "Applied": False,
                                    "Reason": "Original channel locations missing"
                                }
                            else:
                                raise
                        StagesToGo.remove('ChannelInterp')
                else:
                    report["ChannelInterp"] = {
                        "Applied": False,
                    }

                if EpochEvents is not None:
                    from . import pop_epoch
                    assert len(EpochLimits) == 2, "EpochLimits must be a tuple of (min, max) times in seconds."
                    events = EpochEvents
                    if EpochEvents == [] and len(EEG['event']) == 0 or all([e['type'] == 'boundary' for e in EEG['event']]):
                        # trying to epoch around any marker but got no events at all, or only boundary events
                        logger.info(f"Dataset {fn!r} has no (non-boundary) events, nothing to epoch")
                        report["Epoching"] = {
                            "Applied": False,
                            "Reason": "No (non-boundary) events in data"
                        }
                    else:
                        try:
                            with eeg_checkset_strict_mode(False):
                                EEG, *_ = pop_epoch(EEG, types=EpochEvents, lim=EpochLimits)
                                if EpochBaseline is not None:
                                    from . import pop_rmbase
                                    assert len(EpochBaseline) == 2, "EpochBaseline must be a tuple of (min, max) times in seconds or None."
                                    timerange = EpochBaseline
                                    if timerange[0] is None:
                                        timerange = (EEG['times'][0] / 1000, timerange[1])
                                    if timerange[1] is None:
                                        timerange = (timerange[0], EEG['times'][-1] / 1000)
                                    if pop_rmbase_in_ms:
                                        timerange = [timerange[0] * 1000, timerange[1] * 1000]
                                    EEG = pop_rmbase(EEG, timerange=timerange)

                                pop_saveset(EEG, fpath_epoch)

                            report["Epoching"] = {
                                "Applied": True,
                                "TimeLimits": EpochLimits,
                                "EventTypes": EpochEvents,
                                "Baseline": EpochBaseline
                            }
                        except ValueError as e:
                            if 'is empty' in str(e) or 'of an empty dataset' in str(e):
                                report["Epoching"] = {
                                    "Applied": False,
                                    "Reason": "No events retained (possibly all crossing boundaries)"
                                }
                            else:
                                raise

                    StagesToGo.remove('Epoching')
                else:
                    report["Epoching"] = {
                        "Applied": False,
                    }

                if CommonAverageReference:
                    EEG = pop_reref(EEG, [])
                    StagesToGo.remove('CommonAverageRef')
                    report["CommonAverageReference"] = {"Applied": True}

                # optionally write out the final preprocessed file
                if need_final:
                    pop_saveset(EEG, fpath_final)

                # rewrite the events file
                if len(EEG['event']):
                    fpath_events = gen_derived_fpath(fn, outputdir=OutputDir,
                                                     suffix='events', extension='.tsv')
                    have_hed_column = EEG['etc'].get('event_column', None) == 'HED'
                    columns = ['onset', 'duration', 'trial_type'] + (['HED'] if have_hed_column else [])
                    with open(fpath_events, 'w') as fp:
                        print('\t'.join(columns), file=fp)
                        times, srate = EEG['times'], EEG['srate']
                        for e in EEG['event']:
                            ev_type = e['type']
                            try:
                                ev_time = times[e['latency']]/1000.0  # in ms
                            except IndexError:
                                logger.error(f'out-of-bounds event {ev_type} at lat {e["latency"]}; ignoring')
                                continue
                            ev_dur = e.get('duration', 0.0)/srate
                            if np.isnan(ev_dur):
                                ev_dur = 0.0
                            row = [
                                ev_time,
                                ev_dur,
                                ev_type
                            ] + ([ev_type] if have_hed_column else [])
                            print('\t'.join(str(r) for r in row), file=fp)

                # rewrite the channels file
                fpath_channels = gen_derived_fpath(fn, outputdir=OutputDir,
                                                   suffix='channels', extension='.tsv')
                with open(fpath_channels, 'w') as fp:
                    print('name\ttype\tunits', file=fp)
                    for ch in EEG['chanlocs']:
                        ch_type = ch.get('type', 'EEG')
                        ch_unit = 'uV' if ch_type == 'EEG' else 'n/a'
                        print(f"{ch['labels']}\t{ch_type}\t{ch_unit}", file=fp)

                # rewrite the electrodes file
                fpath_elecs = gen_derived_fpath(fn, outputdir=OutputDir,
                                                suffix='electrodes', extension='.tsv')
                with open(fpath_elecs, 'w') as fp:
                    print('name\tx\ty\tz', file=fp)
                    for ch in EEG['chanlocs']:
                        if chanloc_has_coords(ch):
                            print(f"{ch['labels']}\t{ch['X']}\t{ch['Y']}\t{ch['Z']}", file=fp)

                # rewrite/update the coordsystem file
                fpath_coordsystem = gen_derived_fpath(fn, outputdir=OutputDir,
                                                      suffix='coordsystem', extension='.json')
                coordsystem = EEG['etc'].get('BIDSCoordsystem', {})
                coordsystem.update({
                    "EEGCoordinateSystem": "EEGLAB",
                    "EEGCoordinateUnits": "mm",
                    "EEGCoordinateSystemDescription": "ALS | +x is front, +y is left, +z is up",
                })
                with open(fpath_coordsystem, 'w') as fp:
                    json.dump(coordsystem, fp, indent=4)

                # rewrite/update the _eeg.json file
                fpath_eeg = gen_derived_fpath(fn, outputdir=OutputDir,
                                              suffix='eeg', extension='.json')
                if os.path.exists(fpath_eeg):
                    with open(fpath_eeg, 'r') as fp:
                        content = json.load(fp)
                else:
                    content = {}

                # rewrite mandatory fields
                if (ref := EEG.get('ref', 'unknown')) != 'unknown' or 'EEGReference' not in content:
                    content['EEGReference'] = ref
                if 'PowerLineFrequency' not in content:
                    # while possible, it'd be tricky to infer that one on the fly
                    content['PowerLineFrequency'] = 'n/a'
                content['SamplingFrequency'] = EEG['srate']

                # write channel counts based on the modality
                labels = [str(lab['type']).lower() if isinstance(lab['type'], str) else repr(lab['type']) for lab in EEG['chanlocs']]
                content['EEGChannelCount'] = n_eeg = sum(lab == 'eeg' for lab in labels)
                content['ECGChannelCount'] = n_ecg = sum(lab == 'ecg' for lab in labels)
                content['EMGChannelCount'] = n_emg = sum(lab == 'emg' for lab in labels)
                content['EOGChannelCount'] = n_eog = sum(lab == 'eog' for lab in labels)
                content['TriggerChannelCount'] = n_trig = sum(lab == 'trig' for lab in labels)
                content['MISCChannelCount'] = len(EEG['chanlocs']) - (n_eeg+n_ecg+n_emg+n_eog+n_trig)

                # remove misnamed field that may be present from prior json file
                if 'MiscChannelCount' in content:
                    del content['MiscChannelCount']

                # other things that likely changed as a result of preprocessing
                is_epoched = EEG['data'].ndim == 3
                if not is_epoched:
                    content['RecordingDuration'] = EEG['xmax'] - EEG['xmin']
                else:
                    content['EpochLength'] = EEG['xmax'] - EEG['xmin']
                    content['RecordingDuration'] = (EEG['xmax'] - EEG['xmin']) * EEG['data'].shape[-1]
                if is_epoched:
                    content['RecordingType'] = 'epoched'
                elif any(ev['type'] == 'boundary' for ev in EEG['event']):
                    content['RecordingType'] = 'discontinuous'
                else:
                    content['RecordingType'] = 'continuous'

                # complete a few fields that may be missing
                if 'EEGPlacementScheme' not in content:
                    content['EEGPlacementScheme'] = EEG['etc'].get('labelscheme', 'unknown')

                # write information about the applied filters
                if ('SoftwareFilters' not in content) or not isinstance(content['SoftwareFilters'], dict):
                    content['SoftwareFilters'] = sw_filts = {}
                else:
                    sw_filts = content['SoftwareFilters']

                filter_report = deepcopy(report)
                # mapping from the name in the report to the name in the _eeg file
                filter_names = {
                    'Resample': 'pop_resample',
                    'CleanArtifacts': 'clean_artifacts',
                    'ICA': 'eeg_picard',
                    'ChannelInterp': 'eeg_interp',
                    'Epoching': 'pop_epoch+pop_rmbase' if EpochBaseline is not None else 'pop_epoch',
                    'CommonAverageReference': 'pop_reref'
                }
                for in_report, in_filters in filter_names.items():
                    if (flt := filter_report.get(in_report, {'Applied': False})).pop('Applied'):
                        sw_filts[in_filters] = flt

                # write the updated content back
                with open(fpath_eeg, 'w') as fp:
                    json.dump(content, fp, indent=4)

            return EEG if ReturnData else None
        except ExceptionUnlessDebug as e:
            logger.exception(f"Error processing {fn}: {e}")
            if StagesToGo:
                errorStage = StagesToGo.pop(0)
                if errorStage not in report:
                    report[errorStage] = {}
                report[errorStage]["Applied"] = 'Error'
                report[errorStage]["Error"] = {
                    "Message": str(e)
                }
                for remaining in StagesToGo:
                    if remaining not in report:
                        report[remaining] = {}
                    report[remaining]["Applied"] = 'Skipped'
                    report[remaining]["Reason"] = "Previous stage failed"
            report["Errors"].append(
                {
                    "Message": str(e)
                })
            with _lock:
                _n_skipped.value += 1
            return None
        finally:
            for st in SkippedStages:
                if st not in report:
                    report[st] = {}
                report[st]["Applied"] = False

            # rewrite report
            if WithReport:
                logger.info(f"Writing report to {fpath_report}")
                try:
                    os.makedirs(os.path.dirname(fpath_report), exist_ok=True)
                    with open(fpath_report, 'w') as f:
                        json.dump(report, f, indent=4)
                except OSError as e:
                    logger.error(f"Failed to write report file {fpath_report}: {e}")

    elif os.path.isdir(root):
        # process all files under a BIDS root recursively
        all_files = bids_list_eeg_files(
            root, subjects=Subjects, sessions=Sessions, runs=Runs, tasks=Tasks)
        n_jobs = 1 if is_debug() else num_jobs_from_reservation(ReservePerJob)
        n_total = len(all_files)
        t0 = now()

        # copy/move the other files from the root
        if '{root}' in OutputDir:
            OutputDir = OutputDir.replace('{root}', root)
        _copy_misc_root_files(root, OutputDir, exclude=all_files)

        # rewrite the dataset_description.json file
        dataset_desc_path = os.path.join(root, 'dataset_description.json')
        if os.path.exists(dataset_desc_path):
            logger.info(
                f"Updating derived dataset_description.json from {dataset_desc_path}")
            with open(dataset_desc_path, 'r') as f:
                desc = json.load(f)
        else:
            logger.info(f"Creating new derived dataset_description.json")
            desc = {
                # these must be present
                "Name": os.path.basename(root),
                "BIDSVersion": "1.1.1",
            }
        # update the desc
        desc["DatasetType"] = "derivative"
        if "GeneratedBy" not in desc:
            desc["GeneratedBy"] = []

        repo_root = os.path.dirname(os.path.abspath(__file__))
        commit = get_git_commit_id(repo_root)
        desc["GeneratedBy"].append(
            {
                "Name": "eegprep",
                "Description": "The EEGPrep data pipeline",
                "Version": commit or "",
                "CodeURL": "https://github.com/sccn/eegprep"
            })

        orig_doi = desc.get("DatasetDOI", "")

        # determine the dataset URL
        if '/openneuro.ds' in orig_doi:
            # infer from doi
            ds_name = orig_doi.split('/')[1].split('.')[1]
        else:
            # guess from folder name, if conforming to openneuro convention
            folder_name = os.path.basename(root)
            if folder_name.startswith('ds') and folder_name[2:8].isdigit():
                ds_name = folder_name[:8]
            else:
                ds_name = ''
        if ds_name:
            dataset_url = f"https://openneuro.org/datasets/{ds_name}"
        else:
            dataset_url = ""
        desc["SourceDatasets"] = [
            {
                "URL": dataset_url,
                "DOI": orig_doi
            }
        ]
        # note that the actual epoched data *can* be absent if there were no matching 
        # event markers in any study file, which we can't determine at this point
        desc['IsEpoched'] = EpochEvents is not None
        fpath_dataset_desc = gen_derived_fpath(dataset_desc_path, outputdir=OutputDir, keyword='',
                                               extension='.json')
        with open(fpath_dataset_desc, 'w') as f:
            json.dump(desc, f, indent=4)

        with multiprocessing.Manager() as manager:
            n_skipped = manager.Value('i', 0)
            lock = manager.Lock()

            if n_jobs == 1:
                # run sequentially
                results = []
                for k, fn in enumerate(all_files):
                    results.append(bids_preproc(
                        fn,
                        **kwargs,
                        # reserved parameters
                        _lock=lock, _n_skipped=n_skipped, _k=k, _n_total=n_total, _n_jobs=n_jobs, _t0=t0
                    ))

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
                    results = [result.get() for result in results]

            logger.info(f"Processed {n_total - n_skipped.value} files, "
                        f"skipped {n_skipped.value} files; total time: "
                        f"{humanize_seconds(now() - t0)}.")

            return results if ReturnData else None
    else:
        raise ValueError(f"root must be a BIDS root folder or a supported EEG file type, got {root}")
