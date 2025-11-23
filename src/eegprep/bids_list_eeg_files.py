"""BIDS EEG file listing utilities."""

import logging
from typing import List, Sequence
from eegprep.utils.bids import layout_for_fpath

logger = logging.getLogger(__name__)


# list of valid file extensions for raw EEG data files in BIDS format
eeg_extensions = ('.vhdr', '.edf', '.bdf', '.set')


def bids_list_eeg_files(
        root: str,
        subjects: Sequence[str | int] | str | int = (),
        sessions: Sequence[str | int] | str | int = (),
        runs: Sequence[str | int] | str | int = (),
        tasks: Sequence[str | int] | str | int = (),
) -> List[str]:
    """Return a list of all EEG raw-data files in a BIDS dataset.

    Parameters
    ----------
    root : str
        The root directory containing BIDS data.
    subjects : Sequence[str | int], optional
        A sequence of subject identifiers or (zero-based) indices to filter the files by.
        If empty, all subjects are included.
    sessions : Sequence[str | int], optional
        A sequence of session identifiers or (zero-based) indices to filter the files by.
        If empty, all sessions are included.
    runs : Sequence[str | int], optional
        A sequence of run numbers or identifiers to filter the files by. If empty, all runs
        are included. Note that zero-based indexing does not apply to runs, unlike
        subjects and sessions since runs are already integers.
    tasks : Sequence[str] | str, optional
        A sequence of task names or single task to filter the files by. If empty, all
        tasks are included (default is an empty sequence).

    Returns
    -------
    List[str]
        A list of file paths to EEG files in the BIDS dataset.
    """
    layout = layout_for_fpath(root)
    
    # prepare filters, if any
    filters = {
        'subject': [subjects] if isinstance(subjects, (str, int)) else subjects,
        'session': [sessions] if isinstance(sessions, (str, int)) else sessions,
        'run': [runs] if isinstance(runs, (str, int)) else runs,
        'task': [tasks] if isinstance(tasks, (str, int)) else tasks,
    }
    filters = {k: v for k, v in filters.items() if v}  # remove empty filters

    # first get all EEG files
    eeg_files = layout.get(suffix='eeg', return_type='filename' if not filters else 'object')
    
    # apply filters
    if filters:
        for key, query_values in filters.items():
            for v in query_values:
                if isinstance(v, str) and '-' in v and v.split('-')[0] in ('sub', 'ses', 'run', 'task'):
                    raise ValueError("Query values should not be formatted with 'sub-', 'ses-', "
                                     "'run-', or 'task-' prefixes. Use the raw identifiers instead.")
            data_values = [f.entities.get(key, None) for f in eeg_files]
            if all(v is None for v in data_values):
                # key is missing from entire dataset: ignore the filter
                logger.info(f"Dataset at {root} does not contain any files with the key '{key}'; "
                            f"ignoring the filter.")
            elif all(isinstance(v, int) for v in data_values):
                # all items are integers; make sure our queries are also integers if they're not already
                if any(isinstance(v, str) for v in query_values):
                    # uniformize query values to integer
                    try:
                        query_values = [int(v) for v in query_values]
                    except Exception as e:
                        raise ValueError(f"When filtering by {key}, use integers for the query ({e})")
                eeg_files = [f for f in eeg_files if f.entities[key] in query_values]
            else:
                # data values might be some sort of mix of ints, strings, or None
                # querying those strings by index (with integers)
                if all(isinstance(v, int) for v in query_values):
                    # index the applicable values (eg subjects) with integers, alphabetically
                    # strip any missing candidates
                    data_values = [dv for dv in data_values if dv is not None and dv != '']
                    # normalize data values to strings
                    data_values = [str(dv) for dv in data_values]
                    # get unique values, sorted alphabetically
                    uq_values = sorted(set(data_values))
                    # rewrite the query values into the first k sorted unique strings
                    query_values = [uq_values[i] for i in query_values if i < len(uq_values)]
                if all(isinstance(v, str) for v in query_values):
                    # all query values are strings now, can do a direct lookup
                    eeg_files = [f for f in eeg_files if str(f.entities.get(key)) in query_values]
                else:
                    raise ValueError(f"query values for {key} must either all be strings or all "
                                     f"integers, but were: {query_values}")
        # reduce to file paths
        eeg_files = [eeg_file.path for eeg_file in eeg_files]

    # have to filter by extension since the layout will also return the sidecar files
    # (e.g., .fdt and .vmrk) and other files that are not raw EEG data files
    filelist = [fn for fn in eeg_files if fn.endswith(eeg_extensions)]
    return filelist
