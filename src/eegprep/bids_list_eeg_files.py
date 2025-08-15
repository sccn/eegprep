import os
from typing import List, Sequence
from types import NoneType
from eegprep.utils.bids import layout_for_fpath

# list of valid file extensions for raw EEG data files in BIDS format
eeg_extensions = ('.vhdr', '.edf', '.bdf', '.set')


def bids_list_eeg_files(
        root: str,
        subjects: Sequence[str | int] | str | int = (),
        sessions: Sequence[str | int] | str | int = (),
        runs: Sequence[str | int] | str | int = (),
        tasks: Sequence[str | int] | str | int = (),
) -> List[str]:
    """
    Return a list of all EEG raw-data files in a BIDS dataset.

    Parameters:
    -----------
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

    Returns:
    --------

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
        for key, values in filters.items():
            for v in values:
                if isinstance(v, str) and '-' in v and v.split('-')[0] in ('sub', 'ses', 'run', 'task'):
                    raise ValueError("Query values should not be formatted with 'sub-', 'ses-', "
                                     "'run-', or 'task-' prefixes. Use the raw identifiers instead.")
            all_values = [f.entities[key] for f in eeg_files]
            if all(isinstance(v, int) for v in all_values):
                # the items are natively integer-indexed for this key (eg run)
                if any(isinstance(v, str) for v in values):
                    # convert query values to integer
                    try:
                        values = [int(v) for v in values]
                    except Exception as e:
                        raise ValueError(f"When filtering by {key}, use integers for the query ({e})")
                eeg_files = [f for f in eeg_files if f.entities[key] in values]
            else:
                if all(isinstance(v, int) for v in values):
                    # index the applicable values (eg subjects) with integers, alphabetically
                    uq_values = sorted(set(all_values)) 
                    values = [uq_values[i] for i in values]             
                if all(isinstance(v, str) for v in values):
                    eeg_files = [f for f in eeg_files if f.entities[key] in values]
                else:
                    raise ValueError(f"query values for {key} must either all be strings or all "
                                     f"integers, but were: {values}")
        # reduce to file paths
        eeg_files = [eeg_file.path for eeg_file in eeg_files]

    # have to filter by extension since the layout will also return the sidecar files
    # (e.g., .fdt and .vmrk) and other files that are not raw EEG data files
    filelist = [fn for fn in eeg_files if fn.endswith(eeg_extensions)]
    return filelist
