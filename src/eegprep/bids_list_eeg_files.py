from typing import List

# list of valid file extensions for raw EEG data files in BIDS format
eeg_extensions = ('.vhdr', '.edf', '.bdf', '.set')


def bids_list_eeg_files(root: str) -> List[str]:
    """
    Return a list of all EEG raw-data files in a BIDS dataset.

    Parameters:
    -----------
    root : str
        The root directory containing BIDS data.

    Returns:
    --------

    List[str]
        A list of file paths to EEG files in the BIDS dataset.
    """
    from bids import BIDSLayout
    layout = BIDSLayout(root)
    eeg_files = layout.get(suffix='eeg', return_type='filename')

    # have to filter by extension since the layout will also return the sidecar files
    # (e.g., .fdt and .vmrk) and other files that are not raw EEG data files
    filelist = [fn for fn in eeg_files if fn.endswith(eeg_extensions)]
    return filelist
