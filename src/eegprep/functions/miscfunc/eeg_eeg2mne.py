"""EEG to MNE conversion functions."""

from pathlib import Path
import tempfile

import mne

from ..popfunc.pop_saveset import pop_saveset  # in development


def eeg_eeg2mne(EEG):
    """Convert EEG data structure to MNE Raw object.

    Parameters
    ----------
    EEG : dict
        EEG data structure

    Returns
    -------
    raw : mne.io.Raw
        MNE Raw object
    """
    with tempfile.TemporaryDirectory(prefix="eegprep-eeg2mne-") as temp_dir:
        set_path = Path(temp_dir) / "bridge.set"
        pop_saveset(EEG, str(set_path))

        if EEG['trials'] > 1:
            return mne.io.read_epochs_eeglab(str(set_path))
        return mne.io.read_raw_eeglab(str(set_path), preload=True)
