"""EEG to MNE conversion functions."""

import os
import tempfile

import mne

from ..popfunc.pop_saveset import pop_saveset


def eeg_eeg2mne(EEG):
    """Convert EEG data structure to MNE Raw or Epochs object.

    Parameters
    ----------
    EEG : dict
        EEG data structure

    Returns
    -------
    raw : mne.io.Raw | mne.BaseEpochs
        MNE Raw object for continuous data, Epochs for epoched data.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        set_path = os.path.join(tmpdir, "eeg_eeg2mne.set")
        pop_saveset(EEG, set_path)
        if EEG['trials'] > 1:
            return mne.io.read_epochs_eeglab(set_path)
        return mne.io.read_raw_eeglab(set_path, preload=True)
