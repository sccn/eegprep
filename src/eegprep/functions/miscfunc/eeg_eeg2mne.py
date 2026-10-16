"""EEG to MNE conversion functions."""

from pathlib import Path
import tempfile

import mne
import numpy as np

from ..popfunc.pop_saveset import pop_saveset
from .eeg_mne2eeg import _apply_eeg_events_to_mne, _eeg_subject, _set_mne_subject


def _with_readable_chaninfo(EEG):
    chaninfo = EEG.get('chaninfo')
    if isinstance(chaninfo, dict) and chaninfo:
        return EEG
    if isinstance(chaninfo, np.ndarray) and chaninfo.size > 0:
        return EEG
    copied = dict(EEG)
    copied['chaninfo'] = {'nosedir': '+X', 'nodatchans': np.array([])}
    return copied


def eeg_eeg2mne(EEG):
    """Convert an EEG dataset to an MNE Raw or Epochs object.

    Continuous data becomes Raw. Epoched data becomes Epochs. ``EEG['subject']``
    is stored as ``info['subject_info']['his_id']``. Events inside each epoch
    window are kept along with the time-locking event.

    Parameters
    ----------
    EEG : dict
        EEG data structure.

    Returns
    -------
    mne.io.BaseRaw | mne.BaseEpochs
        MNE object for the dataset.
    """
    with tempfile.TemporaryDirectory(prefix="eegprep-eeg2mne-") as temp_dir:
        set_path = Path(temp_dir) / "bridge.set"
        pop_saveset(_with_readable_chaninfo(EEG), str(set_path))

        if int(EEG.get('trials', 1) or 1) > 1:
            converted = mne.io.read_epochs_eeglab(str(set_path))
        else:
            converted = mne.io.read_raw_eeglab(str(set_path), preload=True)

    _set_mne_subject(converted, _eeg_subject(EEG))
    if int(EEG.get('trials', 1) or 1) > 1:
        _apply_eeg_events_to_mne(converted, EEG)
    return converted
