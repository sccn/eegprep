"""EEG to MNE conversion functions."""

from pathlib import Path
import tempfile

import mne
import numpy as np

from ..popfunc.pop_saveset import pop_saveset
from ._eeg_mne import _apply_eeg_events_to_mne, _eeg_subject, _set_mne_subject


def _with_readable_chaninfo(EEG):
    """Fill an empty chaninfo field with a non-empty struct."""
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
    is stored as ``info['subject_info']['his_id']``. Events whose onset falls in
    ``[xmin, xmax + 1/srate)`` are kept with the time-locking event. Nonzero
    ``event['duration']`` values, in samples, become annotation durations in
    seconds.

    The urevent table and ``event['urevent']`` do not survive. Boolean,
    non-finite, and non-scalar event fields do not survive. Annotation fields
    ``onset``, ``description``, ``ch_names``, and ``hed_string`` cannot be
    stored as extra event fields. A duration of zero is omitted on the way
    back to EEG.

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
