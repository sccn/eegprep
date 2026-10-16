"""MNE to EEG conversion functions."""

from pathlib import Path
import tempfile

import mne
import numpy as np
from mne.export import export_epochs, export_raw

from ..popfunc.pop_loadset import pop_loadset
from ._eeg_mne import (
    _duration_samples,
    _mne_epochs_to_eeg_events,
    _mne_subject,
    _rebuild_epoch_structure,
    _restore_numeric_event_types,
)


def _mne_events_to_eeglab_events(raw_or_epochs):
    """Convert MNE annotations or events to EEGLAB event records.

    Nonzero annotation durations are stored in samples. Event types stay
    numbers when every type is numeric.
    """
    events = []
    sfreq = raw_or_epochs.info['sfreq']
    if (
        hasattr(raw_or_epochs, 'annotations')
        and raw_or_epochs.annotations is not None
        and len(raw_or_epochs.annotations) > 0
    ):
        for ann in raw_or_epochs.annotations:
            latency = int(ann['onset'] * sfreq) + 1  # EEGLAB is 1-based
            event = {'latency': latency, 'type': ann['description']}
            if ann['duration']:
                event['duration'] = _duration_samples(ann['duration'], sfreq)
            events.append(event)
    elif hasattr(raw_or_epochs, 'events') and raw_or_epochs.events is not None:
        for ev in raw_or_epochs.events:
            latency = int(ev[0]) + 1  # sample index, 1-based
            event_type = None
            if hasattr(raw_or_epochs, 'event_id'):
                for name, code in raw_or_epochs.event_id.items():
                    if code == ev[2]:
                        event_type = str(name)
                        break
            if event_type is None:
                event_type = str(ev[2])
            events.append({'latency': latency, 'type': event_type})
    return _restore_numeric_event_types(events)


def eeg_mne2eeg(raw):
    """Convert an MNE Raw or Epochs object to an EEG dataset.

    ``info['subject_info']['his_id']`` is copied to ``EEG['subject']``. For
    epochs, the time-locking event is placed at time 0 and other events whose
    onset falls in ``[xmin, xmax + 1/srate)`` are kept. ``eeg_checkset`` builds
    ``EEG['epoch']``. Nonzero annotation durations are stored on
    ``event['duration']`` in samples (seconds times ``srate``). A duration of
    zero is omitted. Event types stay numbers when every type is numeric, and
    are stored as text when any type is not numeric.

    The urevent table and ``event['urevent']`` do not survive. Boolean,
    non-finite, and non-scalar event fields do not survive. Annotation fields
    ``onset``, ``description``, ``ch_names``, and ``hed_string`` cannot be
    stored as extra event fields.

    Parameters
    ----------
    raw : mne.io.BaseRaw | mne.BaseEpochs
        MNE Raw or Epochs object.

    Returns
    -------
    dict
        EEG data structure.
    """
    raw_or_epochs = raw

    with tempfile.TemporaryDirectory(prefix="eegprep-mne2eeg-") as temp_dir:
        set_path = Path(temp_dir) / "bridge.set"
        if isinstance(raw_or_epochs, mne.BaseEpochs):
            export_epochs(str(set_path), raw_or_epochs, fmt='eeglab')
        else:
            export_raw(str(set_path), raw_or_epochs, fmt='eeglab')
        EEG = pop_loadset(str(set_path))

    EEG['subject'] = _mne_subject(raw_or_epochs)
    if isinstance(raw_or_epochs, mne.BaseEpochs):
        EEG['event'] = _mne_epochs_to_eeg_events(raw_or_epochs)
        EEG['urevent'] = np.array([])
        return _rebuild_epoch_structure(EEG)

    eeglab_events = _mne_events_to_eeglab_events(raw_or_epochs)
    if eeglab_events:
        EEG['event'] = eeglab_events
        EEG['urevent'] = np.array([])
    return EEG
