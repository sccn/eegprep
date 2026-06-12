"""MNE to EEG conversion functions."""

from pathlib import Path
import tempfile

import mne
from mne.export import export_raw, export_epochs

from ..popfunc.pop_loadset import pop_loadset


def _mne_events_to_eeglab_events(raw_or_epochs):
    """Convert MNE Annotations or events to EEGLAB event structure (list of dicts)."""
    events = []
    sfreq = raw_or_epochs.info['sfreq']
    # Handle Annotations (Raw)
    if (
        hasattr(raw_or_epochs, 'annotations')
        and raw_or_epochs.annotations is not None
        and len(raw_or_epochs.annotations) > 0
    ):
        for ann in raw_or_epochs.annotations:
            latency = int(ann['onset'] * sfreq) + 1  # EEGLAB is 1-based
            events.append({'latency': latency, 'type': ann['description']})
    # Handle Epochs/events array
    elif hasattr(raw_or_epochs, 'events') and raw_or_epochs.events is not None:
        for ev in raw_or_epochs.events:
            latency = int(ev[0]) + 1  # sample index, 1-based
            # Try to get event type from event_id
            event_type = None
            if hasattr(raw_or_epochs, 'event_id'):
                for k, v in raw_or_epochs.event_id.items():
                    if v == ev[2]:
                        event_type = k
                        break
            if event_type is None:
                event_type = str(ev[2])
            events.append({'latency': latency, 'type': event_type})
    return events


def eeg_mne2eeg(raw):
    """Convert MNE Raw object to EEG data structure.

    Parameters
    ----------
    raw : mne.io.Raw | mne.BaseEpochs
        MNE Raw or Epochs object.

    Returns
    -------
    EEG : dict
        EEG data structure
    """
    raw_or_epochs = raw

    with tempfile.TemporaryDirectory(prefix="eegprep-mne2eeg-") as temp_dir:
        set_path = Path(temp_dir) / "bridge.set"
        if isinstance(raw_or_epochs, mne.BaseEpochs):
            export_epochs(str(set_path), raw_or_epochs, fmt='eeglab')
        else:
            export_raw(str(set_path), raw_or_epochs, fmt='eeglab')
        EEG = pop_loadset(str(set_path))

    # Inject events/annotations from MNE object into EEGLAB structure
    eeglab_events = _mne_events_to_eeglab_events(raw_or_epochs)
    if eeglab_events:
        EEG['event'] = eeglab_events
    return EEG
