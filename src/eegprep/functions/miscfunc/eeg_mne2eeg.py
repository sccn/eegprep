"""MNE to EEG conversion functions."""

from pathlib import Path
import tempfile

import mne
import numpy as np
from mne.export import export_epochs, export_raw

from ..adminfunc.eeg_checkset import eeg_checkset
from ..popfunc.pop_loadset import pop_loadset

_DROP = object()
_RESERVED_EXTRAS = {'onset', 'duration', 'description', 'ch_names', 'hed_string', 'type'}


def _mne_events_to_eeglab_events(raw_or_epochs):
    """Convert MNE Annotations or events to EEGLAB event structure (list of dicts)."""
    events = []
    sfreq = raw_or_epochs.info['sfreq']
    if (
        hasattr(raw_or_epochs, 'annotations')
        and raw_or_epochs.annotations is not None
        and len(raw_or_epochs.annotations) > 0
    ):
        for ann in raw_or_epochs.annotations:
            latency = int(ann['onset'] * sfreq) + 1  # EEGLAB is 1-based
            events.append({'latency': latency, 'type': ann['description']})
    elif hasattr(raw_or_epochs, 'events') and raw_or_epochs.events is not None:
        for ev in raw_or_epochs.events:
            latency = int(ev[0]) + 1  # sample index, 1-based
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


def _subject_text(value):
    if value is None:
        return ''
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ''
        if value.size == 1:
            value = value.item()
        else:
            return ''
    if isinstance(value, str):
        return value
    return str(value)


def _eeg_subject(EEG):
    return _subject_text(EEG.get('subject', ''))


def _mne_subject(inst):
    subject_info = inst.info.get('subject_info') or {}
    return _subject_text(subject_info.get('his_id', ''))


def _set_mne_subject(inst, subject):
    if not subject:
        return
    subject_info = dict(inst.info['subject_info'] or {})
    subject_info['his_id'] = subject
    inst.info['subject_info'] = subject_info


def _iter_events(events):
    if events is None:
        return []
    if isinstance(events, dict):
        return [dict(events)]
    if isinstance(events, np.ndarray) and events.dtype.names:
        return [{name: row[name] for name in events.dtype.names} for row in events]
    return [dict(event) for event in events]


def _coerce_extra(value):
    if isinstance(value, (bool, np.bool_)):
        return _DROP
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return _DROP
        return float(value)
    if isinstance(value, str):
        return value
    return _DROP


def _epoch_index_for_event(event, pnts, trials):
    epoch = event.get('epoch')
    if isinstance(epoch, (int, float, np.integer, np.floating)) and not isinstance(epoch, (bool, np.bool_)):
        index = int(epoch) - 1
        if 0 <= index < trials:
            return index
    latency = event.get('latency')
    if isinstance(latency, (int, float, np.integer, np.floating)) and not isinstance(latency, (bool, np.bool_)):
        index = int((float(latency) - 1.0) // pnts)
        if 0 <= index < trials:
            return index
    return None


def _relative_seconds(event, epoch_index, pnts, srate, xmin):
    return (float(event['latency']) - epoch_index * pnts - 1.0) / srate + xmin


def _latency_from_onset(onset, epoch_index, pnts, srate, xmin):
    return (float(onset) - xmin) * srate + 1.0 + epoch_index * pnts


def _in_window(onset, tmin, tmax, sfreq):
    half_sample = 0.5 / sfreq
    return (tmin - half_sample) <= float(onset) <= (tmax + half_sample)


def _event_type_name(epochs, code):
    for name, value in (epochs.event_id or {}).items():
        if value == code:
            return str(name)
    return str(int(code))


def _mne_epochs_to_eeg_events(epochs):
    """Build EEGLAB events for one MNE Epochs object.

    The time-locking event is placed at time 0. Annotations are added when
    their onset falls inside the epoch window. Onset is seconds relative to
    that time-locking event.
    """
    sfreq = float(epochs.info['sfreq'])
    tmin = float(epochs.tmin)
    tmax = float(epochs.tmax)
    pnts = len(epochs.times)
    grouped = epochs.get_annotations_per_epoch(with_extras=True)
    events = []
    for epoch_index, annotations in enumerate(grouped):
        locking_type = _event_type_name(epochs, int(epochs.events[epoch_index, 2]))
        found_locking = False
        epoch_events = []
        for onset, _duration, description, extras in annotations:
            extras = dict(extras or {})
            declared_epoch = extras.get('epoch')
            if declared_epoch is not None and int(declared_epoch) != epoch_index + 1:
                continue
            if not _in_window(onset, tmin, tmax, sfreq):
                continue
            description = str(description)
            if abs(float(onset)) <= 0.5 / sfreq and description == locking_type:
                found_locking = True
            latency = extras.get('latency')
            if latency is None:
                latency = _latency_from_onset(onset, epoch_index, pnts, sfreq, tmin)
            event = {'type': description, 'latency': latency, 'epoch': epoch_index + 1}
            for key, value in extras.items():
                if key in _RESERVED_EXTRAS or key in {'latency', 'epoch'}:
                    continue
                coerced = _coerce_extra(value)
                if coerced is _DROP:
                    continue
                event[key] = coerced
            epoch_events.append(event)
        if not found_locking:
            epoch_events.append(
                {
                    'type': locking_type,
                    'latency': _latency_from_onset(0.0, epoch_index, pnts, sfreq, tmin),
                    'epoch': epoch_index + 1,
                }
            )
        epoch_events.sort(key=lambda event: float(event['latency']))
        events.extend(epoch_events)
    return events


def _apply_eeg_events_to_mne(epochs, EEG):
    """Attach EEG epoch events to an MNE Epochs object."""
    eeg_events = _iter_events(EEG.get('event'))
    trials = int(EEG.get('trials', 1) or 1)
    if trials <= 1 or not eeg_events or len(epochs) != trials:
        return
    pnts = int(EEG['pnts'])
    srate = float(EEG['srate'])
    xmin = float(EEG['xmin'])
    xmax = float(EEG['xmax'])
    grouped = [[] for _ in range(trials)]
    for event in eeg_events:
        index = _epoch_index_for_event(event, pnts, trials)
        if index is None:
            continue
        grouped[index].append(event)

    id_map = {}
    samples = []
    codes = []
    onsets = []
    descriptions = []
    extras = []
    next_id = 1
    for epoch_index, group in enumerate(grouped):
        if group:
            relatives = [_relative_seconds(event, epoch_index, pnts, srate, xmin) for event in group]
            lock_at = min(range(len(group)), key=lambda index: (abs(relatives[index]), index))
            locking_type = str(group[lock_at]['type'])
        else:
            relatives = []
            locking_type = 'TLE'
        if locking_type not in id_map:
            id_map[locking_type] = next_id
            next_id += 1
        samples.append(epoch_index * pnts)
        codes.append(id_map[locking_type])
        for event, relative in zip(group, relatives):
            if not _in_window(relative, xmin, xmax, srate):
                continue
            extra = {}
            for key, value in event.items():
                if key in _RESERVED_EXTRAS:
                    continue
                coerced = _coerce_extra(value)
                if coerced is _DROP:
                    continue
                extra[key] = coerced
            extra.setdefault('epoch', epoch_index + 1)
            onsets.append(epoch_index * pnts / srate + relative)
            descriptions.append(str(event.get('type', '')))
            extras.append(extra)

    epochs.events = np.column_stack(
        [
            np.asarray(samples, dtype=np.int64),
            np.zeros(trials, dtype=np.int64),
            np.asarray(codes, dtype=np.int64),
        ]
    )
    epochs.event_id = id_map
    if onsets:
        epochs.set_annotations(
            mne.Annotations(
                onset=onsets,
                duration=np.zeros(len(onsets)),
                description=descriptions,
                extras=extras,
            )
        )


def _rebuild_epoch_structure(EEG):
    """Fill ``EEG['epoch']`` from ``EEG['event']`` using ``eeg_checkset``."""
    trials = int(EEG.get('trials', 1) or 1)
    if trials <= 1:
        return EEG
    checked = eeg_checkset(
        {
            'data': EEG['data'],
            'nbchan': int(EEG['nbchan']),
            'pnts': int(EEG['pnts']),
            'trials': trials,
            'srate': float(EEG['srate']),
            'xmin': float(EEG['xmin']),
            'xmax': float(EEG['xmax']),
            'event': EEG.get('event') or [],
            'epoch': [],
        },
        'eventconsistency',
    )
    EEG['event'] = checked['event']
    EEG['epoch'] = checked['epoch']
    return EEG


def eeg_mne2eeg(raw):
    """Convert an MNE Raw or Epochs object to an EEG dataset.

    ``info['subject_info']['his_id']`` is copied to ``EEG['subject']``. For
    epochs, the time-locking event is placed at time 0 and other events inside
    the epoch window are kept. ``eeg_checkset`` builds ``EEG['epoch']``.

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
        return _rebuild_epoch_structure(EEG)

    eeglab_events = _mne_events_to_eeglab_events(raw_or_epochs)
    if eeglab_events:
        EEG['event'] = eeglab_events
    return EEG
