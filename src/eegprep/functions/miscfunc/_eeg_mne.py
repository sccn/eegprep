"""Shared helpers for EEG and MNE conversion."""

import re

import mne
import numpy as np

from ..adminfunc.eeg_checkset import eeg_checkset

_DROP = object()
_RESERVED_EXTRAS = {'onset', 'duration', 'description', 'ch_names', 'hed_string', 'type'}
_EVENT_EXTRA_SKIP = _RESERVED_EXTRAS | {'latency', 'epoch', 'urevent'}
_TIME_LOCK_SAMPLES = 1e-6
_INTEGER_TYPE = re.compile(r'[+-]?\d+')
_FLOAT_TYPE = re.compile(r'[+-]?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|[+-]?\d+[eE][+-]?\d+')


def _subject_text(value):
    """Return one subject id, or '' when the value is empty."""
    if value is None:
        return ''
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ''
        if value.size != 1:
            raise ValueError('subject must be a single value')
        value = value.item()
    if isinstance(value, str):
        return value
    return str(value)


def _eeg_subject(EEG):
    """Return ``EEG['subject']`` as text."""
    return _subject_text(EEG.get('subject', ''))


def _mne_subject(inst):
    """Return ``info['subject_info']['his_id']`` as text."""
    subject_info = inst.info.get('subject_info') or {}
    return _subject_text(subject_info.get('his_id', ''))


def _set_mne_subject(inst, subject):
    """Store a subject id on ``info['subject_info']['his_id']``."""
    if not subject:
        return
    subject_info = dict(inst.info['subject_info'] or {})
    subject_info['his_id'] = subject
    inst.info['subject_info'] = subject_info


def _iter_events(events):
    """Return event records as a list of dicts."""
    if events is None:
        return []
    if isinstance(events, dict):
        return [dict(events)]
    if isinstance(events, np.ndarray) and events.dtype.names:
        return [{name: row[name] for name in events.dtype.names} for row in events]
    return [dict(event) for event in events]


def _coerce_extra(value):
    """Return a scalar extra value, or ``_DROP`` when MNE cannot store it."""
    if isinstance(value, (bool, np.bool_)):
        return _DROP
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return _DROP
        return float(value)
    if isinstance(value, str):
        return str(value)
    return _DROP


def _has_urevent_table(urevent):
    """Return whether ``urevent`` contains records."""
    if urevent is None:
        return False
    if isinstance(urevent, np.ndarray):
        return urevent.size > 0
    return len(urevent) > 0


def _drop_dangling_urevent(events, urevent):
    """Drop ``event['urevent']`` when the urevent table is empty."""
    if _has_urevent_table(urevent):
        return events
    for event in events:
        event.pop('urevent', None)
    return events


def _parse_numeric_token(value):
    """Return an int or float when ``value`` is a numeric token."""
    text = str(value)
    if _INTEGER_TYPE.fullmatch(text):
        return int(text)
    if _FLOAT_TYPE.fullmatch(text):
        number = float(text)
        if np.isfinite(number):
            return number
    return None


def _restore_numeric_event_types(events):
    """Keep numeric event types when every type is numeric."""
    parsed = [_parse_numeric_token(event.get('type', '')) for event in events]
    if events and all(value is not None for value in parsed):
        for event, value in zip(events, parsed):
            event['type'] = value
        return events
    for event in events:
        event['type'] = str(event.get('type', ''))
    return events


def _duration_samples(duration_seconds, srate):
    """Convert an annotation duration in seconds to samples."""
    samples = float(duration_seconds) * float(srate)
    rounded = round(samples)
    if abs(samples - rounded) <= 1e-6:
        return int(rounded)
    return samples


def _annotation_duration_seconds(event, srate):
    """Convert ``event['duration']`` in samples to seconds."""
    duration = event.get('duration', 0)
    if duration in (None, ''):
        return 0.0
    coerced = _coerce_extra(duration)
    if coerced is _DROP or coerced == 0:
        return 0.0
    return float(coerced) / float(srate)


def _epoch_index_for_event(event, pnts, trials):
    """Return the 0-based epoch index for one event."""
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
    """Return seconds from time 0 for one EEGLAB latency."""
    return (float(event['latency']) - epoch_index * pnts - 1.0) / srate + xmin


def _latency_from_onset(onset, epoch_index, pnts, srate, xmin):
    """Return a 1-based EEGLAB latency for an onset in seconds."""
    return (float(onset) - xmin) * srate + 1.0 + epoch_index * pnts


def _in_window(onset, tmin, tmax, sfreq):
    """Return whether onset falls in ``[tmin, tmax + 1/sfreq)``.

    The end is one sample past ``tmax``, matching EEGLAB's half-open epoch
    window so a boundary sample belongs to one epoch.
    """
    return float(tmin) <= float(onset) < float(tmax) + 1.0 / float(sfreq)


def _is_time_lock(onset, sfreq):
    """Return whether an onset is the epoch time-locking sample."""
    return abs(float(onset)) * float(sfreq) < _TIME_LOCK_SAMPLES


def _event_type_name(epochs, code):
    """Return the plain-string event_id name for one event code."""
    for name, value in (epochs.event_id or {}).items():
        if value == code:
            return str(name)
    return str(int(code))


def _mne_epochs_to_eeg_events(epochs):
    """Build EEGLAB events for one MNE Epochs object.

    The time-locking event is placed at time 0. Other annotations are included
    when their onset falls in ``[tmin, tmax + 1/sfreq)``. Nonzero annotation
    durations become ``event['duration']`` in samples.
    """
    sfreq = float(epochs.info['sfreq'])
    raw_sfreq = float(epochs._raw_sfreq)
    tmin = float(epochs.tmin)
    tmax = float(epochs.tmax)
    pnts = len(epochs.times)
    tzeros = np.asarray(epochs.events[:, 0], dtype=float) / raw_sfreq
    annotations = epochs.annotations
    rows = []
    if annotations is not None and len(annotations):
        rows = [annotations[index] for index in range(len(annotations))]
    events = []
    for epoch_index, tzero in enumerate(tzeros):
        locking_type = _event_type_name(epochs, int(epochs.events[epoch_index, 2]))
        found_locking = False
        epoch_events = []
        for ann in rows:
            onset = float(ann['onset']) - float(tzero)
            extras = dict(ann.get('extras') or {})
            declared_epoch = extras.get('epoch')
            if declared_epoch is not None and int(declared_epoch) != epoch_index + 1:
                continue
            if not _in_window(onset, tmin, tmax, sfreq):
                continue
            description = str(ann['description'])
            if _is_time_lock(onset, sfreq) and description == locking_type:
                found_locking = True
            latency = extras.get('latency')
            if latency is None:
                latency = _latency_from_onset(onset, epoch_index, pnts, sfreq, tmin)
            event = {'type': description, 'latency': latency, 'epoch': epoch_index + 1}
            duration_seconds = float(ann['duration'])
            if duration_seconds:
                event['duration'] = _duration_samples(duration_seconds, sfreq)
            for key, value in extras.items():
                if key in _EVENT_EXTRA_SKIP:
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
    return _restore_numeric_event_types(events)


def _apply_eeg_events_to_mne(epochs, EEG):
    """Attach in-window EEG events to an MNE Epochs object."""
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
    durations = []
    descriptions = []
    extras = []
    next_id = 1
    for epoch_index, group in enumerate(grouped):
        relatives = [_relative_seconds(event, epoch_index, pnts, srate, xmin) for event in group]
        lock_at = min(range(len(group)), key=lambda index: (abs(relatives[index]), index))
        locking_type = str(group[lock_at]['type'])
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
                if key in _EVENT_EXTRA_SKIP:
                    continue
                coerced = _coerce_extra(value)
                if coerced is _DROP:
                    continue
                extra[key] = coerced
            extra.setdefault('epoch', epoch_index + 1)
            onsets.append(epoch_index * pnts / srate + relative)
            durations.append(_annotation_duration_seconds(event, srate))
            descriptions.append(str(event.get('type', '')))
            extras.append(extra)

    epochs.events = np.column_stack(
        [
            np.asarray(samples, dtype=np.int64),
            np.zeros(trials, dtype=np.int64),
            np.asarray(codes, dtype=np.int64),
        ]
    )
    epochs.event_id = {str(name): int(code) for name, code in id_map.items()}
    if onsets:
        epochs.set_annotations(
            mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions,
                extras=extras,
            )
        )


def _rebuild_epoch_structure(EEG):
    """Fill ``EEG['epoch']`` from ``EEG['event']`` using ``eeg_checkset``."""
    trials = int(EEG.get('trials', 1) or 1)
    if trials <= 1:
        return EEG
    events = _drop_dangling_urevent(_iter_events(EEG.get('event')), EEG.get('urevent'))
    checked = eeg_checkset(
        {
            'data': EEG['data'],
            'nbchan': int(EEG['nbchan']),
            'pnts': int(EEG['pnts']),
            'trials': trials,
            'srate': float(EEG['srate']),
            'xmin': float(EEG['xmin']),
            'xmax': float(EEG['xmax']),
            'event': events,
            'epoch': [],
        },
        'eventconsistency',
    )
    EEG['event'] = checked['event']
    EEG['epoch'] = checked['epoch']
    return EEG
