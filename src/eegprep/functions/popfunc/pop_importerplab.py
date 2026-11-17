"""Map ERPLAB event-list and bin definitions onto existing EEG events."""

from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
import re
import shlex
from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._file_io import events_to_records
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_numeric_sequence

logger = logging.getLogger(__name__)
_BIN_ID = r'(?:bin|b)?(\d+)'


def pop_importerplab(
    EEG: dict[str, Any],
    file1: str | Path | None = None,
    file2: str | Path | None = None,
    ncbins: bool = False,
    *,
    gui: bool | None = None,
    renderer: Any = None,
    return_com: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Add ERPLAB labels and bin membership to existing events.

    ``file1`` contains numeric event codes, quoted names, and event indices.
    ``file2`` defines simple bins and contrasts. ``ncbins`` also imports
    non-contrast bin membership. Event types and latencies are preserved.
    """
    if gui is True or (gui is None and (file1 is None or file2 is None)):
        result = inputgui(
            pop_importerplab_dialog_spec(),
            initial_values={'file1': str(file1 or ''), 'file2': str(file2 or ''), 'ncbins': ncbins},
            renderer=renderer,
        )
        if result is None:
            return (EEG, '') if return_com else EEG
        file1, file2, ncbins = result['file1'], result['file2'], result['ncbins']
    for filename in (file1, file2):
        if not filename or not Path(filename).is_file():
            raise ValueError(
                f'pop_importerplab: file {filename!r} was not found. Select both an event list and a bin text file.'
            )
    event_map = _read_event_list(file1)
    bins, contrasts = _read_bins(file2)
    out = deepcopy(EEG)
    events = events_to_records(out.get('event'))
    for event in events:
        try:
            label, index = event_map[float(event['type'])]
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(
                f'pop_importerplab: event type {event.get("type")!r} is absent from {file1}. '
                'Include each numeric event type exactly once in the event list.'
            ) from exc
        event.update(newtype=label, eventind=index)
        if ncbins:
            for label, indices in bins.values():
                event[_field_name(label)] = int(index in indices)
    for label, first, second in contrasts:
        if first not in bins or second not in bins:
            logger.warning(
                'pop_importerplab: skipping %s; contrast operands must name simple bins in %s.', label, file2
            )
            continue
        label1, indices1 = bins[first]
        label2, indices2 = bins[second]
        if indices1 & indices2:
            logger.warning('pop_importerplab: skipping %s; use disjoint bins for a contrast in %s.', label, file2)
            continue
        for event in events:
            index = event['eventind']
            event[_field_name(label)] = (
                label1 if index in indices1 else label2 if index in indices2 else 'none of these'
            )
    out['event'] = events
    out['saved'] = 'no'
    command = f'EEG = pop_importerplab(EEG, {format_history_value(str(file1))}, {format_history_value(str(file2))}, {int(ncbins)});'
    out['history'] = command if not EEG.get('history') else f"{EEG['history'].rstrip()}\n{command}"
    return (out, command) if return_com else out


def _read_event_list(filename):
    events = {}
    for number, line in enumerate(Path(filename).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            code, label, index = shlex.split(line)
            code, index = float(code), float(index)
            if code in events:
                raise ValueError('duplicate event code')
        except ValueError as exc:
            raise ValueError(
                f'pop_importerplab: invalid event list {filename}, line {number}. '
                'Use a unique numeric code, quoted label, and numeric event index on each line.'
            ) from exc
        events[code] = (label, index)
    return events


def _read_bins(filename):
    bins, contrasts, identifiers = {}, [], set()
    for number, line in enumerate(Path(filename).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(rf'\s*{_BIN_ID}\s*=\s*(.*?)\s+label\s+(.+?)\s*', line, re.IGNORECASE)
        try:
            if match is None:
                raise ValueError('missing bin identifier, expression or label')
            identifier, expression, label = int(match[1]), match[2], match[3]
            if identifier in identifiers:
                raise ValueError('duplicate bin identifier')
            identifiers.add(identifier)
            if expression.startswith('(') and expression.endswith(')'):
                bins[identifier] = (label, set(parse_numeric_sequence(expression[1:-1])))
            else:
                contrast = re.fullmatch(rf'{_BIN_ID}\s*-\s*{_BIN_ID}', expression, re.IGNORECASE)
                if contrast is None:
                    raise ValueError('invalid contrast')
                contrasts.append((label, int(contrast[1]), int(contrast[2])))
        except ValueError as exc:
            raise ValueError(
                f'pop_importerplab: invalid bin definition in {filename}, line {number}. '
                'Use "b1 = (1 2) label Name" or "b3 = b1-b2 label Contrast", with unique bin numbers.'
            ) from exc
    return bins, contrasts


def _field_name(label):
    name = re.sub(r'[ ()/\-&]', '_', label.strip())
    return 'f' + name if name[0].isdigit() else name


def pop_importerplab_dialog_spec() -> DialogSpec:
    """Return the event-list and bin-file import dialog."""
    controls = []
    for tag, label in [('file1', 'Select an event list text file'), ('file2', 'Select a bin text file')]:
        controls.extend(
            [
                ControlSpec('text', label),
                ControlSpec('edit', tag=tag),
                ControlSpec(
                    'pushbutton',
                    '...',
                    tag=tag + '_browse',
                    callback=CallbackSpec('select_file', {'button': tag + '_browse', 'target': tag, 'caption': label}),
                ),
            ]
        )
    controls.extend(
        [
            ControlSpec('spacer'),
            ControlSpec('checkbox', 'Import all bins including non-contrast bins', tag='ncbins', value=False),
        ]
    )
    return DialogSpec(
        title='Import ERPLAB info files -- pop_importerplab()',
        function_name='pop_importerplab',
        help_text='pop_importerplab',
        eeglab_source='functions/popfunc/pop_importerplab.m',
        geometry=([2, 2, 0.5], [2, 2, 0.5], 1, 1),
        controls=tuple(controls),
        weighted_columns=True,
        size=(640, 174),
    )
