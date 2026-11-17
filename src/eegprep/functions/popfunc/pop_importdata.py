"""Import channel data into an EEGPrep EEG structure."""

from __future__ import annotations

from pathlib import Path
import ast
import re
from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.listdlg2 import listdlg2
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from eegprep.functions.popfunc._file_io import eeg_from_data, infer_dataformat, load_data_array
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args
from eegprep.functions.popfunc.pop_editset import _history_value, pop_editset

_DATA_FORMATS = ('array', 'ascii', 'float32le', 'float32be', 'matlab', 'auto')


def pop_importdata(
    *args: Any, return_com: bool = False, gui: bool | None = None, renderer: Any = None, datasets=(), **kwargs: Any
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import data from an array or supported file into an EEG dataset."""
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if gui is True or (gui is None and not options):
        initial = dict(options)
        if 'dataformat' in initial:
            initial['dataformat'] = _DATA_FORMATS.index(initial['dataformat']) + 1
        result = inputgui(pop_importdata_dialog_spec(datasets), initial_values=initial, renderer=renderer)
        if result is None:
            empty = eeg_emptyset()
            return (empty, '') if return_com else empty
        options = {key: value for key, value in result.items() if value != ''}
        options['dataformat'] = _DATA_FORMATS[int(result['dataformat']) - 1]
        for key in ('srate', 'xmin'):
            options[key] = float(result[key] or 0)
        for key in ('nbchan', 'pnts'):
            options[key] = int(result[key] or 0)
        for key in ('chanlocs', 'icaweights', 'icasphere'):
            if key in options:
                match = re.fullmatch(r'ALLEEG\((\d+)\)\.(\w+)', str(options[key]))
                if match:
                    index, field = int(match[1]) - 1, match[2]
                    if not 0 <= index < len(datasets) or field != key:
                        raise ValueError(
                            f'pop_importdata: invalid dataset selection for {key}. Select an available dataset.'
                        )
                    options[key] = datasets[index][field]
        if options['dataformat'] == 'array':
            try:
                options['data'] = np.asarray(ast.literal_eval(str(options.get('data', ''))))
            except (ValueError, SyntaxError) as exc:
                raise ValueError(
                    'pop_importdata: invalid array. Enter a Python numeric array literal or select a file format.'
                ) from exc
    if "data" not in options:
        raise ValueError("pop_importdata requires a 'data' file or array")
    data_source = options["data"]
    nbchan = _optional_int(options.get("nbchan"))
    dataformat = infer_dataformat(
        data_source if isinstance(data_source, (str, Path)) else None, options.get("dataformat")
    )
    data = load_data_array(
        data_source,
        dataformat=dataformat,
        nbchan=nbchan,
        variable=options.get("variable"),
    )
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.ndim == 2 and (
        (not nbchan and data.shape[0] > data.shape[1])
        or (nbchan and data.shape[0] != nbchan and data.shape[1] == nbchan)
    ):
        data = data.T
    pnts = _optional_int(options.get("pnts"))
    if pnts and data.ndim == 2:
        if data.shape[1] % pnts:
            raise ValueError(
                "pop_importdata: pnts must divide the sample count. Enter an equal-length epoch size or 0."
            )
        data = data.reshape(data.shape[0], -1, pnts).transpose(0, 2, 1)
    options['srate'] = float(options.get('srate', 1) or 1)
    filename = str(Path(data_source).name) if isinstance(data_source, (str, Path)) else ""
    filepath = str(Path(data_source).parent) if isinstance(data_source, (str, Path)) else ""
    eeg = eeg_from_data(
        data,
        srate=float(options.get("srate", 1.0) or 1.0),
        setname=str(options.get("setname") or ""),
        nbchan=nbchan,
        chanlocs=[],
        xmin=float(options.get("xmin", 0.0) or 0.0),
        subject=str(options.get("subject") or ""),
        condition=str(options.get("condition") or ""),
        group=str(options.get("group") or ""),
        session=options.get("session"),
        comments=str(options.get("comments") or ""),
        ref=options.get("ref", "common"),
        filename=filename,
        filepath=filepath,
    )
    metadata = {key: options[key] for key in ('chanlocs', 'icaweights', 'icasphere', 'icaweight') if key in options}
    if metadata:
        eeg = pop_editset(eeg, gui=False, **metadata)
    command = _history_command(options, dataformat)
    eeg["history"] = command
    return (eeg, command) if return_com else eeg


def _history_command(options: dict[str, Any], dataformat: str) -> str:
    values = dict(options)
    values["dataformat"] = dataformat
    pieces = []
    for key in [
        "data",
        "dataformat",
        "setname",
        "srate",
        "pnts",
        "xmin",
        "nbchan",
        "subject",
        "condition",
        "session",
        "group",
        "ref",
        "comments",
        "chanlocs",
        "icaweights",
        "icaweight",
        "icasphere",
    ]:
        if key in values and _history_value_present(values[key]):
            pieces.extend([format_history_value(key), _history_value(values[key])])
    return f"EEG = pop_importdata({', '.join(pieces)});"


def _history_value_present(value: Any) -> bool:
    if value is None:
        return False
    return not (isinstance(value, str) and value == "")


def _optional_int(value: Any) -> int | None:
    if value in (None, "", 0, "0"):
        return None
    return int(value)


def pop_importdata_dialog_spec(datasets=()) -> DialogSpec:
    """Return the data import form, with channel and ICA dataset selectors."""
    controls = [
        ControlSpec('text', 'Data file/array (select the format)', font_weight='bold'),
        ControlSpec(
            'popupmenu',
            'Array|ASCII text file|float32 le file|float32 be file|MATLAB .mat file|Auto detect',
            tag='dataformat',
            value=6,
        ),
        ControlSpec('edit', tag='data'),
        ControlSpec('pushbutton', 'Browse', tag='browse_data', callback=_browse('data')),
        ControlSpec('text', 'Dataset name', font_weight='bold'),
        ControlSpec('edit', tag='setname'),
        ControlSpec('spacer'),
    ]
    for left, left_tag, default, right, right_tag in [
        ('Data sampling rate (Hz)', 'srate', '1', 'Subject code', 'subject'),
        ('Time points per epoch (0->continuous)', 'pnts', '0', 'Task condition', 'condition'),
        ('Start time (sec) (only for data epochs)', 'xmin', '0', 'Session number', 'session'),
        ('Number of channels (0->set from data)', 'nbchan', '0', 'Subject group', 'group'),
        ('Ref. channel indices or mode (see help)', 'ref', '', 'About this dataset', 'comments'),
    ]:
        controls.extend(
            [
                ControlSpec('text', left),
                ControlSpec('edit', tag=left_tag, value=default),
                ControlSpec('text', right),
                ControlSpec('edit', tag=right_tag),
            ]
        )
    controls.append(ControlSpec('spacer'))
    for label, tag in [
        ('Channel location file or info', 'chanlocs'),
        ('ICA weights array or text/binary file (if any)', 'icaweights'),
        ('ICA sphere array or text/binary file (if any)', 'icasphere'),
    ]:
        controls.extend(
            [
                ControlSpec('text', label),
                ControlSpec(
                    'pushbutton',
                    'From other dataset',
                    tag='dataset_' + tag,
                    enabled=bool(datasets),
                    callback=CallbackSpec(
                        'run_dialog_action',
                        {
                            'button': 'dataset_' + tag,
                            'function': 'pop_importdata',
                            'action': _dataset_selector(datasets, tag),
                        },
                    ),
                ),
                ControlSpec('edit', tag=tag),
                ControlSpec('pushbutton', 'Browse', tag='browse_' + tag, callback=_browse(tag)),
            ]
        )
    return DialogSpec(
        title='Import dataset info -- pop_importdata()',
        function_name='pop_importdata',
        help_text='pop_importdata',
        eeglab_source='functions/popfunc/pop_importdata.m',
        geometry=([1.4, 0.7, 0.8, 0.5], [2, 3.02], 1, *[[2.5, 1, 1.5, 1.5]] * 5, 1, *[[1.4, 0.7, 0.8, 0.5]] * 3),
        controls=tuple(controls),
        size=(1000, 480),
        known_differences=('Array input uses a Python literal; dataset selectors use the shared session datasets.',),
    )


def _browse(tag):
    return CallbackSpec('select_file', {'button': 'browse_' + tag, 'target': tag, 'caption': 'Select a file'})


def _dataset_selector(datasets, field):
    def select(values, update):
        selected, accepted, _ = listdlg2(
            liststring=[f"{index}: {eeg.get('setname', '')}" for index, eeg in enumerate(datasets, 1)],
            selectionmode='single',
            name='Select dataset',
            promptstring='Select the source dataset',
        )
        if accepted:
            update({field: f'ALLEEG({selected[0]}).{field}'})

    return select
