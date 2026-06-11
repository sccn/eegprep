"""EEG data saving and loading utilities."""

import os
from pathlib import Path

import numpy as np
import scipy.io

from eegprep.functions.adminfunc.storage import (
    MemmapData,
    OffloadedData,
    dataset_datfile_path,
    dataset_set_path,
    memmap_enabled,
    memmap_fdt,
    savetwofiles_enabled,
    write_fdt,
)
from eegprep.functions.popfunc._pop_utils import parse_key_value_args

# Allows access using . notation
# class EEG:
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)
#     def __getitem__(self, key):
#         return self.__dict__[key]
#     def __setitem__(self, key, value):
#         self.__dict__[key] = value

default_empty = np.array([])


def flatten_dict_sub(d, parent_key='', sep='_'):
    """Flatten a nested dictionary.

    Parameters
    ----------
    d : dict
        Dictionary to flatten.
    parent_key : str, optional
        Parent key.
    sep : str, optional
        Separator.

    Returns
    -------
    dict
        Flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict_sub(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _is_numeric(val):
    """Check if a value is numeric (Python or numpy scalar)."""
    if isinstance(val, (int, float, np.integer, np.floating)):
        return True
    if isinstance(val, str):
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False
    return False


def _string_field(value):
    """Return a scalar MATLAB string field value."""
    if value is None:
        return ''
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ''
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def _matlab_empty_if_missing(EEG, key):
    """Return an EEGLAB empty array for optional fields missing from EEG."""
    value = EEG.get(key, default_empty)
    return default_empty if value is None else value


def _matlab_empty_struct_if_missing(EEG, key):
    """Return an EEGLAB empty array for optional empty struct-like fields."""
    value = _matlab_empty_if_missing(EEG, key)
    if isinstance(value, dict) and not value:
        return default_empty
    return value


def flatten_dict(data):
    """Flatten dictionary data.

    Parameters
    ----------
    data : list
        List of dictionaries.

    Returns
    -------
    np.recarray
        Flattened data.
    """
    # Flatten each dictionary and collect the fields and types
    flat_data = [flatten_dict_sub(item) for item in data]
    fields = list(flat_data[0].keys())
    dtypes = []
    has_object = False

    # Determine data types — EEGLAB expects numeric scalar fields as double
    # Check ALL values per field (not just the first) to handle mixed types
    # e.g., event type can be numeric (2) for regular events and 'boundary' for boundary events
    for field in fields:
        all_values = [item[field] for item in flat_data]
        has_seq = any(isinstance(v, (list, np.ndarray)) for v in all_values)
        if has_seq:
            dtypes.append((field, 'O'))
            has_object = True
        elif all(_is_numeric(v) for v in all_values):
            dtypes.append((field, np.float64))
        else:
            dtypes.append((field, 'U{}'.format(max(len(str(v)) for v in all_values))))

    # If any field requires object dtype, use object for all fields
    # (scipy.io.savemat handles mixed typed/object recarrays poorly)
    if has_object:
        dtype = np.dtype([(f, 'O') for f in fields])
        data_tuples = [tuple(item[field] for field in fields) for item in flat_data]
    else:
        dtype = np.dtype(dtypes)
        data_tuples = []
        for item in flat_data:
            row = []
            for field, (_, dt) in zip(fields, dtypes):
                val = item[field]
                if dt == np.float64:
                    row.append(float(val))
                else:
                    row.append(val)
            data_tuples.append(tuple(row))

    rec_array = np.array(data_tuples, dtype=dtype).view(np.recarray)
    return rec_array


def saveset(EEG, file_name):
    """Save EEG data to file.

    Parameters
    ----------
    EEG : dict
        EEG data.
    file_name : str
        File name.

    Returns
    -------
    dict
        EEG data.
    """
    return pop_saveset(EEG, file_name)


# def dictlist_to_recarray(events):
#     # --- Infer dtype automatically ---
#     dtype_fields = []
#     for key in events[0].keys():
#         values = [e[key] for e in events]

#         # If string: pick Unicode type with max length
#         if all(isinstance(v, str) for v in values):
#             maxlen = max(len(v) for v in values)
#             dtype_fields.append((key, f'U{maxlen}'))

#         # If integer: use int32
#         elif all(isinstance(v, int) for v in values):
#             dtype_fields.append((key, 'i4'))

#         # If float or mixed int/float: use float64
#         elif all(isinstance(v, (float, int)) for v in values):
#             dtype_fields.append((key, 'f8'))

#         else:
#             # fallback: generic object
#             dtype_fields.append((key, object))

#     dtype = np.dtype(dtype_fields)

#     # --- Convert events to recarray ---
#     rec_events = np.array(
#         [tuple(e[k] for k in events[0].keys()) for e in events],
#         dtype=dtype
#     ).view(np.recarray)

#     return rec_events


def pop_saveset_old(EEG, file_path):
    """Save EEG data to file (old version).

    Parameters
    ----------
    EEG : dict
        EEG data.
    file_path : str
        File path.

    Returns
    -------
    dict
        EEG data.
    """
    # convert Events to structured array
    # if 'event' in EEG:
    #     EEG['event'] = flatten_dict(EEG['event'])

    # add 1 to EEG['icachansind'] to make it 1-based
    if 'icachansind' in EEG and EEG['icachansind'].size > 0:
        EEG['icachansind'] = EEG['icachansind'] + 1

    # search for array of dictionaries and convert them to flatten_dicts
    for key in EEG:
        if isinstance(EEG[key], np.ndarray) and len(EEG[key]) > 0 and isinstance(EEG[key][0], dict):
            EEG[key] = flatten_dict(EEG[key])

    EEG['icaact'] = default_empty
    scipy.io.savemat(file_path, EEG)

    return EEG


def _chanlocs_to_struct_array(chanlocs_list):
    """Convert a list/array of chanloc dicts to a structured numpy array.

    scipy.io.savemat writes an object-array of dicts as a MATLAB *cell* array
    of structs, but EEGLAB expects a *struct* array.  Converting to a structured
    numpy array beforehand makes savemat produce the correct MATLAB type.

    Used for both ``EEG.chanlocs`` and ``EEG.chaninfo.removedchans``.
    """
    if chanlocs_list is None or len(chanlocs_list) == 0:
        return np.array([])

    # Normalise: accept numpy object-arrays of dicts as well as plain lists
    items = list(chanlocs_list)
    if not isinstance(items[0], dict):
        return np.array(chanlocs_list)  # already structured, leave as-is

    matlab_null = np.array([])

    # Chanloc fields with their target dtypes
    field_spec = [
        ('labels', 'U100'),
        ('theta', np.float64),
        ('radius', np.float64),
        ('X', np.float64),
        ('Y', np.float64),
        ('Z', np.float64),
        ('sph_theta', np.float64),
        ('sph_phi', np.float64),
        ('sph_radius', np.float64),
        ('type', 'U10'),
        ('urchan', np.int32),
        ('ref', 'U100'),
        ('unit', 'U20'),
    ]

    # Extract values, replacing empty numpy arrays with None
    d_list = []
    for c in items:
        row = {}
        for fld, _ in field_spec:
            val = c.get(fld, matlab_null)
            if isinstance(val, np.ndarray) and val.size == 0:
                row[fld] = None
            else:
                row[fld] = val
        d_list.append(row)

    # Keep only fields where at least one entry is non-None
    retain = [f for f, _ in field_spec if not all(d[f] is None for d in d_list)]
    if not retain:
        return np.array([])

    dtype = np.dtype([(f, t) for f, t in field_spec if f in retain])
    arr = np.array(
        [
            tuple(
                d[f] if d[f] is not None else (0 if np.issubdtype(t, np.number) else '')
                for f, t in field_spec
                if f in retain
            )
            for d in d_list
        ],
        dtype=dtype,
    )
    return arr


def _serialize_chaninfo(chaninfo):
    """Prepare chaninfo dict so savemat produces MATLAB struct arrays.

    The critical sub-field is ``removedchans``: an object-array of dicts that
    savemat would serialize as a cell array of structs.  MATLAB's pop_interp
    expects a struct array, so we convert it here.
    """
    if chaninfo is None or (isinstance(chaninfo, np.ndarray) and chaninfo.size == 0):
        return np.array([])
    if not isinstance(chaninfo, dict):
        return chaninfo

    out = {}
    for k, v in chaninfo.items():
        if k in ('removedchans', 'nodatchans'):
            if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                first = v[0] if isinstance(v, list) else v.flat[0]
                if isinstance(first, dict):
                    out[k] = _chanlocs_to_struct_array(v)
                else:
                    out[k] = v
            else:
                out[k] = np.array([])
        else:
            out[k] = v
    return out


def pop_saveset(EEG, file_name=None, *args, **kwargs):
    """Save EEG data to file.

    Parameters
    ----------
    EEG : dict
        EEG data.
    file_name : str, optional
        File name. EEGLAB-style ``filename``/``filepath``/``savemode`` keyword
        arguments are also accepted.
    """
    file_path, savemode = _save_target(EEG, file_name, args, kwargs)
    file_name = os.fspath(file_path)
    save_dir = os.path.dirname(file_name) or '.'
    save_name = os.path.basename(file_name)
    datfile_name = f"{Path(save_name).stem}.fdt"
    save_two_files = _save_two_files(EEG, savemode)
    data_value = EEG.get('data', default_empty)
    if isinstance(data_value, OffloadedData):
        raise RuntimeError("Cannot save offloaded EEG data; retrieve the dataset before saving it.")
    mat_data = datfile_name if save_two_files else np.asarray(data_value)
    eeglab_dict = {
        'setname': _string_field(EEG.get('setname', '')),
        'filename': save_name,
        'filepath': save_dir,
        'subject': _string_field(EEG.get('subject', '')),
        'group': _string_field(EEG.get('group', '')),
        'condition': _string_field(EEG.get('condition', '')),
        'session': EEG.get('session', np.array([])),
        'comments': _string_field(EEG.get('comments', '')),
        'nbchan': float(EEG['nbchan']),
        'trials': float(EEG['trials']),
        'pnts': float(EEG['pnts']),
        'srate': float(EEG['srate']),
        'xmin': float(EEG['xmin']),
        'xmax': float(EEG['xmax']),
        'times': EEG['times'],
        'data': mat_data,
        'datfile': datfile_name if save_two_files else '',
        'icaact': _matlab_empty_if_missing(EEG, 'icaact'),
        'icawinv': _matlab_empty_if_missing(EEG, 'icawinv'),
        'icasphere': _matlab_empty_if_missing(EEG, 'icasphere'),
        'icaweights': _matlab_empty_if_missing(EEG, 'icaweights'),
        'icachansind': _matlab_empty_if_missing(EEG, 'icachansind').copy(),
        'chanlocs': _matlab_empty_if_missing(EEG, 'chanlocs'),
        'urchanlocs': _matlab_empty_if_missing(EEG, 'urchanlocs'),
        'chaninfo': _serialize_chaninfo(EEG.get('chaninfo', {})),
        'ref': EEG.get('ref', 'common'),
        'event': _matlab_empty_if_missing(EEG, 'event'),
        'urevent': _matlab_empty_if_missing(EEG, 'urevent'),
        'eventdescription': _matlab_empty_if_missing(EEG, 'eventdescription'),
        'epoch': _matlab_empty_if_missing(EEG, 'epoch'),
        'epochdescription': _matlab_empty_if_missing(EEG, 'epochdescription'),
        'reject': _matlab_empty_if_missing(EEG, 'reject'),
        'stats': _matlab_empty_if_missing(EEG, 'stats'),
        'specdata': _matlab_empty_if_missing(EEG, 'specdata'),
        'specicaact': _matlab_empty_if_missing(EEG, 'specicaact'),
        'splinefile': _matlab_empty_if_missing(EEG, 'splinefile'),
        'icasplinefile': _matlab_empty_if_missing(EEG, 'icasplinefile'),
        'dipfit': _matlab_empty_if_missing(EEG, 'dipfit'),
        'history': EEG.get('history', ''),
        'saved': EEG.get('saved', 'yes'),
        'etc': _matlab_empty_struct_if_missing(EEG, 'etc'),
        'run': _matlab_empty_if_missing(EEG, 'run'),
        'roi': _matlab_empty_if_missing(EEG, 'roi'),
    }

    # add 1 to EEG['icachansind'] to make it 1-based
    if isinstance(eeglab_dict['icachansind'], list):
        eeglab_dict['icachansind'] = np.array(eeglab_dict['icachansind'])
    if 'icachansind' in eeglab_dict and eeglab_dict['icachansind'].size > 0:
        eeglab_dict['icachansind'] = eeglab_dict['icachansind'] + 1

    # check if EEG['urchan'] is 0-based
    if len(eeglab_dict['chanlocs']) > 0 and 'urchan' in eeglab_dict['chanlocs'][0]:
        for i in range(len(eeglab_dict['chanlocs'])):
            eeglab_dict['chanlocs'][i]['urchan'] = eeglab_dict['chanlocs'][i]['urchan'] + 1

    # check if EEG['event'][i]['urevent'] is 0-based
    if len(eeglab_dict['event']) > 0 and 'urevent' in eeglab_dict['event'][0]:
        for i in range(len(eeglab_dict['event'])):
            eeglab_dict['event'][i]['urevent'] = eeglab_dict['event'][i]['urevent'] + 1

    # Create the list of dictionaries with a string field
    if 'chanlocs' in EEG and len(EEG['chanlocs']) > 0:
        matlab_null = np.array([])
        d_list = [
            {
                'labels': c['labels'],
                'theta': c['theta'] if not isinstance(c.get('theta', matlab_null), np.ndarray) else None,
                'radius': c['radius'] if not isinstance(c.get('radius', matlab_null), np.ndarray) else None,
                'X': c['X'] if not isinstance(c.get('X', matlab_null), np.ndarray) else None,
                'Y': c['Y'] if not isinstance(c.get('Y', matlab_null), np.ndarray) else None,
                'Z': c['Z'] if not isinstance(c.get('Z', matlab_null), np.ndarray) else None,
                'sph_theta': c['sph_theta'] if not isinstance(c.get('sph_theta', matlab_null), np.ndarray) else None,
                'sph_phi': c['sph_phi'] if not isinstance(c.get('sph_phi', matlab_null), np.ndarray) else None,
                'sph_radius': c['sph_radius'] if not isinstance(c.get('sph_radius', matlab_null), np.ndarray) else None,
                'type': c['type'] if not isinstance(c.get('type', matlab_null), np.ndarray) else None,
                'urchan': c['urchan'] if not isinstance(c.get('urchan', matlab_null), np.ndarray) else None,
                'ref': c['ref'] if not isinstance(c.get('ref', matlab_null), np.ndarray) else None,
            }
            for c in EEG['chanlocs']
        ]

        # build a list of fields to selectively filter out if all entries are None
        retain_fields = [fld for fld in d_list[0].keys() if not all(d[fld] is None for d in d_list)]

        dtype = np.dtype(
            [
                (f, t)
                for f, t in [
                    ('labels', 'U100'),  # String up to 100 characters
                    ('theta', np.float64),
                    ('radius', np.float64),
                    ('X', np.float64),
                    ('Y', np.float64),
                    ('Z', np.float64),
                    ('sph_theta', np.float64),
                    ('sph_phi', np.float64),
                    ('sph_radius', np.float64),
                    ('type', 'U10'),  # String up to 10 characters
                    ('urchan', np.int32),
                    ('ref', 'U100'),  # String up to 100 characters
                ]
                if f in retain_fields
            ]
        )

        # Convert the list of dictionaries to a structured NumPy array
        eeglab_dict['chanlocs'] = np.array([tuple(item[fld] for fld in retain_fields) for item in d_list], dtype=dtype)

    # Normalize event latencies to float before saving so MATLAB loads them
    # as double.  Without this, integer stimulus latencies become int64 and
    # MATLAB's [struct.field] concatenation coerces boundary .5 latencies to
    # int64, losing the fractional part.
    if isinstance(eeglab_dict['event'], (list, np.ndarray)):
        for ev in eeglab_dict['event']:
            if isinstance(ev, dict) and 'latency' in ev:
                ev['latency'] = float(ev['latency'])

    if isinstance(eeglab_dict['event'], list):
        eeglab_dict['event'] = np.array(eeglab_dict['event'])

    for key in eeglab_dict:
        if (
            isinstance(eeglab_dict[key], np.ndarray)
            and len(eeglab_dict[key]) > 0
            and isinstance(eeglab_dict[key][0], dict)
        ):
            eeglab_dict[key] = flatten_dict(eeglab_dict[key])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if save_two_files:
        write_fdt(data_value, Path(save_dir) / datfile_name, EEG)

    # # Step 4: Save the EEGLAB dataset as a .mat file
    try:
        scipy.io.savemat(file_name, eeglab_dict, appendmat=False)
    except ValueError as e:
        if '31 characters' in str(e):
            # try to save with long_field_names option
            scipy.io.savemat(file_name, eeglab_dict, appendmat=False, long_field_names=True)
        else:
            # the file is likely partial and thus invalid -- delete
            if os.path.exists(file_name):
                os.remove(file_name)
            raise
    EEG['filename'] = save_name
    EEG['filepath'] = save_dir
    EEG['datfile'] = datfile_name if save_two_files else ''
    EEG['saved'] = 'yes'
    if save_two_files and (isinstance(data_value, MemmapData) or memmap_enabled()):
        EEG['data'] = memmap_fdt(Path(save_dir) / datfile_name, EEG)
    elif not save_two_files and isinstance(data_value, MemmapData):
        EEG['data'] = np.asarray(data_value).copy()
    return EEG


def _save_target(EEG, file_name, args, kwargs):
    known_keys = {'filename', 'filepath', 'savemode', 'check', 'version'}
    if isinstance(file_name, str) and file_name.lower() in known_keys:
        options = parse_key_value_args((file_name, *args), kwargs, lowercase_keys=True, lowercase_kwargs=True)
    else:
        options = parse_key_value_args(args, kwargs, lowercase_keys=True, lowercase_kwargs=True)
        if file_name is not None:
            options.setdefault('filename', file_name)
    unknown = sorted(set(options) - known_keys)
    if unknown:
        raise ValueError(f"Unsupported pop_saveset option(s): {', '.join(unknown)}")
    savemode = str(options.get('savemode') or '').lower()
    if savemode and savemode not in {'resave', 'onefile', 'twofiles'}:
        raise ValueError("savemode must be 'resave', 'onefile', or 'twofiles'")
    filename = options.get('filename')
    filepath = options.get('filepath')
    if savemode == 'resave' and filename in {None, ''}:
        target = dataset_set_path(EEG)
        if target is None:
            raise ValueError("savemode='resave' requires EEG filename/filepath metadata")
        return target, savemode
    if filename in {None, ''}:
        raise ValueError("file_name argument is required")
    path = Path(os.fspath(filename))
    if filepath not in {None, ''} and not path.is_absolute():
        path = Path(os.fspath(filepath)) / path.name
    return path, savemode


def _save_two_files(EEG, savemode):
    if savemode == 'onefile':
        return False
    if savemode == 'twofiles':
        return True
    datfile = dataset_datfile_path(EEG)
    return bool(datfile and savemode == 'resave') or savetwofiles_enabled()


def test_pop_saveset():
    """Test pop_saveset function."""
    from eegprep.functions.popfunc.pop_loadset import pop_loadset

    file_path = './sample_data/eeglab_data_with_ica_tmp.set'
    EEG = pop_loadset(file_path)
    pop_saveset(EEG, '/Users/arno/Python/eegprep/sample_data/tmp.set')
    pop_saveset_old(
        EEG, '/Users/arno/Python/eegprep/sample_data/tmp2.set'
    )  # does not do events and function above is better
    # print the keys of the EEG dictionary
    print(EEG.keys())


if __name__ == '__main__':
    test_pop_saveset()

# STILL OPEN QUESTION: Better to have empty MATLAB arrays as None for empty numpy arrays (current default).
# The current default is to make it more MALTAB compatible. A lot of MATLAB function start indexing MATLAB
# empty arrays to add values to them. This is not possible with None and would create more conversion and
# bugs. However, None is more pythonic.
