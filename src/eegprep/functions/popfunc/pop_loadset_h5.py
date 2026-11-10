"""Decode MATLAB v7.3 EEG datasets without retaining HDF5 references."""

import h5py
import numpy as np

from eegprep.functions.popfunc._matlab_fields import apply_field_rules


def pop_loadset_h5(file_name):
    """Load a MATLAB v7.3 EEG dataset, including any companion sample file."""
    from eegprep.functions.popfunc.pop_loadset import pop_loadset

    return pop_loadset(file_name)


def _read_set_h5(file_name):
    with h5py.File(file_name, 'r') as file:
        group = file.get('EEG', file)
        return {key: _read_value(value, key) for key, value in group.items() if not key.startswith('#')}


def _read_value(node, field=None):
    if isinstance(node, h5py.Group):
        children = list(node.items())
        if not children:
            return {}
        # MATLAB struct arrays store one object reference per field per element.
        if all(
            isinstance(child, h5py.Dataset) and h5py.check_dtype(ref=child.dtype) and 'MATLAB_class' not in child.attrs
            for _, child in children
        ):
            count = children[0][1].size
            records = [{} for _ in range(count)]
            for key, child in children:
                for record, reference in zip(records, child[()].reshape(-1)):
                    record[key] = _read_value(node.file[reference], key) if reference else np.empty((0, 0))
            value = records[0] if count == 1 else records
        else:
            value = {key: _read_value(child, key) for key, child in children}
        return apply_field_rules(field, value)

    raw = node[()]
    matlab_class = node.attrs.get('MATLAB_class', b'')
    if node.attrs.get('MATLAB_empty', False):
        value = '' if matlab_class == b'char' else np.empty(tuple(int(n) for n in raw.reshape(-1)))
    elif matlab_class == b'char':
        rows = np.atleast_2d(raw.T)
        texts = [np.asarray(row, dtype='<u2').tobytes().decode('utf-16-le') for row in rows]
        value = texts[0] if len(texts) == 1 else np.asarray(texts, dtype=object)
    elif h5py.check_dtype(ref=node.dtype):
        values = [_read_value(node.file[ref]) if ref else np.empty((0, 0)) for ref in raw.reshape(-1)]
        value = np.empty(raw.size, dtype=object)
        value[:] = values
        value = value.reshape(raw.shape).T
        if value.ndim <= 2 and 1 in value.shape:
            value = list(value.reshape(-1))
    else:
        # MATLAB reverses every axis in HDF5, including square and cubic arrays.
        value = np.asarray(raw).T
    return apply_field_rules(field, value)
