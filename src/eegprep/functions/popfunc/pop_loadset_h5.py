"""Load EEG data from MATLAB v7.3 (HDF5) .set files."""

import h5py
import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.popfunc._file_io import normalize_icachansind, ur_indices_to_zero_based

_NUMERIC = {
    "double": np.float64,
    "single": np.float32,
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
}


def pop_loadset_h5(file_name):
    """Load an EEGLAB dataset from a MATLAB v7.3 (HDF5) .set file.

    Parameters
    ----------
    file_name : str
        Path to the .set file.

    Returns
    -------
    dict
        EEG dataset.
    """
    with h5py.File(file_name, "r") as handle:
        source = handle["EEG"] if "EEG" in handle else handle
        eeg = {key: _decode(source[key], squeeze=key != "data") for key in source if not str(key).startswith("#")}
    if "icachansind" in eeg:
        eeg["icachansind"] = normalize_icachansind(eeg["icachansind"], matlab_one_based=True)
    # A 1-by-1 struct array is one record. scipy squeeze_me returns that as a dict.
    for key in ("chanlocs", "event"):
        if isinstance(eeg.get(key), dict):
            eeg[key] = [eeg[key]]
    ur_indices_to_zero_based(eeg)
    return eeg_checkset(eeg)


def _matlab_class(node):
    if "MATLAB_class" not in node.attrs:
        return ""
    value = np.asarray(node.attrs["MATLAB_class"]).reshape(-1)
    if value.size == 0:
        return ""
    item = value[0]
    if isinstance(item, bytes):
        return item.decode()
    return str(item)


def _is_empty(node):
    if "MATLAB_empty" not in node.attrs:
        return False
    return bool(np.asarray(node.attrs["MATLAB_empty"]).reshape(-1)[0])


def _matlab_order(values):
    """Reverse HDF5 axes so the array matches MATLAB's column-major shape."""
    values = np.asarray(values)
    if values.ndim < 2:
        return values
    return np.transpose(values)


def _squeeze_numeric(values):
    squeezed = np.squeeze(values)
    if squeezed.shape == ():
        return squeezed.item()
    return squeezed


def _codes_to_str(codes):
    return np.ascontiguousarray(codes, dtype="<u2").tobytes().decode("utf-16-le")


def _decode_char(values):
    text = _matlab_order(np.asarray(values, dtype=np.uint16))
    if text.ndim <= 1:
        return _codes_to_str(np.atleast_1d(text).reshape(-1))
    rows = [_codes_to_str(text[index]) for index in range(text.shape[0])]
    if len(rows) == 1:
        return rows[0]
    return np.asarray(rows, dtype=object)


def _decode_ref(handle, ref):
    if not ref:
        return np.array([])
    return _decode(handle[ref])


def _decode_refs(node):
    refs = _matlab_order(np.asarray(node[()]))
    decoded = np.empty(refs.shape, dtype=object)
    for index in np.ndindex(refs.shape):
        decoded[index] = _decode_ref(node.file, refs[index])
    decoded = np.squeeze(decoded)
    if decoded.shape == ():
        return decoded.item()
    return decoded


def _is_struct_array_field(node):
    return (
        isinstance(node, h5py.Dataset)
        and "MATLAB_class" not in node.attrs
        and h5py.check_dtype(ref=node.dtype) is not None
    )


def _decode_struct(group):
    fields = [key for key in group if not str(key).startswith("#")]
    if not fields:
        return {}
    if any(_is_struct_array_field(group[field]) for field in fields):
        return _decode_struct_array(group, fields)
    return {field: _decode(group[field]) for field in fields}


def _decode_struct_array(group, fields):
    references = {}
    shape = None
    for field in fields:
        node = group[field]
        if _is_struct_array_field(node):
            references[field] = _matlab_order(np.asarray(node[()]))
            shape = references[field].shape
    records = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        records[index] = {
            field: _decode_ref(group.file, references[field][index]) if field in references else _decode(group[field])
            for field in fields
        }
    records = np.squeeze(records)
    if records.shape == ():
        return records.item()
    return records


def _decode(node, squeeze=True):
    if isinstance(node, h5py.Group):
        return _decode_struct(node)
    if _is_empty(node):
        return np.array([])
    kind = _matlab_class(node)
    if kind == "char":
        return _decode_char(node[()])
    if kind == "logical":
        return _squeeze_numeric(_matlab_order(np.asarray(node[()], dtype=np.uint8)))
    if kind in {"cell", "struct"} or h5py.check_dtype(ref=node.dtype) is not None:
        return _decode_refs(node)
    values = np.asarray(node[()])
    if kind in _NUMERIC:
        values = values.astype(_NUMERIC[kind], copy=False)
    values = _matlab_order(values)
    if not squeeze:
        return values
    return _squeeze_numeric(values)
