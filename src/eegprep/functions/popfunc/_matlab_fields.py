"""Field shapes shared by the v7 and v7.3 EEGLAB ``.set`` readers.

MATLAB ``load`` (``pop_loadset.m``) keeps every dimension. ``eeg_checkset.m``
then requires ``size(EEG.data, 1) == EEG.nbchan`` and
``size(EEG.icasphere, 2) == length(EEG.icachansind)``.
"""

from __future__ import annotations

import numpy as np
from scipy.io.matlab import mat_struct

MATRIX_FIELDS = frozenset({"data", "icaact", "icaweights", "icasphere", "icawinv"})
VECTOR_FIELDS = frozenset({"times", "icachansind"})
SCALAR_FIELDS = frozenset({"nbchan", "pnts", "trials", "srate", "xmin", "xmax"})
STRUCT_ARRAY_FIELDS = frozenset({"chanlocs", "urchanlocs", "event", "urevent", "epoch"})
# eeg_checkset coerces chanlocs, event, and epoch. These stay arrays.
_ARRAY_FIELDS = frozenset(
    {
        "urchanlocs",
        "urevent",
        "comments",
        "eventdescription",
        "epochdescription",
        "filecontent",
    }
)
_MAT_META = frozenset({"__header__", "__version__", "__globals__"})


def dataset_from_mat(loaded: dict) -> dict:
    """Return the EEG dictionary stored in a MAT v5/v7 file."""
    fields = {key: value for key, value in loaded.items() if key not in _MAT_META}
    if "EEG" in fields:
        dataset = convert_matlab(fields["EEG"], "EEG")
        if isinstance(dataset, dict):
            return dataset
    return {key: convert_matlab(value, key) for key, value in fields.items()}


def convert_matlab(value, field: str | None = None):
    """Unwrap one unsqueezed MATLAB value and apply :func:`apply_field_rules`."""
    return apply_field_rules(field, _unwrap(value))


def matlab_empty(value) -> np.ndarray:
    """Keep the shape and dtype of an empty MATLAB array.

    ``[]`` is 0-by-0. A missing channel list can be 0-by-1, and a selection
    can be 32-by-0. The v7 and v7.3 readers share this through
    :func:`apply_field_rules`.
    """
    array = np.asarray(value)
    if array.ndim == 0:
        return np.empty((0, 0), dtype=array.dtype)
    return np.empty(array.shape, dtype=array.dtype)


def apply_field_rules(field: str | None, value):
    """Apply the EEGLAB shape for ``field``.

    Matrix fields keep the MATLAB shape, including a 1-by-1 ``icasphere``.
    Scalar fields and other non-matrix numerics drop length-1 axes. Vector
    fields become 1-d. A 1-by-1 ``chanlocs``, ``event``, or ``epoch`` struct
    becomes a one-element list. Empty arrays keep the dimensions stored in
    the file.
    """
    if isinstance(value, np.ndarray) and value.size == 0:
        return matlab_empty(value)
    if field in STRUCT_ARRAY_FIELDS and isinstance(value, dict):
        value = [value]
    if field in _ARRAY_FIELDS and isinstance(value, list):
        return _object_array(value)
    if field in MATRIX_FIELDS and _is_scalar(value):
        return np.asarray(value).reshape(1, 1)
    if isinstance(value, np.ndarray) and value.dtype.kind in "iufcb":
        return normalize_numeric(field, value)
    return value


def normalize_numeric(field: str | None, value):
    """Return ``value`` with the numeric shape ``field`` uses in an EEG dict."""
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    if array.dtype.kind not in "iufcb":
        return value
    if array.size == 0:
        return matlab_empty(array)
    if field in MATRIX_FIELDS:
        if array.ndim == 0:
            return array.reshape(1, 1)
        return array
    if field in VECTOR_FIELDS:
        return np.ravel(array, order="F")
    if field in SCALAR_FIELDS or array.size == 1:
        return _python_scalar(array)
    squeezed = np.squeeze(array)
    if squeezed.shape == ():
        return _python_scalar(squeezed)
    return squeezed


def _unwrap(value):
    if isinstance(value, mat_struct):
        return _struct_dict(value)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return matlab_empty(value)
        if value.dtype == object:
            return _unwrap_object(value)
        if value.dtype.kind in "SU":
            return _unwrap_char(value)
        return value
    if isinstance(value, bytes):
        return value.decode("latin-1")
    return value


def _vector_shape(shape: tuple[int, ...]) -> bool:
    return len(shape) <= 1 or (len(shape) == 2 and (shape[0] == 1 or shape[1] == 1))


def _unwrap_object(value: np.ndarray):
    if not _vector_shape(value.shape):
        grid = np.empty(value.shape, dtype=object)
        flat = value.reshape(-1, order="F")
        for offset, item in enumerate(flat):
            index = np.unravel_index(offset, value.shape, order="F")
            grid[index] = _struct_dict(item) if isinstance(item, mat_struct) else convert_matlab(item)
        return grid
    items = value.reshape(-1, order="F")
    if len(items) and all(isinstance(item, mat_struct) for item in items):
        records = [_struct_dict(item) for item in items]
        if value.shape in {(1, 1), (1,), ()}:
            return records[0]
        return records
    return [convert_matlab(item) for item in items]


def _struct_dict(value: mat_struct) -> dict:
    converted = {}
    for name in getattr(value, "_fieldnames"):
        if name == "tracking":
            converted[name] = "<unsupported>"
        else:
            converted[name] = convert_matlab(getattr(value, name), name)
    return converted


def _unwrap_char(value: np.ndarray):
    rows = _char_rows(value)
    if len(rows) == 1:
        return rows[0]
    return _object_array(rows)


def _char_rows(value: np.ndarray) -> list[str]:
    if value.dtype.kind == "U" and value.dtype.itemsize > np.dtype("<U1").itemsize:
        return [str(item) for item in value.reshape(-1, order="F")]
    if value.dtype.kind == "S" and value.dtype.itemsize > 1:
        return [item.decode("latin-1").replace("\x00", "") for item in value.reshape(-1, order="F")]
    rows = []
    for row in np.atleast_2d(value):
        if value.dtype.kind == "S":
            rows.append(row.tobytes().decode("latin-1").replace("\x00", ""))
        else:
            rows.append("".join(str(character) for character in row.reshape(-1)))
    return rows


def _python_scalar(array: np.ndarray):
    item = np.asarray(array).reshape(-1)[0]
    kind = np.asarray(array).dtype.kind
    if kind in "iu":
        return int(item)
    if kind == "f":
        return float(item)
    if kind == "c":
        return complex(item)
    if kind == "b":
        return bool(item)
    return item.item() if hasattr(item, "item") else item


def _object_array(items: list) -> np.ndarray:
    array = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        array[index] = item
    return array


def _is_scalar(value) -> bool:
    return not isinstance(value, str) and (np.isscalar(value) or isinstance(value, (int, float, complex, np.number)))
