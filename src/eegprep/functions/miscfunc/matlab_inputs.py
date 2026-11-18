"""Interpret MATLAB-shaped values at EEGPrep function boundaries.

MATLAB structs arrive as NumPy structured arrays or ``numpy.void`` records, cells
as object arrays, and numbers as doubles. Empty arrays are false in ``if``.
A nonempty numeric or logical array is true only when every element is nonzero;
NaN counts as true. These helpers accept those shapes. They do not wrap Python
scalars, lists, or dicts back into MATLAB 1-by-1 arrays or struct arrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def is_matlab_empty(value: Any) -> bool:
    """Return whether MATLAB ``isempty`` is true.

    Numeric zero is not empty.
    """
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (str, bytes, list, tuple, dict, set)):
        return len(value) == 0
    return False


def matlab_true(value: Any) -> bool:
    """Return MATLAB ``if`` truth for scalars, strings, and numeric arrays.

    Raises:
        ValueError: ``value`` is a cell array or structure. MATLAB cannot
            convert those to logical.
    """
    if value is None or is_matlab_empty(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return "\0" not in value
    if isinstance(value, bytes):
        return b"\0" not in value
    if isinstance(value, np.ndarray):
        if value.dtype.names or value.dtype == object:
            raise ValueError("MATLAB cannot use a cell array or structure as a logical condition")
        if value.dtype.kind == "U":
            return bool(np.all(value != "\0"))
        if value.dtype.kind == "S":
            return bool(np.all(value != b"\0"))
        return bool(np.all(np.asarray(value) != 0))
    if isinstance(value, (list, tuple)):
        return all(matlab_true(item) for item in value)
    if isinstance(value, (int, float, np.integer, np.floating, complex, np.complexfloating)):
        return value != 0
    raise ValueError("MATLAB cannot use this value as a logical condition")


def matlab_ints(value: Any) -> list[int] | None:
    """Return integral numbers, or ``None`` when ``value`` is not integral numeric.

    An empty numeric array is ``[]``. Strings, booleans, and non-integral
    numbers, including NaN, are ``None``.
    """
    if isinstance(value, (bool, np.bool_)) or value is None or isinstance(value, (str, bytes, dict)):
        return None
    if isinstance(value, (list, tuple)):
        result: list[int] = []
        for item in value:
            ints = matlab_ints(item)
            if ints is None:
                return None
            result.extend(ints)
        return result
    if isinstance(value, np.ndarray):
        if value.dtype == object or value.dtype.names or not np.issubdtype(value.dtype, np.number):
            return None
        if value.size == 0:
            return []
        flat = np.asarray(value, dtype=float).reshape(-1)
        if not np.all(np.isfinite(flat)) or not np.all(flat == np.floor(flat)):
            return None
        return [int(item) for item in flat]
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or float(value) != int(value):
            return None
        return [int(value)]
    return None


def scalar_index(value: Any) -> int | None:
    """Return one integral number from a scalar or 1-by-1 numeric array."""
    if isinstance(value, (str, bytes, list, tuple, dict)):
        return None
    if isinstance(value, np.ndarray) and value.size != 1:
        return None
    ints = matlab_ints(value)
    if ints is None or len(ints) != 1:
        return None
    return ints[0]


def is_cell(value: Any) -> bool:
    """Return whether ``value`` is a MATLAB cell array."""
    return isinstance(value, np.ndarray) and value.dtype == object


def cell_items(value: Any) -> list[Any]:
    """Return cell contents in MATLAB column-major order.

    A size-1 nested cell or a size-1 character array is unwrapped to its
    element or to ``str``. Numeric 1-by-1 arrays stay arrays.
    """
    if not is_cell(value):
        raise TypeError("expected a MATLAB cell array")
    return [_unwrap_cell_item(item) for item in value.ravel(order="F")]


def struct_records(value: Any) -> list[dict[str, Any]] | None:
    """Return structure records, or ``None`` when ``value`` is not a structure.

    A dict is one record. A structured array, including a scalar ``numpy.void``,
    is one dict per record in column-major order. An object array or sequence
    is records only when every item is a dict. An empty structure is ``[]``.
    """
    if isinstance(value, dict):
        return [value]
    if isinstance(value, np.void) and value.dtype.names:
        return [_record_dict(value)]
    if isinstance(value, np.ndarray) and value.dtype.names:
        if value.size == 0:
            return []
        return [_record_dict(record) for record in value.ravel(order="F")]
    if is_cell(value):
        items = list(value.ravel(order="F"))
        if all(isinstance(item, dict) for item in items):
            return items
        return None
    if isinstance(value, (list, tuple)) and all(isinstance(item, dict) for item in value):
        return list(value)
    return None


def as_dataset_list(value: Any) -> list[Any]:
    """Return an ``ALLEEG`` value as a list of datasets.

    ``None`` and an empty array are no datasets. A structure array or a cell of
    dicts becomes one dict per dataset. A Python list is copied as a list.
    """
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        if value.dtype.names:
            return struct_records(value) or []
        if value.dtype == object:
            items = list(value.ravel(order="F"))
            if items and all(isinstance(item, dict) for item in items):
                return items
        return list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return list(value)


def _unwrap_cell_item(item: Any) -> Any:
    if isinstance(item, np.ndarray) and item.dtype == object and item.size == 1:
        return _unwrap_cell_item(item.reshape(-1)[0])
    if isinstance(item, np.ndarray) and item.dtype.kind in "SU" and item.size == 1:
        return str(item.reshape(-1)[0])
    return item


def _record_dict(record: np.void) -> dict[str, Any]:
    names = record.dtype.names or ()
    return {name: record[name] for name in names}
