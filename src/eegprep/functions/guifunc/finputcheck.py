"""Validation for EEGLAB-style key/value arguments."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np

from eegprep.functions.miscfunc.matlab_inputs import cell_items, is_cell, is_matlab_empty


FieldRule = tuple[str, str | Sequence[str], Any, Any]
logger = logging.getLogger(__name__)


def finputcheck(
    arguments: Mapping[str, Any] | Sequence[Any],
    fieldlist: Sequence[FieldRule],
    callingfunc: str = "",
    mode: str = "error",
    verbose: str = "verbose",
    *,
    return_unrecognized: bool = False,
) -> dict[str, Any] | str | tuple[dict[str, Any] | str, list[Any]]:
    """Validate EEGLAB-style key/value arguments against field rules.

    Each field rule is ``(name, type, allowed_values, default)``. Supported
    types match EEGLAB's ``finputcheck``: ``boolean``, ``integer``, ``real``,
    ``float``, ``string``, ``cell``, ``struct``, and ``function_handle``.
    Arguments may be a mapping, a flat sequence, or a MATLAB cell. Cell values
    may be object arrays, and structure values may be structured arrays.
    Numeric scalars are returned as 1-by-1 arrays. Validation failures are
    returned as strings, as in EEGLAB.

    Args:
        arguments: Mapping or flat key/value sequence to validate.
        fieldlist: Validation rules in EEGLAB's four-column order.
        callingfunc: Optional function name prefixed to error messages.
        mode: Use ``"ignore"`` to retain unrecognized arguments.
        verbose: Use ``"quiet"`` to suppress duplicate-key notices.
        return_unrecognized: Also return a flat key/value list of arguments
            not declared by ``fieldlist``. This is Python's explicit
            equivalent of requesting EEGLAB's second output.

    Returns:
        A validated dictionary or an error string. When
        ``return_unrecognized`` is true, also returns residual key/value pairs.
    """
    values_or_error = _argument_dict(arguments, verbose)
    if isinstance(values_or_error, str):
        return _with_residual(values_or_error, [], return_unrecognized)
    values = values_or_error
    prefix = f"{callingfunc} " if callingfunc else ""

    rule_names: set[str] = set()
    for rule in fieldlist:
        if len(rule) != 4:
            raise ValueError("finputcheck field rules must have four entries")
        name, expected_types, allowed, default = _rule_fields(rule)
        rule_names.add(name)
        if name not in values:
            values[name] = default
        error = _validate_field(name, values[name], expected_types, allowed, prefix)
        if error:
            return _with_residual(error, [], return_unrecognized)

    residual: list[Any] = []
    for name, value in values.items():
        if name in rule_names:
            continue
        if mode.lower() != "ignore":
            error = f"{prefix}error: undefined argument '{name}'"
            return _with_residual(error, [], return_unrecognized)
        residual.extend((name, _matlab_number(value)))
    values = {name: _matlab_number(value) for name, value in values.items()}
    checked = _CheckedArguments(values, residual)
    return _with_residual(checked, residual, return_unrecognized)


class _CheckedArguments(dict):
    """Validated arguments that also expose MATLAB's unrecognized-argument output."""

    matlab_outputs = ("values", "unrecognized")

    def __init__(self, values: dict[str, Any], unrecognized: list[Any]):
        super().__init__(values)
        self.values = self
        row = np.empty((1, len(unrecognized)), dtype=object)
        row[0, :] = unrecognized
        self.unrecognized = row


def _matlab_number(value: Any) -> Any:
    if isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, Real):
        return np.array([[float(value)]])
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number) and value.ndim < 2 and value.size <= 1:
        return np.asarray(value, dtype=float).reshape(1, 1) if value.size else value.reshape(1, 0)
    return value


def _argument_dict(arguments: Mapping[str, Any] | Sequence[Any], verbose: str) -> dict[str, Any] | str:
    if isinstance(arguments, Mapping):
        return dict(arguments)
    if isinstance(arguments, (str, bytes)):
        return "error: bad 'key', 'val' sequence"
    if is_matlab_empty(arguments):
        return {}
    items = cell_items(arguments) if is_cell(arguments) else list(arguments)
    if len(items) % 2:
        return "error: bad 'key', 'val' sequence"
    result: dict[str, Any] = {}
    duplicates: list[str] = []
    for index in range(0, len(items), 2):
        name = items[index]
        if not isinstance(name, str):
            return "error: bad 'key', 'val' sequence"
        if name in result:
            duplicates.append(name)
        result[name] = items[index + 1]
    if duplicates and verbose.lower() == "verbose":
        names = ", ".join(dict.fromkeys(duplicates))
        logger.info("Duplicate '%s' parameter(s); keeping the last value", names)
    return result


def _validate_field(
    name: str,
    value: Any,
    expected_types: str | Sequence[str],
    allowed: Any,
    prefix: str,
) -> str:
    types = _type_names(expected_types)
    allowed_by_type = _allowed_by_type(allowed, len(types))
    errors = [
        _validate_one_type(name, value, expected_type, accepted, prefix)
        for expected_type, accepted in zip(types, allowed_by_type)
    ]
    if any(not error for error in errors):
        return ""
    return "\nor ".join(errors)


def _allowed_by_type(allowed: Any, count: int) -> list[Any]:
    if count == 1:
        return [allowed]
    if isinstance(allowed, (list, tuple)) and len(allowed) == count:
        return list(allowed)
    return [allowed] * count


def _validate_one_type(name: str, value: Any, expected_type: str, allowed: Any, prefix: str) -> str:
    field_type = expected_type.lower()
    if field_type in {"boolean", "integer", "real", "float"}:
        if not _is_numeric(value):
            return f"{prefix}error: argument '{name}' must be numeric"
        array = np.asarray(value)
        if field_type == "boolean" and np.any((array != 0) & (array != 1)):
            return f"{prefix}error: argument '{name}' must be 0 or 1"
        if field_type == "integer" and np.any(array != np.floor(array)):
            return f"{prefix}error: argument '{name}' must contain integers"
        if field_type == "integer" and not _is_empty(allowed) and array.size:
            bounds = np.asarray(allowed).reshape(-1)
            if np.any(array < bounds[0]) or np.any(array > bounds[-1]):
                return f"{prefix}error: value out of range for argument '{name}'"
        if field_type in {"real", "float"} and not _is_empty(allowed) and array.size:
            bounds = np.asarray(allowed).reshape(-1)
            if np.any(array < bounds[0]) or np.any(array > bounds[1]):
                return f"{prefix}error: value out of range for argument '{name}'"
        return ""
    if field_type == "string":
        if not isinstance(value, str) and not _is_empty(value):
            return f"{prefix}error: argument '{name}' must be a string"
        if not _is_empty(allowed):
            choices = [str(item).lower() for item in np.asarray(allowed, dtype=object).reshape(-1)]
            if str(value).lower() not in choices:
                return f"{prefix}error: wrong value for argument '{name}'"
        return ""
    if field_type == "cell":
        if isinstance(value, (list, tuple)) or is_cell(value) or (isinstance(value, np.ndarray) and value.size == 0):
            return ""
        return f"{prefix}error: argument '{name}' must be a cell array"
    if field_type == "struct":
        if isinstance(value, Mapping) or (isinstance(value, np.ndarray) and bool(value.dtype.names)):
            return ""
        return f"{prefix}error: argument '{name}' must be a structure"
    if field_type == "function_handle":
        return "" if callable(value) else f"{prefix}error: argument '{name}' must be a function handle"
    if field_type == "":
        return ""
    raise ValueError(f"finputcheck error: unrecognized type '{expected_type}'")


def _rule_fields(rule: Any) -> tuple[Any, Any, Any, Any]:
    if is_cell(rule) or (isinstance(rule, np.ndarray) and rule.dtype == object):
        fields = list(np.asarray(rule, dtype=object).reshape(-1))
    else:
        fields = list(rule)
    if len(fields) != 4:
        raise ValueError("finputcheck field rules must have four entries")
    return fields[0], fields[1], fields[2], fields[3]


def _type_names(expected_types: Any) -> list[str]:
    if isinstance(expected_types, str):
        return [expected_types]
    if isinstance(expected_types, np.ndarray):
        return [str(item) for item in expected_types.reshape(-1)]
    return [item if isinstance(item, str) else str(item) for item in expected_types]


def _is_numeric(value: Any) -> bool:
    if isinstance(value, (Real, np.number, bool)):
        return True
    if not isinstance(value, (list, tuple, np.ndarray)):
        return False
    return np.asarray(value).dtype.kind in "biufc"


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bytes, Sequence, Mapping)):
        return len(value) == 0
    return isinstance(value, np.ndarray) and value.size == 0


def _with_residual(
    result: dict[str, Any] | str,
    residual: list[Any],
    return_unrecognized: bool,
) -> dict[str, Any] | str | tuple[dict[str, Any] | str, list[Any]]:
    return (result, residual) if return_unrecognized else result
