"""Plain-JSON form of the calibration state of ``clean_asr``.

``clean_asr(EEG, return_state=True)`` returns the state its calibration produced.
``state_to_json`` turns that state into a dictionary that ``json.dumps`` writes,
and ``state_from_json`` reads such a dictionary back into a state that
``clean_asr(EEG, state=...)`` reapplies. Every array travels as float64 and every
number as a Python float, so the text holds the shortest representation that
parses back to the identical value.
"""

from typing import Any, Dict, Optional

import numpy as np

STATE_FORMAT = "eegprep.asr-state.v1"

_COMMON_KEYS = (
    "format",
    "M",
    "T",
    "B",
    "A",
    "cov",
    "carry",
    "last_R",
    "last_trivial",
    "srate",
    "channels",
    "cutoff",
    "window_len",
    "step_size",
    "max_dims",
    "useriemannian",
    "maxmem",
)
_EUCLIDEAN_KEYS = ("sos", "iir_state", "compatibility")
_RIEMANNIAN_KEYS = ("iir",)
_USERIEMANNIAN_VALUES = (None, "calib", "all")
_COMPATIBILITY_VALUES = ("standard", "max")
# Present in a calibration state and never filled before processing starts.
_EMPTY_AT_CALIBRATION = ("cov", "carry", "last_R")
_REQUIRED_ARRAYS = ("M", "T", "B", "A", "iir_state", "iir")
_OPTIONAL_ARRAYS = ("sos",)


def state_to_json(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``state`` as a dictionary that ``json.dumps`` writes without loss.

    Args:
        state: The state ``clean_asr(..., return_state=True)`` returned.

    Returns:
        A new dictionary. Each array is ``{"dtype": "float64", "shape": [...], "data": nested lists}``,
        ``None`` stays ``None``, and numpy scalars become Python numbers.

    Raises:
        ValueError: An array is not float64, or a number is not finite.
        TypeError: A value has a type a state does not hold.
    """
    return {key: _encode(key, value) for key, value in state.items()}


def state_from_json(document: Dict[str, Any]) -> Dict[str, Any]:
    """Read a state back from the dictionary ``state_to_json`` wrote.

    Args:
        document: The parsed JSON, for example ``json.loads(text)``.

    Returns:
        The state, with float64 arrays, for ``clean_asr(EEG, state=...)``.

    Raises:
        ValueError: The document has another ``format``, lacks a key or has an extra one,
            holds an array whose shape does not match its data, or holds a non-finite number.
            The message names the key.
    """
    if not isinstance(document, dict):
        raise ValueError("document: expected a JSON object")
    if document.get("format") != STATE_FORMAT:
        raise ValueError(f"format: expected {STATE_FORMAT!r}, got {document.get('format')!r}")
    if "useriemannian" not in document:
        raise ValueError("useriemannian: missing")
    mode = document["useriemannian"]
    if mode not in _USERIEMANNIAN_VALUES:
        raise ValueError(f"useriemannian: expected one of {_USERIEMANNIAN_VALUES}, got {mode!r}")
    expected = _COMMON_KEYS + (_RIEMANNIAN_KEYS if mode == "all" else _EUCLIDEAN_KEYS)
    for key in expected:
        if key not in document:
            raise ValueError(f"{key}: missing")
    for key in document:
        if key not in expected:
            raise ValueError(f"{key}: not a key of an ASR state with useriemannian={mode!r}")
    return {key: _decode(key, value) for key, value in document.items()}


def _encode(key: str, value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype != np.float64:
            raise ValueError(f"{key}: dtype must be float64, got {value.dtype}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{key}: contains a non-finite number")
        return {"dtype": "float64", "shape": list(value.shape), "data": value.tolist()}
    if isinstance(value, (list, tuple)):
        return [_encode(key, item) for item in value]
    if isinstance(value, (int, float)):
        if not np.isfinite(value):
            raise ValueError(f"{key}: not a finite number")
        return value
    raise TypeError(f"{key}: a state cannot hold a value of type {type(value).__name__}")


def _decode(key: str, value: Any) -> Any:
    if key in _EMPTY_AT_CALIBRATION:
        if value is not None:
            raise ValueError(f"{key}: must be null, a calibration state carries no processing buffers")
        return None
    if key in _REQUIRED_ARRAYS:
        return _decode_array(key, value)
    if key in _OPTIONAL_ARRAYS:
        return None if value is None else _decode_array(key, value)
    if key == "last_trivial":
        if not isinstance(value, bool):
            raise ValueError(f"{key}: expected true or false, got {value!r}")
        return value
    if key == "compatibility":
        if value not in _COMPATIBILITY_VALUES:
            raise ValueError(f"{key}: expected one of {_COMPATIBILITY_VALUES}, got {value!r}")
        return value
    if key == "channels":
        return _decode_channels(value)
    if key == "maxmem" and value is None:
        return None
    if key == "step_size":
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key}: expected a positive integer, got {value!r}")
        return value
    if key in ("format", "useriemannian"):
        return value
    return _decode_number(key, value, positive=key in ("srate", "window_len"))


def _decode_array(key: str, value: Any) -> np.ndarray:
    if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data"}:
        raise ValueError(f"{key}: expected an object with dtype, shape and data")
    if value["dtype"] != "float64":
        raise ValueError(f"{key}: dtype must be 'float64', got {value['dtype']!r}")
    shape = value["shape"]
    if not isinstance(shape, list) or not all(
        isinstance(size, int) and not isinstance(size, bool) and size >= 0 for size in shape
    ):
        raise ValueError(f"{key}: shape must be a list of non-negative integers")
    try:
        array = np.array(value["data"], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key}: data is not a rectangular array of numbers") from error
    if list(array.shape) != shape:
        raise ValueError(f"{key}: shape {shape} does not match data of shape {list(array.shape)}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{key}: contains a non-finite number")
    return array


def _decode_channels(value: Any) -> Optional[list]:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(label, str) for label in value):
        raise ValueError(f"channels: expected null or a list of labels, got {value!r}")
    return value


def _decode_number(key: str, value: Any, positive: bool) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{key}: expected a finite number, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{key}: expected a positive number, got {value!r}")
    return value
