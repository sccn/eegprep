"""Import EGI Net Station MATLAB exports into an EEGPrep dataset."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import scipy.io

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset, strict_mode
from eegprep.functions.popfunc._file_io import eeg_from_data
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.sigprocfunc.readegilocs import readegilocs


_SEGMENT_FIELD = re.compile(r"^(?P<type>.+)_Segment(?P<number>\d+)$")


def pop_importegimat(
    filename: str | Path,
    srate: float | None = None,
    latpoint0: float = 0.0,
    data_field: str = "Session",
    *,
    fileloc: str | Path | None = "auto",
    return_com: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import a segmented or continuous EGI Net Station MATLAB file.

    Net Station segmented exports store one channel-by-sample matrix per
    ``<condition>_Segment<number>`` variable. They become trials ordered first
    by condition name and then by segment number, with one event per trial.
    Continuous exports are read from ``data_field`` (``"Session"`` by
    default). An embedded scalar ``samplingRate`` takes precedence over
    ``srate``.

    Args:
        filename: MATLAB file exported by EGI Net Station.
        srate: Sampling rate in Hz when the file does not contain
            ``samplingRate``.
        latpoint0: Milliseconds from the start of each segment to time zero.
        data_field: Continuous-data variable name or MATLAB-compatible prefix.
        fileloc: EGI montage filename. ``"auto"`` selects the packaged montage
            from the imported channel count; an empty string skips locations.
        return_com: Return ``(EEG, command)`` when true.

    Returns:
        An EEG dictionary, optionally paired with its replayable history
        command.
    """
    path = Path(filename)
    variables = _load_variables(path)
    effective_srate = _sampling_rate(variables.get("samplingRate"), fallback=srate)
    latency_ms = _finite_scalar(latpoint0, "latpoint0")
    field_name = str(data_field)
    if not field_name:
        raise ValueError("data_field must be a non-empty MATLAB variable name")

    segment_fields = _find_segment_fields(variables)
    if segment_fields:
        data, events = _segmented_data(variables, segment_fields, effective_srate, latency_ms)
        if _has_empty_reference(data):
            data = data[:-1]
        eeg = eeg_from_data(
            data,
            srate=effective_srate,
            setname=str(path.with_suffix("")),
            nbchan=data.shape[0],
            xmin=-latency_ms / 1000.0,
            filename=path.name,
            filepath=str(path.parent),
        )
        eeg["event"], eeg["urevent"] = _events_with_urevents(events)
        with strict_mode(False):
            eeg = eeg_checkset(eeg, "eventconsistency")
    else:
        data = _continuous_data(variables, field_name)
        eeg = eeg_from_data(
            data,
            srate=effective_srate,
            nbchan=data.shape[0],
            filename=path.name,
            filepath=str(path.parent),
        )

    eeg = _apply_egi_locations(eeg, fileloc)
    command = _history_command(path, effective_srate, latency_ms, field_name, fileloc)
    eeg["history"] = command
    return (eeg, command) if return_com else eeg


def _load_variables(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"EGI MATLAB file not found: {path}")
    loaded = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
    return {name: value for name, value in loaded.items() if not name.startswith("__")}


def _sampling_rate(embedded: Any, *, fallback: float | None) -> float:
    value = fallback if embedded is None else embedded
    if value is None:
        raise ValueError("srate is required when the MATLAB file has no samplingRate variable")
    rate = _finite_scalar(value, "srate")
    if rate <= 0:
        raise ValueError("srate must be positive")
    return rate


def _finite_scalar(value: Any, name: str) -> float:
    values = np.asarray(value).ravel()
    if values.size != 1:
        raise ValueError(f"{name} must be a scalar")
    try:
        scalar = float(values[0])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _find_segment_fields(variables: dict[str, Any]) -> list[tuple[str, int, str]]:
    segments = []
    for name in variables:
        match = _SEGMENT_FIELD.fullmatch(name)
        if match:
            segments.append((match.group("type"), int(match.group("number")), name))
    return sorted(segments, key=lambda item: (item[0], item[1]))


def _segmented_data(
    variables: dict[str, Any],
    segment_fields: list[tuple[str, int, str]],
    srate: float,
    latency_ms: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    trials = []
    events = []
    expected_shape: tuple[int, int] | None = None
    for trial_index, (event_type, _segment_number, field_name) in enumerate(segment_fields):
        trial = _numeric_matrix(variables[field_name], field_name)
        if expected_shape is None:
            expected_shape = trial.shape
        elif trial.shape != expected_shape:
            raise ValueError(
                f"All EGI segment variables must have the same shape; "
                f"{field_name} has {trial.shape}, expected {expected_shape}"
            )
        trials.append(trial)
        pnts = trial.shape[1]
        events.append(
            {
                "type": event_type,
                "latency": latency_ms / 1000.0 * srate + 1.0 + trial_index * pnts,
                "epoch": trial_index + 1,
            }
        )

    stacked = np.stack(trials, axis=2)
    output_dtype = np.complex64 if np.iscomplexobj(stacked) else np.float32
    return stacked.astype(output_dtype), events


def _continuous_data(variables: dict[str, Any], data_field: str) -> np.ndarray:
    if data_field in variables:
        field_name = data_field
    else:
        matches = [name for name in variables if name.startswith(data_field)]
        if not matches:
            raise ValueError(f"MATLAB data field not found: {data_field}")
        field_name = matches[0]
    return _numeric_matrix(variables[field_name], field_name)


def _numeric_matrix(value: Any, field_name: str) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.ndim != 2:
        raise ValueError(f"EGI MATLAB variable {field_name!r} must be a 2-D channel-by-sample matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"EGI MATLAB variable {field_name!r} must not be empty")
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError(f"EGI MATLAB variable {field_name!r} must contain numeric data")
    return matrix


def _has_empty_reference(data: np.ndarray) -> bool:
    return bool(np.all(data[-1] == 0))


def _events_with_urevents(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized = []
    urevents = []
    for index, event in enumerate(events):
        urevent = dict(event)
        event_with_pointer = dict(event)
        event_with_pointer["urevent"] = index
        normalized.append(event_with_pointer)
        urevents.append(urevent)
    return normalized, urevents


def _apply_egi_locations(eeg: dict[str, Any], fileloc: str | Path | None) -> dict[str, Any]:
    if fileloc == "":
        return eeg
    selected = None if fileloc in {None, "auto"} else str(fileloc)
    return readegilocs(eeg, selected)


def _history_command(
    path: Path,
    srate: float,
    latpoint0: float,
    data_field: str,
    fileloc: str | Path | None,
) -> str:
    arguments = [
        format_history_value(path),
        format_history_value(srate),
        format_history_value(latpoint0),
        format_history_value(data_field),
    ]
    if fileloc not in {None, "auto"}:
        arguments.append(f"fileloc={format_history_value(fileloc)}")
    return f"EEG = pop_importegimat({', '.join(arguments)});"


__all__ = ["pop_importegimat"]
