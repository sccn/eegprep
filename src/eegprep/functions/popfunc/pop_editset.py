"""Edit EEG dataset structure fields with EEGLAB ``pop_editset`` semantics."""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._file_io import infer_dataformat, load_data_array
from eegprep.functions.popfunc._pop_utils import (
    format_history_value,
    is_empty_value as _is_empty_value,
    parse_key_value_args,
)
from eegprep.functions.sigprocfunc.floatread import floatread
from eegprep.functions.sigprocfunc.readlocs import readlocs


logger = logging.getLogger(__name__)

_TEXT_FIELDS = ("setname", "subject", "condition", "group", "comments")
_NUMERIC_FIELDS = ("srate", "pnts", "xmin", "nbchan")
_OPTIONAL_NUMERIC_FIELDS = ("run", "session")
_DIRECT_ASSIGNMENT_FIELDS = ("chanlocs", "icaweights", "icasphere", "icachansind", "data")
_PATH_EXTENSIONS = {
    ".asc",
    ".bdf",
    ".csv",
    ".dat",
    ".elp",
    ".fdt",
    ".mat",
    ".npy",
    ".npz",
    ".sfp",
    ".tsv",
    ".txt",
}
_SUPPORTED_FIELDS = {
    "ref",
    "dataformat",
    *_TEXT_FIELDS,
    *_NUMERIC_FIELDS,
    *_OPTIONAL_NUMERIC_FIELDS,
    *_DIRECT_ASSIGNMENT_FIELDS,
}


def pop_editset(
    EEG: dict[str, Any],
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Edit dataset metadata fields in an EEG dictionary."""
    if not isinstance(EEG, dict):
        raise TypeError("EEG must be an EEG dataset dictionary")

    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if "icaweight" in options and "icaweights" not in options:
        options["icaweights"] = options.pop("icaweight")
    if gui is None:
        gui = not bool(options)

    if gui:
        result = inputgui(pop_editset_dialog_spec(EEG), renderer=renderer)
        if result is None:
            original = copy.copy(EEG)
            return (original, "") if return_com else original
        options = _changed_options_from_gui(EEG, result)
        if not options:
            original = copy.copy(EEG)
            return (original, "") if return_com else original

    unknown = sorted(set(options) - _SUPPORTED_FIELDS)
    if unknown:
        raise ValueError(f"Unknown pop_editset option(s): {', '.join(unknown)}")

    output = copy.deepcopy(EEG)
    if "data" in options:
        _clear_ica_fields(output)
    old_xmin = float(output.get("xmin", 0.0) or 0.0)
    old_srate = float(output.get("srate", 1.0) or 1.0)

    dataformat = str(options.get("dataformat") or "auto")
    for key, value in options.items():
        if key == "dataformat":
            continue
        _assign_option(output, key, value, dataformat=dataformat)

    _normalize_data_dimensions(output, strict_pnts="pnts" in options)
    if "xmin" in options:
        _adjust_event_latencies_for_xmin_change(output, old_xmin, old_srate)
    _normalize_ica_index_field(output)
    _refresh_time_fields(output)
    output = eeg_checkset(output)
    command = _history_command(options) if return_com else ""
    return (output, command) if return_com else output


def pop_editset_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_editset``."""
    return DialogSpec(
        title="Edit dataset information - pop_editset()",
        function_name="pop_editset",
        eeglab_source="functions/popfunc/pop_editset.m",
        size=(701, 389),
        geometry=(
            (2, 3.38),
            1,
            (2.5, 1, 1.5, 1.5),
            (2.5, 1, 1.5, 1.5),
            (2.5, 1, 1.5, 1.5),
            (2.5, 1, 1.5, 1.5),
            (2.5, 1, 1.5, 1.5),
            1,
            (1.4, 0.7, 0.8, 0.5),
            1,
            (1.4, 0.7, 0.8, 0.5),
            (1.4, 0.7, 0.8, 0.5),
            (1.4, 0.7, 0.8, 0.5),
        ),
        content_margins=(36, 20, 36, 8),
        row_spacing=2,
        help_text="pop_editset",
        controls=(
            ControlSpec("text", "Dataset name", font_weight="bold"),
            ControlSpec("edit", tag="setname", value=_display_value(EEG.get("setname", ""))),
            ControlSpec("spacer"),
            ControlSpec("text", "Data sampling rate (Hz)", font_weight="bold"),
            ControlSpec("edit", tag="srate", value=_display_value(EEG.get("srate", ""))),
            ControlSpec("text", "Subject code"),
            ControlSpec("edit", tag="subject", value=_display_value(EEG.get("subject", ""))),
            ControlSpec("text", "Time points per epoch (0->continuous)"),
            ControlSpec("edit", tag="pnts", value=_display_value(EEG.get("pnts", ""))),
            ControlSpec("text", "Task condition"),
            ControlSpec("edit", tag="condition", value=_display_value(EEG.get("condition", ""))),
            ControlSpec("text", "Start time (sec) (only for data epochs)"),
            ControlSpec("edit", tag="xmin", value=_display_value(EEG.get("xmin", ""))),
            ControlSpec("text", "Subject group"),
            ControlSpec("edit", tag="group", value=_display_value(EEG.get("group", ""))),
            ControlSpec("text", "Number of channels (0->set from data)"),
            ControlSpec("edit", tag="nbchan", value=_display_value(EEG.get("nbchan", "")), enabled=False),
            ControlSpec("text", "Run number"),
            ControlSpec("edit", tag="run", value=_display_value(EEG.get("run", ""))),
            ControlSpec("text", "Ref. channel indices or mode (see help)"),
            ControlSpec("edit", tag="ref", value=_display_value(EEG.get("ref", "")), enabled=False),
            ControlSpec("text", "Session number"),
            ControlSpec("edit", tag="session", value=_display_value(EEG.get("session", ""))),
            ControlSpec("spacer"),
            ControlSpec("text", "Channel location file or info", font_weight="bold"),
            ControlSpec("pushbutton", "From other dataset", enabled=False),
            ControlSpec("edit", tag="chanfile", enabled=False),
            ControlSpec("pushbutton", "Browse", enabled=False),
            ControlSpec(
                "text",
                'Note: The file format may be auto-detected from its file extension. See menu "Edit > Channel locations" for other options.',
            ),
            ControlSpec("text", "ICA weights array or text/binary file (if any):"),
            ControlSpec("pushbutton", "From other dataset", enabled=False),
            ControlSpec("edit", tag="weightfile", enabled=False),
            ControlSpec("pushbutton", "Browse", enabled=False),
            ControlSpec("text", "ICA sphere array:"),
            ControlSpec("pushbutton", "From other dataset", enabled=False),
            ControlSpec("edit", tag="sphfile", enabled=False),
            ControlSpec("spacer"),
            ControlSpec("text", "ICA channel indices (by default all):"),
            ControlSpec("pushbutton", "From other dataset", enabled=False),
            ControlSpec("edit", tag="icainds", enabled=False),
            ControlSpec("spacer"),
        ),
        known_differences=(
            "Channel-location and ICA file pickers and fields are visible for parity but disabled until later Phase 1b channel workflows.",
        ),
        extra_stylesheet="""
            QDialog#pop_editset QLabel,
            QDialog#pop_editset QLineEdit,
            QDialog#pop_editset QPushButton {
                font-size: 10px;
            }
            QDialog#pop_editset QLineEdit,
            QDialog#pop_editset QPushButton {
                min-height: 14px;
                max-height: 14px;
            }
        """,
    )


def _changed_options_from_gui(EEG: dict[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for key in ("setname", "srate", "subject", "pnts", "condition", "xmin", "group", "nbchan", "run", "ref", "session"):
        if key not in result:
            continue
        value = result[key]
        if key in _NUMERIC_FIELDS:
            parsed = _parse_required_number(value, key)
            if not _numbers_equal(parsed, EEG.get(key)):
                options[key] = parsed
            continue
        if key in _OPTIONAL_NUMERIC_FIELDS:
            parsed = _parse_optional_number(value)
            if _display_value(parsed) != _display_value(EEG.get(key, "")):
                options[key] = parsed
            continue
        if str(value) != _display_value(EEG.get(key, "")):
            options[key] = str(value)
    return options


def _assign_option(output: dict[str, Any], key: str, value: Any, *, dataformat: str = "auto") -> None:
    if key in _TEXT_FIELDS:
        output[key] = "" if value is None else str(value)
        return
    if key == "srate":
        output[key] = _positive_float(value, key)
        return
    if key == "pnts":
        output[key] = _nonnegative_int(value, key)
        return
    if key == "xmin":
        output[key] = float(value)
        return
    if key == "nbchan":
        nbchan = _nonnegative_int(value, key)
        if nbchan:
            output[key] = nbchan
        return
    if key in _OPTIONAL_NUMERIC_FIELDS:
        output[key] = _parse_optional_number(value)
        return
    if key == "ref":
        output[key] = "" if value is None else str(value)
        logger.warning("Changing EEG.ref only updates metadata; use pop_reref to re-reference data.")
        return
    if key == "chanlocs":
        _assign_chanlocs(output, value)
        return
    if key in {"icaweights", "icasphere"}:
        _assign_ica_matrix(output, key, value, dataformat=dataformat)
        return
    if key == "icachansind":
        if isinstance(value, (str, Path)) and str(value).strip() != "[]":
            path = _existing_path(value)
            if path is not None:
                output[key] = _as_int_array(_load_array_path(path, dataformat=dataformat))
                return
            if _looks_like_path(value):
                raise FileNotFoundError(f"pop_editset icachansind file not found: {value}")
            raise ValueError(
                "pop_editset cannot evaluate MATLAB workspace expressions for icachansind; "
                "pass Python indices directly or provide a numeric file path."
            )
        output[key] = _as_int_array(value)
        return
    if key == "data":
        _assign_data(output, value, dataformat=dataformat)


def _assign_chanlocs(output: dict[str, Any], value: Any) -> None:
    if _is_empty_value(value):
        output["chanlocs"] = []
        return
    if isinstance(value, (str, Path)):
        if str(value).strip() == "[]":
            output["chanlocs"] = []
            return
        path = _existing_path(value)
        if path is None:
            if _looks_like_path(value):
                raise FileNotFoundError(f"pop_editset channel-location file not found: {value}")
            raise ValueError(
                "pop_editset cannot evaluate MATLAB workspace expressions for chanlocs; "
                "pass channel-location structures directly or provide a file path."
            )
        output["chanlocs"] = readlocs(path)
        return
    if _looks_like_chanloc_package(value):
        package = list(value)
        output["chanlocs"] = copy.deepcopy(package[0])
        if len(package) > 1:
            output["chaninfo"] = copy.deepcopy(package[1])
        if len(package) > 2:
            output["urchanlocs"] = copy.deepcopy(package[2])
        return
    output["chanlocs"] = copy.deepcopy(value)


def _assign_ica_matrix(output: dict[str, Any], key: str, value: Any, *, dataformat: str = "auto") -> None:
    if _is_empty_value(value):
        output[key] = np.array([])
    elif isinstance(value, (str, Path)):
        if str(value).strip() == "[]":
            output[key] = np.array([])
        else:
            path = _existing_path(value)
            if path is None:
                if _looks_like_path(value):
                    raise FileNotFoundError(f"pop_editset {key} file not found: {value}")
                raise ValueError(
                    f"pop_editset cannot evaluate MATLAB workspace expressions for {key}; "
                    "pass a Python array directly or provide a matrix file path."
                )
            output[key] = _load_matrix_file(path, output, dataformat=dataformat)
    else:
        output[key] = np.asarray(value)
    output["icawinv"] = np.array([])
    output["icaact"] = np.array([])
    if (
        key == "icaweights"
        and np.asarray(output.get("icaweights", [])).size
        and _is_empty_value(output.get("icasphere"))
    ):
        output["icasphere"] = np.eye(np.asarray(output["icaweights"]).shape[1])


def _assign_data(output: dict[str, Any], value: Any, *, dataformat: str = "auto") -> None:
    if isinstance(value, (str, Path)):
        path = _existing_path(value)
        if path is None:
            if _looks_like_path(value):
                raise FileNotFoundError(f"pop_editset data file not found: {value}")
            raise ValueError(
                "pop_editset cannot evaluate MATLAB workspace expressions for data; "
                "pass a Python array directly or provide a data file path."
            )
        data = load_data_array(
            path,
            dataformat=infer_dataformat(path, None if dataformat == "auto" else dataformat),
            nbchan=int(output.get("nbchan", 0) or 0) or None,
        )
    else:
        data = value
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim not in {2, 3}:
        raise ValueError("data must be a 2-D or 3-D channel-major array")
    output["data"] = data
    output["nbchan"] = int(data.shape[0])
    output["pnts"] = int(data.shape[1])
    output["trials"] = int(data.shape[2]) if data.ndim == 3 else 1


def _clear_ica_fields(output: dict[str, Any]) -> None:
    for key in ("icaweights", "icasphere", "icawinv", "icaact", "icachansind"):
        output[key] = np.array([])


def _normalize_data_dimensions(output: dict[str, Any], *, strict_pnts: bool) -> None:
    data = output.get("data")
    if isinstance(data, str) or data is None:
        return
    data = np.asarray(data)
    if data.size == 0:
        output["data"] = data
        return
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim not in {2, 3}:
        raise ValueError("data must be a 2-D or 3-D channel-major array")

    output["nbchan"] = int(data.shape[0])
    pnts = int(output.get("pnts", data.shape[1]) or 0)
    trials = int(output.get("trials", 1) or 1)
    if pnts <= 0:
        pnts = int(data.shape[1])
        trials = int(data.shape[2]) if data.ndim == 3 else 1

    if data.ndim == 2 and pnts > 1:
        frames = int(data.shape[1])
        if frames % pnts:
            if strict_pnts:
                raise ValueError(f"pnts={pnts} does not evenly divide {frames} data frames")
            if trials > 1 and frames // pnts > 0:
                trials = frames // pnts
                data = data[:, : trials * pnts]
                data = _continuous_to_epoched(data, pnts, trials)
            else:
                pnts = frames
                trials = 1
        elif frames != pnts:
            trials = frames // pnts
            data = _continuous_to_epoched(data, pnts, trials)

    if data.ndim == 3:
        if strict_pnts and int(data.shape[1]) != pnts:
            raise ValueError(f"pnts={pnts} does not match the data epoch length {data.shape[1]}")
        pnts = int(data.shape[1])
        trials = int(data.shape[2])
    output["data"] = data
    output["pnts"] = pnts
    output["trials"] = trials


def _continuous_to_epoched(data: np.ndarray, pnts: int, trials: int) -> np.ndarray:
    return data.reshape(data.shape[0], trials, pnts).transpose(0, 2, 1)


def _adjust_event_latencies_for_xmin_change(output: dict[str, Any], old_xmin: float, old_srate: float) -> None:
    if int(output.get("trials", 1) or 1) <= 1:
        return
    new_xmin = float(output.get("xmin", old_xmin) or 0.0)
    delta_samples = (new_xmin - old_xmin) * old_srate
    if not delta_samples:
        return
    events = output.get("event", [])
    if isinstance(events, np.ndarray):
        iterable = events.tolist()
    elif isinstance(events, dict):
        iterable = [events]
    else:
        iterable = list(events or [])
    for event in iterable:
        if isinstance(event, dict) and "latency" in event:
            event["latency"] = float(event["latency"]) - delta_samples
    if isinstance(events, np.ndarray):
        output["event"] = np.asarray(iterable, dtype=object)
    elif isinstance(events, dict):
        output["event"] = iterable[0] if iterable else {}
    else:
        output["event"] = iterable


def _refresh_time_fields(output: dict[str, Any]) -> None:
    pnts = int(output.get("pnts", 0) or 0)
    srate = float(output.get("srate", 0) or 0)
    xmin = float(output.get("xmin", 0) or 0)
    if pnts <= 0 or srate <= 0:
        output["xmax"] = xmin
        output["times"] = np.array([])
        return
    output["xmax"] = xmin + (pnts - 1) / srate
    output["times"] = np.linspace(xmin * 1000, output["xmax"] * 1000, pnts)


def _normalize_ica_index_field(output: dict[str, Any]) -> None:
    value = output.get("icachansind")
    if value is None:
        return
    array = np.asarray(value)
    if array.size == 0:
        output["icachansind"] = np.array([], dtype=int)
        return
    output["icachansind"] = _as_int_array(array)


def _history_command(options: Mapping[str, Any]) -> str:
    command = "EEG = pop_editset(EEG"
    for key, value in options.items():
        command += f", {format_history_value(key)}, {_history_value(value)}"
    return command + ");"


def _history_value(value: Any) -> str:
    if _is_empty_value(value):
        return "[]"
    if isinstance(value, np.ndarray) and value.dtype == object:
        value = value.tolist()
    if isinstance(value, Mapping):
        return _history_mapping(value)
    if isinstance(value, (list, tuple)) and any(isinstance(item, Mapping) for item in value):
        return "[" + ", ".join(_history_value(item) for item in value) + "]"
    return format_history_value(value)


def _history_mapping(value: Mapping[str, Any]) -> str:
    pieces = []
    for key, item in value.items():
        pieces.append(f"{format_history_value(str(key))}: {_history_value(item)}")
    return "{" + ", ".join(pieces) + "}"


def _existing_path(value: Any) -> Path | None:
    if not isinstance(value, (str, Path)):
        return None
    text = str(value).strip()
    if not text or text == "[]":
        return None
    path = Path(text).expanduser()
    return path if path.exists() else None


def _looks_like_path(value: Any) -> bool:
    if isinstance(value, Path):
        return True
    if not isinstance(value, str):
        return False
    text = str(value).strip()
    if not text or text == "[]":
        return False
    return any(separator in text for separator in ("/", "\\")) or Path(text).suffix.lower() in _PATH_EXTENSIONS


def _load_array_path(path: Path, *, dataformat: str = "auto") -> np.ndarray:
    return load_data_array(path, dataformat=_resolved_dataformat(path, dataformat)).ravel()


def _load_matrix_file(path: Path, output: dict[str, Any], *, dataformat: str = "auto") -> np.ndarray:
    resolved_format = _resolved_dataformat(path, dataformat)
    nbcol = int(np.asarray(output.get("icachansind", [])).size or output.get("nbchan", 0) or 0)
    if resolved_format in {"float32le", "float32be"}:
        values = np.asarray(floatread(path, [1, np.inf], resolved_format), dtype=float).reshape(-1)
        if nbcol <= 0 or values.size % nbcol:
            raise ValueError(f"pop_editset cannot infer matrix shape for {path}")
        matrix = values.reshape(values.size // nbcol, nbcol, order="F")
    else:
        matrix = np.asarray(load_data_array(path, dataformat=resolved_format))
    if matrix.ndim == 1:
        if nbcol <= 0 or matrix.size % nbcol:
            raise ValueError(f"pop_editset cannot infer matrix shape for {path}")
        matrix = matrix.reshape(matrix.size // nbcol, nbcol)
    if matrix.ndim != 2:
        raise ValueError("ICA matrix files must contain a 2-D array")
    return matrix


def _resolved_dataformat(path: Path, dataformat: str) -> str:
    return infer_dataformat(path, None if dataformat == "auto" else dataformat)


def _parse_required_number(value: Any, key: str) -> float | int:
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{key} is required")
    if key in {"pnts", "nbchan"}:
        return _nonnegative_int(value, key)
    if key == "srate":
        return _positive_float(value, key)
    return float(value)


def _parse_optional_number(value: Any) -> str | int | float:
    if _is_empty_value(value):
        return ""
    if isinstance(value, str):
        text = value.strip()
        try:
            numeric = float(text)
        except ValueError:
            return text
    else:
        numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def _positive_float(value: Any, key: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{key} must be positive")
    return parsed


def _nonnegative_int(value: Any, key: str) -> int:
    parsed = int(float(value))
    if parsed < 0:
        raise ValueError(f"{key} must be non-negative")
    return parsed


def _as_int_array(value: Any) -> np.ndarray:
    if _is_empty_value(value):
        return np.array([], dtype=int)
    array = np.asarray(value)
    if np.any(array != np.floor(array)):
        raise ValueError("icachansind must contain integer indices")
    return array.astype(int).ravel()


def _looks_like_chanloc_package(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return False
    if not value:
        return False
    first = value[0]
    return (
        isinstance(first, Sequence)
        and not isinstance(first, Mapping)
        and not isinstance(first, (str, bytes, bytearray))
    )


def _display_value(value: Any) -> str:
    if _is_empty_value(value):
        return ""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return f"{value:g}" if isinstance(value, float) else str(value)


def _numbers_equal(left: Any, right: Any) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


__all__ = ["pop_editset", "pop_editset_dialog_spec"]
