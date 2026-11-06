"""Export EEG data to EDF, BDF, or GDF."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pyedflib

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._file_io import channel_labels, events_to_records
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args
from eegprep.functions.sigprocfunc.writegdf import writegdf

_FORMATS = ("EDF", "GDF", "BDF")
_BDF_WARNING = (
    "pop_writeeeg: BDF file headers have a known BIOSIG issue "
    "(https://sccn.ucsd.edu/bugzilla/show_bug.cgi?id=1020). "
    "The file is still written as BDF+."
)


def pop_writeeeg(
    EEG: dict[str, Any],
    filename: str | Path | None = None,
    *args: Any,
    renderer: Any = None,
    **kwargs: Any,
) -> str:
    """Write continuous EEG data to EDF, BDF, or GDF.

    With no filename, the file-format dialog from ``pop_writeeeg.m`` lines
    52-80 is shown. Cancel returns an empty command. Epoched data is
    concatenated, which is what ``EEG.data(:,:)`` does once the user accepts
    the prompt at lines 53-57. The history command has no ``LASTCOM`` prefix
    (``pop_writeeeg.m`` line 97).
    """
    include_type = False
    if filename is None:
        selected = _writeeeg_dialog(EEG, renderer)
        if selected is None:
            return ""
        filename, output_type = selected
        include_type = True
        if output_type == "bdf":
            # pop_writeeeg.m:76-77 warns only when the dialog selects BDF.
            warnings.warn(_BDF_WARNING, UserWarning, stacklevel=2)
        options: dict[str, Any] = {}
    else:
        options = parse_key_value_args(args, kwargs, lowercase_keys=True, lowercase_kwargs=True)
        passed_type = "type" in options
        output_type = str(options.pop("type", Path(filename).suffix.lstrip("."))).lower()
        include_type = passed_type or bool(args)
    path = Path(filename)
    if options:
        raise ValueError(
            "pop_writeeeg: unsupported option(s): "
            f"{', '.join(sorted(options))}. The only option is TYPE, one of EDF, GDF, or BDF."
        )
    if path.suffix == "":
        path = path.with_suffix(f".{output_type.lower()}")
    if path.suffix.lower() not in {".edf", ".bdf", ".gdf"}:
        raise ValueError(
            "pop_writeeeg: the output file must end in .edf, .bdf, or .gdf. "
            f"Rename {path.name} so the extension matches the format, then call pop_writeeeg again."
        )
    if output_type != path.suffix.lstrip(".").lower():
        raise ValueError(
            f"pop_writeeeg: TYPE {output_type.upper()} does not match the extension {path.suffix}. "
            "Use the same format in TYPE and in the file name."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _continuous_data(EEG)
    if output_type == "gdf":
        writegdf(
            path,
            data,
            EEG["srate"],
            labels=channel_labels(EEG),
            events=events_to_records(EEG.get("event")),
            subject=str(EEG.get("subject", "")),
            recording=str(EEG.get("setname", "")),
        )
    else:
        _write_edf_family(EEG, data, path, output_type)
    return _history_command(path, output_type, include_type)


def pop_writeeeg_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the BIOSIG format dialog from ``pop_writeeeg.m`` lines 67-73."""
    trials = int(EEG.get("trials") or 1)
    controls: list[ControlSpec] = [
        ControlSpec("text", "File name"),
        ControlSpec("edit", tag="filename"),
        ControlSpec(
            "pushbutton",
            "Browse",
            tag="browse",
            callback=CallbackSpec(
                "select_file",
                {
                    "button": "browse",
                    "target": "filename",
                    "mode": "save",
                    "caption": "Enter a file name -- pop_writeeeg()",
                    "filter": "EEG files (*.edf *.gdf *.bdf);;All files (*)",
                },
            ),
        ),
        ControlSpec("text", "File format"),
        ControlSpec("listbox", "EDF|GDF|BDF", tag="format", value=1),
    ]
    geometry: list[Any] = [[1, 2, 0.5], [1], [1]]
    if trials > 1:
        controls.append(
            ControlSpec(
                "checkbox",
                "Export concatenated data epochs",
                tag="concatenate",
                value=True,
            )
        )
        geometry.append([1])
    return DialogSpec(
        title="Write data using BIOSIG -- pop_writeeeg()",
        function_name="pop_writeeeg",
        eeglab_source="functions/popfunc/pop_writeeeg.m",
        geometry=tuple(geometry),
        geomvert=(1, 1, 2.5, *(() if trials <= 1 else (1,))),
        controls=tuple(controls),
    )


def _writeeeg_dialog(EEG: dict[str, Any], renderer: Any) -> tuple[str, str] | None:
    result = inputgui(pop_writeeeg_dialog_spec(EEG), renderer=renderer)
    if not result:
        return None
    if int(EEG.get("trials") or 1) > 1 and not result.get("concatenate", True):
        return None
    filename = str(result.get("filename") or "").strip()
    if not filename:
        raise ValueError("pop_writeeeg: File name required. Enter an output file name, then press Ok.")
    choice = int(result.get("format") or 1)
    if choice < 1 or choice > len(_FORMATS):
        raise ValueError(
            f"pop_writeeeg: format choice {choice} is not EDF, GDF, or BDF. Select one of those three formats."
        )
    return filename, _FORMATS[choice - 1].lower()


def _history_command(path: Path, output_type: str, include_type: bool) -> str:
    # pop_writeeeg.m:97. An empty option list is omitted so the command stays valid.
    pieces = [format_history_value(path)]
    if include_type:
        pieces.extend(["'TYPE'", format_history_value(output_type.upper())])
    return f"pop_writeeeg(EEG, {', '.join(pieces)});"


def _continuous_data(EEG: dict[str, Any]) -> np.ndarray:
    data = np.ascontiguousarray(EEG["data"], dtype=float)
    if data.ndim == 3:
        data = np.ascontiguousarray(data.reshape((data.shape[0], data.shape[1] * data.shape[2])))
    if data.ndim != 2:
        raise ValueError(
            "pop_writeeeg: EEG.data must be channel-by-sample, or channel-by-sample-by-trial. "
            "Epoch the dataset or pass continuous data, then call pop_writeeeg again."
        )
    if not np.isfinite(data).all():
        raise ValueError(
            "pop_writeeeg: EEG.data contains NaN or Inf. Replace those samples, then call pop_writeeeg again."
        )
    return data


def _write_edf_family(EEG: dict[str, Any], data: np.ndarray, path: Path, output_type: str) -> None:
    srate = float(EEG["srate"])
    events = events_to_records(EEG.get("event"))
    duration, annotation_signals = _edf_record_plan(data.shape[1], srate, len(events))
    file_type = pyedflib.FILETYPE_BDFPLUS if output_type == "bdf" else pyedflib.FILETYPE_EDFPLUS
    digital_min, digital_max = (-8_388_608, 8_388_607) if output_type == "bdf" else (-32_768, 32_767)
    writer = pyedflib.EdfWriter(str(path), int(data.shape[0]), file_type=file_type)
    try:
        if abs(duration - 1.0) > 1e-12:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                writer.setDatarecordDuration(duration)
        if annotation_signals != 1:
            writer.set_number_of_annotation_signals(annotation_signals)
        for index, (label, signal) in enumerate(zip(channel_labels(EEG), data)):
            writer.setSignalHeader(index, _signal_header(label, signal, srate, digital_min, digital_max))
        writer.writeSamples(data)
        for event in events:
            onset = (float(event.get("latency", 1)) - 1) / srate
            event_duration = float(event.get("duration", 0) or 0) / srate
            writer.writeAnnotation(onset, event_duration, str(event.get("type", "event")))
    finally:
        writer.close()


def _edf_record_plan(n_samples: int, srate: float, n_events: int) -> tuple[float, int]:
    """Choose a record length that stores every sample and every annotation.

    EDF data records last between 0.001 and 60 seconds. pyedflib stores about
    one annotation per annotation signal per record, so the record count has
    to cover the events or extra annotation signals are used (at most 64).
    """
    if n_samples < 1 or not np.isfinite(srate) or srate <= 0:
        raise ValueError(
            f"pop_writeeeg: cannot write {n_samples} samples at {srate} Hz. "
            "Set EEG.srate to a positive sampling rate and EEG.data to a non-empty array."
        )
    min_samples = max(1, math.ceil(0.001 * srate - 1e-9))
    max_samples = min(n_samples, max(min_samples, math.floor(60.0 * srate + 1e-9)))
    best: tuple[tuple[int, int], float, int] | None = None
    for samples in range(min_samples, max_samples + 1):
        if n_samples % samples:
            continue
        records = n_samples // samples
        duration = samples / srate
        if duration < 0.001 - 1e-9 or duration > 60.0 + 1e-9:
            continue
        signals = 1 if n_events <= 0 else math.ceil(n_events / records)
        if signals > 64:
            continue
        key = (signals, records)
        if best is None or key < best[0]:
            best = (key, duration, int(signals))
    if best is None:
        raise ValueError(
            f"pop_writeeeg: {n_samples} samples at {srate} Hz cannot be stored in an EDF/BDF "
            "data record without padding. EDF records must last between 0.001 and 60 seconds "
            f"and hold an integer number of samples. Resample so the length is a multiple of "
            f"{min_samples} samples, then call pop_writeeeg again."
        )
    return best[1], best[2]


def _signal_header(
    label: str,
    signal: np.ndarray,
    srate: float,
    digital_min: int,
    digital_max: int,
) -> dict[str, Any]:
    physical_min = float(np.floor(np.min(signal)))
    physical_max = float(np.ceil(np.max(signal)))
    if physical_min == physical_max:
        physical_min -= 1
        physical_max += 1
    return {
        "label": label,
        "dimension": "uV",
        "sample_frequency": srate,
        "physical_min": physical_min,
        "physical_max": physical_max,
        "digital_min": digital_min,
        "digital_max": digital_max,
        "transducer": "",
        "prefilter": "",
    }
