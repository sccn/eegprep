"""Dataset loading, inspection, and validation helpers for the EEGPrep CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eegprep.cli.core import EEGPrepCLIError, existing_input, ok, warning
from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset, strict_mode
from eegprep.functions.popfunc.pop_loadset import pop_loadset


def load_dataset(path: str | Path) -> dict[str, Any]:
    """Load an EEGLAB dataset for headless CLI commands."""
    input_path = existing_input(path)
    if input_path.suffix.lower() not in {".set", ".mat"}:
        raise EEGPrepCLIError(
            "UNSUPPORTED_FORMAT",
            f"Unsupported input format: {input_path.suffix or '(none)'}",
            path=input_path,
            suggestion="Use an EEGLAB .set file or convert the source file before running this command.",
        )
    return pop_loadset(str(input_path))


def dataset_summary(EEG: dict[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    data = np.asarray(EEG.get("data", np.array([])))
    pnts = int(EEG.get("pnts", data.shape[1] if data.ndim >= 2 else 0) or 0)
    trials = int(EEG.get("trials", data.shape[2] if data.ndim == 3 else 1) or 1)
    srate = float(EEG.get("srate", 0) or 0)
    duration = (pnts * trials / srate) if srate > 0 else 0.0
    return {
        "file": None if path is None else str(path),
        "format": "eeglab_set",
        "setname": str(EEG.get("setname") or ""),
        "sampling_rate_hz": srate,
        "n_channels": int(EEG.get("nbchan", data.shape[0] if data.ndim else 0) or 0),
        "n_points": pnts,
        "n_trials": trials,
        "n_events": len(_records(EEG.get("event"))),
        "has_ica": _has_ica(EEG),
        "has_channel_locations": bool(_channel_records(EEG)),
        "duration_seconds": duration,
        "data_shape": list(data.shape),
        "saved": str(EEG.get("saved") or ""),
    }


def inspect_dataset(path: str | Path) -> dict[str, Any]:
    input_path = existing_input(path)
    EEG = load_dataset(input_path)
    return ok("eegprep.inspect.dataset.v1", **dataset_summary(EEG, input_path), warnings=validation_warnings(EEG))


def inspect_events(path: str | Path) -> dict[str, Any]:
    input_path = existing_input(path)
    EEG = load_dataset(input_path)
    events = []
    for index, event in enumerate(_records(EEG.get("event")), start=1):
        events.append(
            {
                "index": index,
                "type": _scalar(event.get("type", "")),
                "latency": _scalar(event.get("latency")),
                "duration": _scalar(event.get("duration")) if "duration" in event else None,
                "epoch": _scalar(event.get("epoch")) if "epoch" in event else None,
            }
        )
    return ok("eegprep.inspect.events.v1", file=str(input_path), count=len(events), events=events)


def inspect_channels(path: str | Path) -> dict[str, Any]:
    input_path = existing_input(path)
    EEG = load_dataset(input_path)
    channels = []
    for index, chan in enumerate(_channel_records(EEG), start=1):
        channels.append(
            {
                "index": index,
                "label": str(chan.get("labels") or chan.get("label") or f"Ch{index}"),
                "type": "" if chan.get("type") is None else str(chan.get("type")),
                "has_coordinates": any(
                    key in chan and not _empty(chan.get(key)) for key in ("X", "Y", "Z", "theta", "radius")
                ),
            }
        )
    return ok("eegprep.inspect.channels.v1", file=str(input_path), count=len(channels), channels=channels)


def inspect_ica(path: str | Path) -> dict[str, Any]:
    input_path = existing_input(path)
    EEG = load_dataset(input_path)
    weights = np.asarray(EEG.get("icaweights", np.array([])))
    sphere = np.asarray(EEG.get("icasphere", np.array([])))
    winv = np.asarray(EEG.get("icawinv", np.array([])))
    return ok(
        "eegprep.inspect.ica.v1",
        file=str(input_path),
        has_ica=_has_ica(EEG),
        n_components=int(weights.shape[0]) if weights.ndim >= 2 else 0,
        icaweights_shape=list(weights.shape),
        icasphere_shape=list(sphere.shape),
        icawinv_shape=list(winv.shape),
        icachansind=[int(value) for value in np.asarray(EEG.get("icachansind", []), dtype=int).ravel()],
    )


def validate_dataset(path: str | Path) -> dict[str, Any]:
    input_path = existing_input(path)
    EEG = load_dataset(input_path)
    with strict_mode(False):
        checked = eeg_checkset(EEG)
    checks = validation_checks(checked)
    status = "warning" if any(check["status"] == "warning" for check in checks) else "ok"
    payload = {
        "file": str(input_path),
        "checks": checks,
        "can_continue": not any(check["status"] == "error" for check in checks),
    }
    return warning("eegprep.validate.v1", **payload) if status == "warning" else ok("eegprep.validate.v1", **payload)


def validation_warnings(EEG: dict[str, Any]) -> list[dict[str, Any]]:
    return [check for check in validation_checks(EEG) if check["status"] == "warning"]


def validation_checks(EEG: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [
        _check("data", bool(np.asarray(EEG.get("data", np.array([]))).size), "DATA_MISSING", "Dataset has no data."),
        _check(
            "channel_locations",
            bool(_channel_records(EEG)),
            "MISSING_CHANNEL_LOCATIONS",
            "The dataset does not contain channel location information.",
            status_if_false="warning",
        ),
        _check("events", True, "INVALID_EVENT_LATENCY", ""),
        _check(
            "sampling_rate",
            float(EEG.get("srate", 0) or 0) > 0,
            "INVALID_SAMPLING_RATE",
            "Sampling rate must be positive.",
        ),
    ]
    events = _records(EEG.get("event"))
    bad_latencies = [
        index
        for index, event in enumerate(events, start=1)
        if "latency" in event and not _finite_latency(event["latency"])
    ]
    if bad_latencies:
        checks[2] = {
            "name": "events",
            "status": "error",
            "code": "INVALID_EVENT_LATENCY",
            "message": f"{len(bad_latencies)} event latencies are not finite sample positions.",
            "indices": bad_latencies,
        }
    else:
        checks[2] = {"name": "events", "status": "ok"}
    return checks


def _check(
    name: str,
    condition: bool,
    code: str,
    message: str,
    *,
    status_if_false: str = "error",
) -> dict[str, Any]:
    if condition:
        return {"name": name, "status": "ok"}
    return {"name": name, "status": status_if_false, "code": code, "message": message}


def _has_ica(EEG: dict[str, Any]) -> bool:
    return bool(
        np.asarray(EEG.get("icaweights", np.array([]))).size and np.asarray(EEG.get("icasphere", np.array([]))).size
    )


def _records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, np.ndarray):
        return [item for item in value.ravel().tolist() if isinstance(item, dict)]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _channel_records(EEG: dict[str, Any]) -> list[dict[str, Any]]:
    return _records(EEG.get("chanlocs"))


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return _scalar(value.reshape(-1)[0])
    if isinstance(value, np.generic):
        return value.item()
    return value


def _finite_latency(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(np.asarray(value).reshape(-1)[0])))
    except (TypeError, ValueError, IndexError):
        return False


def _empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, str):
        return not value.strip()
    return False
