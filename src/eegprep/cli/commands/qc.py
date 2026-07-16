"""Agent-friendly EEGPrep QC command support."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from eegprep.cli.core import (
    EEGPrepCLIError as CommandError,
    build_manifest,
    command_error as error_result,
    command_ok as success_result,
    file_sha256,
    json_safe,
    utc_now,
    write_manifest_file,
)
from eegprep.cli.dataset import load_dataset as load_eeg_dataset
from eegprep.cli.reporting import write_report_html


QC_SCHEMA_VERSION = "eegprep.qc.v1"


class _QCArgumentParser(argparse.ArgumentParser):
    """Nested qc parser that can defer JSON-mode errors to the top-level CLI."""

    def __init__(self, *args: Any, json_requested: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.json_requested = json_requested

    def error(self, message: str) -> NoReturn:
        if self.json_requested:
            raise CommandError(
                "CONFIG_SCHEMA_ERROR",
                message,
                suggestion="Run the command with --help or inspect schema command qc.",
                exit_code=2,
            )
        super().error(message)


def compute_qc_metrics(EEG: dict[str, Any], *, dataset_path: str | Path | None = None) -> dict[str, Any]:
    """Compute deterministic QC metrics and recommendation codes for an EEG dict.
    
    The returned dictionary matches the internal QC schema and includes:
      - dataset, recording, channels, and events metadata.
      - ica: Includes ICA shapes and a structured 'iclabel' field containing 
        the classes and their mean probabilities across components (if ICLabel was run).
      - data_quality: Includes non-finite data metrics, flat channel detection, 
        channel RMS, and a structured 'asr' field containing median/max summaries 
        of noisiness and z-scores for channels and temporal windows (if clean_rawdata was run).
      - recommendations: A list of actionable recommendation codes.
    """
    data = np.asarray(EEG.get("data", np.array([])))
    events = _as_list(EEG.get("event", []))
    chanlocs = _as_list(EEG.get("chanlocs", []))
    nbchan = int(EEG.get("nbchan", data.shape[0] if data.ndim else 0) or 0)
    pnts = int(EEG.get("pnts", data.shape[1] if data.ndim >= 2 else 0) or 0)
    trials = int(EEG.get("trials", data.shape[2] if data.ndim == 3 else 1) or 1)
    srate = float(EEG.get("srate", 0.0) or 0.0)
    total_samples = pnts * trials
    duration_seconds = float(total_samples / srate) if srate > 0 else None

    finite_mask = np.isfinite(data) if data.size else np.array([], dtype=bool)
    nonfinite_count = int(data.size - int(np.count_nonzero(finite_mask))) if data.size else 0
    nonfinite_fraction = float(nonfinite_count / data.size) if data.size else 0.0
    channel_stds = _channel_stds(data)
    flat_channel_indices = [int(index) for index, value in enumerate(channel_stds, start=1) if value <= 1e-12]
    channel_rms = _channel_rms(data)
    invalid_events = _invalid_event_latencies(events, total_samples)
    missing_location_indices = _missing_location_indices(chanlocs, nbchan)

    metrics = {
        "schema_version": QC_SCHEMA_VERSION,
        "dataset": {
            "path": str(dataset_path) if dataset_path is not None else None,
            "setname": _clean_scalar(EEG.get("setname", "")),
            "filename": _clean_scalar(EEG.get("filename", "")),
            "filepath": _clean_scalar(EEG.get("filepath", "")),
        },
        "recording": {
            "nbchan": nbchan,
            "pnts": pnts,
            "trials": trials,
            "srate_hz": srate,
            "duration_seconds": duration_seconds,
            "xmin_seconds": _float_or_none(EEG.get("xmin")),
            "xmax_seconds": _float_or_none(EEG.get("xmax")),
            "data_shape": [int(dim) for dim in data.shape],
            "data_dtype": str(data.dtype) if data.size else None,
        },
        "channels": {
            "count": nbchan,
            "chanloc_count": len(chanlocs),
            "missing_location_count": len(missing_location_indices),
            "missing_location_indices": missing_location_indices,
            "index_base": 1,
            "labels": _channel_labels(chanlocs),
        },
        "events": {
            "count": len(events),
            "invalid_latency_count": len(invalid_events),
            "invalid_latency_indices": invalid_events,
            "index_base": 1,
            "unique_types": _unique_event_types(events),
        },
        "ica": _ica_metrics(EEG),
        "data_quality": {
            "nonfinite_count": nonfinite_count,
            "nonfinite_fraction": nonfinite_fraction,
            "flat_channel_count": len(flat_channel_indices),
            "flat_channel_indices": flat_channel_indices,
            "index_base": 1,
            "channel_rms_uv": {
                "median": _float_or_none(np.median(channel_rms)) if channel_rms else None,
                "max": _float_or_none(np.max(channel_rms)) if channel_rms else None,
            },
            "asr": _asr_metrics(EEG),
        },
    }
    metrics["recommendations"] = qc_recommendations(metrics)
    return json_safe(metrics)


def qc_recommendations(metrics: dict[str, Any]) -> list[dict[str, str]]:
    """Return stable recommendation codes derived from QC metrics."""
    recommendations: list[dict[str, str]] = []
    channels = metrics["channels"]
    events = metrics["events"]
    recording = metrics["recording"]
    data_quality = metrics["data_quality"]
    ica = metrics["ica"]

    if channels["missing_location_count"]:
        recommendations.append(
            _recommendation(
                "MISSING_CHANNEL_LOCATIONS",
                "warning",
                f"{channels['missing_location_count']} channels are missing usable location coordinates.",
                "Add or repair channel locations before interpolation, topographies, or ASR channel cleaning.",
            )
        )
    if events["invalid_latency_count"]:
        recommendations.append(
            _recommendation(
                "INVALID_EVENT_LATENCY",
                "error",
                f"{events['invalid_latency_count']} events have latencies outside the dataset sample range.",
                "Inspect event latencies using EEGLAB-facing 1-based sample boundaries before epoching.",
            )
        )
    if data_quality["nonfinite_count"]:
        severity = "error" if data_quality["nonfinite_fraction"] >= 0.01 else "warning"
        recommendations.append(
            _recommendation(
                "NONFINITE_DATA_VALUES",
                severity,
                f"{data_quality['nonfinite_count']} data samples are NaN or infinite.",
                "Replace or reject nonfinite samples before numerical preprocessing.",
            )
        )
    if data_quality["flat_channel_count"]:
        recommendations.append(
            _recommendation(
                "FLAT_CHANNELS_DETECTED",
                "warning",
                f"{data_quality['flat_channel_count']} channels have zero or near-zero variance.",
                "Review flat channels and consider clean_rawdata flatline/channel rejection.",
            )
        )
    if recording["srate_hz"] and recording["srate_hz"] < 100:
        recommendations.append(
            _recommendation(
                "LOW_SAMPLING_RATE",
                "info",
                f"Sampling rate is {recording['srate_hz']:g} Hz.",
                "Confirm this is sufficient for the intended frequency analysis.",
            )
        )
    if events["count"] == 0:
        recommendations.append(
            _recommendation(
                "NO_EVENTS_FOUND",
                "info",
                "No events are present in the dataset.",
                "Import or verify events before event-locked epoching.",
            )
        )
    if not ica["has_ica"]:
        recommendations.append(
            _recommendation(
                "ICA_NOT_FOUND",
                "info",
                "ICA decomposition fields are not present.",
                "Run ICA before component classification or component-level rejection.",
            )
        )
    return recommendations


def qc_dataset(input_path: str | Path) -> dict[str, Any]:
    """Run QC for a dataset path and return a structured result."""
    try:
        EEG = load_eeg_dataset(input_path)
        metrics = compute_qc_metrics(EEG, dataset_path=input_path)
        return success_result("qc", input=str(input_path), qc=metrics)
    except CommandError as exc:
        return error_result("qc", exc)


def qc_report_dataset(
    input_path: str | Path,
    html_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run QC and write an HTML report plus optional manifest."""
    started_at = utc_now()
    try:
        EEG = load_eeg_dataset(input_path)
        metrics = compute_qc_metrics(EEG, dataset_path=input_path)
        html_entry = write_report_html(
            EEG,
            html_path,
            metrics=metrics,
            dataset_path=input_path,
            title="EEGPrep QC Report",
            overwrite=overwrite,
        )
        output_files = [html_entry]
        manifest_entry = None
        if manifest_path is not None:
            finished_at = utc_now()
            manifest = build_manifest(
                command="qc report",
                input_files=[{"path": str(input_path), "sha256": file_sha256(input_path)}],
                output_files=output_files,
                parameters={"html": str(html_path)},
                started_at=started_at,
                finished_at=finished_at,
                deterministic=True,
                warnings=metrics["recommendations"],
            )
            manifest_entry = write_manifest_file(manifest_path, manifest, overwrite=overwrite)
            output_files.append(manifest_entry)
        return success_result(
            "qc report",
            input=str(input_path),
            output_files=output_files,
            manifest=str(manifest_path) if manifest_entry else None,
            qc=metrics,
        )
    except CommandError as exc:
        return error_result("qc report", exc)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Register ``qc`` commands with an argparse dispatcher."""
    parser = subparsers.add_parser("qc", help="Compute dataset QC metrics and recommendations.")
    parser.add_argument("qc_args", nargs=argparse.REMAINDER)
    parser.set_defaults(handler=handle_registered)
    return parser


def handle_registered(args: argparse.Namespace) -> dict[str, Any]:
    """Handle ``qc`` when mounted into a larger dispatcher."""
    return dispatch(args.qc_args)


def dispatch(argv: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    """Dispatch qc command arguments without assuming a global CLI foundation."""
    argv = list(argv or [])
    json_requested = "--json" in argv
    if argv and argv[0] == "report":
        parser = _report_parser(json_requested=json_requested)
        args = parser.parse_args(argv[1:])
        return qc_report_dataset(args.input, args.html, manifest_path=args.manifest, overwrite=args.overwrite)
    parser = _qc_parser(json_requested=json_requested)
    args = parser.parse_args(argv)
    return qc_dataset(args.input)


def main(argv: list[str] | None = None) -> int:
    """Route module execution through the canonical top-level CLI dispatcher."""
    from eegprep.cli.main import main as cli_main

    return cli_main(["qc", *(sys.argv[1:] if argv is None else argv)])


def _qc_parser(*, json_requested: bool = False) -> argparse.ArgumentParser:
    parser = _QCArgumentParser(prog="eegprep qc", json_requested=json_requested)
    parser.add_argument("input", help="Input EEGLAB .set dataset")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON")
    return parser


def _report_parser(*, json_requested: bool = False) -> argparse.ArgumentParser:
    parser = _QCArgumentParser(prog="eegprep qc report", json_requested=json_requested)
    parser.add_argument("input", help="Input EEGLAB .set dataset")
    parser.add_argument("--html", required=True, help="Output HTML report path")
    parser.add_argument("--manifest", help="Optional manifest JSON path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON")
    return parser


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        return [_clean_scalar(item) for item in value.tolist()]
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _channel_labels(chanlocs: list[Any]) -> list[str]:
    labels = []
    for index, chanloc in enumerate(chanlocs, start=1):
        if isinstance(chanloc, dict):
            label = chanloc.get("labels", "")
        else:
            label = getattr(chanloc, "labels", "")
        labels.append(str(_clean_scalar(label) or f"Ch{index}"))
    return labels


def _missing_location_indices(chanlocs: list[Any], nbchan: int) -> list[int]:
    if len(chanlocs) < nbchan:
        return list(range(len(chanlocs) + 1, nbchan + 1))
    missing = []
    for index, chanloc in enumerate(chanlocs[:nbchan], start=1):
        if not _has_usable_coordinates(chanloc):
            missing.append(index)
    return missing


def _has_usable_coordinates(chanloc: Any) -> bool:
    if not isinstance(chanloc, dict):
        return False
    coordinate_sets = (("X", "Y", "Z"), ("theta", "radius"), ("sph_theta", "sph_phi", "sph_radius"))
    for fields in coordinate_sets:
        values = [_float_or_none(chanloc.get(field)) for field in fields]
        if all(value is not None and math.isfinite(value) for value in values):
            return True
    return False


def _invalid_event_latencies(events: list[Any], max_latency: int) -> list[int]:
    invalid = []
    for index, event in enumerate(events, start=1):
        latency = _event_value(event, "latency")
        value = _float_or_none(latency)
        if value is None or value < 1 or (max_latency > 0 and value > max_latency):
            invalid.append(index)
    return invalid


def _unique_event_types(events: list[Any]) -> list[str]:
    values = sorted(
        {str(_clean_scalar(_event_value(event, "type"))) for event in events if _event_value(event, "type")}
    )
    return values


def _event_value(event: Any, key: str) -> Any:
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


def _ica_metrics(EEG: dict[str, Any]) -> dict[str, Any]:
    weights = np.asarray(EEG.get("icaweights", np.array([])))
    sphere = np.asarray(EEG.get("icasphere", np.array([])))
    winv = np.asarray(EEG.get("icawinv", np.array([])))
    has_ica = bool(weights.size and sphere.size and winv.size)
    return {
        "has_ica": has_ica,
        "component_count": int(weights.shape[0]) if weights.ndim >= 2 else 0,
        "icaweights_shape": [int(dim) for dim in weights.shape] if weights.size else [],
        "icasphere_shape": [int(dim) for dim in sphere.shape] if sphere.size else [],
        "icawinv_shape": [int(dim) for dim in winv.shape] if winv.size else [],
        "icachansind_count": len(_as_list(EEG.get("icachansind", []))),
        "iclabel": _iclabel_metrics(EEG),
    }


def _iclabel_metrics(EEG: dict[str, Any]) -> dict[str, Any] | None:
    etc = EEG.get("etc", {})
    if not isinstance(etc, dict):
        return None
    ic_class = etc.get("ic_classification", {})
    iclabel = ic_class.get("ICLabel", {})
    classes = iclabel.get("classes", [])
    classifications = np.asarray(iclabel.get("classifications", []))
    if classifications.size == 0 or len(classes) == 0:
        return None

    # mean probabilities for all classes across components
    mean_probs = np.mean(classifications, axis=0)

    return {
        "classes": _as_list(classes),
        "mean_probabilities": [_float_or_none(p) for p in mean_probs],
    }


def _asr_metrics(EEG: dict[str, Any]) -> dict[str, Any] | None:
    etc = EEG.get("etc", {})
    if not isinstance(etc, dict):
        return None

    metrics = {}
    for key in ["clean_channel_noisiness_median", "clean_channel_noisiness_max", 
                "clean_channel_zscores_median", "clean_channel_zscores_max",
                "clean_window_rms_median", "clean_window_rms_max",
                "clean_window_zscores_median", "clean_window_zscores_max"]:
        if key in etc:
            metrics[key] = _float_or_none(etc[key])

    return metrics if metrics else None


def _channel_stds(data: np.ndarray) -> list[float]:
    if data.size == 0 or data.ndim < 2:
        return []
    flat = data.reshape(data.shape[0], -1)
    return [_float_or_none(value) or 0.0 for value in np.nanstd(flat, axis=1)]


def _channel_rms(data: np.ndarray) -> list[float]:
    if data.size == 0 or data.ndim < 2:
        return []
    flat = data.reshape(data.shape[0], -1)
    with np.errstate(invalid="ignore"):
        values = np.sqrt(np.nanmean(np.square(flat), axis=1))
    return [_float_or_none(value) or 0.0 for value in values]


def _recommendation(code: str, severity: str, message: str, suggestion: str) -> dict[str, str]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "suggestion": suggestion,
    }


def _float_or_none(value: Any) -> float | None:
    value = _clean_scalar(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _clean_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.size == 1:
            return _clean_scalar(value.reshape(-1)[0])
        return value.tolist()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
