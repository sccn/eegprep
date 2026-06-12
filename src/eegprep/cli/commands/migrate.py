"""Migration and compatibility commands for the EEGPrep CLI."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from eegprep.cli.core import EEGPrepCLIError, output_path
from eegprep.cli.dataset import load_dataset


_COMMAND_MAP = {
    "pop_loadset": ("import", 0.9),
    "pop_fileio": ("import", 0.85),
    "pop_saveset": ("export", 0.8),
    "pop_resample": ("resample", 0.95),
    "pop_reref": ("rereference", 0.95),
    "pop_eegfilt": ("filter", 0.82),
    "pop_eegfiltnew": ("filter", 0.95),
    "pop_clean_rawdata": ("clean", 0.9),
    "pop_epoch": ("epoch", 0.95),
    "pop_runica": ("ica", 0.95),
    "pop_iclabel": ("components", 0.85),
    "pop_subcomp": ("components", 0.85),
}


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register migration compatibility commands."""
    parser = subparsers.add_parser("migrate", help="Inspect old EEGLAB histories and migration compatibility.")
    migrate_sub = parser.add_subparsers(dest="migrate_command", required=True)

    history_parser = migrate_sub.add_parser("history", help="Inspect mapped EEGLAB history operations.")
    history_parser.add_argument("input")
    history_parser.add_argument("--json", action="store_true")
    history_parser.set_defaults(handler=lambda args: history(args.input))

    compare_parser = migrate_sub.add_parser("compare", help="Compare two EEGLAB .set datasets.")
    compare_parser.add_argument("left")
    compare_parser.add_argument("right")
    compare_parser.add_argument("--json", action="store_true")
    compare_parser.set_defaults(handler=lambda args: compare(args.left, args.right))

    convert = migrate_sub.add_parser("convert-script", help="Best-effort conversion of simple EEGLAB scripts to YAML.")
    convert.add_argument("script")
    convert.add_argument("--to", choices=("eegprep-yaml",), default="eegprep-yaml")
    convert.add_argument("--output")
    convert.add_argument("--overwrite", action="store_true")
    convert.add_argument("--json", action="store_true")
    convert.set_defaults(
        handler=lambda args: convert_script(args.script, target=args.to, output=args.output, overwrite=args.overwrite)
    )
    return parser


def history(input_path: str | Path) -> dict[str, Any]:
    """Map EEGLAB history lines to EEGPrep CLI concepts."""
    EEG = load_dataset(input_path)
    operations = []
    for index, line in enumerate(_history_lines(EEG.get("history", "")), start=1):
        command = _command_name(line)
        if command is None:
            operations.append(
                {
                    "index": index,
                    "raw": line,
                    "supported": False,
                    "unsupported": {"code": "COMMAND_NOT_RECOGNIZED", "message": "Could not parse a pop_* command."},
                }
            )
            continue
        operation, confidence = _COMMAND_MAP.get(command, ("", 0.0))
        record: dict[str, Any] = {
            "index": index,
            "raw": line,
            "eeglab_command": command,
            "supported": command in _COMMAND_MAP,
        }
        if command in _COMMAND_MAP:
            record["operation"] = operation
            record["mapped_eegprep_command"] = operation
            record["confidence"] = confidence
        else:
            record["unsupported"] = {
                "code": "COMMAND_NOT_IMPLEMENTED",
                "message": f"No EEGPrep CLI mapping is registered for {command}.",
            }
        operations.append(record)
    return {
        "status": "ok",
        "schema_version": "eegprep.migrate.history.v1",
        "input": str(input_path),
        "history_detected": bool(operations),
        "operations": operations,
    }


def compare(left: str | Path, right: str | Path) -> dict[str, Any]:
    """Compare two EEGLAB datasets with structured differences."""
    left_eeg = load_dataset(left)
    right_eeg = load_dataset(right)
    differences = []
    for key in ("nbchan", "pnts", "trials", "srate"):
        left_value = _scalar(left_eeg.get(key))
        right_value = _scalar(right_eeg.get(key))
        if left_value != right_value:
            differences.append(
                {
                    "path": key,
                    "code": "VALUE_MISMATCH",
                    "left": left_value,
                    "right": right_value,
                }
            )
    left_data = np.asarray(left_eeg.get("data", np.array([])))
    right_data = np.asarray(right_eeg.get("data", np.array([])))
    data: dict[str, Any] = {"shape_left": list(left_data.shape), "shape_right": list(right_data.shape)}
    if left_data.shape != right_data.shape:
        differences.append(
            {
                "path": "data",
                "code": "DATA_SHAPE_MISMATCH",
                "left": list(left_data.shape),
                "right": list(right_data.shape),
            }
        )
    elif left_data.size:
        finite_left = np.isfinite(left_data)
        finite_right = np.isfinite(right_data)
        finite_mask_mismatch = bool(np.any(finite_left != finite_right))
        data["finite_mask_mismatch"] = finite_mask_mismatch
        if finite_mask_mismatch:
            differences.append({"path": "data", "code": "DATA_FINITE_MASK_MISMATCH"})
        with np.errstate(invalid="ignore"):
            comparable = finite_left & finite_right
            max_abs_diff = (
                float(np.max(np.abs(left_data[comparable] - right_data[comparable]))) if comparable.any() else 0.0
            )
        data["max_abs_diff"] = max_abs_diff
        if max_abs_diff > 0:
            differences.append({"path": "data", "code": "DATA_VALUE_MISMATCH", "max_abs_diff": max_abs_diff})
    return {
        "status": "ok",
        "schema_version": "eegprep.migrate.compare.v1",
        "left": str(left),
        "right": str(right),
        "equivalent": not differences,
        "differences": differences,
        "data": data,
    }


def convert_script(
    script: str | Path,
    *,
    target: str = "eegprep-yaml",
    output: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert simple EEGLAB history/script lines into an EEGPrep pipeline YAML skeleton."""
    source = Path(script).expanduser()
    if target != "eegprep-yaml":
        raise EEGPrepCLIError("CONFIG_SCHEMA_ERROR", "Only --to eegprep-yaml is supported.", exit_code=2)
    if not source.exists():
        raise EEGPrepCLIError("INPUT_FILE_NOT_FOUND", f"Script not found: {source}", path=source, exit_code=2)
    lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    converted_steps = []
    unsupported = []
    input_path = ""
    for line in lines:
        name = _command_name(line)
        if name is None:
            continue
        if name == "pop_loadset":
            input_path = _first_quoted(line) or input_path
            converted_steps.append({"name": "load", "path": input_path})
        elif name == "pop_resample":
            freq = _first_number_after_command(line)
            converted_steps.append({"name": "resample", "freq": freq})
        elif name in {"pop_eegfilt", "pop_eegfiltnew"}:
            converted_steps.append({"name": "filter", "raw": line})
        elif name == "pop_reref":
            converted_steps.append({"name": "rereference", "method": "average", "raw": line})
        elif name == "pop_epoch":
            converted_steps.append({"name": "epoch", "raw": line})
        elif name == "pop_runica":
            converted_steps.append({"name": "ica", "method": "runica", "raw": line})
        else:
            unsupported.append({"command": name, "code": "COMMAND_NOT_IMPLEMENTED", "raw": line})
    pipeline = {
        "schema_version": "eegprep.pipeline.v1",
        "input": {"path": input_path or "input.set", "format": "eeglab"},
        "output": {"directory": "derivatives/eegprep", "filename": "output.set", "overwrite": False},
        "steps": [step for step in converted_steps if step.get("name") != "load"],
    }
    output_path_value = None
    if output is not None:
        destination = output_path(output, overwrite=overwrite)
        destination.write_text(yaml.safe_dump(pipeline, sort_keys=False), encoding="utf-8")
        output_path_value = str(destination)
    return {
        "status": "ok",
        "schema_version": "eegprep.migrate.convert_script.v1",
        "source": str(source),
        "target": target,
        "output": output_path_value,
        "converted_steps": converted_steps,
        "unsupported_commands": unsupported,
        "pipeline": pipeline,
    }


def main(argv: list[str] | None = None) -> int:
    """Route module execution through the canonical top-level CLI dispatcher."""
    from eegprep.cli.main import main as cli_main

    return cli_main(["migrate", *(sys.argv[1:] if argv is None else argv)])


def _history_lines(history_text: Any) -> list[str]:
    if history_text is None:
        return []
    return [line.strip() for line in str(history_text).splitlines() if line.strip()]


def _command_name(line: str) -> str | None:
    match = re.search(r"\b([A-Za-z]\w*)\s*\(", line)
    if match is None:
        return None
    return match.group(1)


def _first_quoted(line: str) -> str | None:
    match = re.search(r"['\"]([^'\"]+)['\"]", line)
    return None if match is None else match.group(1)


def _first_number_after_command(line: str) -> float | None:
    body = line.split("(", 1)[-1]
    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", body)
    if not numbers:
        return None
    return float(numbers[-1])


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.size == 1:
            return _scalar(value.reshape(-1)[0])
        return value.tolist()
    return value
