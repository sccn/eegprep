"""BIDS commands for the agent-friendly EEGPrep CLI."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from eegprep.cli.core import build_manifest, json_safe, now_iso, output_path, sha256_file, write_manifest_file
from eegprep.cli.dataset import dataset_summary, load_dataset
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.plugins.EEG_BIDS.bids_list_eeg_files import bids_list_eeg_files
from eegprep.plugins.EEG_BIDS.pop_exportbids import pop_exportbids
from eegprep.plugins.EEG_BIDS.pop_importbids import pop_importbids


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register ``bids`` commands."""
    parser = subparsers.add_parser("bids", help="Validate, import, or export BIDS EEG datasets.")
    bids_sub = parser.add_subparsers(dest="bids_command", required=True)

    validate = bids_sub.add_parser("validate", help="Validate a BIDS EEG root.")
    validate.add_argument("root")
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(handler=lambda args: validate_dataset(args.root))

    import_parser = bids_sub.add_parser("import", help="Import a BIDS EEG dataset to EEGLAB .set.")
    import_parser.add_argument("root")
    import_parser.add_argument("--subject")
    import_parser.add_argument("--task")
    import_parser.add_argument("--output", required=True)
    import_parser.add_argument("--manifest")
    import_parser.add_argument("--overwrite", action="store_true")
    import_parser.add_argument("--json", action="store_true")
    import_parser.set_defaults(
        handler=lambda args: import_dataset(
            args.root,
            output=args.output,
            subject=args.subject,
            task=args.task,
            manifest=args.manifest,
            overwrite=args.overwrite,
        )
    )

    export = bids_sub.add_parser("export", help="Export an EEGLAB .set dataset to BIDS EEG.")
    export.add_argument("input")
    export.add_argument("--bids-root", required=True)
    export.add_argument("--subject", default="01")
    export.add_argument("--task", default="eeg")
    export.add_argument("--manifest")
    export.add_argument("--overwrite", action="store_true")
    export.add_argument("--json", action="store_true")
    export.set_defaults(
        handler=lambda args: export_dataset(
            args.input,
            args.bids_root,
            subject=args.subject,
            task=args.task,
            manifest=args.manifest,
            overwrite=args.overwrite,
        )
    )
    return parser


def validate_dataset(root: str | Path) -> dict[str, Any]:
    """Validate a BIDS root enough for EEGPrep CLI recovery logic."""
    root_path = Path(root).expanduser()
    if not root_path.exists():
        return _error(
            "INPUT_FILE_NOT_FOUND",
            f"BIDS root not found: {root_path}",
            path=root_path,
            suggestion="Create the BIDS root or pass the correct path.",
            exit_code=2,
        )
    if not root_path.is_dir():
        return _error("UNSUPPORTED_FORMAT", f"BIDS root is not a directory: {root_path}", path=root_path)
    dependency_error = _require_pybids("validate BIDS datasets")
    if dependency_error is not None:
        return dependency_error

    try:
        files = bids_list_eeg_files(str(root_path))
    except Exception as exc:
        return _error("BIDS_VALIDATION_FAILED", str(exc), path=root_path)
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if not (root_path / "dataset_description.json").exists():
        warnings.append(_issue("BIDS_VALIDATION_WARNING", "dataset_description.json is missing", root_path))
    if not files:
        errors.append(_issue("BIDS_EEG_FILES_MISSING", "No supported EEG files were found", root_path))
    status = "error" if errors else "warning" if warnings else "ok"
    return {
        "status": status,
        "schema_version": "eegprep.bids.validate.v1",
        "command": "bids validate",
        "bids_root": str(root_path),
        "files": files,
        "errors": errors,
        "warnings": warnings,
        "can_continue": not errors,
    }


def import_dataset(
    root: str | Path,
    *,
    output: str | Path,
    subject: str | None = None,
    task: str | None = None,
    manifest: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Import a BIDS EEG dataset and save it as an EEGLAB .set file."""
    root_path = Path(root).expanduser()
    started_at = now_iso()
    if not root_path.exists():
        return _error(
            "INPUT_FILE_NOT_FOUND",
            f"BIDS path not found: {root_path}",
            path=root_path,
            suggestion="Pass an existing BIDS root or BIDS EEG file.",
            exit_code=2,
        )
    dependency_error = _require_pybids("import BIDS datasets")
    if dependency_error is not None:
        return dependency_error

    if subject or task:
        files = bids_list_eeg_files(
            str(root_path),
            subjects=() if subject is None else [_normalize_entity(subject, "sub")],
            tasks=() if task is None else [_normalize_entity(task, "task")],
        )
        if not files:
            return _error(
                "BIDS_EEG_FILES_MISSING",
                "No BIDS EEG files matched the requested subject/task filters.",
                path=root_path,
            )
        EEG, history, warnings = _import_bids_file(Path(files[0]))
    else:
        if root_path.is_file():
            files = [str(root_path)]
            EEG, history, warnings = _import_bids_file(root_path)
        else:
            files = bids_list_eeg_files(str(root_path))
            if not files:
                return _error(
                    "BIDS_EEG_FILES_MISSING",
                    "No supported BIDS EEG files were found.",
                    path=root_path,
                )
            EEG, history, warnings = _import_bids_file(Path(files[0])) if files else ([], "", [])
    if isinstance(EEG, list):
        EEG = EEG[0]
    destination = output_path(output, overwrite=overwrite)
    manifest_error = _preflight_manifest(manifest, overwrite=overwrite)
    if manifest_error is not None:
        return manifest_error
    saved = pop_saveset(EEG, str(destination))
    output_files = _dataset_output_records(destination, saved)
    manifest_payload = build_manifest(
        command="bids import",
        input_files=[Path(path) for path in files if Path(path).exists()],
        output_files=output_files,
        parameters={"subject": subject, "task": task},
        history=history,
        started_at=started_at,
        deterministic=True,
        warnings=warnings,
    )
    manifest_entry = None if manifest is None else write_manifest_file(manifest, manifest_payload, overwrite=overwrite)
    return {
        "status": "ok",
        "schema_version": "eegprep.bids.import.v1",
        "command": "bids import",
        "bids_root": str(root_path),
        "output": str(destination),
        "manifest": None if manifest_entry is None else manifest_entry["path"],
        "output_files": output_files + ([] if manifest_entry is None else [manifest_entry]),
        "history": history,
        "dataset": _dataset_cli_summary(EEG, destination),
        "warnings": warnings,
    }


def export_dataset(
    input_path: str | Path,
    bids_root: str | Path,
    *,
    subject: str = "01",
    task: str = "eeg",
    manifest: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export an EEGLAB .set dataset to a BIDS EEG root."""
    source = Path(input_path).expanduser()
    root = Path(bids_root).expanduser()
    if root.exists() and any(root.iterdir()) and not overwrite:
        return _error(
            "OUTPUT_EXISTS",
            f"BIDS output root already exists and is not empty: {root}",
            path=root,
            suggestion="Use --overwrite or choose an empty BIDS root.",
        )
    manifest_error = _preflight_manifest(manifest, overwrite=overwrite)
    if manifest_error is not None:
        return manifest_error
    started_at = now_iso()
    EEG = load_dataset(source)
    subject_id = _normalize_entity(subject, "sub")
    task_id = _normalize_entity(task, "task")
    output_root, history = pop_exportbids(EEG, root, subject=subject_id, task=task_id, return_com=True)
    files = [path for path in Path(output_root).rglob("*") if path.is_file()]
    manifest_payload = build_manifest(
        command="bids export",
        input_files=[source],
        output_files=[{"path": str(path), "type": _file_type(path), "sha256": sha256_file(path)} for path in files],
        parameters={"subject": subject_id, "task": task_id},
        history=history,
        started_at=started_at,
        deterministic=True,
    )
    manifest_entry = None if manifest is None else write_manifest_file(manifest, manifest_payload, overwrite=overwrite)
    return {
        "status": "ok",
        "schema_version": "eegprep.bids.export.v1",
        "command": "bids export",
        "input": str(source),
        "bids_root": str(root),
        "subject": subject_id,
        "task": task_id,
        "manifest": None if manifest_entry is None else manifest_entry["path"],
        "history": history,
        "files": [str(path) for path in files],
    }


def main(argv: list[str] | None = None) -> int:
    """Standalone module harness for tests and local debugging."""
    parser = argparse.ArgumentParser(prog="eegprep bids")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register(subparsers)
    args = parser.parse_args(["bids", *(sys.argv[1:] if argv is None else argv)])
    result = args.handler(args)
    print(json.dumps(json_safe(result), sort_keys=True))
    return 0 if result.get("status") in {"ok", "warning"} else int(result.get("exit_code", 1))


def _file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".tsv":
        return "tsv"
    if suffix == ".set":
        return "eeglab_set"
    if suffix == ".fdt":
        return "eeglab_fdt"
    return "file"


def _preflight_manifest(path: str | Path | None, *, overwrite: bool) -> dict[str, Any] | None:
    if path is not None and Path(path).expanduser().exists() and not overwrite:
        return _error(
            "OUTPUT_EXISTS",
            f"Manifest already exists: {Path(path).expanduser()}",
            path=Path(path).expanduser(),
            suggestion="Use --overwrite or choose a different --manifest path.",
        )
    return None


def _dataset_output_records(set_path: Path, EEG: dict[str, Any]) -> list[dict[str, Any]]:
    records = [_output_record(set_path)]
    datfile = EEG.get("datfile")
    if datfile:
        dat_path = Path(str(datfile))
        if not dat_path.is_absolute():
            dat_path = set_path.parent / dat_path.name
        if dat_path.exists():
            records.append(_output_record(dat_path))
    return records


def _output_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "type": _file_type(path), "sha256": sha256_file(path)}


def _import_bids_file(eeg_file: Path) -> tuple[dict[str, Any] | list[dict[str, Any]], str, list[dict[str, str]]]:
    # The BIDS metadata importer (pop_importbids -> pop_load_frombids) only applies sidecar
    # metadata to raw recording formats (.edf/.bdf/.vhdr). EEGLAB .set files carry their own
    # metadata and cannot be routed through the sidecar-application path, so load them directly
    # with the EEGLAB .set loader instead of attempting BIDS sidecar import.
    if eeg_file.suffix.lower() == ".set":
        EEG = load_dataset(eeg_file)
        history = f"EEG = pop_importbids('{eeg_file.as_posix()}');"
        EEG["history"] = history
        return (
            EEG,
            history,
            [
                {
                    "code": "BIDS_SIDECARS_SKIPPED",
                    "message": (
                        "EEGLAB .set files are imported with the EEGLAB .set loader; "
                        "BIDS sidecar metadata is not applied for this format."
                    ),
                    "path": str(eeg_file),
                }
            ],
        )
    EEG, history = pop_importbids(eeg_file, return_com=True)
    return EEG, history, []


def _dataset_cli_summary(EEG: dict[str, Any], path: Path) -> dict[str, Any]:
    summary = dataset_summary(EEG, path)
    return {
        "path": str(path),
        "setname": summary["setname"],
        "nbchan": summary["n_channels"],
        "pnts": summary["n_points"],
        "trials": summary["n_trials"],
        "srate": summary["sampling_rate_hz"],
        "events": summary["n_events"],
        "data_shape": summary["data_shape"],
    }


def _normalize_entity(value: str | None, prefix: str) -> str:
    text = "" if value is None else str(value).strip()
    if text.startswith(f"{prefix}-"):
        return text.split("-", 1)[1]
    return text


def _require_pybids(action: str) -> dict[str, Any] | None:
    if importlib.util.find_spec("bids") is not None:
        return None
    return _error(
        "DEPENDENCY_MISSING",
        f"PyBIDS is required to {action}.",
        suggestion="Install EEGPrep with its BIDS dependencies, then rerun the command.",
    )


def _issue(code: str, message: str, path: Path) -> dict[str, str]:
    return {"code": code, "message": str(message), "path": str(path)}


def _error(
    code: str,
    message: str,
    *,
    path: str | Path | None = None,
    suggestion: str | None = None,
    exit_code: int = 1,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "error",
        "schema_version": "eegprep.bids.error.v1",
        "command": "bids",
        "code": code,
        "message": message,
        "error": {"code": code, "message": message},
        "exit_code": exit_code,
    }
    if path is not None:
        payload["path"] = str(path)
        payload["error"]["path"] = str(path)
    if suggestion is not None:
        payload["suggestion"] = suggestion
        payload["error"]["suggestion"] = suggestion
    return payload
