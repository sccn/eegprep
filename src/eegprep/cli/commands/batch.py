"""Batch pipeline command support for the EEGPrep CLI."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import yaml

from eegprep.cli.commands.pipeline import run_pipeline_config
from eegprep.cli.core import EEGPrepCLIError
from eegprep.plugins.EEG_BIDS.bids_list_eeg_files import bids_list_eeg_files


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register ``batch`` commands."""
    parser = subparsers.add_parser("batch", help="Run an EEGPrep pipeline over several datasets.")
    batch_sub = parser.add_subparsers(dest="batch_command", required=True)
    run = batch_sub.add_parser("run", help="Run a pipeline for explicit inputs or BIDS subjects.")
    run.add_argument("inputs", nargs="*", help="Input EEGLAB .set files. Omit when using --bids-root.")
    run.add_argument("--pipeline", required=True, help="Pipeline YAML config to apply.")
    run.add_argument("--output-dir", help="Root directory for per-input outputs.")
    run.add_argument("--bids-root", help="BIDS root to discover subject EEG files from.")
    run.add_argument("--subjects", nargs="+", help="Optional BIDS subjects, with or without sub- prefix.")
    run.add_argument(
        "--jobs", type=int, default=1, help="Reserved for future parallel execution; currently sequential."
    )
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--json", action="store_true")
    run.set_defaults(handler=run_batch)
    return parser


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Run a pipeline across several inputs and return per-input status."""
    inputs = _resolve_inputs(args)
    if not inputs:
        raise EEGPrepCLIError(
            "INPUT_FILE_NOT_FOUND",
            "Batch run requires at least one input file or a BIDS root with EEG files.",
            exit_code=2,
        )
    if args.jobs != 1:
        jobs_warning = {
            "code": "BATCH_JOBS_SEQUENTIAL",
            "message": "--jobs is accepted for pipeline config compatibility, but this implementation runs sequentially.",
        }
    else:
        jobs_warning = None

    from eegprep.cli.cloud import resolve_output
    results = []
    successes = 0
    failures = 0
    
    if args.output_dir and isinstance(args.output_dir, str) and (args.output_dir.startswith("s3://") or args.output_dir.startswith("gs://")):
        resolved_output_root = resolve_output(args.output_dir)
    elif args.output_dir:
        resolved_output_root = Path(args.output_dir).expanduser()
    else:
        resolved_output_root = None

    for input_path in inputs:
        with tempfile.TemporaryDirectory(prefix="eegprep-batch-") as tmpdir:
            config_path = _write_input_config(
                args.pipeline,
                input_path,
                tmpdir,
                output_root=resolved_output_root,
                overwrite=bool(args.overwrite),
            )
            result = run_pipeline_config(config_path, dry_run=bool(args.dry_run), overwrite=bool(args.overwrite))
        item = {
            "input": str(input_path),
            "status": result.get("status"),
            "pipeline_result": result,
        }
        if result.get("status") == "ok":
            successes += 1
            item["manifest"] = result.get("manifest")
        else:
            failures += 1
            item["error"] = result.get("error")
        results.append(item)

    status = "ok" if failures == 0 else "partial_success" if successes else "error"
    warnings = [jobs_warning] if jobs_warning else []
    return {
        "status": status,
        "schema_version": "eegprep.batch.run.v1",
        "command": "batch run",
        "subjects_total": len(inputs),
        "subjects_success": successes,
        "subjects_failed": failures,
        "results": results,
        "warnings": warnings,
        "exit_code": 0 if failures == 0 else 1,
    }


def _resolve_inputs(args: argparse.Namespace) -> list[Path]:
    from eegprep.cli.core import existing_input
    from eegprep.cli.cloud import resolve_input
    if args.bids_root:
        if isinstance(args.bids_root, str) and (args.bids_root.startswith("s3://") or args.bids_root.startswith("gs://")):
            root_path = resolve_input(args.bids_root)
        else:
            root_path = Path(args.bids_root).expanduser()
        subjects = [_strip_prefix(subject, "sub") for subject in (args.subjects or [])]
        return [Path(path) for path in bids_list_eeg_files(str(root_path), subjects=subjects)]
    return [existing_input(path) for path in args.inputs]


def _write_input_config(
    source_config: str | Path,
    input_path: Path,
    tmpdir: str | Path,
    *,
    output_root: Path | None,
    overwrite: bool,
) -> Path:
    config_path = Path(source_config).expanduser()
    if not config_path.exists():
        raise EEGPrepCLIError("INPUT_FILE_NOT_FOUND", f"Pipeline config not found: {config_path}", path=config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise EEGPrepCLIError("CONFIG_SCHEMA_ERROR", "Pipeline config must be a YAML mapping.", path=config_path)
    config.setdefault("input", {})
    config["input"]["path"] = str(input_path.expanduser().resolve())
    config.setdefault("output", {})
    if output_root is not None:
        config["output"]["directory"] = str(output_root / input_path.stem)
    else:
        _resolve_relative_output_paths(config["output"], config_path.parent)
    config["output"]["overwrite"] = bool(config["output"].get("overwrite", False) or overwrite)
    target = Path(tmpdir) / "pipeline.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def _strip_prefix(value: str, prefix: str) -> str:
    text = str(value)
    return text.split("-", 1)[1] if text.startswith(f"{prefix}-") else text


def _resolve_relative_output_paths(output: dict[str, Any], base_dir: Path) -> None:
    for key in ("directory", "dataset", "manifest", "qc", "report"):
        value = output.get(key)
        if isinstance(value, str) and value.strip() and not Path(value).expanduser().is_absolute():
            output[key] = str((base_dir / value).resolve())
