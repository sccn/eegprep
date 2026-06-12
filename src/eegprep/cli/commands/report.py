"""Agent-friendly EEGPrep report command support."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from eegprep.cli.commands.qc import compute_qc_metrics
from eegprep.cli.core import (
    EEGPrepCLIError as CommandError,
    build_manifest,
    command_error as error_result,
    command_ok as success_result,
    file_sha256,
    utc_now,
    write_manifest_file,
)
from eegprep.cli.dataset import load_dataset as load_eeg_dataset
from eegprep.cli.reporting import write_report_html


REPORT_SCHEMA_VERSION = "eegprep.report.v1"


def report_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write an HTML dataset report and return a structured result."""
    started_at = utc_now()
    try:
        EEG = load_eeg_dataset(input_path)
        metrics = compute_qc_metrics(EEG, dataset_path=input_path)
        report_entry = write_report_html(
            EEG,
            output_path,
            metrics=metrics,
            dataset_path=input_path,
            title="EEGPrep Report",
            overwrite=overwrite,
        )
        output_files = [report_entry]
        manifest_entry = None
        if manifest_path is not None:
            finished_at = utc_now()
            manifest = build_manifest(
                command="report",
                input_files=[{"path": str(input_path), "sha256": file_sha256(input_path)}],
                output_files=output_files,
                parameters={"output": str(output_path)},
                started_at=started_at,
                finished_at=finished_at,
                deterministic=True,
                warnings=metrics["recommendations"],
            )
            manifest_entry = write_manifest_file(manifest_path, manifest, overwrite=overwrite)
            output_files.append(manifest_entry)
        return success_result(
            "report",
            input=str(input_path),
            output_files=output_files,
            manifest=str(manifest_path) if manifest_entry else None,
            report={
                "schema_version": REPORT_SCHEMA_VERSION,
                "format": "html",
                "path": str(output_path),
            },
            qc=metrics,
        )
    except CommandError as exc:
        return error_result("report", exc)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Register ``report`` with an argparse dispatcher."""
    parser = subparsers.add_parser("report", help="Generate an HTML dataset report.")
    parser.add_argument("input", help="Input EEGLAB .set dataset")
    parser.add_argument("--output", required=True, help="Output HTML report path")
    parser.add_argument("--manifest", help="Optional manifest JSON path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON")
    parser.set_defaults(func=handle_registered, handler=handle_registered)
    return parser


def handle_registered(args: argparse.Namespace) -> dict[str, Any]:
    """Handle ``report`` when mounted into a larger dispatcher."""
    return report_dataset(args.input, args.output, manifest_path=args.manifest, overwrite=args.overwrite)


def main(argv: list[str] | None = None) -> int:
    """Route module execution through the canonical top-level CLI dispatcher."""
    from eegprep.cli.main import main as cli_main

    return cli_main(["report", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
