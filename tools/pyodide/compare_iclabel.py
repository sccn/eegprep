"""Compare native and ONNX Runtime Web ICLabel classification reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


RTOL = 1e-4
ATOL = 1e-5


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if report.get("schema_version") != 1:
        raise ValueError(f"Unsupported ICLabel parity schema in {path}")
    if report.get("platform") not in {"native", "pyodide"}:
        raise ValueError(f"Invalid ICLabel parity platform in {path}")
    classifications = np.asarray(report.get("classifications"), dtype=np.float32)
    if classifications.ndim != 2 or classifications.shape[1] != 7:
        raise ValueError(f"Unexpected ICLabel classification shape in {path}: {classifications.shape}")
    if list(classifications.shape) != report.get("shape"):
        raise ValueError(f"ICLabel report shape does not match classifications in {path}")
    if not np.isfinite(classifications).all():
        raise ValueError(f"ICLabel report contains non-finite values in {path}")
    return report


def compare_reports(native: dict[str, Any], pyodide: dict[str, Any]) -> dict[str, Any]:
    """Return numeric parity metrics and the Phase 5 pass/fail decision."""
    if native.get("platform") != "native" or pyodide.get("platform") != "pyodide":
        raise ValueError("Reports must be labeled native and pyodide")
    if native.get("dataset") != pyodide.get("dataset"):
        raise ValueError("Native and Pyodide reports use different datasets")

    native_values = np.asarray(native["classifications"], dtype=np.float32)
    pyodide_values = np.asarray(pyodide["classifications"], dtype=np.float32)
    if native_values.shape != pyodide_values.shape:
        raise ValueError(f"Classification shapes differ: {native_values.shape} != {pyodide_values.shape}")
    difference = np.abs(native_values - pyodide_values)
    scale = np.maximum(np.maximum(np.abs(native_values), np.abs(pyodide_values)), np.float32(1e-10))
    max_absolute = float(np.max(difference, initial=0.0))
    max_relative = float(np.max(difference / scale, initial=0.0))
    return {
        "schema_version": 1,
        "dataset": native["dataset"],
        "shape": list(native_values.shape),
        "rtol": RTOL,
        "atol": ATOL,
        "max_absolute_difference": max_absolute,
        "max_relative_difference": max_relative,
        "allclose": bool(np.allclose(native_values, pyodide_values, rtol=RTOL, atol=ATOL)),
    }


def _markdown(comparison: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# Phase 5 ICLabel browser parity",
            "",
            f"Dataset: `{comparison['dataset']}`; shape: `{comparison['shape']}`.",
            "",
            f"- `allclose`: `{comparison['allclose']}`",
            f"- maximum absolute difference: `{comparison['max_absolute_difference']:.3e}`",
            f"- maximum relative difference: `{comparison['max_relative_difference']:.3e}`",
            f"- tolerance: `rtol={comparison['rtol']}`, `atol={comparison['atol']}`",
            "",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--pyodide", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    comparison = compare_reports(_load_report(args.native), _load_report(args.pyodide))
    print(json.dumps(comparison, indent=2, sort_keys=True))
    if args.report:
        args.report.write_text(_markdown(comparison))
    return 0 if comparison["allclose"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
