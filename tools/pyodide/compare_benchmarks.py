"""Compare native and Pyodide Phase 2 benchmark reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BLAS_SPEEDUP_GATE = 1.2
REQUIRED_ICA_ALGORITHMS = ("runica", "picard")
REQUIRED_MATMUL_CASES = ("activation", "weight_update")


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if report.get("schema_version") != 1:
        raise ValueError(f"Unsupported benchmark schema in {path}")
    if tuple(report.get("shape", ())) != (64, 15_000):
        raise ValueError(f"Unexpected benchmark shape in {path}: {report.get('shape')}")
    return report


def _matmul_lookup(report: dict[str, Any], case: str, dtype: str, operation: str) -> dict[str, Any]:
    matches = [
        item
        for item in report["matmul"]
        if item["case"] == case and item["dtype"] == dtype and item["operation"] == operation
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {operation} result for {case}/{dtype}, found {len(matches)}")
    return matches[0]


def compare_reports(native: dict[str, Any], pyodide: dict[str, Any]) -> dict[str, Any]:
    """Return the Phase 2 decisions and ratios for two valid reports."""
    if native.get("platform") != "native" or pyodide.get("platform") != "pyodide":
        raise ValueError("Reports must be labeled native and pyodide")
    if (
        native["seed"] != pyodide["seed"]
        or native["runica_block"] != pyodide["runica_block"]
        or native["thread_count"] != pyodide["thread_count"]
    ):
        raise ValueError("Native and Pyodide reports do not use the same benchmark configuration")

    ica = {}
    for algorithm in REQUIRED_ICA_ALGORITHMS:
        native_summary = native["ica"][algorithm]
        pyodide_summary = pyodide["ica"][algorithm]
        if native_summary["median_seconds"] is None or pyodide_summary["median_seconds"] is None:
            raise ValueError(f"Missing successful {algorithm} timing")
        ica[algorithm] = {
            "native_median_seconds": native_summary["median_seconds"],
            "native_median_iterations": native_summary["median_iterations"],
            "native_all_converged": native_summary["all_converged"],
            "pyodide_median_seconds": pyodide_summary["median_seconds"],
            "pyodide_median_iterations": pyodide_summary["median_iterations"],
            "pyodide_all_converged": pyodide_summary["all_converged"],
            "pyodide_speed_ratio": pyodide_summary["median_seconds"] / native_summary["median_seconds"],
        }

    picard = ica["picard"]
    runica_result = ica["runica"]
    picard_browser_default_retained = bool(
        picard["pyodide_all_converged"]
        and runica_result["pyodide_median_iterations"] is not None
        and picard["pyodide_median_iterations"] is not None
        and picard["pyodide_median_iterations"] < runica_result["pyodide_median_iterations"]
    )

    matmul = []
    phase3_gate_results = []
    for case in REQUIRED_MATMUL_CASES:
        for dtype, blas_name in (("float64", "dgemm"), ("float32", "sgemm")):
            native_numpy_result = _matmul_lookup(native, case, dtype, "numpy_matmul")
            native_blas_result = _matmul_lookup(native, case, dtype, blas_name)
            numpy_result = _matmul_lookup(pyodide, case, dtype, "numpy_matmul")
            blas_result = _matmul_lookup(pyodide, case, dtype, blas_name)
            native_speedup = native_numpy_result["median_seconds"] / native_blas_result["median_seconds"]
            speedup = numpy_result["median_seconds"] / blas_result["median_seconds"]
            phase3_gate_results.append(speedup >= BLAS_SPEEDUP_GATE)
            matmul.append(
                {
                    "case": case,
                    "dtype": dtype,
                    "native_numpy_median_seconds": native_numpy_result["median_seconds"],
                    "native_blas": blas_name,
                    "native_blas_median_seconds": native_blas_result["median_seconds"],
                    "native_blas_speedup_over_numpy": native_speedup,
                    "numpy_median_seconds": numpy_result["median_seconds"],
                    "blas": blas_name,
                    "blas_median_seconds": blas_result["median_seconds"],
                    "blas_speedup_over_numpy": speedup,
                }
            )

    return {
        "schema_version": 1,
        "shape": pyodide["shape"],
        "seed": pyodide["seed"],
        "ica": ica,
        "decisions": {
            "picard_browser_default_retained": picard_browser_default_retained,
            "phase3_recommended": all(phase3_gate_results),
            "phase3_gate": f"scipy BLAS >= {BLAS_SPEEDUP_GATE:.1f}x faster than NumPy @ for both Pyodide runica shapes and dtypes",
        },
        "matmul": matmul,
    }


def _markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Phase 2 Pyodide benchmark comparison",
        "",
        f"Input: `{comparison['shape'][0]} x {comparison['shape'][1]}`, seed `{comparison['seed']}`.",
        "",
        "## ICA",
        "",
        "| Algorithm | Native median (s) | Native median iterations | Native converged | Pyodide median (s) | Pyodide median iterations | Pyodide converged |",
        "| --- | ---: | ---: | :---: | ---: | ---: | :---: |",
    ]
    for algorithm, result in comparison["ica"].items():
        lines.append(
            f"| {algorithm} | {result['native_median_seconds']:.6f} | {result['native_median_iterations']:.1f} | "
            f"{result['native_all_converged']} | {result['pyodide_median_seconds']:.6f} | "
            f"{result['pyodide_median_iterations']:.1f} | {result['pyodide_all_converged']} |"
        )
    lines.extend(
        [
            "",
            "## Matrix multiplication",
            "",
            "| runica product | dtype | Native NumPy @ (s) | Native BLAS (s) | Native speedup | Pyodide NumPy @ (s) | Pyodide BLAS (s) | Pyodide speedup |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in comparison["matmul"]:
        lines.append(
            f"| {result['case']} | {result['dtype']} | {result['native_numpy_median_seconds']:.6f} | "
            f"{result['native_blas_median_seconds']:.6f} | {result['native_blas_speedup_over_numpy']:.2f}x | "
            f"{result['numpy_median_seconds']:.6f} | {result['blas_median_seconds']:.6f} | "
            f"{result['blas_speedup_over_numpy']:.2f}x |"
        )
    decisions = comparison["decisions"]
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            f"- Retain Picard as the browser default: `{decisions['picard_browser_default_retained']}`.",
            f"- Recommend Phase 3 BLAS work: `{decisions['phase3_recommended']}`.",
            f"- Gate: {decisions['phase3_gate']}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--pyodide", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    comparison = compare_reports(_load_report(args.native), _load_report(args.pyodide))
    output = json.dumps(comparison, indent=2, sort_keys=True)
    print(output)
    if args.report:
        args.report.write_text(_markdown(comparison) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
