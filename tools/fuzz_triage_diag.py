"""Replay the nightly fuzz repros for issues 464-468 against matlab-batch.

Prints both sides, full MATLAB stacks, and the NaN-mask locations. Not part of
the package.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np

from tests.fuzz.harness import (
    MatlabBackend,
    PythonBackend,
    case_from_payload,
    dispatch_matlab,
    dispatch_python,
    materialize_eeg,
)

REPRO_DIR = Path(__file__).resolve().parent / "fuzz_triage_repros"


def _load_cases() -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(REPRO_DIR.glob("*.json"))]


def _nan_indices(data: np.ndarray) -> list[int]:
    bad = ~np.isfinite(np.asarray(data))
    return [int(i) for i in np.flatnonzero(bad.ravel(order="C"))]


def _finite_gap(python_data, oracle_data) -> str:
    python = np.asarray(python_data, dtype=np.float64)
    oracle = np.asarray(oracle_data, dtype=np.float64)
    if python.shape != oracle.shape:
        return f"shape {python.shape} != {oracle.shape}"
    finite = np.isfinite(python) & np.isfinite(oracle)
    if not np.any(finite):
        return "no shared finite samples"
    diff = np.abs(python[finite] - oracle[finite])
    return f"finite max abs {float(np.max(diff)):.6g} n {int(np.count_nonzero(finite))}"


def main() -> None:
    python = PythonBackend()
    oracle = MatlabBackend()
    engine = oracle.engine()
    for statement in (
        "exist('yulewalk')",
        "which('yulewalk')",
        "exist('resample')",
        "which('resample')",
    ):
        try:
            value = engine.engine.eval(statement, nargout=1)
        except Exception as exc:
            value = f"{type(exc).__name__}: {exc}"
        print(f"PROBE {statement} -> {value!r}", flush=True)

    for payload in _load_cases():
        print("=" * 72, flush=True)
        print(payload["signature"], flush=True)
        case = case_from_payload(payload)
        eeg = materialize_eeg(case)
        data = np.asarray(eeg["data"])
        print(
            f"input shape {data.shape} srate {eeg['srate']} nans {int(np.count_nonzero(~np.isfinite(data)))}",
            flush=True,
        )
        try:
            py = dispatch_python(case, eeg)
            py_data = np.asarray(py["data"])
            print(
                f"python ok shape {py_data.shape} srate {py.get('srate')} nans {int(np.count_nonzero(~np.isfinite(py_data)))} idx {_nan_indices(py_data)[:40]}",
                flush=True,
            )
            changed = float(np.nanmax(np.abs(py_data - data))) if py_data.shape == data.shape else "shape-changed"
            print(f"python vs input max {changed}", flush=True)
        except Exception as exc:
            py = None
            print(f"python FAIL {type(exc).__name__}: {exc}", flush=True)
        try:
            mat = dispatch_matlab(engine, case, eeg)
            mat_data = np.asarray(mat["data"])
            print(
                f"matlab ok shape {mat_data.shape} srate {mat.get('srate')} nans {int(np.count_nonzero(~np.isfinite(mat_data)))} idx {_nan_indices(mat_data)[:40]}",
                flush=True,
            )
            if mat_data.shape == data.shape:
                gap = np.abs(mat_data.astype(np.float64) - data.astype(np.float64))
                print(
                    f"matlab vs input max {float(np.nanmax(gap))} equal {bool(np.array_equal(mat_data, data, equal_nan=True))}",
                    flush=True,
                )
            else:
                print("matlab vs input shape-changed", flush=True)
            if py is not None:
                print(_finite_gap(py["data"], mat["data"]), flush=True)
                py_bad = ~np.isfinite(np.asarray(py["data"]))
                mat_bad = ~np.isfinite(mat_data)
                if py_bad.shape == mat_bad.shape:
                    only_py = np.flatnonzero(py_bad.ravel() & ~mat_bad.ravel())
                    only_mat = np.flatnonzero(mat_bad.ravel() & ~py_bad.ravel())
                    print(f"nan only python {only_py.tolist()} only matlab {only_mat.tolist()}", flush=True)
        except Exception:
            print("matlab FAIL", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
