"""One-shot MATLAB diagnostics for fuzz issues 451, 452, and 453.

Prints calibration masks, NaN counts, and min-phase coefficient paths.
Not part of the test suite.
"""

from __future__ import annotations

import json
import traceback

import numpy as np

from eegprep.plugins.clean_rawdata.clean_asr import clean_asr
from eegprep.plugins.clean_rawdata.clean_windows import clean_windows
from eegprep.plugins.firfilt._filtering import apply_fir_filter, design_eegfiltnew
from eegprep.plugins.firfilt.minphaserceps import minphaserceps
from tests.fuzz.harness import (
    MatlabBackend,
    case_from_payload,
    dispatch_matlab,
    dispatch_python,
    materialize_eeg,
)

REPROS = {
    "clean_asr": {
        "function": "clean_asr",
        "params": {"cutoff": 5.0, "entry": "clean_asr", "ref_maxbadchannels": 0.075, "window_len": None},
        "spec": {
            "channels": list(range(11)),
            "decimate": 1,
            "length": 256,
            "nan_channel": -1,
            "nan_length": 0,
            "nan_start": 0,
            "start": 8,
        },
    },
    "pop_reref": {
        "function": "pop_reref",
        "params": {"exclude": [], "keepref": "off", "ref": [3]},
        "spec": {
            "channels": list(range(11, 28)),
            "decimate": 1,
            "length": 1399,
            "nan_channel": 0,
            "nan_length": 3,
            "nan_start": 1396,
            "start": 512,
        },
    },
    "pop_eegfiltnew": {
        "function": "pop_eegfiltnew",
        "params": {
            "filtorder": 100,
            "hicutoff": 25,
            "locutoff": None,
            "minphase": True,
            "revfilt": False,
            "usefftfilt": False,
        },
        "spec": {
            "channels": [7],
            "decimate": 1,
            "length": 256,
            "nan_channel": -1,
            "nan_length": 0,
            "nan_start": 0,
            "start": 252,
        },
    },
}


def _case(name: str):
    return case_from_payload(REPROS[name])


def _max_abs(a, b) -> tuple[float, int]:
    diff = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    if diff.size == 0:
        return 0.0, -1
    return float(np.max(diff)), int(np.argmax(diff))


def _spans(mask: np.ndarray) -> list[tuple[int, int]]:
    flags = np.asarray(mask, dtype=bool).ravel()
    edges = np.diff(np.r_[False, flags, False].astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def diagnose_asr(engine) -> None:
    case = _case("clean_asr")
    eeg = materialize_eeg(case)
    data = np.asarray(eeg["data"])
    print("ASR input", data.shape, data.dtype, "srate", eeg["srate"], "has etc", "etc" in eeg)
    py_win = np.asarray(dispatch_python(case, eeg)["data"])
    py_off = np.asarray(clean_asr(eeg, cutoff=5.0, ref_maxbadchannels="off")["data"])
    _cleaned, py_mask = clean_windows(eeg, 0.075, (-3.5, 5.5), 1.0)
    print("python clean_windows kept", int(np.count_nonzero(py_mask)), "spans", _spans(py_mask))
    print("python win vs off", _max_abs(py_win, py_off))

    try:
        ml = dispatch_matlab(engine, case, eeg)
        ml_data = np.asarray(ml["data"])
        print("matlab clean_asr", ml_data.shape, ml_data.dtype)
        print("matlab vs python win", _max_abs(py_win, ml_data))
        print("matlab vs python off", _max_abs(py_off, ml_data))
        print("matlab vs float32(python off)", _max_abs(py_off.astype(np.float32), ml_data))
    except Exception:
        print("matlab clean_asr FAILED")
        traceback.print_exc()

    try:
        ml_clean, ml_mask = engine.clean_windows(eeg, 0.075, np.array([-3.5, 5.5]), 1.0, nargout=2)
        ml_mask = np.asarray(ml_mask).astype(bool).ravel()
        print(
            "matlab clean_windows kept",
            int(np.count_nonzero(ml_mask)),
            "len",
            ml_mask.size,
            "data",
            np.asarray(ml_clean["data"]).shape,
            "spans",
            _spans(ml_mask),
        )
        print("mask equal", bool(np.array_equal(ml_mask, np.asarray(py_mask).ravel())))
    except Exception:
        print("matlab clean_windows FAILED")
        traceback.print_exc()


def diagnose_reref(engine) -> None:
    case = _case("pop_reref")
    eeg = materialize_eeg(case)
    data = np.asarray(eeg["data"], dtype=np.float64)
    py = np.asarray(dispatch_python(case, eeg)["data"], dtype=np.float64)
    nchan = data.shape[0]
    ref = 3
    refmatrix = np.eye(nchan)
    refmatrix[:, ref] -= 1.0
    matrix = np.delete(refmatrix @ data, ref, axis=0)
    print("input nans", int(np.count_nonzero(~np.isfinite(data))))
    print(
        "python nans",
        int(np.count_nonzero(~np.isfinite(py))),
        "matrix nans",
        int(np.count_nonzero(~np.isfinite(matrix))),
    )
    print("python vs matrix mask", int(np.count_nonzero(~np.isfinite(py) != ~np.isfinite(matrix))))
    try:
        ml = np.asarray(dispatch_matlab(engine, case, eeg)["data"], dtype=np.float64)
        print("matlab shape", ml.shape, "nans", int(np.count_nonzero(~np.isfinite(ml))))
        print("matlab vs python mask", int(np.count_nonzero(~np.isfinite(py) != ~np.isfinite(ml))))
        print("matlab vs matrix mask", int(np.count_nonzero(~np.isfinite(matrix) != ~np.isfinite(ml))))
        finite = np.isfinite(py) & np.isfinite(ml)
        if np.any(finite):
            print("finite python vs matlab", _max_abs(py[finite], ml[finite]))
    except Exception:
        print("matlab pop_reref FAILED")
        traceback.print_exc()


def diagnose_filter(engine) -> None:
    case = _case("pop_eegfiltnew")
    eeg = materialize_eeg(case)
    srate = float(eeg["srate"])
    py = np.asarray(dispatch_python(case, eeg)["data"], dtype=np.float64)
    b_lin, _meta = design_eegfiltnew(srate, hicutoff=25, filtorder=100, minphase=False)
    b_ceps = minphaserceps(b_lin)
    ceps_eeg = apply_fir_filter(eeg, b_ceps, causal=True)
    ceps = np.asarray(ceps_eeg["data"], dtype=np.float64)
    print("scipy vs minphaserceps", _max_abs(py, ceps))
    try:
        ml = np.asarray(dispatch_matlab(engine, case, eeg)["data"], dtype=np.float64)
        print("matlab vs scipy", _max_abs(py, ml))
        print("matlab vs minphaserceps", _max_abs(ceps, ml))
        print("matlab vs float32(minphaserceps)", _max_abs(ceps.astype(np.float32), ml))
    except Exception:
        print("matlab pop_eegfiltnew FAILED")
        traceback.print_exc()


def main() -> None:
    engine = MatlabBackend().engine()
    print("=== clean_asr #451 ===")
    diagnose_asr(engine)
    print("=== pop_reref #452 ===")
    diagnose_reref(engine)
    print("=== pop_eegfiltnew #453 ===")
    diagnose_filter(engine)
    print(json.dumps({"done": True}))


if __name__ == "__main__":
    main()
