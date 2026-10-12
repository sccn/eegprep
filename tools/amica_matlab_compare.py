"""Compare AMICA float32 input with MATLAB fwrite, then both decompositions.

Temporary CI check for the frame-order fix. MATLAB is the batch engine in
EEGPREP_MATLAB_BATCH_DIR. The AMICA source tree supplies runamica15.m and must
contain the same amica15ex binary EEGPrep runs.
"""

from __future__ import annotations

import argparse
import importlib
import os
import traceback
from pathlib import Path

import numpy as np

from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.sigprocfunc.runamica import (
    _write_data_file,
    is_amica_available,
    runamica,
)
from tests.matlab_batch_engine import MatlabExecutionError, install

# Overrides that match the MATLAB runamica15 defaults used for this comparison.
MATCHING = {
    "max_iter": 200,
    "max_threads": 1,
    "fix_init": 1,
    "mineig": 1e-12,
    "use_min_dll": 1,
    "use_grad_norm": 1,
    "write_nd": 0,
    "invsigmin": 1e-8,
}


def _max_abs(path_a: Path, path_b: Path) -> str:
    if not path_a.is_file() or not path_b.is_file():
        return f"missing python={path_a.is_file()} matlab={path_b.is_file()}"
    left = np.fromfile(path_a, dtype=np.float64)
    right = np.fromfile(path_b, dtype=np.float64)
    if left.shape != right.shape:
        return f"shape {left.shape} vs {right.shape}"
    if left.size == 0:
        return "empty"
    return f"{float(np.max(np.abs(left - right))):.6e} n={left.size}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amica-src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    eeg = pop_loadset("sample_data/eeglab_data.set")
    data = np.asarray(eeg["data"], dtype=np.float64)
    if data.ndim == 3:
        data = np.reshape(data, (data.shape[0], -1), order="F")
    data = np.ascontiguousarray(data[:4, :3000])
    print(f"data_shape={tuple(data.shape)} dtype={data.dtype}")

    py_path = (args.out / "python.fdt").resolve()
    mat_path = (args.out / "matlab.fdt").resolve()
    _write_data_file(data, py_path)

    install(os.environ["EEGPREP_MATLAB_BATCH_DIR"])
    matlab_engine = importlib.import_module("matlab.engine")
    eng = matlab_engine.start_matlab()
    eng.workspace["dat"] = np.array(data, dtype=np.float64, order="F")
    shape = np.asarray(eng.eval("size(dat)", nargout=1)).astype(int).ravel()
    print(f"matlab_size={shape.tolist()}")
    eng.eval(
        f"fid = fopen('{mat_path.as_posix()}', 'wb'); count = fwrite(fid, dat, 'float'); fclose(fid);",
        nargout=0,
    )
    count = int(eng.eval("count", nargout=1))
    print(f"fwrite_count={count}")

    written = np.fromfile(py_path, dtype="<f4")
    matlab = np.fromfile(mat_path, dtype="<f4")
    print(
        f"input_n_python={written.size} input_n_matlab={matlab.size} "
        f"bytes_equal={py_path.read_bytes() == mat_path.read_bytes()}"
    )
    failed = written.shape != matlab.shape or written.size != data.size
    if failed:
        print("INPUT_FAIL shape mismatch")
    else:
        max_abs = float(np.max(np.abs(written.astype(np.float64) - matlab.astype(np.float64))))
        read = written.reshape(data.shape, order="F")
        mean_diff = float(np.max(np.abs(read.mean(axis=1) - data.astype("<f4").mean(axis=1))))
        print(f"input_max_abs={max_abs:.6e}")
        print(f"channel_mean_max_abs={mean_diff:.6e}")
        failed = max_abs != 0.0

    print(f"amica_binary_available={is_amica_available()}")
    if not is_amica_available():
        print("AMICA_BINARY infra: amica15ub did not start on this runner. Decomposition comparison skipped.")
    else:
        py_out = args.out / "python_amica"
        mat_out = args.out / "matlab_amica"
        py_out.mkdir()
        mat_out.mkdir()
        try:
            runamica(data, outdir=str(py_out), cleanup=False, **MATCHING)
            print("python_runamica=ok")
        except Exception:
            print("python_runamica=FAIL")
            traceback.print_exc()
            failed = True
        else:
            eng.addpath(str(args.amica_src.resolve()), nargout=0)
            pairs: list[object] = []
            for key, value in MATCHING.items():
                pairs.extend((key, value))
            try:
                eng.runamica15(data, *pairs, "outdir", f"{mat_out.resolve().as_posix()}/", nargout=0)
                print("matlab_runamica15=ok")
            except MatlabExecutionError as exc:
                print("matlab_runamica15=FAIL")
                print(exc)
                print("AMICA_DECOMPOSITION matlab runamica15 failed.")
            else:
                for name in ("mean", "S", "W", "LL"):
                    print(f"{name}_max_abs={_max_abs(py_out / name, mat_out / name)}")

    print("RESULT=fail" if failed else "RESULT=pass")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
