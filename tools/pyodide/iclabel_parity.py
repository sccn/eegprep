"""Run ICLabel on the checked-in ICA sample data for native/browser parity.

The browser loads the 8-bit ``iclabel_int8.onnx`` and the native backend loads the full-precision
``iclabel.onnx``. This script compares the browser with the native runtime on the same 8-bit
weights, so the two reports differ only by the runtime. The distance between the two networks is
the frozen quantization gate's (``tools/iclabel/quantize_iclabel_onnx.py``).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.resources
import json
import os
from pathlib import Path
import sys

import numpy as np

from eegprep.functions.popfunc.pop_loadset import pop_loadset


SCHEMA_VERSION = 2
DATASET_NAME = "eeglab_data_with_ica_tmp.set"
CLASS_COUNT = 7
MODEL_PACKAGE = "eegprep.plugins.ICLabel"
MODEL_NAME = "iclabel_int8.onnx"


def packaged_model_sha256() -> str:
    """Return the SHA-256 of the ``iclabel_int8.onnx`` this installation classifies with.

    Native runs hash the working-tree file; the Pyodide run hashes the copy
    inside the installed wheel, which is also where ``run_pyodide.sh`` extracts
    the bytes it gives the ONNX Runtime Web bridge. Equal digests in the two
    reports show that browser parity was measured on the model the tree ships.
    """
    model = importlib.resources.files(MODEL_PACKAGE).joinpath(MODEL_NAME)
    return hashlib.sha256(model.read_bytes()).hexdigest()


def _native_int8_classifications(eeg: dict) -> np.ndarray:
    """Classify with the packaged 8-bit network in native onnxruntime, around the production network call."""
    import onnxruntime

    from eegprep import ICL_feature_extractor
    from eegprep.plugins.ICLabel.iclabel import postprocess_network_output, prepare_network_inputs
    from eegprep.plugins.ICLabel.iclabel_net_onnx import INPUT_NAMES, OUTPUT_NAME

    model = importlib.resources.files(MODEL_PACKAGE).joinpath(MODEL_NAME)
    session = onnxruntime.InferenceSession(model.read_bytes(), providers=["CPUExecutionProvider"])
    inputs = dict(zip(INPUT_NAMES, prepare_network_inputs(ICL_feature_extractor(eeg, True))))
    (output,) = session.run([OUTPUT_NAME], inputs)
    return postprocess_network_output(output)


async def _classifications(eeg: dict, platform: str) -> np.ndarray:
    if platform == "native":
        classifications = _native_int8_classifications(eeg)
    else:
        from eegprep.plugins.ICLabel.iclabel import iclabel_async

        classified = await iclabel_async(eeg)
        classifications = classified["etc"]["ic_classification"]["ICLabel"]["classifications"]
    classifications = np.asarray(classifications, dtype=np.float32)
    if classifications.ndim != 2 or classifications.shape[1] != CLASS_COUNT:
        raise ValueError(f"Unexpected ICLabel classification shape: {classifications.shape}")
    if not np.isfinite(classifications).all():
        raise ValueError("ICLabel classifications contain non-finite values")
    return classifications


async def _run_parity(platform: str, sample_data_dir: Path) -> dict:
    """Await the classification and return the JSON report.

    This coroutine is the shared core: ``run_parity`` drives it with
    ``asyncio.run`` on native Python, and the Pyodide entry point below hands
    it to the host to await.
    """
    dataset_path = sample_data_dir / DATASET_NAME
    eeg = pop_loadset(dataset_path)
    classifications = await _classifications(eeg, platform)
    return {
        "schema_version": SCHEMA_VERSION,
        "platform": platform,
        "dataset": DATASET_NAME,
        "model_sha256": packaged_model_sha256(),
        "shape": list(classifications.shape),
        "classifications": classifications.tolist(),
    }


def run_parity(platform: str, sample_data_dir: Path) -> dict:
    """Synchronous wrapper for native callers: classify the sample dataset and return the report."""
    return asyncio.run(_run_parity(platform, sample_data_dir))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("native", "pyodide"), required=True)
    parser.add_argument(
        "--sample-data-dir",
        type=Path,
        default=Path(os.environ.get("EEGPREP_SAMPLE_DATA", "sample_data")),
    )
    return parser.parse_args()


async def _async_main() -> int:
    args = _parse_args()
    print(json.dumps(await _run_parity(args.platform, args.sample_data_dir), sort_keys=True))
    return 0


def main() -> int:
    args = _parse_args()
    print(json.dumps(run_parity(args.platform, args.sample_data_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    if sys.platform == "emscripten":
        # run_pyodide.mjs awaits this global after runpy returns, because asyncio.run cannot
        # block Pyodide's already-running event loop.
        __eegprep_async_result__ = _async_main()
    else:
        raise SystemExit(main())
