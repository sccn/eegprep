"""Run ICLabel on the checked-in ICA sample data for native/browser parity."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import numpy as np

from eegprep.functions.popfunc.pop_loadset import pop_loadset


DATASET_NAME = "eeglab_data_with_ica_tmp.set"
CLASS_COUNT = 7


async def _classifications(eeg: dict, platform: str) -> np.ndarray:
    if platform == "native":
        from eegprep.plugins.ICLabel.iclabel import iclabel

        classified = iclabel(eeg)
    else:
        from eegprep.plugins.ICLabel.iclabel import iclabel_async

        classified = await iclabel_async(eeg)
    classifications = np.asarray(
        classified["etc"]["ic_classification"]["ICLabel"]["classifications"],
        dtype=np.float32,
    )
    if classifications.ndim != 2 or classifications.shape[1] != CLASS_COUNT:
        raise ValueError(f"Unexpected ICLabel classification shape: {classifications.shape}")
    if not np.isfinite(classifications).all():
        raise ValueError("ICLabel classifications contain non-finite values")
    return classifications


async def _run_parity(platform: str, sample_data_dir: Path) -> dict:
    """Classify the same sample dataset on one platform and return JSON data."""
    dataset_path = sample_data_dir / DATASET_NAME
    eeg = pop_loadset(dataset_path)
    classifications = await _classifications(eeg, platform)
    return {
        "schema_version": 1,
        "platform": platform,
        "dataset": DATASET_NAME,
        "shape": list(classifications.shape),
        "classifications": classifications.tolist(),
    }


def run_parity(platform: str, sample_data_dir: Path) -> dict:
    """Classify the same sample dataset on one platform and return JSON data."""
    return asyncio.run(_run_parity(platform, sample_data_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("native", "pyodide"), required=True)
    parser.add_argument(
        "--sample-data-dir",
        type=Path,
        default=Path(os.environ.get("EEGPREP_SAMPLE_DATA", "sample_data")),
    )
    args = parser.parse_args()
    print(json.dumps(run_parity(args.platform, args.sample_data_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
