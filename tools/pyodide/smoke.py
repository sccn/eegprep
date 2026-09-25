"""Run a small continuous EEGPrep pipeline against the checked-in sample data."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_reref import pop_reref


def main() -> int:
    sample_dir = Path(os.environ["EEGPREP_SAMPLE_DATA"])
    dataset_path = sample_dir / "eeglab_data.set"
    eeg = pop_loadset(dataset_path)
    eeg = eeg_checkset(eeg)
    rereferenced, command = pop_reref(eeg, [], return_com=True)
    data = np.asarray(rereferenced["data"])
    if data.shape != (int(eeg["nbchan"]), int(eeg["pnts"])):
        raise AssertionError(f"Unexpected continuous data shape after rereferencing: {data.shape}")
    if not np.isfinite(data).all():
        raise AssertionError("Rereferenced sample data contains non-finite values")
    print(
        json.dumps(
            {
                "dataset": dataset_path.name,
                "nbchan": int(rereferenced["nbchan"]),
                "pnts": int(rereferenced["pnts"]),
                "trials": int(rereferenced["trials"]),
                "command": command,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
