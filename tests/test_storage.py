from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.adminfunc.storage import MemmapData
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_saveset import pop_saveset


@pytest.fixture(autouse=True)
def restore_eeg_options():
    old_options = dict(EEG_OPTIONS)
    try:
        yield
    finally:
        EEG_OPTIONS.clear()
        EEG_OPTIONS.update(old_options)


def _eeg(data: np.ndarray, *, name: str = "storage") -> dict:
    eeg = eeg_emptyset()
    trials = int(data.shape[2]) if data.ndim == 3 else 1
    eeg.update(
        {
            "setname": name,
            "data": data,
            "nbchan": int(data.shape[0]),
            "pnts": int(data.shape[1]),
            "trials": trials,
            "srate": 100.0,
            "xmin": 0.0,
            "xmax": (int(data.shape[1]) - 1) / 100.0,
            "times": np.arange(int(data.shape[1]), dtype=float),
            "chanlocs": [{"labels": f"Ch{index}"} for index in range(1, int(data.shape[0]) + 1)],
            "event": [],
            "saved": "no",
        }
    )
    return eeg


def test_pop_saveset_twofiles_roundtrips_epoched_data_as_memmap(tmp_path: Path):
    data = np.arange(2 * 3 * 2, dtype=np.float32).reshape((2, 3, 2))
    set_file = tmp_path / "epoched.set"

    pop_saveset(_eeg(data, name="epoched"), set_file, savemode="twofiles")
    EEG_OPTIONS["option_memmapdata"] = 1
    loaded = pop_loadset(set_file)

    assert (tmp_path / "epoched.fdt").exists()
    assert isinstance(loaded["data"], MemmapData)
    assert loaded["data"].shape == data.shape
    assert loaded["data"][0, 0, 0] == data[0, 0, 0]
    assert loaded["data"][1, 2, 1] == data[1, 2, 1]
    np.testing.assert_allclose(np.asarray(loaded["data"]), data)

    loaded["data"][1, 2, 1] = -123.0
    loaded["data"].flush()
    reloaded = pop_loadset(set_file, memmap=True)

    assert isinstance(reloaded["data"], MemmapData)
    assert reloaded["data"][1, 2, 1] == -123.0


def test_pop_saveset_default_single_file_keeps_data_inline(tmp_path: Path):
    data = np.arange(6, dtype=np.float32).reshape((2, 3))
    set_file = tmp_path / "inline.set"

    pop_saveset(_eeg(data), set_file)
    loaded = pop_loadset(set_file, memmap=True)

    assert not (tmp_path / "inline.fdt").exists()
    assert not isinstance(loaded["data"], MemmapData)
    np.testing.assert_allclose(loaded["data"], data)


def test_option_savetwofiles_defaults_to_sidecar_storage(tmp_path: Path):
    EEG_OPTIONS["option_savetwofiles"] = 1
    data = np.arange(12, dtype=np.float32).reshape((3, 4))
    set_file = tmp_path / "option.set"

    saved = pop_saveset(_eeg(data), set_file)
    loaded = pop_loadset(set_file)

    assert saved["datfile"] == "option.fdt"
    assert (tmp_path / "option.fdt").exists()
    np.testing.assert_allclose(loaded["data"], data)


def test_memmap_resave_to_same_sidecar_preserves_data(tmp_path: Path):
    data = np.arange(12, dtype=np.float32).reshape((3, 4))
    set_file = tmp_path / "resave.set"
    pop_saveset(_eeg(data, name="resave"), set_file, savemode="twofiles")
    loaded = pop_loadset(set_file, memmap=True)

    loaded["data"][2, 3] = -12.5
    loaded["data"].flush()
    pop_saveset(loaded, savemode="resave")
    reloaded = pop_loadset(set_file)

    expected = data.copy()
    expected[2, 3] = -12.5
    np.testing.assert_allclose(reloaded["data"], expected)


def test_pop_loadset_missing_sidecar_fails_clearly(tmp_path: Path):
    set_file = tmp_path / "missing.set"
    pop_saveset(_eeg(np.ones((2, 3), dtype=np.float32)), set_file, savemode="twofiles")
    (tmp_path / "missing.fdt").unlink()

    with pytest.raises(FileNotFoundError, match="sidecar"):
        pop_loadset(set_file)


@pytest.mark.slow
def test_memmap_load_large_sidecar_smoke(tmp_path: Path):
    data = np.zeros((128, 12000), dtype=np.float32)
    set_file = tmp_path / "large.set"
    pop_saveset(_eeg(data, name="large"), set_file, savemode="twofiles")
    EEG_OPTIONS["option_memmapdata"] = 1

    start = time.perf_counter()
    loaded = pop_loadset(set_file)
    elapsed = time.perf_counter() - start

    assert isinstance(loaded["data"], MemmapData)
    assert loaded["data"][0, 0] == 0
    assert elapsed < 2.0
