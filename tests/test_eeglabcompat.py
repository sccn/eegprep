"""
Test suite for functions/adminfunc/eeglabcompat.py - EEGLAB compatibility layer.

This module tests the EEGLAB compatibility functions that provide
Python interfaces to MATLAB/Octave EEGLAB functions.
"""

import unittest
from pathlib import Path

import numpy as np
import scipy.io

from eegprep.functions.adminfunc.eeglabcompat import (
    MatlabWrapper,
    _prepare_matlab_arg,
    _restore_double_eeg_data,
    _stash_double_eeg_data,
)
from eegprep.functions.miscfunc.misc import matlab_mtimes
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.utils.testing import DebuggableTestCase
import eegprep.functions.adminfunc.eeglabcompat as eeglabcompat


def test_eeglab_clean_artifacts_roundtrip_uses_private_tempdir(monkeypatch, tmp_path):
    paths: dict[str, list[Path]] = {"save": [], "matlab_load": [], "matlab_save": [], "load": []}

    class DummyEeglab:
        def pop_loadset(self, filename):
            paths["matlab_load"].append(Path(filename))
            return {"loaded": filename}

        def clean_artifacts(self, EEG, *_args):
            return {"cleaned": EEG}

        def pop_saveset(self, EEG, filename):
            paths["matlab_save"].append(Path(filename))
            Path(filename).write_text("cleaned", encoding="utf-8")
            return EEG

    def fake_pop_saveset(EEG, filename):
        paths["save"].append(Path(filename))
        Path(filename).write_text("input", encoding="utf-8")
        return EEG

    def fake_pop_loadset(filename):
        paths["load"].append(Path(filename))
        return {"loaded_from": str(filename)}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(eeglabcompat, "get_eeglab", lambda auto_file_roundtrip=False: DummyEeglab())
    monkeypatch.setattr(eeglabcompat, "pop_saveset", fake_pop_saveset)
    monkeypatch.setattr(eeglabcompat, "pop_loadset", fake_pop_loadset)

    result = eeglabcompat.clean_artifacts({"data": np.zeros((1, 4))}, BurstCriterion="off")

    assert result["loaded_from"].endswith("output.set")
    assert not (tmp_path / "tmp.set").exists()
    assert not (tmp_path / "tmp2.set").exists()
    assert all(path.parent != tmp_path for values in paths.values() for path in values)
    assert {path.name for values in paths.values() for path in values} == {"input.set", "output.set"}


def test_picard_plugin_paths_follow_the_installed_folder_name(tmp_path):
    plugin = tmp_path / "plugins" / "PICARD"
    sources = plugin / "matlab_octave"
    sources.mkdir(parents=True)

    paths = eeglabcompat._picard_plugin_paths(str(tmp_path))

    assert paths == [str(plugin), str(sources)]


class _RecordingEngine:
    """Run the stash ``eval`` the way MATLAB would save the in-memory fields."""

    def __init__(self, data, icaact):
        self.data = np.asarray(data)
        self.icaact = np.array(icaact, copy=True)

    def eval(self, code, nargout=0):
        assignment = code.split("eegprep_roundtrip_icaact =", 1)[1].split(";", 1)[0]
        assert "double(" not in assignment
        path = code.split("save('", 1)[1].split("'", 1)[0]
        scipy.io.savemat(
            path,
            {
                "eegprep_roundtrip_data": np.asarray(self.data, dtype=np.float64),
                "eegprep_roundtrip_icaact": self.icaact,
            },
        )


def test_matlab_bridge_round_trip_preserves_icaact(tmp_path):
    data = np.arange(12, dtype=np.float64).reshape(2, 6)
    single = np.array([[1.25, -2.5], [0.5, 4.0], [0.0, 1.0]], dtype=np.float32)
    epoched = np.arange(24, dtype=np.float64).reshape(2, 4, 3)
    cases = (single, epoched, np.empty((0, 0), dtype=np.float64))
    for index, icaact in enumerate(cases):
        path = tmp_path / f"roundtrip-{index}.mat"
        _stash_double_eeg_data(_RecordingEngine(data, icaact), "OUT", str(path))
        loaded = {
            "data": np.zeros(data.shape, dtype=np.float32),
            "icaact": np.ones((3, 6), dtype=np.float64),
        }

        _restore_double_eeg_data(loaded, str(path))

        assert loaded["data"].dtype == np.float64
        assert loaded["icaact"].dtype == icaact.dtype
        assert loaded["icaact"].shape == icaact.shape
        np.testing.assert_array_equal(loaded["icaact"], icaact)


def test_restore_double_eeg_data_uses_matlab_layout(tmp_path):
    original = np.arange(24, dtype=np.float64).reshape((4, 3, 2), order="F")
    path = tmp_path / "roundtrip.mat"
    scipy.io.savemat(path, {"eegprep_roundtrip_data": np.asfortranarray(original)})
    loaded = {"data": np.zeros((4, 6), dtype=np.float32)}

    _restore_double_eeg_data(loaded, str(path))

    restored = np.reshape(original, (4, 6), order="F")
    np.testing.assert_array_equal(loaded["data"], restored)
    assert loaded["data"].dtype == np.float64


def test_matlab_bridge_restores_icaact_from_the_data(eeglab_matlab_engine):
    """Round-trip ICA through MatlabWrapper and match MATLAB's activation class."""
    # eeg_checkset.m casts the decomposition to double, then multiplies by data.
    data_class = np.dtype(np.float32)
    sample = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data_with_ica_tmp.set"
    eeg = pop_loadset(sample)
    eeg["data"] = np.asarray(eeg["data"], dtype=data_class)
    eeg["icaact"] = np.full((np.asarray(eeg["icaweights"]).shape[0], int(eeg["pnts"])), 9.0, dtype=data_class)

    restored = MatlabWrapper(eeglab_matlab_engine).eeg_checkset(eeg)

    activations = np.asarray(restored["icaact"])
    assert activations.dtype == data_class
    assert activations.size == np.asarray(eeg["icaweights"]).shape[0] * int(eeg["pnts"]) * int(eeg["trials"])
    channels = np.asarray(restored["icachansind"], dtype=int)
    selected = np.asarray(restored["data"], dtype=data_class)[channels]
    flat = np.reshape(selected, (selected.shape[0], -1), order="F")
    unmixing = matlab_mtimes(np.asarray(restored["icaweights"]), np.asarray(restored["icasphere"]))
    assert unmixing.dtype == np.dtype(np.float64)
    expected = matlab_mtimes(unmixing, flat)
    terms = int(flat.shape[0])
    unit = float(np.finfo(data_class).eps)
    gamma = terms * unit / (1.0 - terms * unit)
    scale = np.abs(np.asarray(unmixing, dtype=np.float64)) @ np.abs(np.asarray(flat, dtype=np.float64))
    bound = gamma * scale
    actual = np.reshape(np.asarray(activations, dtype=np.float64), expected.shape, order="F")
    residual = np.abs(actual - np.asarray(expected, dtype=np.float64))
    assert np.all(residual <= bound)


class TestMatlabWrapper(DebuggableTestCase):
    """Test cases for MatlabWrapper class."""

    def test_prepare_matlab_arg_keeps_empty_lists_numeric(self):
        """Empty Python lists should marshal as EEGLAB [] rather than {}."""
        empty = _prepare_matlab_arg([])
        strings = _prepare_matlab_arg(["square", "rt"])
        numbers = _prepare_matlab_arg([1, 2])
        highpass = _prepare_matlab_arg((0.25, 0.75))
        tolerances = _prepare_matlab_arg((-np.inf, 7.0))

        self.assertEqual(empty.shape, (0,))
        self.assertEqual(empty.dtype, np.dtype("float64"))
        self.assertEqual(strings.shape, (1, 2))
        self.assertEqual(strings.dtype, np.dtype("O"))
        np.testing.assert_array_equal(numbers, np.asarray([1.0, 2.0]))
        np.testing.assert_array_equal(highpass, np.asarray([0.25, 0.75]))
        np.testing.assert_array_equal(tolerances, np.asarray([-np.inf, 7.0]))
        self.assertEqual(highpass.dtype, np.dtype("float64"))


if __name__ == '__main__':
    unittest.main()
