"""
Test suite for functions/adminfunc/eeglabcompat.py - EEGLAB compatibility layer.

This module tests the EEGLAB compatibility functions that provide
Python interfaces to MATLAB/Octave EEGLAB functions.
"""

import unittest
from pathlib import Path

import numpy as np
import scipy.io

from eegprep.functions.adminfunc.eeglabcompat import _prepare_matlab_arg, _restore_double_eeg_data
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


def test_restore_double_eeg_data_uses_matlab_layout(tmp_path):
    original = np.arange(24, dtype=np.float64).reshape((4, 3, 2), order="F")
    path = tmp_path / "roundtrip.mat"
    scipy.io.savemat(path, {"eegprep_roundtrip_data": np.asfortranarray(original)})
    loaded = {"data": np.zeros((4, 6), dtype=np.float32)}

    _restore_double_eeg_data(loaded, str(path))

    restored = np.reshape(original, (4, 6), order="F")
    np.testing.assert_array_equal(loaded["data"], restored)
    assert loaded["data"].dtype == np.float64


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
