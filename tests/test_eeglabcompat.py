"""
Test suite for functions/adminfunc/eeglabcompat.py - EEGLAB compatibility layer.

This module tests the EEGLAB compatibility functions that provide
Python interfaces to MATLAB/Octave EEGLAB functions.
"""

import struct
import unittest
from pathlib import Path

import numpy as np
import scipy.io

from eegprep.functions.adminfunc.eeglabcompat import (
    MatlabWrapper,
    _matlab_args_cell,
    _prepare_matlab_arg,
    _restore_double_eeg_data,
    _stash_double_eeg_data,
    load_seen_input,
    matlab_held_data,
    matlab_restore_input,
    save_double_input,
)
from eegprep.functions.miscfunc.misc import matlab_mtimes
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.utils.testing import DebuggableTestCase
import eegprep.functions.adminfunc.eeglabcompat as eeglabcompat
from tests.eeg_options_snapshot import eeg_options_snapshot, reload_eeg_options
from tests.fuzz.harness import resume_eeg_options


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
    monkeypatch.setattr(eeglabcompat, "get_eeglab", lambda auto_file_roundtrip=False, **_kwargs: DummyEeglab())
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


def test_double_input_round_trip_keeps_values_float32_cannot_store(tmp_path):
    signal = np.array(
        [
            [np.nextafter(np.float64(1.0), np.float64(2.0)), np.float64(-26.488021850585973)],
            [np.float64(-26.488021850585902), np.nan],
            [np.inf, -np.inf],
        ],
        dtype=np.float64,
    )
    assert not np.array_equal(signal, signal.astype(np.float32).astype(np.float64), equal_nan=True)
    path = tmp_path / "input.mat"
    seen = tmp_path / "seen.mat"
    save_double_input(signal, path)
    saved = scipy.io.loadmat(path, squeeze_me=False)
    scipy.io.savemat(seen, {"eegprep_seen_data": saved["eegprep_input_data"]}, do_compression=False)

    loaded = load_seen_input(seen)

    assert loaded.dtype == np.dtype(np.float64)
    assert loaded.shape == signal.shape
    assert np.array_equal(loaded, signal, equal_nan=True)
    restore = matlab_restore_input(path)
    assert "EEG.data = eegprep_input.eegprep_input_data" in restore
    assert "eegprep:InputTransport" in restore


def _saved_input(path):
    return scipy.io.loadmat(path, squeeze_me=False, mat_dtype=True)["eegprep_input_data"]


def test_epoched_input_is_saved_as_matlab_holds_it(tmp_path):
    from tests.fuzz.harness import transport_mismatch

    flat = np.arange(24, dtype=np.float64).reshape(4, 6)
    epoch = {"data": flat, "nbchan": 4, "pnts": 3, "trials": 2, "datfile": ""}
    held = matlab_held_data(flat, epoch)
    fortran = np.reshape(flat, (4, 3, 2), order="F")
    assert not np.array_equal(fortran, np.reshape(flat, (4, 3, 2), order="C"))
    np.testing.assert_array_equal(held, fortran)
    flat_path = tmp_path / "flat.mat"
    save_double_input(flat, flat_path, epoch)
    saved = _saved_input(flat_path)
    assert saved.shape == (4, 3, 2)
    assert saved.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(saved, fortran)
    changed = np.array(fortran, copy=True)
    changed[0, 0, 0] += 1
    assert transport_mismatch(held, changed) == "values differ"
    shape = transport_mismatch(held, np.zeros((4, 3), dtype=np.float64))
    assert shape is not None and shape.startswith("shape ")

    page = np.arange(12, dtype=np.float64).reshape(4, 3, 1)
    one_trial = {"data": page, "nbchan": 4, "pnts": 3, "trials": 1, "datfile": "epoch.fdt"}
    one_path = tmp_path / "one.mat"
    save_double_input(page, one_path, one_trial)
    dropped = _saved_input(one_path)
    assert dropped.shape == (4, 3)
    np.testing.assert_array_equal(dropped, np.reshape(page, (4, 3), order="F"))
    kept = {"data": page, "nbchan": 4, "pnts": 3, "trials": 1, "datfile": ""}
    save_double_input(page, tmp_path / "kept.mat", kept)
    assert _saved_input(tmp_path / "kept.mat").shape == (4, 3, 1)

    base = np.arange(48, dtype=np.float64).reshape(4, 3, 4)
    view = base[:, :, ::2]
    assert not view.flags.c_contiguous and not view.flags.f_contiguous
    cube = {"data": view, "nbchan": 4, "pnts": 3, "trials": 2, "datfile": "epoch.fdt"}
    cube_path = tmp_path / "cube.mat"
    save_double_input(view, cube_path, cube)
    stored = _saved_input(cube_path)
    assert stored.shape == (4, 3, 2)
    np.testing.assert_array_equal(stored, np.array(view, copy=True))

    swapped = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    mismatched = {"data": swapped, "nbchan": 4, "pnts": 3, "trials": 2, "datfile": "epoch.fdt"}
    save_double_input(swapped, tmp_path / "swapped.mat", mismatched)
    assert _saved_input(tmp_path / "swapped.mat").shape == (2, 3, 4)

    split = np.arange(48, dtype=np.float64).reshape(4, 6, 2)
    other = {"data": split, "nbchan": 4, "pnts": 3, "trials": 4, "datfile": "epoch.fdt"}
    save_double_input(split, tmp_path / "split.mat", other)
    np.testing.assert_array_equal(_saved_input(tmp_path / "split.mat"), split)

    odd = np.arange(20, dtype=np.float64).reshape(4, 5)
    indivisible = {"data": odd, "nbchan": 4, "pnts": 3, "trials": 2, "datfile": ""}
    save_double_input(odd, tmp_path / "odd.mat", indivisible)
    assert _saved_input(tmp_path / "odd.mat").shape == (4, 5)


_MI_MATRIX = 14
_MX_DOUBLE_CLASS = 6
_MX_UINT8_CLASS = 9


def _integer_double_packed_as_uint8(path, name: str, values: np.ndarray) -> None:
    """Write a MATLAB double whose ``-v7`` payload is uint8 storage.

    ``save -v7`` keeps the array class as double and stores integer values in
    the smallest integer type that holds them. SciPy writes the storage class,
    so the class byte is then set to double.
    """
    scipy.io.savemat(path, {name: np.asarray(values, dtype=np.uint8)}, do_compression=False)
    raw = bytearray(Path(path).read_bytes())
    assert raw[126:128] == b"IM"
    kind, _size = struct.unpack_from("<II", raw, 128)
    assert kind == _MI_MATRIX
    assert raw[144] == _MX_UINT8_CLASS
    raw[144] = _MX_DOUBLE_CLASS
    Path(path).write_bytes(raw)


def test_packed_integer_double_round_trips_as_float64(tmp_path):
    from tests.fuzz.harness import transport_mismatch

    values = (np.arange(32, dtype=np.float64) % 7)[:, None] * np.arange(1, 5, dtype=np.float64)
    seen = tmp_path / "seen.mat"
    _integer_double_packed_as_uint8(seen, "eegprep_seen_data", values)
    stored = scipy.io.loadmat(seen, squeeze_me=False)
    assert stored["eegprep_seen_data"].dtype == np.dtype(np.uint8)

    loaded = load_seen_input(seen)

    assert loaded.dtype == np.dtype(np.float64)
    assert loaded.shape == values.shape
    np.testing.assert_array_equal(loaded, values)
    assert transport_mismatch(values, loaded) is None

    genuine = tmp_path / "uint8.mat"
    scipy.io.savemat(genuine, {"eegprep_seen_data": values.astype(np.uint8)}, do_compression=False)
    raw = load_seen_input(genuine)
    assert raw.dtype == np.dtype(np.uint8)
    assert transport_mismatch(values, raw) == "dtype uint8"

    stashed = tmp_path / "stashed.mat"
    _integer_double_packed_as_uint8(stashed, "eegprep_roundtrip_data", values)
    restored = {"data": np.zeros(values.shape, dtype=np.float32)}
    _restore_double_eeg_data(restored, str(stashed))
    assert restored["data"].dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(restored["data"], values)


def test_matlab_bridge_restores_double_input_before_the_target(monkeypatch):
    signal = np.array(
        [[np.nextafter(np.float64(1.0), np.float64(2.0)), np.nan], [np.inf, -np.inf]],
        dtype=np.float64,
    )
    eeg = {"data": signal, "nbchan": 2, "pnts": 2, "trials": 1, "srate": 128.0, "event": []}

    class _Engine:
        def __init__(self):
            self.evals = []

        def eval(self, code, nargout=0):
            del nargout
            self.evals.append(code)
            if "eegprep_identity" in code:
                self.last_caught_error = {
                    "identifier": "MATLAB:badsubscript",
                    "message": "Index in position 1 is invalid.",
                    "stack": "  asr_calibrate.m>fit_eeg_distribution (line 395)\n",
                    "printed": "",
                }
            else:
                self.last_caught_error = None
            if "eegprep_seen_data = EEG.data" in code:
                seen = code.split("save('", 1)[1].split("'", 1)[0]
                saved = scipy.io.loadmat(seen.replace(".seen.mat", ".input.mat"), squeeze_me=False)
                scipy.io.savemat(seen, {"eegprep_seen_data": saved["eegprep_input_data"]}, do_compression=False)
            if "eegprep_roundtrip_data = double(" in code:
                path = code.split("save('", 1)[1].split("'", 1)[0]
                saved = scipy.io.loadmat(path.replace(".double.mat", ".input.mat"), squeeze_me=False)
                scipy.io.savemat(
                    path,
                    {
                        "eegprep_roundtrip_data": saved["eegprep_input_data"],
                        "eegprep_roundtrip_icaact": np.array([]),
                    },
                    do_compression=False,
                )

    def _save(_eeg, _path):
        return None

    def _load(_path):
        return {"data": np.zeros(signal.shape, dtype=np.float32), "trials": 1, "srate": 128.0, "event": []}

    monkeypatch.setattr(eeglabcompat, "pop_saveset", _save)
    monkeypatch.setattr(eeglabcompat, "pop_loadset", _load)
    engine = _Engine()

    wrapper = MatlabWrapper(engine)
    returned = wrapper.eegprep_identity(eeg)

    load_at = next(index for index, code in enumerate(engine.evals) if "pop_loadset" in code)
    restore_at = next(index for index, code in enumerate(engine.evals) if "eegprep_input_data" in code)
    capture_at = next(index for index, code in enumerate(engine.evals) if "eegprep_seen_data = EEG.data" in code)
    target_at = next(index for index, code in enumerate(engine.evals) if "eegprep_identity" in code)
    assert load_at < restore_at < capture_at < target_at
    seen = wrapper.last_input_data
    assert isinstance(seen, np.ndarray)
    assert seen.dtype == np.dtype(np.float64)
    assert np.array_equal(seen, signal, equal_nan=True)
    assert np.array_equal(returned["data"], signal, equal_nan=True)
    assert wrapper.last_caught_error["identifier"] == "MATLAB:badsubscript"
    assert engine.last_caught_error is None


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
    """Round-trip ICA through MatlabWrapper and match MATLAB's activation class.

    The product class follows ``option_single``. Pin that option on for this
    call so an earlier test cannot leave the shared session in double.
    """
    # eeg_checkset.m:1030 forms W*S in double; MATLAB rounds it to single for the product with single data. gamma uses eps = 2u, which covers both single evaluations.
    data_class = np.dtype(np.float32)
    sample = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data_with_ica_tmp.set"
    eeg = pop_loadset(sample)
    eeg["data"] = np.asarray(eeg["data"], dtype=data_class)
    eeg["icaact"] = np.full((np.asarray(eeg["icaweights"]).shape[0], int(eeg["pnts"])), 9.0, dtype=data_class)
    snapshot = eeg_options_snapshot(eeglab_matlab_engine)
    try:
        eeglab_matlab_engine.eval("pop_editoptions('option_single', 1);", nargout=0)
        reload_eeg_options(eeglab_matlab_engine)
        assert int(eeglab_matlab_engine.eval("option_single", nargout=1)) == 1
        restored = MatlabWrapper(eeglab_matlab_engine).eeg_checkset(eeg)
    finally:
        resume_eeg_options(eeglab_matlab_engine, snapshot)

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

    def test_empty_args_stay_a_one_row_cell(self):
        cell = _matlab_args_cell([np.array([]), np.array([]), np.array([])])
        stacked = np.array([np.array([]), np.array([]), np.array([])], dtype=object)

        self.assertEqual(stacked.shape, (3, 0))
        self.assertEqual(cell.shape, (1, 3))
        self.assertEqual(cell.dtype, np.dtype("O"))
        for item in cell.ravel():
            self.assertEqual(np.asarray(item).size, 0)


if __name__ == '__main__':
    unittest.main()
