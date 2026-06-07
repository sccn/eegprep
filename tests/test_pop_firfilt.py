from __future__ import annotations

import copy
import os
import unittest

import numpy as np
import pytest

from eegprep.functions.adminfunc.console import _console_python_command
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.popfunc.pop_eegfilt import pop_eegfilt
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.plugins.firfilt._filtering import design_eegfiltnew
from eegprep.plugins.firfilt._pop_common import bool_value
from eegprep.plugins.firfilt.pop_eegfiltnew import pop_eegfiltnew
from eegprep.plugins.firfilt.pop_firma import pop_firma
from eegprep.plugins.firfilt.pop_firpm import pop_firpm
from eegprep.plugins.firfilt.pop_firpmord import pop_firpmord
from eegprep.plugins.firfilt.pop_firws import pop_firws
from eegprep.plugins.firfilt.pop_firwsord import pop_firwsord
from eegprep.plugins.firfilt.pop_kaiserbeta import pop_kaiserbeta
from eegprep.plugins.firfilt.pop_xfirws import pop_xfirws

try:
    from .fixtures import SAMPLE_DATASET_PATH, create_test_eeg
except (ImportError, ValueError):
    from fixtures import SAMPLE_DATASET_PATH, create_test_eeg


def _continuous_eeg() -> dict:
    eeg = create_test_eeg(n_channels=3, n_samples=600, srate=200.0, n_trials=1)
    rng = np.random.default_rng(20)
    eeg["data"] = rng.standard_normal((3, 600))
    eeg["chanlocs"] = [
        {"labels": "Cz", "type": "EEG"},
        {"labels": "Pz", "type": "EEG"},
        {"labels": "EOG", "type": "EOG"},
    ]
    eeg["event"] = []
    eeg["urevent"] = []
    return eeg


def test_pop_eegfiltnew_sample_data_lowpass_preserves_core_metadata():
    eeg = pop_loadset(SAMPLE_DATASET_PATH)

    out, command = pop_eegfiltnew(eeg, hicutoff=40, filtorder=100, return_com=True)

    assert out["data"].shape == eeg["data"].shape
    assert out["pnts"] == eeg["pnts"]
    assert out["trials"] == eeg["trials"]
    assert out["srate"] == eeg["srate"]
    np.testing.assert_allclose(out["times"], eeg["times"])
    assert len(out.get("event", [])) == len(eeg.get("event", []))
    assert len(out.get("urevent", [])) == len(eeg.get("urevent", []))
    assert out["icaact"].size == 0
    assert out["saved"] == "no"
    assert command == "EEG = pop_eegfiltnew(EEG, 'hicutoff', 40, 'filtorder', 100);"
    assert _console_python_command(command) == "EEG = pop_eegfiltnew(EEG, hicutoff=40, filtorder=100)"


def test_pop_eegfiltnew_one_based_channel_subset_only_changes_requested_channel():
    eeg = _continuous_eeg()
    before = eeg["data"].copy()

    out, command = pop_eegfiltnew(eeg, hicutoff=30, filtorder=80, channels=[2], return_com=True)

    np.testing.assert_allclose(out["data"][0], before[0])
    assert not np.allclose(out["data"][1], before[1])
    np.testing.assert_allclose(out["data"][2], before[2])
    assert "'channels', [2]" in command


def test_pop_eegfiltnew_chantype_filters_matching_channels():
    eeg = _continuous_eeg()
    before = eeg["data"].copy()

    out = pop_eegfiltnew(eeg, hicutoff=30, filtorder=80, chantype=["EOG"])

    np.testing.assert_allclose(out["data"][:2], before[:2])
    assert not np.allclose(out["data"][2], before[2])


def test_pop_eegfiltnew_chantype_history_uses_matlab_cell_array():
    eeg = _continuous_eeg()

    _out, command = pop_eegfiltnew(eeg, hicutoff=30, filtorder=80, chantype=["EOG"], return_com=True)

    assert "'chantype', {'EOG'}" in command
    assert _console_python_command(command) == "EEG = pop_eegfiltnew(EEG, hicutoff=30, filtorder=80, chantype=['EOG'])"


def test_pop_eegfiltnew_rejects_filter_order_below_eeglab_minimum():
    with pytest.raises(ValueError, match="Filter order too low"):
        design_eegfiltnew(250, locutoff=1, hicutoff=40, filtorder=100)


def test_pop_firma_respects_continuous_boundary_events():
    eeg = create_test_eeg(n_channels=1, n_samples=80, srate=100.0, n_trials=1)
    eeg["data"] = np.zeros((1, 80), dtype=float)
    eeg["data"][0, 38] = 1.0
    eeg["event"] = [{"type": "boundary", "latency": 40.5}]

    out = pop_firma(eeg, forder=4)

    assert np.any(out["data"][0, :40])
    np.testing.assert_allclose(out["data"][0, 40:], 0, atol=1e-12)


def test_pop_firma_filters_epoched_trials_independently():
    eeg = create_test_eeg(n_channels=1, n_samples=5, srate=100.0, n_trials=2)
    eeg["data"] = np.zeros((1, 5, 2), dtype=float)
    eeg["data"][0, 4, 0] = 1.0

    out = pop_firma(eeg, forder=2)

    assert np.any(out["data"][0, :, 0])
    np.testing.assert_allclose(out["data"][0, :, 1], 0, atol=1e-12)


def test_pop_firws_and_firpm_return_replayable_history():
    eeg = _continuous_eeg()

    ws_out, ws_command = pop_firws(eeg, fcutoff=[8, 30], forder=120, ftype="bandpass", return_com=True)
    pm_out, pm_command = pop_firpm(eeg, fcutoff=[8, 30], ftrans=4, ftype="bandpass", forder=120, return_com=True)

    assert ws_out["data"].shape == eeg["data"].shape
    assert pm_out["data"].shape == eeg["data"].shape
    assert _console_python_command(ws_command) == (
        "EEG = pop_firws(EEG, fcutoff=[8, 30], forder=120, ftype='bandpass', wtype='hamming')"
    )
    assert _console_python_command(pm_command) == (
        "EEG = pop_firpm(EEG, fcutoff=[8, 30], ftrans=4, ftype='bandpass', forder=120)"
    )


def test_pop_firws_usefftfilt_matches_time_domain_filtering():
    eeg = _continuous_eeg()

    time_domain = pop_firws(eeg, fcutoff=30, forder=80, ftype="lowpass")
    fft_domain = pop_firws(eeg, fcutoff=30, forder=80, ftype="lowpass", usefftfilt=True)

    np.testing.assert_allclose(fft_domain["data"], time_domain["data"], atol=1e-10, rtol=1e-10)


def test_pop_firws_and_firpm_filter_requested_channels_only():
    eeg = _continuous_eeg()
    before = eeg["data"].copy()

    ws_out = pop_firws(eeg, fcutoff=30, forder=80, ftype="lowpass", channels=[2])
    pm_out, pm_command = pop_firpm(
        eeg, fcutoff=30, ftrans=4, ftype="lowpass", forder=80, chantype=["EOG"], return_com=True
    )

    np.testing.assert_allclose(ws_out["data"][0], before[0])
    assert not np.allclose(ws_out["data"][1], before[1])
    np.testing.assert_allclose(ws_out["data"][2], before[2])
    np.testing.assert_allclose(pm_out["data"][:2], before[:2])
    assert not np.allclose(pm_out["data"][2], before[2])
    assert "'chantype', {'EOG'}" in pm_command


def test_pop_eegfilt_legacy_history_replays_with_same_boolean_semantics():
    eeg = _continuous_eeg()

    out, command = pop_eegfilt(eeg, 1, 40, [100], [0], 0, 0, "firls", 0, return_com=True)
    replayed = eval(
        _console_python_command(command).replace("EEG = ", ""),
        {"EEG": eeg, "pop_eegfilt": pop_eegfilt},
    )

    assert out["data"].shape == eeg["data"].shape
    np.testing.assert_allclose(replayed["data"], out["data"])


def test_pop_eegfilt_legacy_firtype_changes_filter_coefficients():
    eeg = _continuous_eeg()

    firls_out = pop_eegfilt(eeg, 1, 40, 100, 0, 0, 0, "firls", 0)
    fir1_out = pop_eegfilt(eeg, 1, 40, 100, 0, 0, 0, "fir1", 0)

    assert not np.allclose(firls_out["data"], fir1_out["data"])


def test_pop_eegfilt_legacy_usefft_fails_clearly():
    with pytest.raises(ValueError, match="Legacy pop_eegfilt FFT filtering is not a standalone EEGPrep path"):
        pop_eegfilt(_continuous_eeg(), 1, 40, 100, 0, 1, 0, "firls", 0)


def test_pop_eegfiltnew_legacy_usefft_errors_like_eeglab():
    with pytest.raises(ValueError, match="FFT filtering is not supported"):
        pop_eegfiltnew(_continuous_eeg(), hicutoff=40, filtorder=100, usefft=True)


def test_bool_value_matches_eeglab_singleton_numeric_flags():
    assert bool_value([0]) is False
    assert bool_value(np.asarray([0])) is False
    assert bool_value([1]) is True
    assert bool_value("off") is False


def test_pop_firws_logs_filter_report_and_can_plot_response(caplog, monkeypatch):
    calls = []

    def fake_plotfresp(coefficients, *args, **kwargs):
        calls.append((coefficients, args, kwargs))
        return object(), [], {}

    monkeypatch.setattr("eegprep.plugins.firfilt.pop_firws.plotfresp", fake_plotfresp)
    caplog.set_level("INFO", logger="eegprep.plugins.firfilt.pop_firws")

    out, command = pop_firws(
        _continuous_eeg(), fcutoff=30, forder=100, ftype="lowpass", plotfresp=True, return_com=True
    )

    assert out["data"].shape == (3, 600)
    assert "'plotfresp', 1" in command
    assert calls
    assert any("pop_firws() - lowpass filtering data" in record.message for record in caplog.records)


def test_pop_eegfiltnew_progress_output_mentions_transition_band(caplog):
    caplog.set_level("INFO", logger="eegprep.plugins.firfilt.pop_eegfiltnew")

    pop_eegfiltnew(_continuous_eeg(), hicutoff=30, filtorder=100, plotfreqz=False)

    messages = "\n".join(record.message for record in caplog.records)
    assert "pop_eegfiltnew() - performing 101 point lowpass filtering" in messages
    assert "transition band width" in messages


def test_pop_order_helpers_return_values_and_replayable_history():
    beta, beta_command = pop_kaiserbeta(0.001, return_com=True)
    order_result, order_command = pop_firwsord("kaiser", 500, 2, 0.001, return_dev=True, return_com=True)
    pm_result, pm_command = pop_firpmord([0, 40, 48, 125], [1, 0], [0.01, 0.001], 250, return_com=True)

    assert beta == pytest.approx(5.65326, abs=1e-10)
    assert beta_command == "beta = pop_kaiserbeta(0.001);"
    assert order_result == (908, pytest.approx(0.001))
    assert order_command == "m = pop_firwsord('kaiser', 500, 2, 0.001);"
    order, wtpass, wtstop = pm_result
    assert order > 0
    assert wtpass > 0
    assert wtstop > 0
    assert pm_command.startswith("[m, wtpass, wtstop] = pop_firpmord(")


def test_pop_order_helpers_gui_results():
    class KaiserRenderer:
        def run(self, spec, initial_values=None):
            return {"dev": "0.001"}

    class FirwsRenderer:
        def run(self, spec, initial_values=None):
            return {"fs": "500", "wtype": 5, "df": "2", "dev": "0.001"}

    beta = pop_kaiserbeta(gui=True, renderer=KaiserRenderer())
    order, dev = pop_firwsord(gui=True, renderer=FirwsRenderer(), return_dev=True)

    assert beta == pytest.approx(5.65326, abs=1e-10)
    assert order == 908
    assert dev == pytest.approx(0.001)


def test_pop_xfirws_designs_and_exports_filter_file(tmp_path):
    path = tmp_path / "demo.fir"

    (b, a), command = pop_xfirws(
        srate=250,
        fcutoff=[1, 40],
        ftype="bandpass",
        wtype="hamming",
        forder=100,
        filename=path.name,
        pathname=tmp_path,
        return_com=True,
    )

    assert a == 1
    assert b.shape == (101,)
    assert path.is_file()
    text = path.read_text()
    assert "[fir design]" in text
    assert "type    bandpass" in text
    assert "[fir]" in text
    assert "'filename', 'demo.fir'" in command


@unittest.skipIf(os.getenv("EEGPREP_SKIP_MATLAB") == "1", "MATLAB not available")
class TestPopFirfiltParity(unittest.TestCase):
    def setUp(self):
        try:
            self.eeglab = get_eeglab("MAT")
        except Exception as exc:
            self.skipTest(f"MATLAB not available: {exc}")
        self.eeg = pop_loadset(SAMPLE_DATASET_PATH)

    def test_parity_pop_eegfiltnew_lowpass_sample_data(self):
        py_eeg = pop_eegfiltnew(copy.deepcopy(self.eeg), hicutoff=40, filtorder=100)
        ml_eeg = self.eeglab.pop_eegfiltnew(copy.deepcopy(self.eeg), "hicutoff", 40, "filtorder", 100, "plotfreqz", 0)

        self.assertEqual(py_eeg["data"].shape, ml_eeg["data"].shape)
        np.testing.assert_allclose(py_eeg["data"], ml_eeg["data"], atol=1e-7, rtol=1e-7)

    def test_parity_pop_eegfiltnew_minphase_sample_data(self):
        py_eeg = pop_eegfiltnew(copy.deepcopy(self.eeg), hicutoff=40, filtorder=100, minphase=True)
        ml_eeg = self.eeglab.pop_eegfiltnew(
            copy.deepcopy(self.eeg), "hicutoff", 40, "filtorder", 100, "minphase", 1, "plotfreqz", 0
        )

        self.assertEqual(py_eeg["data"].shape, ml_eeg["data"].shape)
        np.testing.assert_allclose(py_eeg["data"], ml_eeg["data"], atol=3e-3, rtol=3e-3)

    def test_parity_pop_eegfiltnew_boundary_split(self):
        eeg = create_test_eeg(n_channels=1, n_samples=200, srate=100.0, n_trials=1)
        eeg["data"] = np.zeros((1, 200), dtype=float)
        eeg["data"][0, 92] = 1.0
        eeg["event"] = [{"type": "boundary", "latency": 100.5}]

        py_eeg = pop_eegfiltnew(copy.deepcopy(eeg), hicutoff=20, filtorder=20)
        ml_eeg = self.eeglab.pop_eegfiltnew(copy.deepcopy(eeg), "hicutoff", 20, "filtorder", 20, "plotfreqz", 0)

        self.assertEqual(py_eeg["data"].shape, ml_eeg["data"].shape)
        np.testing.assert_allclose(py_eeg["data"], ml_eeg["data"], atol=3e-8, rtol=3e-8)

    def test_parity_pop_firws_bandpass_sample_data(self):
        py_eeg = pop_firws(copy.deepcopy(self.eeg), fcutoff=[8, 30], forder=120, ftype="bandpass", wtype="hamming")
        ml_eeg = self.eeglab.pop_firws(
            copy.deepcopy(self.eeg),
            "fcutoff",
            [8, 30],
            "forder",
            120,
            "ftype",
            "bandpass",
            "wtype",
            "hamming",
            "plotfresp",
            0,
        )

        self.assertEqual(py_eeg["data"].shape, ml_eeg["data"].shape)
        np.testing.assert_allclose(py_eeg["data"], ml_eeg["data"], atol=2e-5, rtol=2e-5)

    def test_parity_pop_eegfilt_legacy_firls_sample_data(self):
        py_eeg = pop_eegfilt(copy.deepcopy(self.eeg), 1, 40, 100, 0, 0, 0, "firls", 0)
        ml_eeg = self.eeglab.pop_eegfilt(copy.deepcopy(self.eeg), 1, 40, 100, 0, 0, 0, "firls", 0)

        self.assertEqual(py_eeg["data"].shape, ml_eeg["data"].shape)
        np.testing.assert_allclose(py_eeg["data"], ml_eeg["data"], atol=2e-4, rtol=2e-4)

    def test_parity_pop_firma_boundary_split(self):
        eeg = create_test_eeg(n_channels=1, n_samples=80, srate=100.0, n_trials=1)
        eeg["data"] = np.zeros((1, 80), dtype=float)
        eeg["data"][0, 38] = 1.0
        eeg["event"] = [{"type": "boundary", "latency": 40.5}]

        py_eeg = pop_firma(copy.deepcopy(eeg), forder=4)
        ml_eeg = self.eeglab.pop_firma(copy.deepcopy(eeg), "forder", 4)

        self.assertEqual(py_eeg["data"].shape, ml_eeg["data"].shape)
        np.testing.assert_allclose(py_eeg["data"], ml_eeg["data"], atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
