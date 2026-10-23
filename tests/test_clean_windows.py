import logging
import unittest

import numpy as np
import pytest

from eegprep.plugins.clean_rawdata.clean_windows import clean_windows


class TestCleanWindows(unittest.TestCase):
    """Test the clean_windows function."""

    def setUp(self):
        """Set up test fixtures with synthetic EEG data."""
        np.random.seed(42)  # For reproducible tests

        self.n_channels = 8
        self.n_samples = 2500  # 10 seconds at 250 Hz
        self.srate = 250.0

        self.clean_data = np.random.randn(self.n_channels, self.n_samples) * 0.5

        # Add some realistic structure
        for ch in range(self.n_channels):
            # Add some low-frequency trend
            t = np.linspace(0, self.n_samples / self.srate, self.n_samples)
            self.clean_data[ch] += 0.2 * np.sin(2 * np.pi * 0.5 * t)

        # Add some artifacts to specific windows
        self.data_with_artifacts = self.clean_data.copy()

        # Add high-amplitude artifacts to specific channels/times
        self.data_with_artifacts[2, 500:750] += np.random.randn(250) * 5.0  # Large artifact
        self.data_with_artifacts[5, 1500:1750] += np.random.randn(250) * 4.0  # Another artifact

        self.EEG_artifacts = {
            'data': self.data_with_artifacts,
            'srate': self.srate,
            'pnts': self.n_samples,
            'nbchan': self.n_channels,
            'xmin': 0.0,
            'xmax': (self.n_samples - 1) / self.srate,
        }

    def test_pop_select_success_preserves_events_and_inserts_boundaries(self):
        """pop_select success path keeps events and inserts boundaries at cuts.

        Mirrors EEGLAB's clean_windows.m, which only wipes event metadata in
        the manual fallback branch. On the success path, pop_select / eeg_eegrej
        shift event latencies and insert a 'boundary' event at each cut with
        duration equal to the excised sample count.
        """
        EEG_in = self.EEG_artifacts.copy()
        # Pre-populate events at known sample latencies, covering survivors,
        # an event inside an artifact region, and a marker after the second
        # artifact so we can verify post-cut latency shifting.
        EEG_in['event'] = [
            {'type': 'S1', 'latency': 100.0, 'duration': 0.0},
            {'type': 'S2', 'latency': 600.0, 'duration': 0.0},  # inside first artifact (500:750)
            {'type': 'S3', 'latency': 1000.0, 'duration': 0.0},
            {'type': 'S4', 'latency': 2000.0, 'duration': 0.0},
        ]
        EEG_in['urevent'] = [dict(ev) for ev in EEG_in['event']]
        for i, ev in enumerate(EEG_in['event'], start=1):
            ev['urevent'] = i
        EEG_in['nbchan'] = self.n_channels
        EEG_in['trials'] = 1

        EEG_out, sample_mask = clean_windows(EEG_in)

        # The success path should not wipe event metadata.
        self.assertIn('event', EEG_out)
        events = list(EEG_out['event'])
        self.assertGreater(len(events), 0)

        # At least one boundary event should have been inserted (artifacts
        # produced cuts, so sample_mask has False stretches).
        boundary_events = [ev for ev in events if str(ev.get('type', '')).lower() == 'boundary']
        self.assertGreaterEqual(len(boundary_events), 1)

        # Surviving events must lie within the new sample grid.
        new_pnts = EEG_out['pnts']
        for ev in events:
            if 'latency' in ev:
                self.assertGreaterEqual(float(ev['latency']), 0.0)
                self.assertLessEqual(float(ev['latency']), float(new_pnts) + 1)

        # Boundary durations should be positive and not exceed the total
        # number of removed samples.
        total_removed = int(np.sum(~sample_mask))
        for ev in boundary_events:
            self.assertGreater(float(ev.get('duration', 0.0)), 0.0)
            self.assertLessEqual(float(ev.get('duration', 0.0)), float(total_removed))

        self.assertEqual(EEG_out['data'].dtype, EEG_in['data'].dtype)


def _eeg(data: np.ndarray, srate: float = 128.0) -> dict:
    pnts = int(data.shape[1])
    return {
        "data": np.array(data, dtype=np.float64, copy=True),
        "srate": srate,
        "nbchan": int(data.shape[0]),
        "pnts": pnts,
        "trials": 1,
        "xmin": 0.0,
        "xmax": (pnts - 1) / srate,
        "event": [],
    }


def test_unidentified_window_fit_keeps_median_mad_scale():
    """A zero fit scale uses the median and MAD, so a modest outlier stays.

    Twenty-three non-overlapping windows is the smallest default grid
    fit_eeg_distribution can index. Treating that zero scale as infinite
    would reject the louder window.
    """
    from eegprep.plugins.clean_rawdata.private.stats import fit_eeg_distribution

    mu, sig, *_rest = fit_eeg_distribution(np.array([]))
    assert not np.isfinite(mu)
    assert not np.isfinite(sig)

    samples = 2301
    data = np.ones((4, samples), dtype=np.float64)
    data[:, 2201:2301] = 1.3
    _cleaned, mask = clean_windows(
        _eeg(data, 100.0),
        0.075,
        (-3.5, 2.0),
        1.0,
        window_overlap=0.0,
    )
    assert mask.shape == (samples,)
    assert mask.all()


def test_too_few_windows_raises():
    """The width loop errors when its smallest rounded width is 1 or less.

    Ten windows at these defaults round to 1. Eleven windows round to 2, and
    MATLAB fits them.
    """
    from pathlib import Path

    from eegprep.functions.popfunc.pop_loadset import pop_loadset

    data = np.ones((4, 180), dtype=np.float64)
    with pytest.raises(ValueError, match="Too few windows"):
        clean_windows(_eeg(data), 0.075, (-3.5, 2.0), 1.0)

    with pytest.raises(ValueError, match="Too few windows"):
        clean_windows(_eeg(np.ones((4, 521), dtype=np.float64)), 0.075, (-3.5, 2.0), 1.0)

    source = pop_loadset(Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data.set")
    short = np.asarray(source["data"][:11, 8:264], dtype=np.float64)
    with pytest.raises(ValueError, match="Too few windows"):
        clean_windows(_eeg(short), 0.075, (-3.5, 5.5), 1.0)


def test_eleven_windows_still_fit():
    cleaned, mask = clean_windows(_eeg(np.ones((4, 565), dtype=np.float64)), 0.075, (-3.5, 2.0), 1.0)
    assert mask.shape == (565,)
    assert cleaned["pnts"] == 565


def test_cleaned_xmax_follows_the_checkset_round():
    """eeg_checkset.m:849 keeps a proportional xmax when it already counts pnts."""
    from eegprep.functions.miscfunc.misc import round_mat

    eeg = _eeg(np.random.default_rng(0).normal(size=(4, 2500)), 128.0)
    eeg["data"][:, 400:900] += 80.0
    old_pnts = int(eeg["pnts"])
    old_span = float(eeg["xmax"]) - float(eeg["xmin"])
    cleaned, mask = clean_windows(eeg)
    new_pnts = int(cleaned["pnts"])
    srate = float(cleaned["srate"])
    proportional = float(eeg["xmin"]) + old_span * (new_pnts / old_pnts)
    assert int(mask.sum()) < old_pnts
    assert int(round_mat(srate * proportional + 1)) == new_pnts
    assert cleaned["xmax"] == pytest.approx(proportional)
    assert cleaned["xmax"] != pytest.approx(float(eeg["xmin"]) + (new_pnts - 1) / srate)
    assert np.asarray(cleaned["times"])[-1] == pytest.approx(proportional * 1000.0)

    heavy = _eeg(np.random.default_rng(0).normal(size=(4, 2500)), 128.0)
    heavy["data"][:, 200:1900] += 80.0
    snapped, _mask = clean_windows(heavy)
    snapped_pnts = int(snapped["pnts"])
    snapped_prop = old_span * (snapped_pnts / old_pnts)
    assert int(round_mat(srate * snapped_prop + 1)) != snapped_pnts
    assert snapped["xmax"] == pytest.approx((snapped_pnts - 1) / srate)


def test_prior_mask_of_a_different_length_is_not_replaced(caplog):
    eeg = _eeg(np.ones((4, 2945), dtype=np.float64), 128.0)
    eeg["data"][:, 128:256] += 40.0
    original = np.array([True, False, True])
    eeg["etc"] = {"clean_sample_mask": original.copy()}
    # 2945 samples at 128 Hz and overlap 0 is 23 windows, so the fit runs.
    with caplog.at_level(logging.WARNING):
        cleaned, _mask = clean_windows(eeg, window_len=1.0, window_overlap=0.0)
    np.testing.assert_array_equal(cleaned["etc"]["clean_sample_mask"], original)
    assert "clean_sample is present" in caplog.text


def test_prior_mask_is_written_into_the_kept_indices(caplog):
    samples = 2945
    eeg = _eeg(np.ones((4, samples), dtype=np.float64), 128.0)
    eeg["data"][:, 128:256] += 40.0
    previous = np.concatenate([np.array([False]), np.ones(samples, dtype=bool)])
    eeg["etc"] = {"clean_sample_mask": previous.copy()}
    with caplog.at_level(logging.WARNING):
        cleaned, sample_mask = clean_windows(eeg, window_len=1.0, window_overlap=0.0)
    updated = np.asarray(cleaned["etc"]["clean_sample_mask"])
    assert updated.shape == (samples + 1,)
    assert not updated[0]
    np.testing.assert_array_equal(updated[1:], sample_mask)
    assert "overwritten" not in caplog.text


if __name__ == '__main__':
    unittest.main()
