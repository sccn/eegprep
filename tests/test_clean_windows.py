import unittest
import numpy as np

from eegprep.plugins.clean_rawdata.clean_windows import clean_windows


class TestCleanWindows(unittest.TestCase):
    """Test the clean_windows function."""

    def setUp(self):
        """Set up test fixtures with synthetic EEG data."""
        np.random.seed(42)  # For reproducible tests

        # Create synthetic EEG data structure
        self.n_channels = 8
        self.n_samples = 2500  # 10 seconds at 250 Hz
        self.srate = 250.0

        # Create clean EEG data
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


if __name__ == '__main__':
    unittest.main()
