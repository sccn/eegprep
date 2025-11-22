"""Tests for test fixtures module."""

import unittest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from . import fixtures
from .fixtures import (
    mpl_use_agg, rng_seed, create_test_eeg, create_test_eeg_with_ica,
    create_test_events, cleanup_matplotlib, TestFixturesContextManager, small_eeg
)


class TestFixturesFunctions(unittest.TestCase):
    """Test individual fixture functions."""
    
    def test_mpl_use_agg(self):
        """Test matplotlib backend setting."""
        original_backend = matplotlib.get_backend()
        
        try:
            mpl_use_agg()
            self.assertEqual(matplotlib.get_backend(), 'Agg')
        finally:
            # Restore original backend
            matplotlib.use(original_backend)
    
    def test_rng_seed(self):
        """Test random seed setting."""
        rng_seed(123)
        val1 = np.random.random()
        
        rng_seed(123)
        val2 = np.random.random()
        
        self.assertEqual(val1, val2, "Random seed should produce reproducible results")
    
    def test_create_test_eeg_basic(self):
        """Test basic EEG creation."""
        eeg = create_test_eeg()
        
        # Check basic structure
        self.assertIn('data', eeg)
        self.assertIn('srate', eeg)
        self.assertIn('pnts', eeg)
        self.assertIn('nbchan', eeg)
        self.assertIn('trials', eeg)
        
        # Check default values
        self.assertEqual(eeg['nbchan'], 32)
        self.assertEqual(eeg['pnts'], 1000)
        self.assertEqual(eeg['srate'], 250.0)
        self.assertEqual(eeg['trials'], 1)
        
        # Check data shape
        self.assertEqual(eeg['data'].shape, (32, 1000))
        
        # Check that data is finite
        self.assertTrue(np.all(np.isfinite(eeg['data'])))
    
    def test_create_test_eeg_custom_params(self):
        """Test EEG creation with custom parameters."""
        eeg = create_test_eeg(n_channels=16, n_samples=500, srate=125.0, n_trials=3)
        
        self.assertEqual(eeg['nbchan'], 16)
        self.assertEqual(eeg['pnts'], 500)
        self.assertEqual(eeg['srate'], 125.0)
        self.assertEqual(eeg['trials'], 3)
        
        # Check data shape for multi-trial
        self.assertEqual(eeg['data'].shape, (16, 500, 3))
    
    def test_create_test_eeg_with_ica(self):
        """Test EEG creation with ICA components."""
        eeg = create_test_eeg_with_ica(n_channels=16, n_components=12)
        
        # Check ICA fields
        self.assertIn('icawinv', eeg)
        self.assertIn('icaweights', eeg)
        self.assertIn('icaact', eeg)
        self.assertIn('icachansind', eeg)
        
        # Check ICA shapes
        self.assertEqual(eeg['icawinv'].shape, (16, 12))
        self.assertEqual(eeg['icaweights'].shape, (12, 16))
        self.assertEqual(eeg['icaact'].shape, (12, 1000))
        
        # Check reference is set to average
        self.assertEqual(eeg['ref'], 'averef')
        
        # Check channel locations
        self.assertEqual(len(eeg['chanlocs']), 16)
        for i, ch in enumerate(eeg['chanlocs']):
            self.assertIn('labels', ch)
            self.assertEqual(ch['labels'], f'Ch{i+1}')
    
    def test_create_test_events(self):
        """Test event creation."""
        events = create_test_events(n_events=5, max_latency=500)
        
        self.assertEqual(len(events), 5)
        
        for event in events:
            self.assertIn('type', event)
            self.assertIn('latency', event)
            self.assertGreaterEqual(event['latency'], 1)
            self.assertLessEqual(event['latency'], 500)
    
    def test_create_test_events_custom_types(self):
        """Test event creation with custom event types."""
        event_types = ['A', 'B', 'C']
        events = create_test_events(n_events=6, event_types=event_types)
        
        # Should cycle through event types
        types_used = [e['type'] for e in events]
        self.assertEqual(types_used, ['A', 'B', 'C', 'A', 'B', 'C'])
    
    def test_cleanup_matplotlib(self):
        """Test matplotlib cleanup."""
        # Record initial state
        initial_figs = len(plt.get_fignums())
        
        # Create some figures
        plt.figure()
        plt.figure()
        
        self.assertGreater(len(plt.get_fignums()), initial_figs)
        
        cleanup_matplotlib()
        
        # Should be back to initial state or better
        self.assertLessEqual(len(plt.get_fignums()), initial_figs)
    
    def test_small_eeg(self):
        """Test small_eeg convenience function."""
        eeg = small_eeg()
        
        self.assertEqual(eeg['nbchan'], 8)
        self.assertEqual(eeg['pnts'], 250)
        self.assertEqual(eeg['data'].shape, (8, 250))


class TestFixturesContextManager(unittest.TestCase):
    """Test TestFixtures context manager."""
    
    def test_context_manager_basic(self):
        """Test basic context manager functionality."""
        with TestFixturesContextManager(seed=456) as fixtures:
            eeg = fixtures.create_eeg(n_channels=4)
            self.assertEqual(eeg['nbchan'], 4)
    
    def test_context_manager_reproducible(self):
        """Test context manager provides reproducible results."""
        with TestFixturesContextManager(seed=789) as fixtures:
            eeg1 = fixtures.create_eeg(n_channels=4, n_samples=100)
        
        with TestFixturesContextManager(seed=789) as fixtures:
            eeg2 = fixtures.create_eeg(n_channels=4, n_samples=100)
        
        np.testing.assert_array_equal(eeg1['data'], eeg2['data'])
    
    def test_context_manager_matplotlib_backend(self):
        """Test context manager sets matplotlib backend."""
        original_backend = matplotlib.get_backend()
        
        with TestFixturesContextManager(mpl_backend='Agg'):
            self.assertEqual(matplotlib.get_backend(), 'Agg')
        
        # Should restore original backend
        self.assertEqual(matplotlib.get_backend(), original_backend)
    
    def test_context_manager_methods(self):
        """Test context manager methods."""
        with TestFixturesContextManager() as fixtures:
            # Test EEG creation
            eeg = fixtures.create_eeg(n_channels=8)
            self.assertEqual(eeg['nbchan'], 8)
            
            # Test EEG with ICA creation
            eeg_ica = fixtures.create_eeg_with_ica(n_channels=8, n_components=6)
            self.assertEqual(eeg_ica['icawinv'].shape, (8, 6))
            
            # Test event creation
            events = fixtures.create_events(n_events=3)
            self.assertEqual(len(events), 3)


class TestFixturesIntegration(unittest.TestCase):
    """Test fixtures integration with real use cases."""
    
    def test_deterministic_across_tests(self):
        """Test that fixtures provide deterministic results across tests."""
        # This test verifies that using the same seed produces same results
        # even when called from different test methods
        
        with TestFixturesContextManager(seed=999) as fixtures:
            eeg = fixtures.create_eeg(n_channels=3, n_samples=50)
            first_sample = eeg['data'][0, 0]
        
        # Call from different context but same seed
        with TestFixturesContextManager(seed=999) as fixtures:
            eeg = fixtures.create_eeg(n_channels=3, n_samples=50)
            second_sample = eeg['data'][0, 0]
        
        self.assertEqual(first_sample, second_sample)
    
    def test_fixtures_with_real_processing(self):
        """Test fixtures work with actual data processing."""
        with TestFixturesContextManager(seed=111, mpl_backend='Agg') as fixtures:
            eeg = fixtures.create_eeg_with_ica(n_channels=4, n_samples=100)
            
            # Test that we can perform basic operations
            mean_data = np.mean(eeg['data'])
            self.assertTrue(np.isfinite(mean_data))
            
            # Test ICA components are reasonable
            self.assertTrue(np.all(np.isfinite(eeg['icawinv'])))
            self.assertTrue(np.all(np.isfinite(eeg['icaact'])))
    
    def test_fixtures_memory_efficiency(self):
        """Test that fixtures don't cause memory leaks."""
        initial_figs = len(plt.get_fignums())
        
        for i in range(5):
            with TestFixturesContextManager() as fixtures:
                eeg = fixtures.create_eeg(n_channels=2, n_samples=10)
                plt.figure()  # Create figure that should be cleaned up
        
        # Should not accumulate figures
        final_figs = len(plt.get_fignums())
        self.assertEqual(final_figs, initial_figs)


if __name__ == '__main__':
    unittest.main()
