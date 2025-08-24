"""Common test fixtures for eegprep tests.

This module provides common test fixtures and utilities that can be reused
across different test modules.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def mpl_use_agg():
    """Set matplotlib backend to 'Agg' for headless testing.
    
    This should be called before importing matplotlib.pyplot or other
    matplotlib modules that require a display.
    """
    matplotlib.use('Agg')


def rng_seed(seed=42):
    """Set numpy random seed for deterministic testing.
    
    Args:
        seed (int): Random seed value. Default is 42.
    """
    np.random.seed(seed)


def create_test_eeg(n_channels=32, n_samples=1000, srate=250.0, n_trials=1):
    """Create a synthetic EEG structure for testing.
    
    Args:
        n_channels (int): Number of EEG channels. Default is 32.
        n_samples (int): Number of time samples. Default is 1000.
        srate (float): Sampling rate in Hz. Default is 250.0.
        n_trials (int): Number of trials/epochs. Default is 1.
        
    Returns:
        dict: EEG structure with synthetic data and metadata.
    """
    # Generate synthetic data
    data = np.random.randn(n_channels, n_samples, n_trials) * 0.5
    if n_trials == 1:
        data = data.squeeze(axis=2)  # Remove trial dimension for continuous data
    
    # Create basic EEG structure
    eeg = {
        'data': data,
        'srate': srate,
        'pnts': n_samples,
        'nbchan': n_channels,
        'trials': n_trials,
        'xmin': 0.0,
        'xmax': (n_samples - 1) / srate,
        'times': np.arange(n_samples) / srate,
        'event': [],
        'ref': 'unknown',
        'setname': 'test_dataset',
        'filename': '',
        'filepath': '',
        'subject': '',
        'group': '',
        'condition': '',
        'session': '',
        'comments': '',
        'icaact': None,
        'icawinv': None,
        'icasphere': None,
        'icaweights': None,
        'icachansind': None,
        'chanlocs': [],
        'urchanlocs': [],
        'chaninfo': {},
        'urevent': [],
        'eventdescription': {},
        'epoch': [],
        'epochdescription': {},
        'reject': {},
        'stats': {},
        'specdata': [],
        'specicaact': [],
        'splinefile': '',
        'icasplinefile': '',
        'dipfit': {},
        'history': '',
        'saved': 'no',
        'etc': {},
        'datfile': '',
        'run': [],
        'roi': {}
    }
    
    return eeg


def create_test_eeg_with_ica(n_channels=32, n_samples=1000, srate=250.0, 
                           n_components=None, n_trials=1):
    """Create a synthetic EEG structure with ICA decomposition for testing.
    
    Args:
        n_channels (int): Number of EEG channels. Default is 32.
        n_samples (int): Number of time samples. Default is 1000.
        srate (float): Sampling rate in Hz. Default is 250.0.
        n_components (int): Number of ICA components. Default is n_channels.
        n_trials (int): Number of trials/epochs. Default is 1.
        
    Returns:
        dict: EEG structure with synthetic data, ICA decomposition, and metadata.
    """
    if n_components is None:
        n_components = n_channels
    
    # Create base EEG structure
    eeg = create_test_eeg(n_channels, n_samples, srate, n_trials)
    
    # Add ICA decomposition
    eeg['icawinv'] = np.random.randn(n_channels, n_components) * 0.5
    eeg['icaweights'] = np.linalg.pinv(eeg['icawinv'])
    eeg['icasphere'] = np.eye(n_channels)
    eeg['icachansind'] = np.arange(n_channels)
    
    # Generate ICA activations
    if n_trials == 1:
        eeg['icaact'] = np.random.randn(n_components, n_samples) * 0.5
    else:
        eeg['icaact'] = np.random.randn(n_components, n_samples, n_trials) * 0.5
    
    # Set reference to average (often required for ICA analysis)
    eeg['ref'] = 'averef'
    
    # Add basic channel locations
    eeg['chanlocs'] = []
    for i in range(n_channels):
        eeg['chanlocs'].append({
            'theta': i * (360 / n_channels),  # Distribute evenly around head
            'radius': 0.3,
            'X': 0.3 * np.cos(np.radians(i * (360 / n_channels))),
            'Y': 0.3 * np.sin(np.radians(i * (360 / n_channels))),
            'Z': 0.0,
            'labels': f'Ch{i+1}',
            'type': 'EEG'
        })
    
    return eeg


def create_test_events(n_events=10, max_latency=1000, event_types=None):
    """Create synthetic event list for testing.
    
    Args:
        n_events (int): Number of events to create. Default is 10.
        max_latency (int): Maximum event latency in samples. Default is 1000.
        event_types (list): List of event types to use. Default is ['stim', 'resp'].
        
    Returns:
        list: List of event dictionaries.
    """
    if event_types is None:
        event_types = ['stim', 'resp']
    
    events = []
    latencies = np.sort(np.random.uniform(1, max_latency, n_events))
    
    for i, latency in enumerate(latencies):
        event_type = event_types[i % len(event_types)]
        events.append({
            'type': event_type,
            'latency': float(latency),
            'duration': 0.0,
            'channel': 0,
            'bvtime': [],
            'bvmknum': 1,
            'visible': [1],
            'code': event_type,
            'urevent': i + 1
        })
    
    return events


def cleanup_matplotlib():
    """Clean up matplotlib figures and reset state.
    
    This should be called in test tearDown methods to prevent
    memory leaks and interference between tests.
    """
    plt.close('all')
    plt.clf()
    plt.cla()


class TestFixtures:
    """Context manager for common test fixtures.
    
    Usage:
        with TestFixtures(seed=42, mpl_backend='Agg') as fixtures:
            eeg = fixtures.create_eeg(n_channels=64)
            # ... run tests ...
    """
    
    def __init__(self, seed=42, mpl_backend='Agg'):
        """Initialize test fixtures.
        
        Args:
            seed (int): Random seed for reproducible tests. Default is 42.
            mpl_backend (str): Matplotlib backend to use. Default is 'Agg'.
        """
        self.seed = seed
        self.mpl_backend = mpl_backend
        self.original_backend = None
    
    def __enter__(self):
        """Enter context manager and set up fixtures."""
        # Set random seed
        if self.seed is not None:
            rng_seed(self.seed)
        
        # Set matplotlib backend
        if self.mpl_backend is not None:
            self.original_backend = matplotlib.get_backend()
            matplotlib.use(self.mpl_backend)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and clean up."""
        # Clean up matplotlib
        cleanup_matplotlib()
        
        # Restore original backend if changed
        if self.original_backend is not None:
            matplotlib.use(self.original_backend)
    
    def create_eeg(self, **kwargs):
        """Create test EEG data with fixtures applied."""
        return create_test_eeg(**kwargs)
    
    def create_eeg_with_ica(self, **kwargs):
        """Create test EEG data with ICA and fixtures applied."""
        return create_test_eeg_with_ica(**kwargs)
    
    def create_events(self, **kwargs):
        """Create test events with fixtures applied."""
        return create_test_events(**kwargs)


# Legacy functions for backward compatibility
small_eeg = lambda: create_test_eeg(n_channels=8, n_samples=250)  # Small EEG for quick tests
