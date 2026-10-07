"""Common test fixtures for eegprep tests.

This module provides common test fixtures and utilities that can be reused
across different test modules.
"""

import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np


SAMPLE_DATASET_PATH = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data.set"


def matlab_engine_available():
    """Check if MATLAB engine is available for Python.

    Returns:
        bool: True if matlab.engine can be imported and started, False otherwise.
    """
    # Check if MATLAB tests should be skipped via environment variable
    if os.environ.get('EEGPREP_SKIP_MATLAB', '0') == '1':
        return False

    try:
        return importlib.util.find_spec("matlab.engine") is not None
    except (ImportError, ValueError):
        return False


def skip_without_matlab(test_func):
    """Decorator to skip tests that require MATLAB engine.

    Usage:
        @skip_without_matlab
        def test_matlab_function(self):
            # Test code that requires MATLAB
            pass
    """
    return unittest.skipUnless(matlab_engine_available(), "MATLAB engine not available or skipped")(test_func)


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

    # Create events and epoch info if epoched data
    events = []
    epochs = []
    if n_trials > 1:
        # Create one event per epoch
        for i in range(n_trials):
            events.append(
                {
                    'type': 'epoch',
                    'latency': i * n_samples + 1,  # 1-based indexing for EEGLAB
                    'duration': 0,
                    'urevent': i + 1,
                }
            )
            epochs.append({'event': [i], 'eventtype': ['epoch'], 'eventlatency': [0], 'eventduration': [0]})

    # Create basic channel locations
    chanlocs = []
    for i in range(n_channels):
        chanlocs.append(
            {
                'labels': f'Ch{i + 1}',
                'type': 'EEG',
                'theta': i * (360 / n_channels),
                'radius': 0.5,
                'X': 0.5 * np.cos(np.radians(i * (360 / n_channels))),
                'Y': 0.5 * np.sin(np.radians(i * (360 / n_channels))),
                'Z': 0.0,
                'sph_theta': i * (360 / n_channels),
                'sph_phi': 0.0,
                'sph_radius': 1.0,
                'urchan': i + 1,
                'ref': '',
            }
        )

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
        'event': events,
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
        'chanlocs': chanlocs,
        'urchanlocs': [],
        'chaninfo': {'filename': '', 'plotrad': [], 'shrink': [], 'nosedir': '+X', 'nodatchans': []},
        'urevent': [],
        'eventdescription': {},
        'epoch': epochs,
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
        'roi': {},
    }

    return eeg


def create_test_eeg_with_ica(n_channels=32, n_samples=1000, srate=250.0, n_components=None, n_trials=1):
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
        eeg['chanlocs'].append(
            {
                'theta': i * (360 / n_channels),  # Distribute evenly around head
                'radius': 0.3,
                'X': 0.3 * np.cos(np.radians(i * (360 / n_channels))),
                'Y': 0.3 * np.sin(np.radians(i * (360 / n_channels))),
                'Z': 0.0,
                'labels': f'Ch{i + 1}',
                'type': 'EEG',
            }
        )

    return eeg
