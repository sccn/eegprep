.. _advanced_topics:

===============
Advanced Topics
===============

This guide covers advanced topics for experienced users, including custom preprocessing chains, extending the pipeline, MNE-Python integration, parallel processing, and performance optimization.

Custom Preprocessing Chains
===========================

Building Custom Pipelines
--------------------------

Create a custom preprocessing pipeline tailored to your specific needs:

.. code-block:: python

    from eegprep import (
        clean_flatlines,
        clean_channels,
        pop_resample,
        pop_eegfiltnew,
        eeg_picard,
        iclabel,
        eeg_interp
    )

    def custom_pipeline(eeg, params=None):
        """Custom preprocessing pipeline with logging"""
        
        if params is None:
            params = {}
        
        # Set defaults
        flatline_crit = params.get('flatline_criterion', 5)
        highpass = params.get('highpass', 1)
        lowpass = params.get('lowpass', 100)
        resample_rate = params.get('resample_rate', 250)
        asr_crit = params.get('asr_criterion', 20)
        
        print(f"Starting preprocessing with {eeg.nbchan} channels")
        
        # Step 1: Remove flatlines
        print("Step 1: Removing flatlines...")
        eeg = clean_flatlines(eeg, flatline_criterion=flatline_crit)
        print(f"  Channels remaining: {eeg.nbchan}")
        
        # Step 2: Remove noisy channels
        print("Step 2: Removing noisy channels...")
        eeg = clean_channels(eeg)
        print(f"  Channels remaining: {eeg.nbchan}")
        
        # Step 3: Interpolate removed channels
        print("Step 3: Interpolating removed channels...")
        eeg = eeg_interp(eeg)
        
        # Step 4: Resample
        print(f"Step 4: Resampling to {resample_rate} Hz...")
        eeg = pop_resample(eeg, resample_rate)
        
        # Step 5: Filter
        print(f"Step 5: Filtering {highpass}-{lowpass} Hz...")
        eeg = pop_eegfiltnew(eeg, locutoff=highpass, hicutoff=lowpass)
        
        # Step 6: ICA
        print("Step 6: Running ICA...")
        eeg = eeg_picard(eeg)
        print(f"  Components: {eeg.icaweights.shape[0]}")
        
        # Step 7: Component classification
        print("Step 7: Classifying components...")
        eeg = iclabel(eeg)
        
        print("Preprocessing complete!")
        return eeg

    # Use custom pipeline
    params = {
        'flatline_criterion': 5,
        'highpass': 1,
        'lowpass': 100,
        'resample_rate': 250,
        'asr_criterion': 20
    }
    eeg = custom_pipeline(eeg, params)

Conditional Preprocessing
--------------------------

Apply different preprocessing based on data characteristics:

.. code-block:: python

    from eegprep import clean_artifacts, eeg_rpsd

    def adaptive_preprocessing(eeg):
        """Adapt preprocessing based on data quality"""
        
        # Assess data quality
        psd = eeg_rpsd(eeg)
        noise_level = psd[50:100].mean()
        
        if noise_level > 100:
            # High noise: aggressive preprocessing
            print("High noise detected: using aggressive preprocessing")
            eeg = clean_artifacts(
                eeg,
                asr_criterion=15,
                flatline_criterion=3
            )
        elif noise_level > 50:
            # Medium noise: standard preprocessing
            print("Medium noise detected: using standard preprocessing")
            eeg = clean_artifacts(eeg)
        else:
            # Low noise: conservative preprocessing
            print("Low noise detected: using conservative preprocessing")
            eeg = clean_artifacts(
                eeg,
                asr_criterion=25,
                flatline_criterion=10
            )
        
        return eeg

    eeg = adaptive_preprocessing(eeg)

Extending the Pipeline
======================

Creating Custom Functions
--------------------------

Create custom preprocessing functions that integrate with eegprep:

.. code-block:: python

    from eegprep import EEGobj
    import numpy as np

    def custom_artifact_removal(eeg, threshold=3):
        """Custom artifact removal based on amplitude threshold"""
        
        if not isinstance(eeg, EEGobj):
            raise TypeError("Input must be an EEGobj")
        
        # Find samples exceeding threshold
        artifact_samples = np.where(
            np.abs(eeg.data).max(axis=0) > threshold * np.std(eeg.data)
        )[0]
        
        # Mark artifacts
        if not hasattr(eeg, 'removed_windows'):
            eeg.removed_windows = []
        
        eeg.removed_windows.extend(artifact_samples)
        
        print(f"Marked {len(artifact_samples)} artifact samples")
        return eeg

    # Use custom function
    eeg = custom_artifact_removal(eeg, threshold=5)

Creating Preprocessing Decorators
----------------------------------

Use decorators to add functionality to preprocessing functions:

.. code-block:: python

    import time
    from functools import wraps

    def timing_decorator(func):
        """Decorator to measure function execution time"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"{func.__name__} took {elapsed:.2f} seconds")
            return result
        return wrapper

    def logging_decorator(func):
        """Decorator to log function calls"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"Calling {func.__name__}")
            result = func(*args, **kwargs)
            print(f"Completed {func.__name__}")
            return result
        return wrapper

    # Apply decorators
    @timing_decorator
    @logging_decorator
    def my_preprocessing(eeg):
        from eegprep import clean_artifacts
        return clean_artifacts(eeg)

    eeg = my_preprocessing(eeg)

Integration with MNE-Python
============================

Converting Between Formats
---------------------------

Convert between eegprep and MNE-Python formats:

.. code-block:: python

    from eegprep import eeg_eeg2mne, eeg_mne2eeg
    import mne

    # Convert eegprep to MNE
    raw = eeg_eeg2mne(eeg)

    # Use MNE functions
    raw.plot()
    raw.compute_psd().plot()

    # Convert back to eegprep
    eeg = eeg_mne2eeg(raw)

Using MNE Preprocessing
-----------------------

Combine eegprep and MNE preprocessing:

.. code-block:: python

    from eegprep import eeg_eeg2mne, eeg_mne2eeg, clean_artifacts
    import mne

    # Preprocess with eegprep
    eeg = clean_artifacts(eeg)

    # Convert to MNE
    raw = eeg_eeg2mne(eeg)

    # Apply MNE preprocessing
    raw.filter(l_freq=1, h_freq=100)
    raw.set_eeg_reference('average')

    # Convert back
    eeg = eeg_mne2eeg(raw)

Epoching with MNE
-----------------

Create epochs using MNE and convert to eegprep:

.. code-block:: python

    from eegprep import eeg_eeg2mne, eeg_mne2eeg_epochs
    import mne

    # Convert to MNE
    raw = eeg_eeg2mne(eeg)

    # Create epochs
    events = mne.find_events(raw)
    epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.5)

    # Convert back to eegprep
    eeg = eeg_mne2eeg_epochs(epochs)

Parallel Processing
===================

Batch Processing with Multiprocessing
--------------------------------------

Process multiple subjects in parallel:

.. code-block:: python

    from multiprocessing import Pool
    from eegprep import pop_loadset, clean_artifacts, pop_saveset
    import os

    def process_subject(subject_id):
        """Process a single subject"""
        
        # Load data
        input_file = f'data/sub-{subject_id:02d}.set'
        eeg = pop_loadset(input_file)
        
        # Preprocess
        eeg = clean_artifacts(eeg)
        
        # Save
        output_file = f'data/preprocessed/sub-{subject_id:02d}_preprocessed.set'
        pop_saveset(eeg, output_file)
        
        return f"Processed subject {subject_id}"

    # Process subjects in parallel
    subject_ids = range(1, 11)  # Subjects 1-10
    
    with Pool(processes=4) as pool:
        results = pool.map(process_subject, subject_ids)
    
    for result in results:
        print(result)

Using joblib for Parallel Processing
-------------------------------------

Use joblib for more flexible parallel processing:

.. code-block:: python

    from joblib import Parallel, delayed
    from eegprep import pop_loadset, clean_artifacts, pop_saveset

    def process_subject(subject_id):
        """Process a single subject"""
        input_file = f'data/sub-{subject_id:02d}.set'
        eeg = pop_loadset(input_file)
        eeg = clean_artifacts(eeg)
        output_file = f'data/preprocessed/sub-{subject_id:02d}_preprocessed.set'
        pop_saveset(eeg, output_file)
        return f"Processed subject {subject_id}"

    # Process with joblib
    results = Parallel(n_jobs=4)(
        delayed(process_subject)(i) for i in range(1, 11)
    )

    for result in results:
        print(result)

GPU Acceleration
----------------

Use GPU acceleration for faster processing:

.. code-block:: python

    import torch
    from eegprep import clean_artifacts

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("GPU not available, using CPU")
        device = 'cpu'

    # Preprocess with GPU
    eeg = clean_artifacts(eeg, device=device)

Performance Optimization
========================

Memory Optimization
-------------------

Reduce memory usage for large datasets:

.. code-block:: python

    from eegprep import pop_loadset, pop_saveset
    import numpy as np

    def process_in_chunks(filename, chunk_size=10):
        """Process data in chunks to reduce memory usage"""
        
        # Load data
        eeg = pop_loadset(filename)
        
        # Process in chunks
        n_chunks = int(np.ceil(eeg.pnts / (chunk_size * eeg.srate)))
        
        for i in range(n_chunks):
            start = i * chunk_size * eeg.srate
            end = min((i + 1) * chunk_size * eeg.srate, eeg.pnts)
            
            print(f"Processing chunk {i+1}/{n_chunks}")
            # Process chunk
            chunk_data = eeg.data[:, start:end]
            # ... process chunk ...
        
        return eeg

Computation Optimization
------------------------

Speed up preprocessing:

.. code-block:: python

    from eegprep import clean_artifacts, EEG_OPTIONS

    # Use optimized parameters
    options = EEG_OPTIONS()
    options.ica_ncomps = 30  # Reduce components
    options.filter_order = 2  # Reduce filter order
    options.asr_wlen = 1.0   # Increase window length

    # Preprocess with optimized settings
    eeg = clean_artifacts(eeg, options=options)

Caching Results
---------------

Cache preprocessing results to avoid recomputation:

.. code-block:: python

    import pickle
    import hashlib
    from eegprep import pop_loadset, clean_artifacts

    def get_preprocessed_data(filename, params):
        """Get preprocessed data with caching"""
        
        # Create cache key
        cache_key = hashlib.md5(
            f"{filename}{str(params)}".encode()
        ).hexdigest()
        cache_file = f"cache/{cache_key}.pkl"
        
        # Check cache
        try:
            with open(cache_file, 'rb') as f:
                eeg = pickle.load(f)
            print(f"Loaded from cache: {cache_file}")
            return eeg
        except FileNotFoundError:
            pass
        
        # Preprocess
        eeg = pop_loadset(filename)
        eeg = clean_artifacts(eeg, **params)
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(eeg, f)
        
        return eeg

Profiling and Benchmarking
--------------------------

Profile preprocessing to identify bottlenecks:

.. code-block:: python

    import cProfile
    import pstats
    from eegprep import clean_artifacts

    def profile_preprocessing(eeg):
        """Profile preprocessing function"""
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Run preprocessing
        eeg = clean_artifacts(eeg)
        
        profiler.disable()
        
        # Print statistics
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # Print top 10 functions
        
        return eeg

Best Practices
==============

Code Organization
-----------------

Organize custom preprocessing code:

.. code-block:: python

    # preprocessing/pipelines.py
    from eegprep import clean_artifacts

    class PreprocessingPipeline:
        """Base class for preprocessing pipelines"""
        
        def __init__(self, params=None):
            self.params = params or {}
        
        def run(self, eeg):
            raise NotImplementedError

    class RestingStatePipeline(PreprocessingPipeline):
        """Resting state preprocessing pipeline"""
        
        def run(self, eeg):
            return clean_artifacts(
                eeg,
                highpass=1,
                lowpass=100,
                asr_criterion=20
            )

    class ERPPipeline(PreprocessingPipeline):
        """ERP preprocessing pipeline"""
        
        def run(self, eeg):
            return clean_artifacts(
                eeg,
                highpass=0.1,
                lowpass=30,
                asr_criterion=15
            )

    # Usage
    pipeline = RestingStatePipeline()
    eeg = pipeline.run(eeg)

Error Handling
--------------

Implement robust error handling:

.. code-block:: python

    from eegprep import pop_loadset, clean_artifacts, pop_saveset
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def safe_preprocessing(filename, output_file):
        """Preprocess with error handling"""
        
        try:
            # Load data
            logger.info(f"Loading {filename}")
            eeg = pop_loadset(filename)
            
            # Preprocess
            logger.info("Preprocessing...")
            eeg = clean_artifacts(eeg)
            
            # Save
            logger.info(f"Saving to {output_file}")
            pop_saveset(eeg, output_file)
            
            logger.info("Success!")
            return True
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False

Documentation
--------------

Document custom functions:

.. code-block:: python

    def custom_preprocessing(eeg, threshold=3):
        """
        Apply custom artifact removal.
        
        Parameters
        ----------
        eeg : EEGobj
            Input EEG data
        threshold : float, optional
            Amplitude threshold in standard deviations (default: 3)
        
        Returns
        -------
        eeg : EEGobj
            Preprocessed EEG data
        
        Examples
        --------
        >>> eeg = custom_preprocessing(eeg, threshold=5)
        """
        # Implementation
        return eeg

Testing
-------

Test custom preprocessing functions:

.. code-block:: python

    import unittest
    from eegprep import pop_loadset

    class TestCustomPreprocessing(unittest.TestCase):
        
        def setUp(self):
            """Load test data"""
            self.eeg = pop_loadset('test_data.set')
        
        def test_preprocessing_runs(self):
            """Test that preprocessing runs without error"""
            eeg = custom_preprocessing(self.eeg)
            self.assertIsNotNone(eeg)
        
        def test_preprocessing_preserves_shape(self):
            """Test that preprocessing preserves data shape"""
            eeg = custom_preprocessing(self.eeg)
            self.assertEqual(eeg.nbchan, self.eeg.nbchan)

    if __name__ == '__main__':
        unittest.main()

Next Steps
==========

Now that you understand advanced topics:

1. Review the :ref:`preprocessing_pipeline` guide for detailed preprocessing steps
2. Explore the :ref:`configuration` guide for parameter tuning
3. Check the :ref:`bids_workflow` for batch processing
4. Review the :ref:`api_reference` for detailed function documentation
