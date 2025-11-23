.. _configuration:

=============
Configuration
=============

This guide covers configuration options for eegprep, including the EEG_OPTIONS object, common parameters, and custom preprocessing chains.

EEG_OPTIONS Overview
====================

The :class:`eegprep.EEG_OPTIONS` class provides a centralized way to configure eegprep behavior:

.. code-block:: python

    from eegprep import EEG_OPTIONS

    # Access default options
    options = EEG_OPTIONS()

    # View all options
    print(options)

    # Modify options
    options.ica_method = 'picard'
    options.asr_criterion = 20

Common Configuration Parameters
===============================

Preprocessing Parameters
------------------------

**Artifact Detection**

.. code-block:: python

    from eegprep import EEG_OPTIONS

    options = EEG_OPTIONS()

    # Flatline detection
    options.flatline_criterion = 5  # Duration in seconds

    # ASR (Artifact Subspace Reconstruction)
    options.asr_criterion = 20      # Standard deviation threshold
    options.asr_wlen = 0.5          # Window length in seconds

    # Channel noise detection
    options.ransac_criterion = 0.8  # Correlation threshold
    options.max_broken_time = 0.5   # Max proportion of broken time

**Filtering**

.. code-block:: python

    # High-pass filter
    options.highpass = 1            # Frequency in Hz

    # Low-pass filter
    options.lowpass = 100           # Frequency in Hz

    # Filter order
    options.filter_order = 4        # FIR filter order

**Resampling**

.. code-block:: python

    # Target sampling rate
    options.resample_rate = 250     # Hz

**ICA**

.. code-block:: python

    # ICA method
    options.ica_method = 'picard'   # 'picard' or 'infomax'

    # Number of components
    options.ica_ncomps = None       # None = number of channels

    # Maximum iterations
    options.ica_max_iter = 500

**Component Classification**

.. code-block:: python

    # Use ICLabel for component classification
    options.use_iclabel = True

    # ICLabel threshold for artifact removal
    options.iclabel_threshold = 0.5

Reference and Re-referencing
-----------------------------

.. code-block:: python

    # Reference type
    options.reference = 'average'   # 'average', 'common', or channel name

    # Exclude channels from reference
    options.reference_exclude = ['HEOG', 'VEOG']

Data Handling
-------------

.. code-block:: python

    # Memory mode
    options.memory_mode = 'disk'    # 'memory' or 'disk'

    # Verbose output
    options.verbose = True

    # Random seed for reproducibility
    options.random_seed = 42

Creating Custom Preprocessing Chains
====================================

Using Configuration Objects
----------------------------

Create a custom configuration for your preprocessing:

.. code-block:: python

    from eegprep import EEG_OPTIONS, clean_artifacts

    # Create custom options
    custom_options = EEG_OPTIONS()
    custom_options.highpass = 0.5
    custom_options.lowpass = 50
    custom_options.asr_criterion = 15
    custom_options.ica_method = 'picard'

    # Use custom options in preprocessing
    eeg = clean_artifacts(eeg, options=custom_options)

Preprocessing Presets
---------------------

Create preset configurations for different use cases:

**Resting State EEG**

.. code-block:: python

    from eegprep import EEG_OPTIONS

    def get_resting_state_options():
        """Configuration for resting state EEG preprocessing"""
        options = EEG_OPTIONS()
        options.highpass = 1
        options.lowpass = 100
        options.asr_criterion = 20
        options.ica_method = 'picard'
        options.use_iclabel = True
        return options

    # Use preset
    eeg = clean_artifacts(eeg, options=get_resting_state_options())

**Event-Related Potentials (ERP)**

.. code-block:: python

    from eegprep import EEG_OPTIONS

    def get_erp_options():
        """Configuration for ERP preprocessing"""
        options = EEG_OPTIONS()
        options.highpass = 0.1
        options.lowpass = 30
        options.asr_criterion = 15
        options.ica_method = 'picard'
        options.use_iclabel = True
        return options

    # Use preset
    eeg = clean_artifacts(eeg, options=get_erp_options())

**High-Frequency Activity**

.. code-block:: python

    from eegprep import EEG_OPTIONS

    def get_hfa_options():
        """Configuration for high-frequency activity analysis"""
        options = EEG_OPTIONS()
        options.highpass = 1
        options.lowpass = 200
        options.asr_criterion = 25
        options.resample_rate = 500
        options.ica_method = 'picard'
        return options

    # Use preset
    eeg = clean_artifacts(eeg, options=get_hfa_options())

**Clinical EEG**

.. code-block:: python

    from eegprep import EEG_OPTIONS

    def get_clinical_options():
        """Configuration for clinical EEG preprocessing"""
        options = EEG_OPTIONS()
        options.highpass = 0.5
        options.lowpass = 70
        options.asr_criterion = 20
        options.flatline_criterion = 10
        options.ica_method = 'picard'
        options.use_iclabel = True
        return options

    # Use preset
    eeg = clean_artifacts(eeg, options=get_clinical_options())

Custom Preprocessing Functions
------------------------------

Create custom preprocessing functions:

.. code-block:: python

    from eegprep import (
        clean_flatlines,
        clean_channels,
        pop_resample,
        pop_eegfiltnew,
        eeg_picard,
        iclabel
    )

    def custom_preprocessing_pipeline(eeg, options=None):
        """Custom preprocessing pipeline"""
        
        # Step 1: Remove flatlines
        eeg = clean_flatlines(eeg, flatline_criterion=5)
        
        # Step 2: Remove noisy channels
        eeg = clean_channels(eeg)
        
        # Step 3: Resample
        eeg = pop_resample(eeg, 250)
        
        # Step 4: Filter
        eeg = pop_eegfiltnew(eeg, locutoff=1, hicutoff=100)
        
        # Step 5: ICA
        eeg = eeg_picard(eeg)
        
        # Step 6: Component classification
        eeg = iclabel(eeg)
        
        return eeg

    # Use custom pipeline
    eeg = custom_preprocessing_pipeline(eeg)

Advanced Settings
=================

ICA Configuration
-----------------

**Picard Algorithm**

.. code-block:: python

    from eegprep import eeg_picard

    eeg = eeg_picard(
        eeg,
        ncomps=None,        # Number of components
        max_iter=500,       # Maximum iterations
        tol=1e-7,          # Convergence tolerance
        ortho=True,        # Orthogonalize components
        extended=False     # Extended ICA
    )

**Infomax Algorithm**

.. code-block:: python

    from eegprep import eeg_picard

    # Picard is recommended, but you can adjust parameters
    eeg = eeg_picard(eeg, ncomps=eeg.nbchan)

ASR Configuration
-----------------

**Standard ASR**

.. code-block:: python

    from eegprep import clean_asr

    eeg = clean_asr(
        eeg,
        asr_criterion=20,   # Standard deviation threshold
        asr_wlen=0.5,      # Window length in seconds
        asr_overlap=0.5    # Window overlap
    )

**Aggressive ASR**

.. code-block:: python

    from eegprep import clean_asr

    eeg = clean_asr(
        eeg,
        asr_criterion=10,   # Lower threshold = more aggressive
        asr_wlen=0.5,
        asr_overlap=0.5
    )

**Conservative ASR**

.. code-block:: python

    from eegprep import clean_asr

    eeg = clean_asr(
        eeg,
        asr_criterion=30,   # Higher threshold = more conservative
        asr_wlen=0.5,
        asr_overlap=0.5
    )

Filter Configuration
--------------------

**FIR Filters**

.. code-block:: python

    from eegprep import pop_eegfiltnew

    # FIR filter (default)
    eeg = pop_eegfiltnew(
        eeg,
        locutoff=1,
        hicutoff=100,
        filtorder=4,  # Filter order
        revfilt=0     # Forward filter
    )

**IIR Filters**

.. code-block:: python

    from eegprep import pop_eegfiltnew

    # IIR filter
    eeg = pop_eegfiltnew(
        eeg,
        locutoff=1,
        hicutoff=100,
        filtorder=4,
        revfilt=0,
        iir=True  # Use IIR filter
    )

Resampling Configuration
------------------------

.. code-block:: python

    from eegprep import pop_resample

    # Resample to 250 Hz
    eeg = pop_resample(eeg, 250)

    # Resample with specific method
    eeg = pop_resample(
        eeg,
        newrate=250,
        method='sinc'  # Sinc interpolation
    )

Configuration Files
===================

Saving Configuration
--------------------

Save your configuration to a file:

.. code-block:: python

    import json
    from eegprep import EEG_OPTIONS

    # Create configuration
    config = {
        'highpass': 1,
        'lowpass': 100,
        'asr_criterion': 20,
        'ica_method': 'picard',
        'use_iclabel': True,
        'resample_rate': 250
    }

    # Save to JSON
    with open('preprocessing_config.json', 'w') as f:
        json.dump(config, f, indent=2)

Loading Configuration
---------------------

Load configuration from a file:

.. code-block:: python

    import json
    from eegprep import EEG_OPTIONS, clean_artifacts

    # Load configuration
    with open('preprocessing_config.json', 'r') as f:
        config = json.load(f)

    # Create options from configuration
    options = EEG_OPTIONS()
    for key, value in config.items():
        setattr(options, key, value)

    # Use configuration
    eeg = clean_artifacts(eeg, options=options)

Example Configuration Files
----------------------------

**resting_state_config.json**

.. code-block:: json

    {
        "highpass": 1,
        "lowpass": 100,
        "asr_criterion": 20,
        "asr_wlen": 0.5,
        "flatline_criterion": 5,
        "ransac_criterion": 0.8,
        "ica_method": "picard",
        "ica_ncomps": null,
        "use_iclabel": true,
        "resample_rate": 250,
        "reference": "average"
    }

**erp_config.json**

.. code-block:: json

    {
        "highpass": 0.1,
        "lowpass": 30,
        "asr_criterion": 15,
        "asr_wlen": 0.5,
        "flatline_criterion": 5,
        "ica_method": "picard",
        "use_iclabel": true,
        "resample_rate": 250,
        "reference": "average"
    }

Parameter Recommendations
=========================

By Data Type
------------

**Resting State EEG**

- Highpass: 1 Hz
- Lowpass: 100 Hz
- ASR criterion: 20
- Resample: 250 Hz

**Event-Related Potentials (ERP)**

- Highpass: 0.1 Hz
- Lowpass: 30 Hz
- ASR criterion: 15
- Resample: 250 Hz

**High-Frequency Activity**

- Highpass: 1 Hz
- Lowpass: 200 Hz
- ASR criterion: 25
- Resample: 500 Hz

**Clinical EEG**

- Highpass: 0.5 Hz
- Lowpass: 70 Hz
- ASR criterion: 20
- Resample: 250 Hz

By Artifact Type
----------------

**Muscle Artifacts**

- ASR criterion: 15 (more aggressive)
- Flatline criterion: 5
- Use ICLabel: True

**Eye Movement Artifacts**

- Highpass: 0.5 Hz
- Use ICLabel: True
- Manual component removal

**Line Noise (50/60 Hz)**

- Notch filter: 50 or 60 Hz
- Lowpass: 100 Hz

**Drift**

- Highpass: 0.5 Hz
- ASR criterion: 20

Troubleshooting Configuration
=============================

Configuration Not Applied
--------------------------

**Problem**: Configuration changes don't affect preprocessing

**Solution**:

1. Verify options are passed to preprocessing function
2. Check that options object is correctly created
3. Ensure parameter names are correct

.. code-block:: python

    from eegprep import EEG_OPTIONS, clean_artifacts

    # Correct way
    options = EEG_OPTIONS()
    options.highpass = 1
    eeg = clean_artifacts(eeg, options=options)

    # Incorrect way (won't work)
    eeg = clean_artifacts(eeg, highpass=1)

Unexpected Results
------------------

**Problem**: Preprocessing produces unexpected results

**Solution**:

1. Visualize data before and after preprocessing
2. Check parameter values
3. Try different parameter combinations
4. Review the preprocessing pipeline steps

.. code-block:: python

    import matplotlib.pyplot as plt

    # Plot before and after
    plt.figure(figsize=(12, 6))
    plt.plot(eeg.data[0, :1000])
    plt.title('Preprocessed Data')
    plt.show()

Performance Issues
------------------

**Problem**: Preprocessing is slow

**Solution**:

1. Reduce number of components in ICA
2. Increase resampling rate
3. Use parallel processing for batch jobs
4. Reduce filter order

.. code-block:: python

    from eegprep import EEG_OPTIONS

    options = EEG_OPTIONS()
    options.ica_ncomps = 30  # Reduce components
    options.resample_rate = 250  # Increase rate
    options.filter_order = 2  # Reduce filter order

Next Steps
==========

Now that you understand configuration:

1. Read the :ref:`preprocessing_pipeline` guide for detailed preprocessing steps
2. Explore the :ref:`advanced_topics` for custom pipelines
3. Check the :ref:`quickstart` for practical examples
4. Review the :ref:`api_reference` for detailed function documentation
