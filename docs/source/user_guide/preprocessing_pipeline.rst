.. _preprocessing_pipeline:

====================
Preprocessing Pipeline
====================

This guide provides a comprehensive overview of the eegprep preprocessing pipeline, including the order of operations, parameter tuning, and quality control.

Pipeline Overview
=================

The eegprep preprocessing pipeline is designed to systematically clean and prepare raw EEG data for analysis. The pipeline removes artifacts, interpolates bad channels, resamples data, applies filtering, performs ICA decomposition, and classifies independent components.

Key Features:

- **Automated artifact detection and removal**: Identifies and removes noisy channels and time windows
- **Flexible component classification**: Uses ICLabel for automatic ICA component classification
- **Customizable parameters**: Adjust thresholds and methods for your specific needs
- **Quality control**: Built-in checks and visualizations to assess preprocessing quality
- **Batch processing**: Process multiple subjects efficiently with :func:`eegprep.bids_preproc`

Pipeline Steps
==============

The preprocessing pipeline follows these steps in order:

1. Channel Selection
2. Artifact Removal (ASR and clean_artifacts)
3. Channel Interpolation
4. Resampling
5. Filtering
6. ICA Decomposition
7. Component Classification (ICLabel)

Step 1: Channel Selection
-------------------------

Select the channels to include in preprocessing:

.. code-block:: python

    from eegprep import pop_select

    # Select only EEG channels
    eeg = pop_select(eeg, 'type', 'EEG')

    # Select specific channels by name
    eeg = pop_select(eeg, 'channel', ['Cz', 'Pz', 'Oz', 'Fz'])

    # Remove specific channels
    eeg = pop_select(eeg, 'nochannel', ['HEOG', 'VEOG'])

**When to use**: Always perform channel selection first to ensure you're working with the correct data.

Step 2: Artifact Removal
------------------------

Remove noisy channels and time windows using multiple methods:

Flatline Detection
~~~~~~~~~~~~~~~~~~

Remove channels with no variation (dead channels):

.. code-block:: python

    from eegprep import clean_flatlines

    eeg = clean_flatlines(
        eeg,
        flatline_criterion=5  # Flatline duration in seconds
    )

Noisy Channel Removal
~~~~~~~~~~~~~~~~~~~~~

Remove channels with excessive noise:

.. code-block:: python

    from eegprep import clean_channels

    eeg = clean_channels(
        eeg,
        ransac_criterion=0.8,  # RANSAC correlation threshold
        max_broken_time=0.5    # Max proportion of broken time
    )

Artifact Subspace Reconstruction (ASR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove bursts of high-amplitude artifacts:

.. code-block:: python

    from eegprep import clean_asr

    eeg = clean_asr(
        eeg,
        asr_criterion=20,      # Standard deviation threshold
        asr_wlen=0.5           # Window length in seconds
    )

Comprehensive Artifact Removal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the all-in-one :func:`eegprep.clean_artifacts` function:

.. code-block:: python

    from eegprep import clean_artifacts

    eeg = clean_artifacts(
        eeg,
        flatline_criterion=5,
        highpass=1,
        lowpass=100,
        asr_criterion=20,
        asr_wlen=0.5,
        remove_channels=True,
        remove_windows=True
    )

**Parameters**:

- ``flatline_criterion``: Duration (seconds) of flatline to detect (default: 5)
- ``asr_criterion``: Standard deviation threshold for ASR (default: 20)
- ``asr_wlen``: Window length for ASR in seconds (default: 0.5)
- ``highpass``: High-pass filter frequency in Hz (default: 1)
- ``lowpass``: Low-pass filter frequency in Hz (default: 100)

Step 3: Channel Interpolation
------------------------------

Interpolate removed channels using spherical spline interpolation:

.. code-block:: python

    from eegprep import eeg_interp

    # Interpolate removed channels
    eeg = eeg_interp(eeg)

    # Interpolate specific channels
    eeg = eeg_interp(eeg, channels=[1, 5, 10])

**When to use**: After removing noisy channels, interpolate them to maintain spatial coverage.

Step 4: Resampling
------------------

Resample data to a lower sampling rate to reduce file size and computation:

.. code-block:: python

    from eegprep import pop_resample

    # Resample to 250 Hz
    eeg = pop_resample(eeg, 250)

    # Resample to 500 Hz
    eeg = pop_resample(eeg, 500)

**Common sampling rates**:

- 250 Hz: Standard for most EEG analysis
- 500 Hz: Higher resolution for detailed analysis
- 100 Hz: Lower resolution for quick analysis

**When to use**: Resample early in the pipeline to reduce computation time for subsequent steps.

Step 5: Filtering
-----------------

Apply frequency filtering to remove noise outside the frequency band of interest:

High-Pass Filtering
~~~~~~~~~~~~~~~~~~~

Remove slow drifts and DC offset:

.. code-block:: python

    from eegprep import pop_eegfiltnew

    # High-pass filter at 1 Hz
    eeg = pop_eegfiltnew(eeg, locutoff=1)

Low-Pass Filtering
~~~~~~~~~~~~~~~~~~

Remove high-frequency noise:

.. code-block:: python

    # Low-pass filter at 100 Hz
    eeg = pop_eegfiltnew(eeg, hicutoff=100)

Band-Pass Filtering
~~~~~~~~~~~~~~~~~~~

Apply both high-pass and low-pass filters:

.. code-block:: python

    # Band-pass filter 1-100 Hz
    eeg = pop_eegfiltnew(eeg, locutoff=1, hicutoff=100)

**Common filter settings**:

- **Resting state**: 1-100 Hz
- **Event-related potentials (ERP)**: 0.1-30 Hz
- **Oscillatory analysis**: 1-100 Hz
- **High-frequency activity**: 1-200 Hz

**When to use**: Apply filtering after resampling but before ICA for best results.

Step 6: ICA Decomposition
-------------------------

Decompose the data into independent components:

Using Picard Algorithm
~~~~~~~~~~~~~~~~~~~~~~

Fast and reliable ICA decomposition:

.. code-block:: python

    from eegprep import eeg_picard

    eeg = eeg_picard(
        eeg,
        ncomps=None,  # Number of components (None = number of channels)
        max_iter=500  # Maximum iterations
    )

Using Extended Infomax ICA
~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternative ICA algorithm:

.. code-block:: python

    from eegprep import eeg_picard

    # Picard is recommended, but you can adjust parameters
    eeg = eeg_picard(eeg, ncomps=eeg.nbchan)

**Parameters**:

- ``ncomps``: Number of components to extract (default: number of channels)
- ``max_iter``: Maximum iterations (default: 500)

**When to use**: After filtering, before component classification.

Step 7: Component Classification (ICLabel)
-------------------------------------------

Automatically classify ICA components using ICLabel:

.. code-block:: python

    from eegprep import iclabel

    eeg = iclabel(eeg)

    # Access component labels
    print(eeg.etc.ic_classification.ICLabel.classes)
    print(eeg.etc.ic_classification.ICLabel.classifications)

**Component types**:

- Brain: Neural activity
- Muscle: Muscle artifacts
- Eye: Eye movement artifacts
- Heart: Cardiac artifacts
- Line Noise: 50/60 Hz noise
- Channel Noise: Noisy channels
- Other: Unclassified

**Removing artifact components**:

.. code-block:: python

    # Remove muscle and eye components
    artifact_components = []
    for i, label in enumerate(eeg.etc.ic_classification.ICLabel.classes):
        if label in ['Muscle', 'Eye']:
            artifact_components.append(i)

    # Remove components
    eeg.icaact = None  # Clear cached ICA activity
    eeg = pop_select(eeg, 'nochannel', artifact_components)

Pipeline Visualization
======================

Here's a text-based flowchart of the preprocessing pipeline:

.. code-block:: text

    Raw EEG Data
         |
         v
    Channel Selection
         |
         v
    Flatline Detection
         |
         v
    Noisy Channel Removal
         |
         v
    Channel Interpolation
         |
         v
    Resampling
         |
         v
    High-Pass Filtering
         |
         v
    Low-Pass Filtering
         |
         v
    ICA Decomposition
         |
         v
    Component Classification (ICLabel)
         |
         v
    Artifact Component Removal
         |
         v
    Preprocessed EEG Data

Parameter Tuning
================

Key Parameters and Their Effects
---------------------------------

**Flatline Criterion**

- **Default**: 5 seconds
- **Lower values**: More aggressive channel removal
- **Higher values**: More lenient, may keep noisy channels
- **Recommendation**: 5 seconds for most applications

**ASR Criterion**

- **Default**: 20 (standard deviations)
- **Lower values**: More aggressive artifact removal
- **Higher values**: More lenient, may keep artifacts
- **Recommendation**: 20 for standard EEG, 15-25 for sensitive applications

**High-Pass Filter**

- **Default**: 1 Hz
- **Lower values**: Preserve slow oscillations
- **Higher values**: Remove more low-frequency noise
- **Recommendation**: 0.5-1 Hz for most applications

**Low-Pass Filter**

- **Default**: 100 Hz
- **Lower values**: Remove more high-frequency noise
- **Higher values**: Preserve high-frequency activity
- **Recommendation**: 100 Hz for standard EEG, 200 Hz for high-frequency analysis

**Resampling Rate**

- **Default**: 250 Hz
- **Lower values**: Smaller file size, faster processing
- **Higher values**: Better temporal resolution
- **Recommendation**: 250 Hz for most applications

Tuning Strategy
---------------

1. **Start with defaults**: Use the default parameters as a baseline
2. **Visualize results**: Plot the data before and after preprocessing
3. **Adjust parameters**: Modify parameters based on visual inspection
4. **Validate**: Check that preprocessing doesn't remove important signals
5. **Document**: Record the parameters used for reproducibility

Quality Control
===============

Assessing Preprocessing Quality
--------------------------------

Visual Inspection
~~~~~~~~~~~~~~~~~

Plot the data before and after preprocessing:

.. code-block:: python

    import matplotlib.pyplot as plt

    # Plot raw data
    plt.figure(figsize=(12, 6))
    plt.plot(eeg.data[0, :1000])
    plt.title('Raw EEG Data')
    plt.show()

    # Plot preprocessed data
    plt.figure(figsize=(12, 6))
    plt.plot(eeg.data[0, :1000])
    plt.title('Preprocessed EEG Data')
    plt.show()

Spectral Analysis
~~~~~~~~~~~~~~~~~

Compare power spectral density before and after preprocessing:

.. code-block:: python

    from eegprep import eeg_rpsd
    import matplotlib.pyplot as plt

    # Compute power spectral density
    psd = eeg_rpsd(eeg)

    plt.figure(figsize=(12, 6))
    plt.semilogy(psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (µV²/Hz)')
    plt.title('Power Spectral Density')
    plt.show()

Component Inspection
~~~~~~~~~~~~~~~~~~~~

Visualize ICA components:

.. code-block:: python

    from eegprep import topoplot
    import matplotlib.pyplot as plt

    # Plot component topographies
    topoplot(eeg, components=[0, 1, 2, 3])
    plt.title('ICA Component Topographies')
    plt.show()

    # Check component classifications
    if hasattr(eeg, 'etc') and 'ic_classification' in eeg.etc:
        classifications = eeg.etc.ic_classification.ICLabel.classifications
        for i, probs in enumerate(classifications):
            print(f"Component {i}: {probs}")

Data Loss Assessment
~~~~~~~~~~~~~~~~~~~~

Check how much data was removed:

.. code-block:: python

    # Check removed channels
    if hasattr(eeg, 'removed_channels'):
        print(f"Removed channels: {eeg.removed_channels}")

    # Check removed windows
    if hasattr(eeg, 'removed_windows'):
        print(f"Removed windows: {eeg.removed_windows}")

    # Calculate percentage of data retained
    if hasattr(eeg, 'removed_windows'):
        pct_retained = (1 - len(eeg.removed_windows) / eeg.pnts) * 100
        print(f"Data retained: {pct_retained:.1f}%")

Quality Metrics
~~~~~~~~~~~~~~~

Compute quality metrics:

.. code-block:: python

    # Signal-to-noise ratio
    from eegprep import eeg_rpsd

    psd = eeg_rpsd(eeg)
    snr = psd[1:50].mean() / psd[50:100].mean()
    print(f"SNR: {snr:.2f}")

    # Autocorrelation
    from eegprep import eeg_autocorr

    acf = eeg_autocorr(eeg, maxlag=100)
    print(f"Autocorrelation: {acf}")

Common Issues and Solutions
============================

Too Many Channels Removed
--------------------------

**Problem**: Preprocessing removes too many channels

**Solutions**:

1. Increase flatline criterion
2. Increase ASR criterion
3. Check data quality before preprocessing
4. Verify channel locations are correct

Too Few Artifacts Removed
--------------------------

**Problem**: Preprocessing doesn't remove enough artifacts

**Solutions**:

1. Decrease ASR criterion
2. Decrease flatline criterion
3. Apply additional filtering
4. Manually inspect and remove bad components

ICA Fails to Converge
---------------------

**Problem**: ICA decomposition doesn't converge

**Solutions**:

1. Increase max_iter parameter
2. Ensure data is properly filtered
3. Check for remaining artifacts
4. Try different ICA algorithm

Next Steps
==========

Now that you understand the preprocessing pipeline:

1. Read the :ref:`configuration` guide for advanced parameter tuning
2. Explore the :ref:`bids_workflow` for batch processing
3. Check the :ref:`advanced_topics` for custom pipelines
4. Review the :ref:`api_reference` for detailed function documentation
