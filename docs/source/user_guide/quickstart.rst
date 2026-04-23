.. _quickstart:

===========
Quick Start
===========

This guide will get you up and running with eegprep in just a few minutes. We'll cover the basic workflow for loading, preprocessing, and saving EEG data.

Basic Preprocessing (5-Minute Example)
======================================

Here's a complete example that demonstrates the core eegprep workflow:

.. code-block:: python

    import eegprep
    from eegprep import pop_loadset, pop_saveset, clean_artifacts, iclabel

    # Load EEG data
    eeg = pop_loadset('sample_data.set')
    print(f"Loaded EEG with {eeg.nbchan} channels and {eeg.pnts} points")

    # Run preprocessing
    eeg = clean_artifacts(eeg)
    print("Artifacts cleaned")

    # Save results
    pop_saveset(eeg, 'sample_data_preprocessed.set')
    print("Data saved")

Loading EEG Data
================

Using pop_loadset
-----------------

The :func:`eegprep.pop_loadset` function loads EEGLAB .set files:

.. code-block:: python

    from eegprep import pop_loadset

    # Load a .set file
    eeg = pop_loadset('data/subject_01.set')

    # Access basic information
    print(f"Channels: {eeg.nbchan}")
    print(f"Sampling rate: {eeg.srate} Hz")
    print(f"Duration: {eeg.pnts / eeg.srate} seconds")
    print(f"Channel names: {eeg.chanlocs}")

**Expected Output:**

.. code-block:: text

    Channels: 64
    Sampling rate: 500 Hz
    Duration: 120.0 seconds
    Channel names: [Fp1, Fp2, F3, F4, ...]

Loading from BIDS Format
-------------------------

For BIDS-formatted datasets, use :func:`eegprep.pop_load_frombids`:

.. code-block:: python

    from eegprep import pop_load_frombids

    # Load from BIDS dataset
    eeg = pop_load_frombids(
        bids_root='data/bids_dataset',
        subject='01',
        session='01',
        task='rest'
    )

Running Preprocessing
======================

Basic Artifact Removal
----------------------

The :func:`eegprep.clean_artifacts` function performs comprehensive artifact removal:

.. code-block:: python

    from eegprep import clean_artifacts

    # Run artifact removal with default settings
    eeg = clean_artifacts(eeg)
    print("Preprocessing complete")

    # Check what was removed
    print(f"Channels removed: {eeg.removed_channels}")
    print(f"Windows rejected: {eeg.removed_windows}")

**Expected Output:**

.. code-block:: text

    Preprocessing complete
    Channels removed: []
    Windows rejected: 12

Advanced Preprocessing with Custom Parameters
----------------------------------------------

Customize the preprocessing pipeline:

.. code-block:: python

    from eegprep import clean_artifacts

    # Custom preprocessing parameters
    eeg = clean_artifacts(
        eeg,
        flatline_criterion=5,  # Flatline detection threshold
        highpass=1,            # High-pass filter at 1 Hz
        lowpass=100,           # Low-pass filter at 100 Hz
        asr_criterion=20,      # ASR threshold
        ica=True,              # Enable ICA
        iclabel=True           # Enable ICLabel classification
    )

Step-by-Step Preprocessing
---------------------------

For more control, apply preprocessing steps individually:

.. code-block:: python

    from eegprep import (
        clean_flatlines,
        clean_channels,
        pop_resample,
        pop_eegfiltnew,
        eeg_picard,
        iclabel
    )

    # 1. Remove flatline channels
    eeg = clean_flatlines(eeg, flatline_criterion=5)
    print(f"Channels after flatline removal: {eeg.nbchan}")

    # 2. Remove noisy channels
    eeg = clean_channels(eeg)
    print(f"Channels after noise removal: {eeg.nbchan}")

    # 3. Resample if needed
    eeg = pop_resample(eeg, 250)  # Resample to 250 Hz
    print(f"New sampling rate: {eeg.srate} Hz")

    # 4. Filter the data
    eeg = pop_eegfiltnew(eeg, locutoff=1, hicutoff=100)
    print("Data filtered")

    # 5. Run ICA
    eeg = eeg_picard(eeg)
    print(f"ICA components: {eeg.icaweights.shape[0]}")

    # 6. Classify components with ICLabel
    eeg = iclabel(eeg)
    print("Components classified")

Saving Results
==============

Using pop_saveset
-----------------

Save preprocessed data back to EEGLAB format:

.. code-block:: python

    from eegprep import pop_saveset

    # Save to .set file
    pop_saveset(eeg, 'data/subject_01_preprocessed.set')
    print("Data saved successfully")

Saving with Compression
------------------------

Save with compression to reduce file size:

.. code-block:: python

    from eegprep import pop_saveset

    # Save with compression
    pop_saveset(
        eeg,
        'data/subject_01_preprocessed.set',
        savemode='onefile'  # Save as single file
    )

Saving to HDF5 Format
---------------------

For large datasets, save to HDF5 format:

.. code-block:: python

    from eegprep import pop_saveset

    # Save to HDF5
    pop_saveset(
        eeg,
        'data/subject_01_preprocessed.h5',
        fmt='h5'
    )

Visualization
=============

Topographic Plots
------------------

Visualize channel locations and data:

.. code-block:: python

    from eegprep import topoplot
    import matplotlib.pyplot as plt

    # Plot channel locations
    topoplot(eeg)
    plt.title('Channel Locations')
    plt.show()

    # Plot component topographies
    topoplot(eeg, components=[0, 1, 2, 3])
    plt.title('ICA Component Topographies')
    plt.show()

Plotting Preprocessed Data
---------------------------

Visualize the preprocessed signal:

.. code-block:: python

    import matplotlib.pyplot as plt

    # Plot first 5 seconds of data
    start_sample = 0
    end_sample = int(eeg.srate * 5)  # 5 seconds

    plt.figure(figsize=(12, 8))
    for ch in range(min(10, eeg.nbchan)):  # Plot first 10 channels
        plt.plot(eeg.data[ch, start_sample:end_sample] + ch * 100)
    plt.xlabel('Sample')
    plt.ylabel('Channel')
    plt.title('Preprocessed EEG Data (First 5 seconds)')
    plt.show()

Complete Workflow Example
=========================

Here's a complete example combining all steps:

.. code-block:: python

    from eegprep import (
        pop_loadset,
        pop_saveset,
        clean_artifacts,
        iclabel,
        topoplot
    )
    import matplotlib.pyplot as plt

    # 1. Load data
    print("Loading data...")
    eeg = pop_loadset('raw_data.set')
    print(f"Loaded: {eeg.nbchan} channels, {eeg.pnts} samples")

    # 2. Preprocess
    print("Preprocessing...")
    eeg = clean_artifacts(
        eeg,
        highpass=1,
        lowpass=100,
        ica=True,
        iclabel=True
    )
    print("Preprocessing complete")

    # 3. Visualize
    print("Visualizing...")
    topoplot(eeg)
    plt.show()

    # 4. Save
    print("Saving...")
    pop_saveset(eeg, 'preprocessed_data.set')
    print("Done!")

Common Tasks
============

Selecting Specific Channels
----------------------------

.. code-block:: python

    from eegprep import pop_select

    # Select only EEG channels (exclude EOG, EMG, etc.)
    eeg = pop_select(eeg, 'type', 'EEG')

    # Select specific channels by name
    eeg = pop_select(eeg, 'channel', ['Cz', 'Pz', 'Oz'])

Epoching Data
-------------

.. code-block:: python

    from eegprep import pop_epoch

    # Epoch data around event markers
    eeg = pop_epoch(eeg, [1, 2, 3], [-1, 2])  # Events 1,2,3; -1 to 2 seconds
    print(f"Epochs: {eeg.trials}")

Re-referencing
---------------

.. code-block:: python

    from eegprep import pop_reref

    # Re-reference to average
    eeg = pop_reref(eeg, [])  # Empty list = average reference

    # Re-reference to specific channel
    eeg = pop_reref(eeg, 32)  # Reference to channel 32

Next Steps
==========

Now that you understand the basics:

1. Read the :ref:`preprocessing_pipeline` guide for detailed information about each preprocessing step
2. Explore the :ref:`bids_workflow` for batch processing
3. Check the :ref:`configuration` guide for advanced parameter tuning
4. Review the :ref:`advanced_topics` for custom pipelines and optimization

For detailed API documentation, see the :ref:`api_reference`.
