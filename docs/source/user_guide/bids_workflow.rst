.. _bids_workflow:

=============
BIDS Workflow
=============

This guide covers working with Brain Imaging Data Structure (BIDS) formatted datasets in eegprep. BIDS is a standardized format for organizing neuroimaging data, making it easier to share and process datasets consistently.

BIDS Dataset Structure
======================

A typical BIDS EEG dataset has the following structure:

.. code-block:: text

    dataset/
    ├── sub-01/
    │   ├── ses-01/
    │   │   └── eeg/
    │   │       ├── sub-01_ses-01_task-rest_eeg.set
    │   │       ├── sub-01_ses-01_task-rest_eeg.fdt
    │   │       ├── sub-01_ses-01_task-rest_channels.tsv
    │   │       ├── sub-01_ses-01_task-rest_eeg.json
    │   │       └── sub-01_ses-01_task-rest_events.tsv
    │   └── ses-02/
    │       └── eeg/
    │           └── ...
    ├── sub-02/
    │   └── ses-01/
    │       └── eeg/
    │           └── ...
    ├── derivatives/
    │   └── eegprep/
    │       ├── sub-01/
    │       │   └── ses-01/
    │       │       └── eeg/
    │       │           └── sub-01_ses-01_task-rest_eeg_preprocessed.set
    │       └── sub-02/
    │           └── ...
    ├── README
    ├── CHANGES
    ├── dataset_description.json
    ├── participants.tsv
    └── participants.json

Key BIDS Files:

- **_eeg.set**: EEGLAB format EEG data
- **_eeg.fdt**: EEGLAB data file (binary)
- **_channels.tsv**: Channel information (name, type, units)
- **_eeg.json**: EEG metadata (sampling rate, reference, etc.)
- **_events.tsv**: Event markers and timing
- **dataset_description.json**: Dataset metadata
- **participants.tsv**: Participant information

Loading BIDS Data
=================

Using pop_load_frombids
-----------------------

Load a single file from a BIDS dataset:

.. code-block:: python

    from eegprep import pop_load_frombids

    # Load a specific file
    eeg = pop_load_frombids(
        bids_root='data/bids_dataset',
        subject='01',
        session='01',
        task='rest'
    )

    print(f"Loaded: {eeg.nbchan} channels, {eeg.pnts} samples")
    print(f"Sampling rate: {eeg.srate} Hz")

**Parameters**:

- ``bids_root``: Path to the BIDS dataset root directory
- ``subject``: Subject ID (without 'sub-' prefix)
- ``session``: Session ID (optional, without 'ses-' prefix)
- ``task``: Task name (optional)
- ``run``: Run number (optional)

Loading with Additional Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from eegprep import pop_load_frombids

    # Load with specific run and additional options
    eeg = pop_load_frombids(
        bids_root='data/bids_dataset',
        subject='01',
        session='01',
        task='oddball',
        run='01',
        preload=True  # Load data into memory
    )

Listing Available Files
-----------------------

Find all EEG files in a BIDS dataset:

.. code-block:: python

    from eegprep import bids_list_eeg_files

    # List all EEG files
    files = bids_list_eeg_files('data/bids_dataset')

    for file_info in files:
        print(f"Subject: {file_info['subject']}")
        print(f"Session: {file_info['session']}")
        print(f"Task: {file_info['task']}")
        print(f"File: {file_info['file']}")
        print()

Running Batch Preprocessing
============================

Using bids_preproc
------------------

Process all files in a BIDS dataset with a single command:

.. code-block:: python

    from eegprep import bids_preproc

    # Run preprocessing on entire dataset
    bids_preproc(
        bids_root='data/bids_dataset',
        output_dir='data/bids_dataset/derivatives/eegprep',
        overwrite=False
    )

**Parameters**:

- ``bids_root``: Path to BIDS dataset root
- ``output_dir``: Output directory for preprocessed data
- ``overwrite``: Whether to overwrite existing files
- ``n_jobs``: Number of parallel jobs (default: 1)

Batch Processing with Custom Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from eegprep import bids_preproc

    # Custom preprocessing parameters
    bids_preproc(
        bids_root='data/bids_dataset',
        output_dir='data/bids_dataset/derivatives/eegprep',
        preproc_params={
            'flatline_criterion': 5,
            'highpass': 1,
            'lowpass': 100,
            'asr_criterion': 20,
            'ica': True,
            'iclabel': True
        },
        n_jobs=4  # Use 4 parallel jobs
    )

Parallel Processing
~~~~~~~~~~~~~~~~~~~

Process multiple subjects in parallel:

.. code-block:: python

    from eegprep import bids_preproc

    # Process with 8 parallel jobs
    bids_preproc(
        bids_root='data/bids_dataset',
        output_dir='data/bids_dataset/derivatives/eegprep',
        n_jobs=8,
        verbose=True
    )

**Note**: The number of jobs should not exceed the number of CPU cores available.

Processing Specific Subjects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from eegprep import bids_preproc

    # Process only specific subjects
    bids_preproc(
        bids_root='data/bids_dataset',
        output_dir='data/bids_dataset/derivatives/eegprep',
        subjects=['01', '02', '03']
    )

Output Structure
================

After running :func:`eegprep.bids_preproc`, the output is organized in the derivatives directory:

.. code-block:: text

    dataset/derivatives/eegprep/
    ├── sub-01/
    │   ├── ses-01/
    │   │   └── eeg/
    │   │       ├── sub-01_ses-01_task-rest_eeg_preprocessed.set
    │   │       ├── sub-01_ses-01_task-rest_eeg_preprocessed.fdt
    │   │       ├── sub-01_ses-01_task-rest_channels.tsv
    │   │       └── sub-01_ses-01_task-rest_eeg.json
    │   └── ses-02/
    │       └── eeg/
    │           └── ...
    ├── sub-02/
    │   └── ...
    ├── dataset_description.json
    └── README

Derivatives Format
------------------

The derivatives directory follows BIDS format with:

- **_preprocessed.set**: Preprocessed EEG data
- **_preprocessed.fdt**: Preprocessed data file
- **channels.tsv**: Updated channel information
- **eeg.json**: Updated metadata
- **dataset_description.json**: Derivatives dataset description

Loading Preprocessed Data
--------------------------

Load preprocessed data from derivatives:

.. code-block:: python

    from eegprep import pop_load_frombids

    # Load preprocessed data
    eeg = pop_load_frombids(
        bids_root='data/bids_dataset/derivatives/eegprep',
        subject='01',
        session='01',
        task='rest'
    )

Integration with Other Tools
=============================

Integration with MNE-Python
----------------------------

Convert eegprep data to MNE format:

.. code-block:: python

    from eegprep import eeg_eeg2mne
    import mne

    # Convert to MNE Raw object
    raw = eeg_eeg2mne(eeg)

    # Now use MNE functions
    raw.plot()
    raw.compute_psd().plot()

Converting Back to eegprep
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from eegprep import eeg_mne2eeg

    # Convert MNE Raw back to eegprep format
    eeg = eeg_mne2eeg(raw)

Integration with EEGLAB
-----------------------

Save preprocessed data in EEGLAB format:

.. code-block:: python

    from eegprep import pop_saveset

    # Save as EEGLAB .set file
    pop_saveset(eeg, 'preprocessed_data.set')

Load EEGLAB files:

.. code-block:: python

    from eegprep import pop_loadset

    # Load EEGLAB .set file
    eeg = pop_loadset('data.set')

Working with BIDS Metadata
===========================

Accessing Channel Information
------------------------------

.. code-block:: python

    from eegprep import pop_load_frombids

    eeg = pop_load_frombids(
        bids_root='data/bids_dataset',
        subject='01',
        session='01',
        task='rest'
    )

    # Access channel information
    for i, chan in enumerate(eeg.chanlocs):
        print(f"Channel {i}: {chan['labels']}")
        print(f"  Type: {chan['type']}")
        print(f"  Location: ({chan['X']}, {chan['Y']}, {chan['Z']})")

Accessing Event Information
----------------------------

.. code-block:: python

    # Access events
    if hasattr(eeg, 'event'):
        for event in eeg.event:
            print(f"Event type: {event['type']}")
            print(f"Latency: {event['latency']} samples")
            print(f"Duration: {event['duration']} samples")

Accessing Metadata
-------------------

.. code-block:: python

    # Access BIDS metadata
    if hasattr(eeg, 'etc') and 'bids' in eeg.etc:
        bids_info = eeg.etc.bids
        print(f"Task: {bids_info.get('task')}")
        print(f"Sampling rate: {bids_info.get('srate')} Hz")

Common BIDS Workflows
=====================

Complete Preprocessing Workflow
-------------------------------

.. code-block:: python

    from eegprep import (
        pop_load_frombids,
        clean_artifacts,
        iclabel,
        pop_saveset
    )

    # 1. Load data
    eeg = pop_load_frombids(
        bids_root='data/bids_dataset',
        subject='01',
        session='01',
        task='rest'
    )

    # 2. Preprocess
    eeg = clean_artifacts(
        eeg,
        highpass=1,
        lowpass=100,
        ica=True,
        iclabel=True
    )

    # 3. Save to derivatives
    pop_saveset(
        eeg,
        'data/bids_dataset/derivatives/eegprep/sub-01/ses-01/eeg/sub-01_ses-01_task-rest_eeg_preprocessed.set'
    )

Batch Processing with Quality Control
--------------------------------------

.. code-block:: python

    from eegprep import bids_preproc, bids_list_eeg_files
    import json

    # 1. List all files
    files = bids_list_eeg_files('data/bids_dataset')
    print(f"Found {len(files)} EEG files")

    # 2. Run preprocessing
    bids_preproc(
        bids_root='data/bids_dataset',
        output_dir='data/bids_dataset/derivatives/eegprep',
        n_jobs=4
    )

    # 3. Create processing report
    report = {
        'total_files': len(files),
        'preprocessing_date': '2024-01-01',
        'parameters': {
            'highpass': 1,
            'lowpass': 100,
            'ica': True
        }
    }

    with open('preprocessing_report.json', 'w') as f:
        json.dump(report, f, indent=2)

Troubleshooting BIDS Workflows
==============================

File Not Found
--------------

**Problem**: ``FileNotFoundError`` when loading BIDS data

**Solution**:

1. Verify BIDS dataset structure
2. Check subject and session IDs
3. Use :func:`eegprep.bids_list_eeg_files` to find available files

.. code-block:: python

    from eegprep import bids_list_eeg_files

    files = bids_list_eeg_files('data/bids_dataset')
    for f in files:
        print(f"sub-{f['subject']}_ses-{f['session']}_task-{f['task']}")

Invalid BIDS Format
-------------------

**Problem**: Data doesn't conform to BIDS standard

**Solution**:

1. Validate BIDS dataset using the BIDS Validator
2. Check dataset_description.json
3. Verify file naming conventions

Parallel Processing Errors
---------------------------

**Problem**: Errors when using ``n_jobs > 1``

**Solution**:

1. Start with ``n_jobs=1`` to identify the issue
2. Check for file locking issues
3. Ensure output directory is writable
4. Reduce ``n_jobs`` if system resources are limited

Memory Issues
-------------

**Problem**: Out of memory errors during batch processing

**Solution**:

1. Reduce ``n_jobs`` to process fewer files in parallel
2. Process subjects in smaller batches
3. Increase available system RAM
4. Use a machine with more memory

Best Practices
==============

1. **Validate BIDS format**: Use the BIDS Validator before processing
2. **Backup original data**: Keep a copy of raw data before preprocessing
3. **Document parameters**: Record preprocessing parameters in a configuration file
4. **Quality control**: Visually inspect preprocessed data
5. **Version control**: Track eegprep version used for reproducibility
6. **Parallel processing**: Use ``n_jobs`` to speed up batch processing
7. **Monitor progress**: Use ``verbose=True`` to track processing status

Next Steps
==========

Now that you understand BIDS workflows:

1. Read the :ref:`preprocessing_pipeline` guide for detailed preprocessing steps
2. Explore the :ref:`configuration` guide for parameter tuning
3. Check the :ref:`advanced_topics` for custom pipelines
4. Review the :ref:`api_reference` for detailed function documentation
