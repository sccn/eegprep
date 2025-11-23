.. _faq:

==========================
Frequently Asked Questions
==========================

Installation FAQ
================

What Python versions does EEGPrep support?
------------------------------------------

EEGPrep supports Python 3.8 and higher. We recommend using Python 3.9 or 3.10 for the best compatibility with all dependencies.

How do I install EEGPrep?
-------------------------

The easiest way is to use pip:

.. code-block:: bash

    pip install eegprep

For development installation from source:

.. code-block:: bash

    git clone https://github.com/NeuroTechX/eegprep.git
    cd eegprep
    pip install -e ".[dev]"

What are the system requirements?
---------------------------------

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB+ recommended for large datasets)
- **Disk Space**: 500MB for installation and dependencies

Can I use EEGPrep on Windows?
-----------------------------

Yes, EEGPrep works on Windows. However, some optional features may require additional setup. We recommend using Windows Subsystem for Linux (WSL) for better compatibility.

What if I get dependency conflicts?
-----------------------------------

Try updating pip and reinstalling:

.. code-block:: bash

    pip install --upgrade pip
    pip install --force-reinstall eegprep

Or create a fresh virtual environment:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate
    pip install eegprep

Does EEGPrep support GPU acceleration?
--------------------------------------

EEGPrep can leverage GPU acceleration through MNE-Python and PyTorch for ICA and other computations. Install GPU support:

.. code-block:: bash

    pip install torch  # For GPU support
    pip install mne[cuda]  # For MNE GPU support

Usage FAQ
=========

How do I load EEG data?
-----------------------

EEGPrep supports multiple formats:

.. code-block:: python

    import eegprep
    
    # Load EEGLAB .set file
    eeg = eegprep.EEGobj.load('data.set')
    
    # Load from BIDS dataset
    eeg = eegprep.pop_load_frombids('/path/to/bids', 'sub-001')
    
    # Load from MNE-Python
    import mne
    raw = mne.io.read_raw_edf('data.edf')
    eeg = eegprep.eeg_mne2eeg(raw)

What data formats are supported?
--------------------------------

EEGPrep supports:

- **EEGLAB**: .set and .fdt files
- **BIDS**: Brain Imaging Data Structure format
- **MNE-Python**: Raw and Epochs objects
- **EDF**: European Data Format
- **BrainVision**: .vhdr, .vmrk, .eeg files
- **Neuroscan**: .cnt files

How do I apply preprocessing?
-----------------------------

Apply preprocessing steps in sequence:

.. code-block:: python

    import eegprep
    
    # Load data
    eeg = eegprep.EEGobj.load('data.set')
    
    # Apply preprocessing pipeline
    eeg = eegprep.clean_flatlines(eeg)
    eeg = eegprep.clean_channels(eeg)
    eeg = eegprep.clean_artifacts(eeg)
    eeg = eegprep.clean_drifts(eeg)
    
    # Save processed data
    eeg.save('data_processed.set')

How do I save processed data?
-----------------------------

Save in EEGLAB format:

.. code-block:: python

    eeg.save('data_processed.set')

Save in HDF5 format:

.. code-block:: python

    eeg.save('data_processed.h5')

Export to MNE-Python:

.. code-block:: python

    raw = eegprep.eeg_eeg2mne(eeg)
    raw.save('data_processed_raw.fif')

Can I use EEGPrep with MNE-Python?
----------------------------------

Yes! EEGPrep integrates seamlessly with MNE-Python:

.. code-block:: python

    import eegprep
    import mne
    
    # Load with MNE
    raw = mne.io.read_raw_edf('data.edf')
    
    # Convert to EEGPrep
    eeg = eegprep.eeg_mne2eeg(raw)
    
    # Process with EEGPrep
    eeg = eegprep.clean_artifacts(eeg)
    
    # Convert back to MNE
    raw = eegprep.eeg_eeg2mne(eeg)

How do I work with BIDS datasets?
---------------------------------

Load and process BIDS data:

.. code-block:: python

    import eegprep
    
    # Load from BIDS
    eeg = eegprep.pop_load_frombids('/path/to/bids', 'sub-001', 'ses-01')
    
    # Process
    eeg = eegprep.clean_artifacts(eeg)
    
    # Save back to BIDS
    eeg.save_bids('/path/to/bids', 'sub-001', 'ses-01')

Performance FAQ
===============

Why is preprocessing slow?
--------------------------

Preprocessing speed depends on:

- **Data size**: Larger datasets take longer
- **Sampling rate**: Higher sampling rates require more computation
- **Number of channels**: More channels = more computation
- **Algorithm complexity**: Some algorithms (ICA, ASR) are computationally intensive

To speed up processing:

1. Downsample data if appropriate
2. Use GPU acceleration
3. Process in parallel (for multiple subjects)
4. Use faster algorithms (e.g., ASR instead of ICA)

How much memory does EEGPrep use?
---------------------------------

Memory usage depends on:

- **Data size**: Roughly 8 bytes per sample per channel
- **Number of channels**: More channels = more memory
- **Sampling rate**: Higher rates = more samples = more memory

Example: 64 channels, 500 Hz sampling, 1 hour of data ≈ 1.2 GB

To reduce memory usage:

- Downsample data
- Process shorter segments
- Use memory-efficient algorithms

Can I process data in parallel?
-------------------------------

Yes, you can process multiple subjects in parallel:

.. code-block:: python

    from multiprocessing import Pool
    import eegprep
    
    def process_subject(subject_id):
        eeg = eegprep.pop_load_frombids('/bids', subject_id)
        eeg = eegprep.clean_artifacts(eeg)
        return eeg
    
    with Pool(4) as p:
        results = p.map(process_subject, ['sub-001', 'sub-002', 'sub-003'])

How can I optimize preprocessing?
---------------------------------

Tips for optimization:

1. **Choose appropriate parameters**: Use defaults as starting point
2. **Skip unnecessary steps**: Only apply needed preprocessing
3. **Use faster algorithms**: ASR is faster than ICA
4. **Downsample if appropriate**: Reduces computation
5. **Use GPU acceleration**: For ICA and other algorithms
6. **Process in batches**: More efficient than one-by-one

Troubleshooting FAQ
===================

I get "ModuleNotFoundError: No module named 'eegprep'"
------------------------------------------------------

**Solution**: Install EEGPrep:

.. code-block:: bash

    pip install eegprep

Or if developing from source:

.. code-block:: bash

    pip install -e .

I get "ValueError: Data shape mismatch"
---------------------------------------

**Cause**: Data dimensions don't match expected format

**Solution**: Check data shape:

.. code-block:: python

    print(eeg.data.shape)  # Should be (channels, samples)
    print(eeg.nbchan)  # Number of channels
    print(eeg.pnts)  # Number of samples

I get "RuntimeError: CUDA out of memory"
----------------------------------------

**Cause**: GPU memory exhausted

**Solutions**:

1. Use CPU instead:

.. code-block:: python

    eeg = eegprep.clean_artifacts(eeg, use_gpu=False)

2. Process smaller segments
3. Reduce batch size
4. Upgrade GPU memory

My data has NaN values
----------------------

**Solution**: Handle NaN values:

.. code-block:: python

    import numpy as np
    
    # Remove NaN values
    eeg.data = np.nan_to_num(eeg.data)
    
    # Or interpolate
    eeg = eegprep.eeg_interp(eeg)

How do I debug preprocessing issues?
------------------------------------

Enable logging:

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Now run preprocessing with debug output
    eeg = eegprep.clean_artifacts(eeg)

Check data at each step:

.. code-block:: python

    print(f"Original shape: {eeg.data.shape}")
    eeg = eegprep.clean_flatlines(eeg)
    print(f"After flatlines: {eeg.data.shape}")
    eeg = eegprep.clean_channels(eeg)
    print(f"After channels: {eeg.data.shape}")

Comparison FAQ
==============

How does EEGPrep compare to EEGLAB?
-----------------------------------

**EEGPrep**:

- Python-based (easier integration with modern tools)
- Open-source and actively maintained
- Scriptable and reproducible
- Good for batch processing

**EEGLAB**:

- MATLAB-based (established in neuroscience)
- Extensive GUI
- Large community and plugins
- Better for interactive exploration

**Recommendation**: Use EEGPrep for reproducible pipelines, EEGLAB for interactive exploration.

How does EEGPrep compare to MNE-Python?
---------------------------------------

**EEGPrep**:

- Specialized for EEG preprocessing
- EEGLAB-compatible
- Comprehensive artifact removal
- BIDS-native support

**MNE-Python**:

- General neuroimaging (EEG, MEG, fMRI)
- Extensive analysis tools
- Large community
- Better for source localization

**Recommendation**: Use EEGPrep for preprocessing, MNE for analysis.

How does EEGPrep compare to Fieldtrip?
--------------------------------------

**EEGPrep**:

- Python-based
- Modern, actively maintained
- Good for batch processing
- BIDS support

**Fieldtrip**:

- MATLAB-based
- Established in neuroscience
- Extensive documentation
- Good for MEG and EEG

**Recommendation**: Use EEGPrep for Python workflows, Fieldtrip for MATLAB workflows.

Data Format FAQ
===============

What is BIDS?
-------------

BIDS (Brain Imaging Data Structure) is a standard for organizing neuroimaging data. It ensures:

- Consistency across datasets
- Reproducibility
- Easy sharing
- Automated processing

Learn more: `BIDS Documentation <https://bids-standard.github.io/>`_

How do I convert data to BIDS?
------------------------------

Use the BIDS converter:

.. code-block:: python

    import eegprep
    
    eeg = eegprep.EEGobj.load('data.set')
    eeg.save_bids('/path/to/bids', 'sub-001', 'ses-01')

What's the difference between .set and .fdt files?
--------------------------------------------------

- **.set**: EEGLAB header file (contains metadata)
- **.fdt**: EEGLAB data file (contains actual EEG data)

Both are needed for complete EEGLAB datasets.

Can I use EEGPrep with other data formats?
------------------------------------------

Yes, EEGPrep supports:

- EDF (European Data Format)
- BrainVision (.vhdr, .vmrk, .eeg)
- Neuroscan (.cnt)
- MNE-Python formats

Convert between formats:

.. code-block:: python

    import eegprep
    import mne
    
    # Load from any MNE-supported format
    raw = mne.io.read_raw('data.edf')
    
    # Convert to EEGPrep
    eeg = eegprep.eeg_mne2eeg(raw)
    
    # Save in EEGLAB format
    eeg.save('data.set')

Getting Help
============

- Check the :doc:`user_guide/index`
- Review :doc:`examples/index`
- Search `GitHub Issues <https://github.com/NeuroTechX/eegprep/issues>`_
- Ask in GitHub Discussions
- Contact the maintainers

Still have questions? Open an issue on GitHub!
