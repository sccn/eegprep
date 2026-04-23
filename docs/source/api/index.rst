.. _api_reference:

=============
API Reference
=============

This section contains the complete API documentation for eegprep. The API is organized into logical modules covering core functionality, preprocessing, independent component analysis, signal processing, input/output operations, and utility functions.

.. toctree::
   :maxdepth: 2

   core
   preprocessing
   ica
   signal_processing
   io
   utils

Core Classes
============

.. autosummary::
   :toctree: generated/

   eegprep.EEGobj

Data Loading and Saving
========================

.. autosummary::
   :toctree: generated/

   eegprep.pop_loadset
   eegprep.loadset
   eegprep.pop_loadset_h5
   eegprep.pop_saveset
   eegprep.pop_load_frombids

Preprocessing Functions
=======================

Artifact Removal
----------------

.. autosummary::
   :toctree: generated/

   eegprep.clean_artifacts
   eegprep.clean_asr
   eegprep.clean_flatlines
   eegprep.clean_drifts
   eegprep.clean_windows

Channel Operations
------------------

.. autosummary::
   :toctree: generated/

   eegprep.clean_channels
   eegprep.clean_channels_nolocs
   eegprep.eeg_interp
   eegprep.pop_reref

Signal Processing
-----------------

.. autosummary::
   :toctree: generated/

   eegprep.pop_resample
   eegprep.pop_eegfiltnew
   eegprep.eeg_picard

Independent Component Analysis
===============================

.. autosummary::
   :toctree: generated/

   eegprep.iclabel
   eegprep.ICL_feature_extractor

Spectral Analysis
=================

.. autosummary::
   :toctree: generated/

   eegprep.eeg_rpsd
   eegprep.eeg_autocorr
   eegprep.eeg_autocorr_welch
   eegprep.eeg_autocorr_fftw

Epoching and Selection
======================

.. autosummary::
   :toctree: generated/

   eegprep.pop_epoch
   eegprep.pop_select
   eegprep.eeg_eegrej
   eegprep.eegrej

Visualization
=============

.. autosummary::
   :toctree: generated/

   eegprep.topoplot

Format Conversion
=================

.. autosummary::
   :toctree: generated/

   eegprep.eeg_mne2eeg
   eegprep.eeg_mne2eeg_epochs
   eegprep.eeg_eeg2mne

Utilities
=========

.. autosummary::
   :toctree: generated/

   eegprep.eeg_checkset
   eegprep.eeg_compare
   eegprep.eeg_decodechan
   eegprep.eeg_lat2point
   eegprep.eeg_point2lat
   eegprep.bids_list_eeg_files
   eegprep.bids_preproc

BIDS Pipeline
=============

.. autosummary::
   :toctree: generated/

   eegprep.bids_preproc
   eegprep.bids_list_eeg_files
   eegprep.pop_load_frombids

Configuration
==============

.. autosummary::
   :toctree: generated/

   eegprep.EEG_OPTIONS
