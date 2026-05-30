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
   eegprep.EEGPrepSession

Interactive GUI and Console
===========================

.. autosummary::
   :toctree: generated/

   eegprep.gui
   eegprep.eeglab
   eegprep.EEGPrepConsoleWorkspace
   eegprep.ConsolePopResult
   eegprep.ConsoleDatasetResult
   eegprep.inputgui
   eegprep.listdlg2
   eegprep.pophelp
   eegprep.DialogSpec
   eegprep.ControlSpec
   eegprep.CallbackSpec

Dataset Workspace Helpers
=========================

.. autosummary::
   :toctree: generated/

   eegprep.eeg_emptyset
   eegprep.eeg_store
   eegprep.eeg_retrieve
   eegprep.pop_newset
   eegprep.pop_delset
   eegprep.pop_editoptions

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
   eegprep.pop_interp
   eegprep.pop_reref

Signal Processing
-----------------

.. autosummary::
   :toctree: generated/

   eegprep.pop_resample
   eegprep.pop_eegfiltnew
   eegprep.eeg_amica
   eegprep.eeg_picard
   eegprep.eeg_runica

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
   eegprep.newtimef
   eegprep.newcrossf
   eegprep.signalstat
   eegprep.pop_newtimef
   eegprep.pop_newcrossf
   eegprep.pop_signalstat
   eegprep.pop_eventstat

Epoching and Selection
======================

.. autosummary::
   :toctree: generated/

   eegprep.pop_adjustevents
   eegprep.pop_epoch
   eegprep.pop_select
   eegprep.eeg_eegrej
   eegprep.eegrej

Visualization
=============

.. autosummary::
   :toctree: generated/

   eegprep.cart2topo
   eegprep.pop_topoplot
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
   eegprep.pop_importbids
   eegprep.pop_exportbids

Bundled Plugins
===============

EEGPrep exposes metadata for bundled in-repo plugin ports. External EEGLAB
plugin install/update/remove workflows are intentionally outside the public API
for now.

.. autosummary::
   :toctree: generated/

   eegprep.bundled_plugins
   eegprep.plugin_status
   eegprep.plugin_menu
   eegprep.format_plugin_menu
   eegprep.pop_clean_rawdata
   eegprep.pop_iclabel
   eegprep.pop_icflag
   eegprep.pop_viewprops
   eegprep.pop_eegfiltnew
   eegprep.pop_firws
   eegprep.pop_firpm
   eegprep.pop_firma
   eegprep.firws
   eegprep.firwsord
   eegprep.pop_dipfit_settings
   eegprep.pop_dipplot
   eegprep.pop_dipfit_headmodel
   eegprep.pop_dipfit_gridsearch
   eegprep.pop_dipfit_nonlinear
   eegprep.pop_multifit
   eegprep.pop_leadfield
   eegprep.pop_dipfit_loreta

STUDY Workflows
===============

The current STUDY wrappers are exported for parity with EEGLAB menu workflows.
Phase 5a owns the final STUDY data/session contracts, so avoid treating helper
objects beyond these wrappers as stable until that phase lands.

.. autosummary::
   :toctree: generated/

   eegprep.pop_study
   eegprep.pop_studywizard
   eegprep.pop_studyerp
   eegprep.pop_loadstudy
   eegprep.pop_savestudy
   eegprep.pop_chanplot

Configuration
==============

.. autosummary::
   :toctree: generated/

   eegprep.EEG_OPTIONS
