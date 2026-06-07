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
   statistics
   io
   extensions
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

GUI and Session Entry Points
============================

.. autosummary::
   :toctree: generated/

   eegprep.select_multiple_datasets

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
   eegprep.timefreq
   eegprep.timef
   eegprep.crossf
   eegprep.pac
   eegprep.pac_cont
   eegprep.newtimefbaseln
   eegprep.newtimeftrialbaseln
   eegprep.newtimefitc
   eegprep.newtimefpowerunit
   eegprep.bootstat
   eegprep.correct_mc
   eegprep.correctfit
   eegprep.rsadjust
   eegprep.rsfit
   eegprep.rsget
   eegprep.rspdfsolv
   eegprep.rspfunc
   eegprep.signalstat
   eegprep.pop_newtimef
   eegprep.pop_newcrossf
   eegprep.pop_timef
   eegprep.pop_crossf
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
   eegprep.eegplot
   eegprep.eeg_multieegplot
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
   eegprep.pop_prop_extended
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

Extension SDK
=============

EEGPrep external extensions are Python packages discovered through the
``eegprep.extensions`` entry-point group. The registry validates declarative
specs and keeps extension callables lazy until a later runtime surface uses
them.

.. autosummary::
   :toctree: generated/

   eegprep.CATALOG_SCHEMA_VERSION
   eegprep.CatalogValidationIssue
   eegprep.CatalogValidationOptions
   eegprep.CatalogValidationReport
   eegprep.EXTENSION_COMPATIBILITY_POLICY
   eegprep.EXTENSION_CURATION_POLICY_URL
   eegprep.EXTENSION_NAMING_PREFIX
   eegprep.EXTENSION_TRUST_MESSAGE
   eegprep.ExtensionSpec
   eegprep.ExtensionRegistry
   eegprep.ExtensionRecord
   eegprep.ExtensionStatus
   eegprep.ExtensionAction
   eegprep.ExtensionPopFunction
   eegprep.ExtensionResource
   eegprep.ExtensionLoadError
   eegprep.ExtensionValidationResult
   eegprep.ExtensionTestHarness
   eegprep.LazyImport
   eegprep.assert_extension_entry_point_loads
   eegprep.check_extension_compatibility
   eegprep.discover_extensions
   eegprep.extension_version_satisfies
   eegprep.load_catalog_entries
   eegprep.validate_catalog_entries
   eegprep.validate_catalog_file
   eegprep.validate_extension_spec

STUDY Workflows
===============

The STUDY wrappers and helpers below cover the integrated standalone STUDY
workflow: metadata/design creation, study load/save, measure precompute,
measure plotting, component preclustering, clustering, and cluster editing.

.. autosummary::
   :toctree: generated/

   eegprep.pop_study
   eegprep.pop_studywizard
   eegprep.pop_studyerp
   eegprep.pop_loadstudy
   eegprep.pop_savestudy
   eegprep.pop_addindepvar
   eegprep.pop_importgroupvar
   eegprep.pop_listfactors
   eegprep.pop_precomp
   eegprep.pop_chanplot
   eegprep.pop_preclust
   eegprep.pop_clust
   eegprep.pop_clustedit
   eegprep.std_addvarlevel
   eegprep.std_builddesignmat
   eegprep.std_rebuilddesign
   eegprep.std_saveindvar
   eegprep.std_readdata
   eegprep.std_readerp
   eegprep.std_readspec
   eegprep.std_readersp
   eegprep.std_readitc
   eegprep.std_readtopo
   eegprep.std_readpac
   eegprep.std_pac
   eegprep.std_pacplot
   eegprep.std_checkfiles
   eegprep.std_checkdatasession
   eegprep.std_uniformfiles
   eegprep.std_uniformsetinds
   eegprep.std_erpplot
   eegprep.std_specplot
   eegprep.std_erspplot
   eegprep.std_itcplot
   eegprep.optimal_kmeans
   eegprep.robust_kmeans
   eegprep.std_apcluster
   eegprep.std_centroid
   eegprep.std_findoutlierclust

Configuration
==============

.. autosummary::
   :toctree: generated/

   eegprep.EEG_OPTIONS
