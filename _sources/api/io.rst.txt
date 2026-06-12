.. _api_io:

=======================
Input/Output Functions
=======================

This section documents the input/output functions for loading and saving EEG data in various formats.

BIDS Loading
============

.. autofunction:: eegprep.pop_load_frombids
   :no-index:

.. autofunction:: eegprep.pop_importbids
   :no-index:

.. autofunction:: eegprep.pop_exportbids
   :no-index:

Generic Import
==============

.. autofunction:: eegprep.pop_importdata
   :no-index:

.. autofunction:: eegprep.pop_fileio
   :no-index:

.. autofunction:: eegprep.pop_biosig
   :no-index:

.. autofunction:: eegprep.pop_importevent
   :no-index:

.. autofunction:: eegprep.pop_importepoch
   :no-index:

.. autofunction:: eegprep.pop_chanevent
   :no-index:

.. autofunction:: eegprep.pop_importpres
   :no-index:

.. autofunction:: eegprep.pop_importerplab
   :no-index:

Channel Locations
=================

.. autofunction:: eegprep.pop_readlocs
   :no-index:

.. autofunction:: eegprep.pop_writelocs
   :no-index:

.. autofunction:: eegprep.readlocs
   :no-index:

.. autofunction:: eegprep.writelocs
   :no-index:

.. autofunction:: eegprep.convertlocs
   :no-index:

.. autofunction:: eegprep.chancenter
   :no-index:

.. autofunction:: eegprep.pop_chancenter
   :no-index:

.. autofunction:: eegprep.pop_chancoresp
   :no-index:

.. autofunction:: eegprep.readegilocs
   :no-index:

.. autofunction:: eegprep.readelp
   :no-index:

.. autofunction:: eegprep.readeetraklocs
   :no-index:

Long-Tail Import Helpers
========================

.. autofunction:: eegprep.pop_loadbci
   :no-index:

.. autofunction:: eegprep.pop_snapread
   :no-index:

.. autofunction:: eegprep.snapread
   :no-index:

.. autofunction:: eegprep.floatread
   :no-index:

.. autofunction:: eegprep.floatwrite
   :no-index:

EEGLAB Format
=============

.. autofunction:: eegprep.pop_loadset
   :no-index:

.. autofunction:: eegprep.pop_loadset_h5
   :no-index:

.. autofunction:: eegprep.pop_saveset
   :no-index:

``pop_saveset(..., savemode="twofiles")`` writes a ``.set`` header plus
``.fdt`` float32 sidecar. ``EEG_OPTIONS["option_savetwofiles"]``
uses that layout by default, and ``EEG_OPTIONS["option_memmapdata"]`` makes
``pop_loadset`` expose sidecar data through a NumPy-compatible memory map. See
:ref:`large_dataset_storage` for storedisk session behavior and limitations.

Text And External Export
========================

.. autofunction:: eegprep.pop_export
   :no-index:

``pop_export`` supports text export options including ICA export,
time/electrode rows, transpose, ERP averaging, precision, separator, and a
standalone numeric ``expr`` transform applied to the exported array ``x``.
Most expression function calls are positional; ``clip`` and ``nan_to_num`` also
accept documented safe numeric keywords. Power operators require small constant
exponents.

.. autofunction:: eegprep.pop_expica
   :no-index:

.. autofunction:: eegprep.pop_expevents
   :no-index:

.. autofunction:: eegprep.pop_writeeeg
   :no-index:

History And STUDY Files
=======================

.. autofunction:: eegprep.pop_saveh
   :no-index:

.. autofunction:: eegprep.pop_runscript
   :no-index:

.. autofunction:: eegprep.pop_study
   :no-index:

.. autofunction:: eegprep.pop_studywizard
   :no-index:

.. autofunction:: eegprep.pop_studyerp
   :no-index:

.. autofunction:: eegprep.pop_studydesign
   :no-index:

.. autofunction:: eegprep.pop_loadstudy
   :no-index:

.. autofunction:: eegprep.pop_savestudy
   :no-index:

.. autofunction:: eegprep.pop_precomp
   :no-index:

.. autofunction:: eegprep.pop_chanplot
   :no-index:

.. autofunction:: eegprep.std_editset
   :no-index:

.. autofunction:: eegprep.std_checkset
   :no-index:

.. autofunction:: eegprep.std_checkdatasetinfo
   :no-index:

.. autofunction:: eegprep.std_checkconsist
   :no-index:

.. autofunction:: eegprep.std_checkdesign
   :no-index:

.. autofunction:: eegprep.std_makedesign
   :no-index:

.. autofunction:: eegprep.std_addvarlevel
   :no-index:

.. autofunction:: eegprep.std_builddesignmat
   :no-index:

.. autofunction:: eegprep.std_limodesign
   :no-index:

.. autofunction:: eegprep.std_getindvar
   :no-index:

.. autofunction:: eegprep.std_indvarmatch
   :no-index:

.. autofunction:: eegprep.std_selectdataset
   :no-index:

.. autofunction:: eegprep.std_gettrialsind
   :no-index:

.. autofunction:: eegprep.std_maketrialinfo
   :no-index:

.. autofunction:: eegprep.std_combtrialinfo
   :no-index:

.. autofunction:: eegprep.std_rebuilddesign
   :no-index:

.. autofunction:: eegprep.std_saveindvar
   :no-index:

.. autofunction:: eegprep.pop_addindepvar
   :no-index:

.. autofunction:: eegprep.pop_importgroupvar
   :no-index:

.. autofunction:: eegprep.pop_listfactors
   :no-index:

.. autofunction:: eegprep.std_precomp
   :no-index:

.. autofunction:: eegprep.std_readdata
   :no-index:

.. autofunction:: eegprep.std_readerp
   :no-index:

.. autofunction:: eegprep.std_readspec
   :no-index:

.. autofunction:: eegprep.std_readersp
   :no-index:

.. autofunction:: eegprep.std_readitc
   :no-index:

.. autofunction:: eegprep.std_readtopo
   :no-index:

.. autofunction:: eegprep.std_readpac
   :no-index:

.. autofunction:: eegprep.std_pac
   :no-index:

.. autofunction:: eegprep.std_pacplot
   :no-index:

.. autofunction:: eegprep.std_prepare_neighbors
   :no-index:

.. autofunction:: eegprep.std_interp
   :no-index:

.. autofunction:: eegprep.std_dipplot
   :no-index:

.. autofunction:: eegprep.std_dipoleclusters
   :no-index:

.. autofunction:: eegprep.std_savedat
   :no-index:

.. autofunction:: eegprep.std_checkfiles
   :no-index:

.. autofunction:: eegprep.std_checkdatasession
   :no-index:

.. autofunction:: eegprep.std_uniformfiles
   :no-index:

.. autofunction:: eegprep.std_uniformsetinds
   :no-index:

.. autofunction:: eegprep.std_findsameica
   :no-index:

.. autofunction:: eegprep.std_selsubject
   :no-index:

.. autofunction:: eegprep.std_substudy
   :no-index:

.. autofunction:: eegprep.std_rmdat
   :no-index:

.. autofunction:: eegprep.std_rmalldatafields
   :no-index:

.. autofunction:: eegprep.std_erpplot
   :no-index:

.. autofunction:: eegprep.std_specplot
   :no-index:

.. autofunction:: eegprep.std_erspplot
   :no-index:

.. autofunction:: eegprep.std_itcplot
   :no-index:

.. autofunction:: eegprep.optimal_kmeans
   :no-index:

.. autofunction:: eegprep.robust_kmeans
   :no-index:

.. autofunction:: eegprep.std_apcluster
   :no-index:

.. autofunction:: eegprep.std_centroid
   :no-index:

.. autofunction:: eegprep.std_findoutlierclust
   :no-index:

.. autofunction:: eegprep.pop_limo
   :no-index:

.. autofunction:: eegprep.pop_limoresults
   :no-index:

.. autofunction:: eegprep.std_selectdesign
   :no-index:

Format Conversion
=================

.. autofunction:: eegprep.eeg_eeg2mne
   :no-index:

.. autofunction:: eegprep.eeg_mne2eeg
   :no-index:

.. autofunction:: eegprep.eeg_mne2eeg_epochs
   :no-index:
