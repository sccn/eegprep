.. _api_io:

=======================
Input/Output Functions
=======================

This section documents the input/output functions for loading and saving EEG data in various formats.

BIDS Loading
============

.. autofunction:: eegprep.pop_load_frombids

.. autofunction:: eegprep.pop_importbids

.. autofunction:: eegprep.pop_exportbids

Generic Import
==============

.. autofunction:: eegprep.pop_importdata

.. autofunction:: eegprep.pop_fileio

.. autofunction:: eegprep.pop_biosig

.. autofunction:: eegprep.pop_importevent

.. autofunction:: eegprep.pop_importepoch

.. autofunction:: eegprep.pop_chanevent

.. autofunction:: eegprep.pop_importpres

.. autofunction:: eegprep.pop_importerplab

Channel Locations
=================

.. autofunction:: eegprep.pop_readlocs

.. autofunction:: eegprep.pop_writelocs

.. autofunction:: eegprep.readlocs

.. autofunction:: eegprep.writelocs

.. autofunction:: eegprep.convertlocs

.. autofunction:: eegprep.chancenter

.. autofunction:: eegprep.pop_chancenter

.. autofunction:: eegprep.pop_chancoresp

.. autofunction:: eegprep.readegilocs

.. autofunction:: eegprep.readelp

.. autofunction:: eegprep.readeetraklocs

Long-Tail Import Helpers
========================

.. autofunction:: eegprep.pop_loadbci

.. autofunction:: eegprep.pop_snapread

.. autofunction:: eegprep.snapread

.. autofunction:: eegprep.floatread

.. autofunction:: eegprep.floatwrite

EEGLAB Format
=============

.. autofunction:: eegprep.pop_loadset

.. autofunction:: eegprep.pop_loadset_h5

.. autofunction:: eegprep.pop_saveset

Text And External Export
========================

.. autofunction:: eegprep.pop_export

``pop_export`` supports EEGLAB-style text export options including ICA export,
time/electrode rows, transpose, ERP averaging, precision, separator, and a
standalone numeric ``expr`` transform applied to the exported array ``x``.
Most expression function calls are positional; ``clip`` and ``nan_to_num`` also
accept documented safe numeric keywords. Power operators require small constant
exponents.

.. autofunction:: eegprep.pop_expica

.. autofunction:: eegprep.pop_expevents

.. autofunction:: eegprep.pop_writeeeg

History And STUDY Files
=======================

.. autofunction:: eegprep.pop_saveh

.. autofunction:: eegprep.pop_runscript

.. autofunction:: eegprep.pop_study

.. autofunction:: eegprep.pop_studywizard

.. autofunction:: eegprep.pop_studyerp

.. autofunction:: eegprep.pop_studydesign

.. autofunction:: eegprep.pop_loadstudy

.. autofunction:: eegprep.pop_savestudy

.. autofunction:: eegprep.pop_precomp

.. autofunction:: eegprep.pop_chanplot

.. autofunction:: eegprep.std_editset

.. autofunction:: eegprep.std_checkset

.. autofunction:: eegprep.std_checkdatasetinfo

.. autofunction:: eegprep.std_makedesign

.. autofunction:: eegprep.std_addvarlevel

.. autofunction:: eegprep.std_builddesignmat

.. autofunction:: eegprep.std_rebuilddesign

.. autofunction:: eegprep.std_saveindvar

.. autofunction:: eegprep.pop_addindepvar

.. autofunction:: eegprep.pop_importgroupvar

.. autofunction:: eegprep.pop_listfactors

.. autofunction:: eegprep.std_precomp

.. autofunction:: eegprep.std_readdata

.. autofunction:: eegprep.std_readerp

.. autofunction:: eegprep.std_readspec

.. autofunction:: eegprep.std_readersp

.. autofunction:: eegprep.std_readitc

.. autofunction:: eegprep.std_readtopo

.. autofunction:: eegprep.std_readpac

.. autofunction:: eegprep.std_savedat

.. autofunction:: eegprep.std_checkfiles

.. autofunction:: eegprep.std_checkdatasession

.. autofunction:: eegprep.std_uniformfiles

.. autofunction:: eegprep.std_uniformsetinds

.. autofunction:: eegprep.std_erpplot

.. autofunction:: eegprep.std_specplot

.. autofunction:: eegprep.std_erspplot

.. autofunction:: eegprep.std_itcplot

.. autofunction:: eegprep.optimal_kmeans

.. autofunction:: eegprep.robust_kmeans

.. autofunction:: eegprep.std_apcluster

.. autofunction:: eegprep.std_centroid

.. autofunction:: eegprep.std_findoutlierclust

.. autofunction:: eegprep.pop_limo

.. autofunction:: eegprep.pop_limoresults

.. autofunction:: eegprep.std_selectdesign

Format Conversion
=================

.. autofunction:: eegprep.eeg_eeg2mne

.. autofunction:: eegprep.eeg_mne2eeg

.. autofunction:: eegprep.eeg_mne2eeg_epochs
