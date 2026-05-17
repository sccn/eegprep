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

EEGLAB Format
=============

.. autofunction:: eegprep.pop_loadset

.. autofunction:: eegprep.pop_loadset_h5

.. autofunction:: eegprep.pop_saveset

Text And External Export
========================

.. autofunction:: eegprep.pop_export

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

.. autofunction:: eegprep.pop_loadstudy

.. autofunction:: eegprep.pop_savestudy

Format Conversion
=================

.. autofunction:: eegprep.eeg_eeg2mne

.. autofunction:: eegprep.eeg_mne2eeg

.. autofunction:: eegprep.eeg_mne2eeg_epochs
