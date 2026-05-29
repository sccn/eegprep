.. _api_signal_processing:

===========================
Signal Processing Functions
===========================

This section documents the signal processing functions for spectral analysis, resampling, and baseline removal.

Spectral Analysis
=================

.. autofunction:: eegprep.eeg_autocorr

.. autofunction:: eegprep.eeg_autocorr_welch

.. autofunction:: eegprep.eeg_rpsd

Time-Frequency And Statistics
=============================

.. autofunction:: eegprep.newtimef

.. autofunction:: eegprep.newcrossf

.. autofunction:: eegprep.signalstat

.. autofunction:: eegprep.pop_newtimef

.. autofunction:: eegprep.pop_newcrossf

.. autofunction:: eegprep.pop_signalstat

.. autofunction:: eegprep.pop_eventstat

Resampling
==========

.. autofunction:: eegprep.pop_resample

Baseline Removal
================

.. autofunction:: eegprep.rmbase

.. autofunction:: eegprep.pop_rmbase

Topography
==========

.. autofunction:: eegprep.cart2topo

.. autofunction:: eegprep.pop_topoplot

.. autofunction:: eegprep.topoplot
