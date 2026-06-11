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

.. autofunction:: eegprep.timefreq

.. autofunction:: eegprep.timef

.. autofunction:: eegprep.crossf

.. autofunction:: eegprep.dftfilt

.. autofunction:: eegprep.dftfilt2

.. autofunction:: eegprep.dftfilt3

.. autofunction:: eegprep.timewarp

.. autofunction:: eegprep.angtimewarp

.. autofunction:: eegprep.tf_cycle_calc

.. autofunction:: eegprep.newtimefbaseln

.. autofunction:: eegprep.newtimeftrialbaseln

.. autofunction:: eegprep.newtimefitc

.. autofunction:: eegprep.newtimefpowerunit

.. autofunction:: eegprep.bootstat

.. autofunction:: eegprep.correct_mc

.. autofunction:: eegprep.correctfit

.. autofunction:: eegprep.rsadjust

.. autofunction:: eegprep.rsfit

.. autofunction:: eegprep.rsget

.. autofunction:: eegprep.rspdfsolv

.. autofunction:: eegprep.rspfunc

.. autofunction:: eegprep.signalstat

.. autofunction:: eegprep.pop_newtimef

.. autofunction:: eegprep.pop_newcrossf

.. autofunction:: eegprep.pop_timef

.. autofunction:: eegprep.pop_crossf

.. autofunction:: eegprep.pop_signalstat

.. autofunction:: eegprep.pop_eventstat

Browser
========

.. autofunction:: eegprep.eegplot

.. autofunction:: eegprep.eeg_multieegplot

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
