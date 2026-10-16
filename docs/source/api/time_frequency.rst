.. _api_time_frequency:

Time-Frequency
==============

Time-frequency decomposition, coherence, phase-amplitude coupling, and time warping.

``make_timewarp`` accepts ``baselineLatency``, ``eventConditions``,
``maxSTDForAbsolute``, and ``maxSTDForRelative`` as keywords or positional
name/value pairs. The Python names ``baseline_latency``, ``event_conditions``,
``max_std_for_absolute``, and ``max_std_for_relative`` do the same thing.
Returned epoch indices are zero-based. Defaults are a baseline of 0 and no
standard-deviation rejection.

.. autosummary::
   :toctree: generated/

   eegprep.angtimewarp
   eegprep.bootstat
   eegprep.crossf
   eegprep.newcrossf
   eegprep.newtimef
   eegprep.newtimefbaseln
   eegprep.newtimefitc
   eegprep.newtimefpowerunit
   eegprep.newtimeftrialbaseln
   eegprep.make_timewarp
   eegprep.pac
   eegprep.pac_cont
   eegprep.tf_cycle_calc
   eegprep.timef
   eegprep.timefreq
   eegprep.timewarp
