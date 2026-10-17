.. _api_time_frequency:

Time-Frequency
==============

Time-frequency decomposition, coherence, phase-amplitude coupling, and time warping.

``newtimef`` accepts the ``finputcheck`` names from ``newtimef.m`` as keywords
or positional name/value pairs, including ``chaninfo``, ``ntimesout``,
``plotamp``, ``maxfreq``, ``cycleinc``, and ``newfig``. Other names that
``timefreq`` accepts (``ffttaper``, ``wavelet``, ``timestretch``) are
forwarded to it. A forwarded ``wavelet`` replaces ``cycles`` inside
``timefreq`` (``newtimef.m`` lines 1186-1188). EEGLAB still applies the
line 1207 FFT scale, ``2/0.375/winsize``, from the original ``cycles``;
EEGPrep applies it only to the decomposition that ran. Any further name,
including ``overlap``, is an undefined ``timefreq`` argument, reported
with the caller's spelling. ``rng`` is an EEGPrep-only name that seeds
the bootstrap.
Defaults follow the ``newtimef`` list, including ``baseline`` 0,
``scale`` ``"log"``, ``timesout`` 200, and ``padratio`` 2. ``ntimesout`` is
the output time count when ``timesout`` is not a vector. ``itctype`` replaces
``type`` unless it is ``phasecoher``. The default frequency range is
``[0, 50]``; ``maxfreq`` above the Nyquist rate is capped and then written
into that range, so a sampling rate below 100 Hz yields ``[0, Nyquist]``.
A figure is drawn when ``plotersp`` or ``plotitc`` is on. ``plotamp`` off
turns ``plotersp`` off.

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
