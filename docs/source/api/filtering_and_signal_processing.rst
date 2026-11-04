.. _api_filtering_and_signal_processing:

Filtering and Signal Processing
===============================

FIR filter design, filtering helpers, and window functions.

``dftfilt3`` accepts ``cycleinc``, ``winsize``, and ``timesupport`` as keywords
or positional name/value pairs. Defaults are linear cycle spacing, a
variable-length Morlet wavelet, and a time support of seven standard
deviations. ``type`` is not an option.

.. autosummary::
   :toctree: generated/

   eegprep.blockave
   eegprep.dftfilt
   eegprep.dftfilt2
   eegprep.dftfilt3
   eegprep.eegfilt
   eegprep.env
   eegprep.fir_filterdcpadded
   eegprep.firfiltreport
   eegprep.firfiltsplit
   eegprep.firgauss
   eegprep.firws
   eegprep.firwsord
   eegprep.getipsph
   eegprep.invfirwsord
   eegprep.invkaiserbeta
   eegprep.kaiserbeta
   eegprep.minphaserceps
   eegprep.movav
   eegprep.plotfresp
   eegprep.rmart
   eegprep.rmsave
   eegprep.windows
