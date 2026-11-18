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
lines 1207-1209 FFT scale, ``2/0.375/winsize`` (default ``winsize`` at
line 453), from the original ``cycles``; EEGPrep applies it only to the
decomposition that ran. Any other name, including ``overlap``, is an
undefined ``timefreq`` argument, reported with the caller's spelling.
``rng`` is an EEGPrep-only name that seeds the bootstrap.
Defaults follow the ``newtimef`` list, including ``baseline`` 0,
``scale`` ``"log"``, ``timesout`` 200, and ``padratio`` 2. ``ntimesout`` is
the output time count when ``timesout`` is not a vector. Times that round to
the same sample are kept, so that count is the requested grid after the
window clamp, not the number of distinct samples. Grid points are spaced as
MATLAB ``linspace`` (``start + i * (stop - start) / (n - 1)``, last point
equal to the end limit). ``itctype`` replaces
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

.. _spectrum_summaries:

Spectrum summaries
------------------

``compute_spectra`` is the Welch estimator behind ``spectopo``. It does not
draw a figure, and importing it does not import Matplotlib. The result is
power spectral density in dB: ``10 * log10`` of SciPy's
``welch(..., scaling="density")`` samples. It is not
``scaling="spectrum"`` bin power, and it is not the integral of density over
a 1 Hz band. The frequency grid is ``k * srate / nfft``. With the default
segment length ``min(round(srate), pnts)`` (half away from zero) and the
default ``nfft`` equal to that length, the spacing is 1 Hz when ``srate`` is
an integer and the recording contains at least one second. The grid starts
at 0 Hz and stops at the Nyquist rate when ``nfft`` is even, or at the
greatest grid frequency at or below Nyquist when ``nfft`` is odd. No sample
is above the Nyquist rate.

To read the 1–70 Hz portion, index the returned frequencies. The selected
columns are still density samples at those frequencies:

.. code-block:: python

   from eegprep import channel_robust_amplitude, compute_spectra, line_noise_excess_db

   spectra, freqs, _std = compute_spectra(EEG["data"], EEG["pnts"], EEG["srate"])
   in_band = (freqs >= 1.0) & (freqs <= 70.0)
   band_db = spectra[:, in_band]

Epochs are averaged in linear density and converted to dB once. A 3-D array
is ``(channels, samples, epochs)``. A 2-D array whose sample count is a
multiple of ``pnts`` is unstacked in column-major order, so epoch ``k``
occupies samples ``k * pnts`` through ``(k + 1) * pnts``. dB values are not
averaged across epochs.

``line_noise_excess_db`` uses that default spectrum. For each requested
frequency the excess is the line bin minus the arithmetic mean of
``neighbours`` bins on each side, with the line bin itself excluded. The mean
is taken on the dB values. The default frequencies are 50 and 60 Hz and the
default neighborhood is one bin on each side. The result has shape
``(channels, len(freqs))``. A frequency above ``srate / 2`` raises
``ValueError``. A frequency that misses every returned bin by more than
1e-6 Hz also raises, as does a neighbor window that would leave the grid.
A line on the
Nyquist bin has no upper neighbor.

``channel_robust_amplitude`` pools every sample of a channel, including every
epoch, and returns ``median(|x - median(x)|) / norm.ppf(0.75)``. The scale
factor is approximately 1.4826, from
``scipy.stats.median_abs_deviation(..., scale="normal")``, so the value
estimates a normal standard deviation. Units are the units of ``EEG["data"]``.
A channel that contains a NaN has amplitude NaN. Neither helper modifies the
dataset.

.. autosummary::
   :toctree: generated/

   eegprep.compute_spectra
   eegprep.line_noise_excess_db
   eegprep.channel_robust_amplitude
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
