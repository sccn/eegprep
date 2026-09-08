.. _changelog:

=========
Changelog
=========

Notable changes to EEGPrep, newest first. Full release notes and downloads are on
the `GitHub Releases <https://github.com/sccn/eegprep/releases>`_ page.

Unreleased
==========

- ``pop_newtimef`` / ``newtimef`` now render EEGLAB's single-condition ERSP/ITC time-frequency
  figure. The ERSP and ITC images use a symmetric color axis, a stimulus-onset (time 0) marker,
  right-hand colorbars titled with the power unit, and the ``turbo`` colormap (EEGPrep's house
  colormap, consistent with ``topoplot`` and ``erpimage``); ITC is colored by its phase sign by
  default (``plotphasesign``), with ``plotphaseonly`` showing the phase angle in degrees. The
  figure adds EEGLAB's marginal panels -- the ERSP min/max envelope and ERP trace below the
  images, and the rotated baseline power spectrum and mean ITC to their left -- using EEGLAB's
  value-axis limits and two ticks per panel. When channel locations are available it also draws
  the channel or component scalp-map inset and caption (the channel label or ``IC n``), marking
  the selected electrode for channels. The ``erspmax`` / ``itcmax`` dialog fields set the ERSP
  and ITC color limits, and the default frequency range stops at 50 Hz (EEGLAB's ``maxfreq``,
  capped at Nyquist).
- ``pop_newtimef`` / ``newtimef`` bootstrap significance now matches EEGLAB. With a significance
  level (``alpha``), each time-frequency point is ranked against a per-frequency baseline null
  built by permuting the baseline time course (EEGLAB's ``shuffle`` permutation, averaged over
  trials), and the p-value folds to a two-sided tail (EEGLAB's ``compute_pvals``) for both ERSP
  and ITC -- so the scattered baseline false positives EEGLAB shows are reproduced and genuine
  event-related effects survive masking. Non-significant regions are shown as the colormap
  midpoint, or outlined with contours instead when ``pcontour`` is set; the FDR (``mcorrect``)
  path is supported. A non-default ``boottype`` (``'rand'`` / ``'randall'``) is now rejected with
  a clear error rather than silently treated as ``'shuffle'``.
- ``timefreq`` -- and the ``newtimef`` / ``pop_newtimef`` plots built on it -- now match EEGLAB's
  decomposition numerics. Requested output frequencies are no longer de-duplicated, so
  ``freqs``/``nfreqs`` requests that snap several values onto the same FFT bin return one output
  frequency per request (as in EEGLAB) -- this changes the ``pac`` output shape (for example
  ``(5, 3, 8)`` becomes ``(5, 5, 8)``); ``detrend`` is now a no-op on the FFT (``cycles=0``)
  path; output time windows are centered with EEGLAB's
  ``eeg_lat2point`` rounding and the negative-``ntimesout`` subsample grid no longer includes a
  spurious trailing time point; ``subitc`` returns the pre-subtraction inter-trial coherence; and
  exact-zero spectral bins are guarded as in EEGLAB. Some outputs change shape or value versus
  earlier EEGPrep releases.
- ``pop_prop`` (Plot > Channel/Component properties) now matches EEGLAB's three-panel
  layout: a scalp map, an ERP image, and the activity power spectrum. The ERP panel is a
  full ERP image (reusing ``erpimage``) instead of a single averaged trace, the spectrum is
  computed from the raw per-epoch data, and component spectra are scaled by the component
  map power (``mapnorm``) so their levels match EEGLAB. Channel maps mark the selected
  channel's location, and the spectrum y-axis stays tight to the plotted frequency band
  instead of stretching to out-of-view frequencies. The ERP average panel is labeled "ERP"
  for channels and left blank for components, since ICA component activations are unitless,
  matching EEGLAB. The ERP trace now sits flush beneath the ERP image (only the trace carries
  the time axis), and the "pop_prop() - <name> properties" label is shown in the window title
  bar rather than on the canvas, as EEGLAB does.
- Scalp maps now render with EEGLAB's left-right orientation. ``topoplot``, ``plottopo``,
  and ``erpimage`` previously mirrored electrode markers, labels, and interpolated data
  along the left-right axis; channel positions now match EEGLAB (F4 on the right, F3 on the
  left). Dataset contents and history are unaffected.

Version 0.3.0
=============

*Released 2026-08-11*

- ``clean_rawdata`` now processes data at any sampling rate. ASR calibration previously
  relied on a table of pre-computed spectral-shaping filter coefficients covering only
  100, 128, 200, 250, 256, 300, 500 and 512 Hz, and raised an error otherwise, so common
  clinical rates such as 258 Hz could not be cleaned without resampling first. The filter
  is now designed for the recording's own rate with a ``yulewalk`` port, matching what
  EEGLAB's ``asr_calibrate.m`` does. See :ref:`preprocessing_pipeline`.
- ``eegprep.rmbase`` now exposes the low-level EEGLAB-style numeric baseline helper; use
  ``pop_rmbase`` for EEG dataset dictionaries.
