.. _changelog:

=========
Changelog
=========

Notable changes to EEGPrep, newest first. Full release notes and downloads are on
the `GitHub Releases <https://github.com/sccn/eegprep/releases>`_ page.

Unreleased
==========

- ``pop_newtimef`` / ``newtimef`` now draw the ERSP and ITC images in EEGLAB's style: the
  ``jet`` colormap, a symmetric color axis, non-significant regions shown as the green colormap
  midpoint (rather than blanked to white), right-hand colorbars titled with the power unit, and
  a stimulus-onset (time 0) marker. Curve-mode plots are unchanged. (Marginal panels, the scalp
  inset, and phase-sign coloring follow in subsequent changes.)
- ``pop_newtimef`` now honors the dialog's "ERSP color limits" and "ITC color limits"
  fields (the ``erspmax`` / ``itcmax`` options); previously these edit boxes were collected
  but ignored. ``erspmax`` sets a symmetric ERSP image scale (``[-erspmax, erspmax]``) and
  ``itcmax`` the ITC image maximum.
- ``timefreq`` -- and the ``newtimef`` / ``pop_newtimef`` time-frequency plots built on it --
  now match EEGLAB's decomposition numerics. Requested output frequencies are no longer
  de-duplicated, so ``freqs``/``nfreqs`` requests that snap several values onto the same FFT
  bin return one output frequency per request (as in EEGLAB, duplicates included); ``detrend``
  is now correctly a no-op on the FFT (``cycles=0``) path; output time windows are centered
  with EEGLAB's ``eeg_lat2point`` rounding and the negative-``ntimesout`` subsample grid no
  longer includes a spurious trailing time point; ``subitc`` returns the pre-subtraction
  inter-trial coherence; and exact-zero spectral bins are guarded as in EEGLAB. Some outputs
  change shape or value versus earlier EEGPrep releases.
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
