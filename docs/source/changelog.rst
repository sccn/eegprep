.. _changelog:

=========
Changelog
=========

Notable changes to EEGPrep, newest first. Full release notes and downloads are on
the `GitHub Releases <https://github.com/sccn/eegprep/releases>`_ page.

Unreleased
==========

- ``pop_prop`` (Plot > Channel/Component properties) now matches EEGLAB's three-panel
  layout: a scalp map, an ERP image, and the activity power spectrum. The ERP panel is a
  full ERP image (reusing ``erpimage``) instead of a single averaged trace, the spectrum is
  computed from the raw per-epoch data, and component spectra are scaled by the component
  map power (``mapnorm``) so their levels match EEGLAB. Channel maps mark the selected
  channel's location, and the spectrum y-axis stays tight to the plotted frequency band
  instead of stretching to out-of-view frequencies. The ERP average panel is labeled "ERP"
  for channels and left blank for components, since ICA component activations are unitless,
  matching EEGLAB.
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
