.. _changelog:

=========
Changelog
=========

Notable changes to EEGPrep, newest first. Full release notes and downloads are on
the `GitHub Releases <https://github.com/sccn/eegprep/releases>`_ page.

Unreleased
==========

- ``pop_autorej`` (Tools > Automatic epoch rejection) now runs EEGLAB's probability loop
  exactly: a pass rejects its flagged epochs only when they are fewer than ``maxrej``
  percent of the remaining epochs (5% of 80 epochs is not fewer, so the threshold is
  raised instead), and once a pass flags nothing the threshold walks back down in
  0.5 s.d. steps toward 5 s.d. for up to eight pruning rounds instead of stopping at the
  first clean pass. The final kurtosis pass is applied in channel mode as well; EEGLAB
  currently skips it there because it reads the component rejection field. Rejected
  epochs on the epoched sample dataset now match ``tests/matlab/pop_autorej_reference.m``.
- Deleting datasets (``pop_delset``, Edit > Delete dataset(s) from memory) now empties
  the slot in place like EEGLAB instead of shifting later datasets down, so dataset
  numbers in the Datasets menu, ``CURRENTSET``, and the history stay valid. ``eeg_store``
  stores new datasets in the lowest empty slot, trailing empty slots are dropped, and
  ``pop_delset`` raises for a dataset number that does not exist. A dataset selection can
  no longer land on an emptied slot; it falls back to a remaining dataset the way EEGLAB's
  redraw does. Creating or editing a STUDY still compacts the empty slots away, because a
  STUDY needs contiguous dataset numbers, but the selection now follows the dataset it was
  on instead of its old number. Editing a STUDY design, precomputing measures, and
  preclustering no longer modify ``ALLEEG``, matching EEGLAB's ``e_plot_study``.
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
