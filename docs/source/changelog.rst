.. _changelog:

=========
Changelog
=========

Notable changes to EEGPrep, newest first. Full release notes and downloads are on
the `GitHub Releases <https://github.com/sccn/eegprep/releases>`_ page.

Unreleased
==========

- Added the asynchronous Pyodide/Emscripten ICLabel path through pinned ONNX
  Runtime Web. ``iclabel_async`` and ``pop_iclabel_async`` preserve the native
  post-processing, console history, and GUI/console session synchronization;
  browser execution is gated by native-to-browser classification parity.
- Added a pinned Pyodide 0.29.5 harness and CI gate that installs the working-tree
  wheel, runs a continuous sample-data smoke pipeline, and records one-thread
  ``runica``/Picard and runica-shaped BLAS benchmarks. See :doc:`pyodide_benchmark`;
  Pyodide Web Workers are documented as the browser responsiveness and independent-job
  concurrency boundary, not as intra-ICA threading.
- Installing ``eegprep`` no longer pulls ``oct2py``, ``psutil``, or ``pyedflib``.
  The Octave parity engine now needs ``eegprep[eeglab]`` and the system-RAM helper in
  ``num_jobs_from_reservation`` now needs ``eegprep[sys]``; both raise an ``ImportError``
  naming the extra when it is missing, and ``eegprep[all]`` still installs everything.
  ``pyedflib`` was unused by the package and is gone from every published install.
  This removes the last base dependencies that have no WebAssembly build, so the base
  requirement set can resolve under Pyodide.
- ICLabel now ships a gate-selected weight-only int8 ONNX artifact (2,932,897
  bytes) while retaining a reproducible float32 reference and calibrated int8
  candidate under ``tools/iclabel/artifacts/``. On the frozen, subject-disjoint
  217-component real-data evaluation set, the shipped artifact matched the
  float32 teacher on 100% of top-1 labels and 100% of existing
  ``pop_icflag`` keep-or-reject decisions; the calibrated candidate measured
  98.1567% and 100%, respectively. Feature extraction, normalization,
  augmentation, softmax, class set, and rejection thresholds are unchanged.
- ``asr_process`` now resolves ``max_mem=None`` to a fixed 64 MB instead of probing free
  system RAM through ``psutil``.
  This matches the ``maxmem=64`` default that ``asr_calibrate`` and ``clean_asr`` already
  use, so the whole ASR pipeline assumes one memory budget and block sizes no longer vary
  with the machine's free memory.
  ``clean_asr`` already passed 64, so the standard cleaning pipeline is unchanged; only
  direct ``asr_process(..., max_mem=None)`` calls see different block splitting, and
  because the reconstruction matrix is refreshed on a per-block grid their output changes
  accordingly.
  Pass ``max_mem`` explicitly to pin the previous behavior.
- ICLabel (``iclabel``/``pop_iclabel``) now classifies the default network through
  ``onnxruntime`` instead of torch. The package ships ``iclabel.onnx`` in place of
  ``netICL.mat``; install the new ``iclabel`` extra (``eegprep[iclabel]``) to run
  classification. torch is now needed only to regenerate the ONNX artifact from
  ``netICL.mat`` (``tools/iclabel/export_iclabel_onnx.py``), not to run ICLabel.
  The preserved float32 reference remains the probability-parity artifact for the
  previous torch and MATLAB paths; the shipped int8 artifact is validated separately
  by the frozen top-1 and keep-or-reject semantic gate.
- ``pop_autorej`` (Tools > Automatic epoch rejection) now runs EEGLAB's probability loop
  exactly: a pass rejects its flagged epochs only when they are fewer than ``maxrej``
  percent of the remaining epochs (5% of 80 epochs is not fewer, so the threshold is
  raised instead), and once a pass flags nothing the threshold walks back down in
  0.5 s.d. steps toward 5 s.d. for up to eight pruning rounds instead of stopping at the
  first clean pass. The final kurtosis pass is applied in channel mode as well; EEGLAB
  currently skips it there because it reads the component rejection field. Rejected
  epochs on the epoched sample dataset now match ``tests/matlab/pop_autorej_reference.m``.
- Deleting a dataset that belongs to a STUDY no longer shifts its STUDY metadata
  (subject, condition, group, session, run, and components) onto the following
  datasets. The deleted dataset's ``datasetinfo`` row is dropped together with its
  ``ALLEEG`` slot, as EEGLAB's ``std_editset`` does, so ``std_checkset`` and the STUDY
  editor keep each dataset's metadata with that dataset.
- ``newtimef`` / ``pop_newtimef`` now return the baseline power spectrum (``powbase``) in dB for the
  default log power scale with a baseline, matching EEGLAB's ``mbase``; it was previously returned in
  absolute power. It stays in absolute power for absolute-scale, ``basenorm``, and ``trialbase`` runs
  and when no baseline is used (as in EEGLAB), and the plotted figure is unchanged. ``std_precomp``
  correspondingly caches the STUDY ``erspbase`` field in dB for its default log/baseline settings;
  rerun precomputation with ``recompute='on'`` to refresh values cached by an earlier version.
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
