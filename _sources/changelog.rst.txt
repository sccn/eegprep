.. _changelog:

=========
Changelog
=========

Notable changes to EEGPrep, newest first. Full release notes and downloads are on
the `GitHub Releases <https://github.com/sccn/eegprep/releases>`_ page.

Unreleased
==========

- Breaking change: ``abspeak``, ``matcorr``, ``mapcorr``, ``matperm``,
  ``hungarian``, ``getchanlist``, ``shuffle``, and ``eeg_amplitudearea`` use
  1-based indices. ``abspeak`` frames, ``hungarian`` row assignments
  (``assignment[j]`` is the row of column ``j + 1``), ``matcorr``, ``mapcorr``,
  and ``matperm`` row numbers, ``getchanlist`` channel numbers, and
  ``eeg_amplitudearea`` channel numbers start at 1. ``shuffle``'s dimension
  argument is now ``dim`` and is 1-based
  (it was ``axis``, 0-based); its forward and inverse indices also start at 1.
  Subtract 1 before using any of these values as a NumPy index. An ``abspeak``
  channel or epoch that is all NaN still returns frame ``-1``.
- ``eeg_hist``, ``is_sccn``, ``iseeglabdeployed``, ``eeg_helphelp``,
  ``helpforexe``, and ``disp`` are available from ``eegprep``. ``eeg_hist``
  appends a command on its own line and skips commands that contain
  ``pop_loadset``. History stored as a character matrix or a list of rows
  is joined into lines first. ``is_sccn`` returns 1 when the stripped Unix
  ``hostname`` output contains ``ucsd.edu``, and 0 when that command times
  out or fails. ``iseeglabdeployed``
  returns 1 when Python is running a frozen executable (``sys.frozen``),
  which is the packaged-application counterpart of MATLAB ``isdeployed``.
  ``eeg_helphelp`` prints its docstring, and ``helpforexe`` writes one
  ``disp`` script per function from that function's docstring.
  ``eegh(n)`` in ``eegprep-console`` runs a command from the session
  history and records that command once. ``gethelpvar`` now returns
  description and name rows with shape ``(1, N)`` instead of Python lists.
- ``eegh(command, EEG)`` returns that dataset and records the command with
  ``eeg_hist``, including a character-matrix or row-list history.
  ``getkeyval`` keeps the space after the comma in a parent full-mode
  fragment. Presentation logs pad short rows, epoch and event field tables
  follow the file columns, and ``eeg_from_data`` keeps an extra channel when
  ``nbchan`` does not match the row count. ``eegplot2event`` rounds half away
  from zero and returns an empty array when no marks remain. ``rejtrend``
  returns no rejections when its window is longer than the trial.
  ``celltomat`` returns a numeric array for an empty cell, and
  ``eeg_point2lat`` accepts a column time window.
- ``statcond`` accepts EEGLAB name/value pairs (``mode``, ``verbose``,
  ``returnresamplingarray``, ``structoutput``, ``cluster``, and the other
  documented options) as well as keywords. A single MATLAB output is the
  statistic array, the resampling cell array, or the ``structoutput`` struct.
  Python callers still receive ``StatcondResult`` or ``SurrogateDistribution``.
  Unpaired two-way MATLAB cells stay in column, row, interaction order.
  ``verbose`` still defaults to off. Unpaired two-condition tests now default
  to Welch's inhomogenous variance, matching ``statcond`` ``finputcheck``, so
  calls that omit ``variance`` change. ``ttest2_cell`` still defaults to equal
  variance. Cluster correction uses the test degrees of freedom. EEGLAB forms
  that threshold with ``tcdf`` at ``statcond.m`` lines 470 and 490, using the
  feature count as the degrees of freedom.
  Resampling cells follow MATLAB ``rng('default')``; pass ``rng`` to control
  the Python surrogate stream.
- ``MemmapData.matlab_assign`` grows when an index extends past the end of an
  axis and fills the new samples with zeros. An empty value deletes those
  entries, folding trailing dimensions when the key is shorter than the rank.
  Boolean masks use ``find``: only ``True`` positions are removed, and an
  all-false mask leaves the data unchanged. Ordinary indexing stays NumPy's.
  Shared copies stay unchanged until they are written, and ``debug=True``
  prints whether a ``matlab_assign`` reused the file. ``pop_firws`` keeps
  single-precision EEG in float32: the filter runs in float64 and the result
  is cast back, so a memory-mapped dataset and the same dataset in memory
  match. Legacy ``pop_eegfilt`` returns float64, except continuous data with
  a boundary event keeps the input dtype for both in-memory and memory-mapped
  data. The filter runs in float64. EEGLAB eegfilt.m:209-210 runs filter/filtfilt
  on single-precision input in single precision (falling back to double only on
  error, :213-215), so its result differs from this one by single-precision
  rounding.
- ``erpimage`` is now a faithful port of EEGLAB's ``erpimage``: it takes
  EEGLAB's arguments and options (as tokens or keywords) and returns its fifteen
  outputs plus the figure. Sorting, smoothing and decimation, phase and
  amplitude sorting, coherence with bootstrap significance, amplitude images,
  time warping and axis limits match MATLAB R2025b, computed by a NumPy-only
  engine. ``erpimage(data, times=..., sort_values=..., smooth=...)`` calls
  become ``erpimage(data, sortvar, times, title, avewidth, decimate, ...)``,
  and ``plot_erp``, ``chan_locs`` and ``channel_index`` become ``erp`` and
  ``topo``. Defaults now follow EEGLAB (no ERP trace, no colorbar). The
  ``erpalpha`` permutation test uses unbiased random signs, so its thresholds
  differ slightly from EEGLAB's
  (`sccn/eeglab#976 <https://github.com/sccn/eeglab/issues/976>`_).
  Invalid inputs raise ``ValueError`` instead of producing a wrong image,
  ``'allcohers'`` raises, ``NoShow`` cannot be combined with ``target``, and
  ``rng`` also accepts an integer seed. The notes EEGLAB prints and continues
  past (clamped widths, removed NaN trials, an ``'erp'`` above 1 reduced to one
  ERP by ``'erpalpha'``, ignored options) are logged; see :ref:`erpimage_api`
  for every difference.
- ``pop_erpimage`` follows EEGLAB's ``pop_erpimage``:
  it takes ``(EEG, typeplot, channel, projchan, titleplot, smooth, decimate,
  sortingtype, sortingwin, sortingeventfield, ...)`` followed by any ``erpimage`` option,
  and returns the ``erpimage`` result instead of a ``{"figure", "image"}`` dictionary
  (``result.outdata`` is ``(frames, trials)``, the transpose of the old ``"image"``).
  ``index=`` becomes ``channel`` (numbers, ranges or labels; several channels are averaged),
  ``sort_values`` and the ``field``, ``type`` and ``eventrange`` aliases are gone,
  and ``limits`` sets the axes instead of cropping the data.
  The defaults are EEGLAB's (``smooth=5``, ``decimate=0``),
  and the ERP trace, colorbar and scalp map are drawn by default,
  where EEGLAB's command line leaves them off.
  The scalp map is left out, with a warning, when several channels are averaged.
  Trials are sorted with ``eeg_getepochevent``
  (first latency, last value of other fields, text sorted alphabetically, stable order);
  events without the field, or no epoch with a match, raise clear errors,
  and epochs left out are listed in a warning.
  A string ``sortingtype`` is one event type, even with spaces,
  and numeric event types match by value.
  ``nosort`` and ``noplot`` are passed to ``erpimage``,
  and phase sorting, coherence, alignment and renormalization formulas are available
  from both the command line and the dialog.
  An option name given as ``projchan`` or ``titleplot`` now raises instead of being misread.
  Smoothing and decimation follow EEGLAB's moving window, so images change shape:
  on ``sample_data/eeglab_data_epochs_ica.set`` channel 14 (80 epochs),
  ``smooth=10, decimate=2`` gives 36 rows (previously 40),
  ``smooth=10, decimate=20`` gives 18 (previously 4)
  and ``smooth=3, decimate=1`` gives 78 (previously 80).
  Smoothing fewer than 4 epochs raises ``ValueError``,
  and component images leave the ERP axis label empty, as EEGLAB does (previously "uV").
  Default titles are built as in EEGLAB:
  joined channel labels, ``Comp. N`` or ``Comp. N -> Chan. X``.
- The ``pop_erpimage`` dialog now follows EEGLAB's layout and behavior:
  the component variant has the ``Plot scalp map`` checkbox and draws the component map,
  the event-field and event-type buttons open list dialogs,
  the phase and coherence frequencies mirror each other,
  and every field (limits, phase sorting, coherence, spectrum, marks, More options) reaches ``erpimage``.
  As in EEGLAB, the fields you fill replace the same options in More options.
  The menu reopens the dialog with the last matching command, as EEGLAB's menu does,
  and ``lastcom=``, or ``gui=True`` with arguments, fills it from the console.
  The event-type list is sorted like EEGLAB's ``unique()``.
  Its history command replays in ``eegprep-console``.
- ``a:b`` ranges in EEGLAB-style number fields count up only, as in MATLAB, so ``5:3`` is empty.
- ``pop_prop`` draws its ERP image with EEGLAB's ``pop_prop`` arguments,
  including the sortvar trace at the right edge and rounded ERP axis ticks,
  and raises ``ValueError`` when the activity is all NaN.
- BIDS EDF and BDF files are imported with ``pop_biosig``. The previous
  neo/pyEDFlib reader rejected the OpenNeuro recordings ds003194, ds003195,
  and ds004279.
- BIDS TSV and JSON sidecars are decoded as UTF-8, and a leading UTF-8 BOM
  is removed. A TSV that is not valid UTF-8 is decoded as Latin-1, so the
  ``0xB5`` unit in ds003620 ``channels.tsv`` stays µ. JSON that is not UTF-8
  is rejected.
- Importing a recording or time window with fewer samples than channels no
  longer raises an orientation error. ``pop_biosig``, ``pop_fileio``,
  ``pop_read_erpss``, and ``pop_snapread`` pass the channel count they already
  know into the dataset builder. A one-second BioSemi window stays
  channel-major. EEGLAB leaves that square matrix sample-major.
- ``clean_windows``, ``asr_calibrate``, and ``asr_calibrate_r`` place RMS
  windows with MATLAB's colon operator. A fractional overlap such as 0.66 no
  longer adds one extra window at the end of some recordings.
  ``clean_windows`` keeps ``window_len * srate`` unrounded, as
  ``clean_windows.m`` does: ``0:N-1`` indexes ``floor(N)`` samples and the
  RMS divisor is that unrounded product. ``asr_calibrate`` still rounds the
  window length, as ``asr_calibrate.m`` does.
- ``pop_clust``, ``kmeans_st``, and ``kmeanscluster`` no longer sort clusters
  by coordinates, and ``kmeans_st`` now follows EEGLAB's UPGMA Hartigan-Wong
  algorithm. ``Cls`` numbers and memberships can change on existing studies
  for the same ``random_state``. On the sample epochs STUDY, the ``kmeans``
  path swaps ``Cls 2`` and ``Cls 3`` relative to develop.
- ``pop_clust(algorithm="kmeanscluster")`` now calls ``kmeanscluster()``.
  It previously used the same Lloyd starts as ``kmeans``, so those
  memberships change too.
- ``pop_select`` finishes by checking event consistency. A leading boundary
  shorter than one sample is removed, events with an out-of-range epoch
  number are removed, and the remaining events are sorted by epoch, then
  latency. Consecutive entries of the boundary list that share a latency
  (event order, before that sort) chain their durations into the last entry
  of the run. A later boundary at the same latency stays separate when a
  different latency stands between the two entries.
- The top-level ``eegprep`` package exports the public statistics helpers
  (``statcond``, ``fdr``, ``statcondfieldtrip``, and the condition-test,
  surrogate, and concatenation helpers) and the public signal helpers
  ``numdim``, ``eeg_ms2f``, ``spher``, ``eegplot2event``,
  ``eegplot2trial``, ``envtopo``, ``erpimage``, ``timtopo``, ``plottopo``,
  ``spectopo``, ``axcopy``, and ``runamica``.
- ASR, Riemannian ASR, and spherical-spline interpolation now use MATLAB's
  pseudoinverse cutoff ``max(size)*eps(norm)``. NumPy's default cutoff could
  invert a numerical null direction and amplify the signal by about 1e13.

Version 0.4.0
=============

*Released 2026-09-23*

- Added a source-driven developer audit for the current EEGLAB test-port
  project. It discovers all MATLAB wrapper and regression methods from the
  pinned suite, collects pytest provenance, resolves same-directory scenario
  scripts to their wrapper methods, and reports exact missing or stale ports.
- Ported the current EEGLAB tutorial-wrapper workflows using
  deterministic generated EEG and BIDS fixtures. Coverage now follows scalp
  movies, grouped ERP and N400-style STUDY analyses, spherical source fitting,
  all-channel time-frequency analysis, and full face-recognition and P300 BIDS
  preprocessing paths.
  Picard now accepts EEGLAB-style positive and negative PCA dimensions, and
  multi-dataset ICA grouping handles empty NumPy-backed BIDS session metadata.
- STUDY measure designs can now use categorical factors stored per trial. With
  ``savetrials="on"``, ERP, spectrum, ERSP, and ITC plots select the matching
  trials within each dataset and then aggregate one case per subject or
  component. Spectrum and ERSP trials are averaged in linear power before dB
  conversion, and ITC phases are averaged circularly.
- ``std_dipplot`` now selects existing DIPFIT models through one-based STUDY
  cluster membership, returns their coordinates, moments, residual variances,
  and centroids, and renders joined, centroid, member, and separated views.
- STUDY ERP, spectrum, ERSP, ITC, and scalp-map functions now arrange cached
  measures by condition and group, retain subject or component cases, return
  parametric or resampled statistics and FDR masks, reconstruct saved
  time-frequency trials, and plot grouped traces, images, and topographies.
- Added standalone miscellaneous visualization utilities for spatial scalp-map
  gradients, logarithmic-axis matrix images, epoched event rasters, recursive
  Matplotlib font styling, and replayable 2-D/3-D scalp movies. Movie generation
  returns direct RGB frames for headless notebooks and tests while
  preserving EEGLAB's 1-based ``movieframes`` boundary and legacy camera paths.
  Current tests for obsolete ``eegplotgold``/``eegplotsold`` and MATLAB menu
  handle introspection instead exercise EEGPrep's maintained browser and
  declarative menu model; MATLAB help-site builders and callback-string
  ``textgui`` remain explicit exclusions.
- Added a standalone ``statcondfieldtrip`` backend for paired and unpaired
  t-tests and unpaired one-way ANOVA, with analytic or seeded permutation
  inference and Bonferroni, Holm, FDR, max-statistic, and max-cluster-mass
  correction. Cluster inference uses explicit dense or sparse adjacency,
  correct paired/unpaired label exchangeability, a direct two-sided null, and
  either seeded Monte Carlo sampling or bounded exact enumeration.
- Added standalone ``corrmap`` and ``pop_corrmap`` component-template
  matching for STUDY workflows. Two-pass correlation handles ICA polarity,
  searches deterministic automatic thresholds, can store matched bad
  components and child clusters, and RMS-normalizes average maps so arbitrary
  ICA scaling cannot change the second-pass template.
- Added a standalone first- and second-level LIMO-compatible workflow.
  ``pop_limo`` now fits OLS, robust WLS, and Tukey-bisquare IRLS models from
  active STUDY designs, persists safe versioned results, and exposes model
  diagnostics. Contrasts, core group tests, regression, ANOVA/ANCOVA,
  repeated-measures ANOVA, and weighted summaries are available through
  ``std_limoresults`` and ``pop_limoresults``. MATLAB ``.mat`` interchange,
  bootstrap, TFCE, and LIMO plotting remain explicit boundaries.
- ``statcond`` now accepts EEGLAB's ``arraycomp`` switch. Batched mode returns
  all requested resampling grids, while off mode exposes one compatibility
  grid or computes inference iteratively to bound peak memory without changing
  seeded statistics.
- Added standalone ``pop_importegimat`` support for continuous and segmented
  EGI Net Station MATLAB exports, including embedded sampling rates, trial
  events and timing, empty-reference removal, packaged EGI montages, and
  replayable command history. Unlike EEGLAB, segment numbering may contain
  gaps or exceed the number of file variables, and a 1 ms pre-stimulus offset
  is preserved correctly. A reference channel is removed only when all of its
  trials and samples are empty.
- EGI Simple Binary RAW versions 2 through 7 can now be read directly through
  ``readegihdr``, ``readegi``, ``pop_readegi``, and ``pop_readsegegi``. The
  readers support 1-based frame or segment selection, A/D scaling, event
  channels, segment categories, equal-length epoched datasets, and validated
  numbered continuous-file series without requiring EEGLAB or large test
  fixtures. Multi-file imports now report a correct continuous point count
  instead of copying the zero-valued segmented header fields. Leading-edge
  events are placed on the first nonzero sample, correcting EEGLAB's one-sample
  early event-channel conversion, and a trailing channel is removed as a
  reference only when the whole channel is empty rather than just ten samples.
  Numbered imports reject gaps before later files instead of silently dropping
  the remainder of the recording.
- ERPSS ``.RAW`` and ``.RDF`` import now works without the legacy compiled
  MEX decompressor. ``read_erpss`` and ``pop_read_erpss`` support compressed
  and uncompressed little- and big-endian recordings, preserve channel labels
  and events, apply valid microvolt calibration, and fail clearly on truncated
  blocks.
- ``pop_loadbv`` now provides a standalone BrainVision reader for binary and
  ASCII recordings in multiplexed or vectorized orientation. It applies
  per-channel resolutions, normalizes voltage channels to microvolts, retains
  channel coordinates and marker metadata, supports 1-based sample/channel
  selection and metadata-only reads, and rejects truncated or contradictory
  files instead of returning an inconsistent EEG dataset.
- ``pop_writeeeg`` now writes standards-compatible GDF 1.25 files with
  float64 signal samples, channel labels, rational sampling rates, and GDF
  event tables. Numeric event types round-trip through independent readers;
  free-text event labels receive explicit file-local uint16 codes.
- Added the low-level EEGLAB plotting helpers ``cbar``, ``copyaxis``,
  ``forcelocs``, ``plotcurve``, ``sbplot``, and ``slider`` as standalone
  Matplotlib APIs. They support partial color scales, copied scientific axes,
  exact montage rotations, confidence-region highlighting, spanning subplot
  layouts, and pannable magnified figures. ``headplot`` also supports its
  ``example`` and ``cartesian`` command modes, and lighting-off plots now have
  explicit wireframe coverage.
- Neuroscan ``loadcnt`` and ``pop_loadcnt`` now provide standalone 16/32-bit
  CNT import with per-channel microvolt calibration, channel-blocked reads,
  partial sample ranges, response and boundary events, and optional ``.fdt``
  memory mapping. The File-IO CNT action uses this reader instead of MNE's
  ambiguous auto-detection path.
- Current EEGLAB legacy ICA/decomposition tests now have substantive Python
  ports for PCA and whitening, Varimax/Promax rotation, component orientation,
  variance and z-score ordering, deterministic k-means, spatial channel subsets,
  event-sequence time-warp selection, and historical ``runica`` entry points.
  The single maintained infomax engine backs all variants. EEGPrep corrects
  EEGLAB's all-negative ``posact`` orientation, dimension-reduced ``varsort``
  inverse, and multi-epoch ``zica`` peak-selection defects; interactive demo-only
  wrappers are covered by numerical source-recovery checks without becoming
  public APIs.
- Added standalone ``openbdf`` and ``readbdf`` record-level BDF access. The
  reader decodes signed 24-bit samples, supports physical calibration and
  variable per-channel sample counts, and correctly infers unknown record
  counts from three-byte BDF samples.
- ``MemmapData`` now covers current EEGLAB ``mmo`` construction, logical
  indexing over normal and transposed files, copy-on-write mutation, resizing,
  and deletion. Disk-backed EEG data remains disk-backed through rejection,
  epoching, baseline removal, FIR filtering, rereferencing, selection, and
  resampling without modifying the source mapping.
- Added standalone low-level signal and text helpers ``blockave``, ``eegfilt``,
  ``env``, ``movav``, ``loadeeg``, ``loadtxt``, ``parsetxt``,
  ``readneurodat``, and ``readtxtfile``. Legacy ``eegfilt`` now supports the
  even-tap least-squares filters and shortest valid epochs accepted by EEGLAB.
  Zero-phase filtering pads each segment with three times the filter order,
  matching MATLAB ``filtfilt``. Segments long enough for SciPy's previous
  default pad, including the continuous EEGLAB sample recording, keep the
  same samples; the shorter pad is what admits those short epochs.
  Truncated Neuroscan files retain complete sweeps without returning an
  incomplete zero-filled tail.
- Regular epoch generation, preceding-event timing, event alignment, BioSig
  event conversion, sphering, RMS summaries, int16 matrix reads, legacy
  lagged-regression artifact removal, ordered string selection, and
  superimposed histograms are now public standalone utilities. Their current
  EEGLAB tests now have deterministic ports, including substantive assertions
  where the MATLAB tests are commented out or ignore their own result.
- Added standalone numerical utilities for peak extraction, average reference,
  Gaussian/Gabor/Laplacian kernels, distance and assignment problems,
  correlation-based map matching, grouped summaries, PCA compression and
  reconstruction, interpolation, quantiles, matrix selection, and permutation
  handling. Python-facing indices are zero-based, and each function is covered
  by ports of the corresponding current EEGLAB test wrapper. Numerical edge
  contracts reject fractional indices and lossy complex casts, support MATLAB
  cell arrays loaded through SciPy, preserve complex PCA and interpolation, and
  avoid catastrophic cancellation in NaN-aware standard deviations. Grouped
  standard errors also use the finite sample count, correcting an EEGLAB helper
  defect when groups contain missing observations.
- Current EEGLAB visual-wrapper test ports now exercise plots through
  deterministic headless numerical and figure assertions. ``pop_plotdata``
  supports channel/component mode, 1-based trial selection, trial averaging,
  and single-trial overlays; ``pop_plottopo`` honors its positional title and
  single-trial arguments. ``pop_spectopo`` supports Blackman-Harris windows and
  ``blckhn``, and ``pop_envtopo`` accepts the legacy ``compnums=-N`` top-N
  shorthand. ``pop_topoplot`` renders signed ``EEG.chanmatrix`` grids even when
  channel locations are unavailable.
- The current EEGLAB GUI-helper tests now have deterministic Python ports.
  ``supergui`` builds declarative dialogs without entering a modal loop,
  ``inputgui`` supports a non-blocking ``mode="plot"`` workflow and renders
  radio buttons, and ``listdlg2`` accepts multi-line prompts and custom OK and
  Cancel labels.
- Core event and channel helpers now cover EEGLAB-compatible event insertion,
  event histograms and type counts, per-epoch field extraction, boundary-aware
  original latencies and context queries, channel-type lookup and montage
  matching/merging, ERP-window amplitude, and bad-time interpolation. These
  helpers are available directly from ``eegprep`` and run without an EEGLAB
  checkout.
- ``pop_loadset`` now supports EEGLAB's metadata-only and 1-based channel
  loading modes, while ``pop_fileio`` applies 1-based channel and inclusive
  sample/trial selections with consistent dataset bookkeeping. ``pop_writeeeg``
  now writes real EDF and BDF files through the installed writer instead of
  routing unsupported BDF format names into MNE's EDF-only exporter.
- Added ``pop_dipparams``, ``pop_erpimparams``, ``pop_erpparams``,
  ``pop_erspparams``, ``pop_specparams``, and ``pop_statparams`` for storing
  EEGLAB-compatible STUDY plotting and statistics settings. Analysis-defining
  range changes now invalidate stale measure caches. ``std_maketrialinfo`` now
  derives custom fields such as reaction time from each epoch's time-locking
  event and preserves those rows across later STUDY operations. Saving ``.set``
  files now supports event dictionaries with heterogeneous custom fields by
  encoding absent struct values as MATLAB empty arrays.
- ``pop_clean_rawdata`` now routes channel cleaning to the location-free
  algorithm when ``EEG["chanlocs"]`` is empty. It previously failed with an
  indexing error before the documented fallback could run.
- Current EEGLAB rejection-workflow test ports now cover threshold, statistical,
  spectral, trend, channel, epoch, and automatic rejection. Scripted
  ``pop_rejchan`` calls use EEGLAB's default threshold of 400; the interactive
  dialog continues to default to a normalized threshold of 5.
- Topographic interpolation no longer leaks NumPy 2 floating-point warnings for
  finite biharmonic matrix products.
- ``eeg_interp`` now accepts MATLAB-loaded object arrays of channel locations when
  restoring a full montage. ``pop_editeventfield`` now reads values from delimited
  text files, accepts MATLAB colon expressions for event indices, rebuilds ``urevent``
  after ``delold='yes'``, and ignores invalid rename/description requests like EEGLAB.
- ``pop_reref`` now accepts EEGLAB's ``method='standard'`` option explicitly. Other
  rereferencing methods fail clearly instead of being silently treated as standard,
  and NumPy 2's spurious finite-matrix warnings no longer leak from average reference.
- ``pop_selectevent`` now supports ``erroronempty='off'`` when a selection removes every
  epoch, returning an empty EEG instead of raising. Component ``pop_topoplot`` colorbars
  now use polarity labels only for scales that span zero; positive-only and negative-only
  scales show their exact numeric ``maplimits`` endpoints. ``eeg_decodechan`` now accepts
  EEGLAB's whitespace-separated channel-name form as well as Python sequences. ICA rejection
  marks can also derive their row count from ``icachansind`` when stored ICA matrices are absent.
- Current EEGLAB admin/GUI test ports exposed and fixed three parity defects:
  ``eeg_retrieve(ALLEEG, 0)`` now returns the empty no-current dataset,
  multi-dataset commands receive EEGLAB's marker in each dataset history, and
  ``listdlg2`` treats a scalar string as one selectable item while preserving
  its 1-based initial selection.
- Compatibility helpers now cover EEGLAB key/value validation, option-file and
  help-header parsing, history-value extraction and formatting, and the generic
  ``inputdlg2`` and ``errordlg2`` dialogs without runtime access to EEGLAB.
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
