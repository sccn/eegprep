.. _changelog:

=========
Changelog
=========

Notable changes to EEGPrep, newest first. Full release notes and downloads are on
the `GitHub Releases <https://github.com/sccn/eegprep/releases>`_ page.

Unreleased
==========

- Channel locations: enable Read, Look up, Plot 2-D/3-D, head-center optimization,
  rotation, transforms, coordinate conversions, head radius, type/reference
  editing, insertion/deletion, and both Save actions. Navigation and action
  history preserve edits in order, and datasets retain their data channels.
  Channel labels without coordinates trigger a lookup prompt using a packaged
  montage. Channel-edit history uses replayable MATLAB cell arguments
  (Stream 5, #550; F36, F49).
- Standard montage lookup applies the EEGLAB nose-axis rotation for
  ``standard_1005.elc`` and recognizes BESA ``.elp`` files. Reading polar
  ``.locs`` files uses an 85 mm head radius and preserves labels after repeated
  tabs. A lookup with no matching labels preserves existing coordinates
  (#550; F45, F46).
- ``pop_reref`` reference, exclusion, and related channel numbers, and
  ``eeg_interp`` channel numbers, are now **1-based** in scripts and GUI fields,
  matching ``pop_interp``. Existing scripts using NumPy row indices must add 1
  at these API boundaries; array indexing and low-level ``reref`` remain
  zero-based. GUI entry ``2 6`` excludes channels 2 and 6 and records those
  numbers in history. Console replay, CLI rereferencing, and spectral rejection
  use the same convention. Empty reference state displays as ``unknown``
  (#550, #569; F12, F61, F62, F70).
- Channel-coordinate edits reject nonnumeric or nonfinite values with a field
  name and correction instructions. This deliberately differs from the
  audited EEGLAB behavior that accepted ``theta='abc'``; a regression test
  locks the validation (#550; F43).

- Event selection records numeric criteria for numeric event codes and preserves
  string codes as text. Data selection preserves input precision; embedded
  save precision remains with the S1 stream (F50, #548).
- Cropping epoched data recalculates event latencies per epoch and preserves
  cached ICA alignment. The tutorial selection retains 129 inclusive samples
  and 8 events; the documented EEGLAB endpoint difference remains intentional
  (F44, #548).
- Select events warns when no events match and explains how to adjust the
  selection, including when boundary events are retained (F40, #548).
- About this dataset preserves comment line breaks, including blank lines, in
  the editor and saved comments (F48, #548).
- Extract epochs supports selecting several event types together in its picker
  (F124, #548).
- Parks–McClellan Estimate opens the ripple/attenuation sub-dialog and leaves
  the filter settings unchanged on Cancel (F101, #548).
- Event fields displays readable descriptions, supports editing them, and opens
  file pickers for existing and new fields, including paths with spaces (F100, #548).
- Data and ICA rejection menus fit within 1073×994 pixels and enable calculation,
  plotting, Help, and color controls. Calculated marks and history are retained
  when closing the menu. Probability and kurtosis PLOT open the scrolling
  rejection browser with calculated marks (F102/F105, #548).
- Select epochs or events opens tutorial datasets with MATLAB array descriptions
  and displays each readable description beside its corresponding field (F39, #548).
- Select data opens tutorial datasets whose channel types are empty MATLAB
  arrays. The channel-type picker is disabled when all types are absent or blank
  (release-readiness F37, #548).
- The Select data dialog validates channel numbers before conversion to
  Python indices. Range errors preserve the entered numbers and give the
  valid one-based range.
- The fuzz harness accepts platform-specific baseline-mean summation roundoff
  for ``pop_rmbase`` (#542) within four float64 eps times the largest finite
  output magnitude (floored at one). Larger gaps and other mismatch types
  remain findings. Baseline processing and stored tolerances are unchanged.
- The fuzz adapter converts EEGLAB's 1-based channel indices to Python
  ``pop_select``'s 0-based indices (#543). Duplicate-channel draws use the
  cropped channel count. Python out-of-range errors name ``pop_select``, the
  invalid indices, ``nbchan``, and the valid Python and EEGLAB/GUI ranges.
- Replaying a folder import records ``[STUDY, ALLEEG] = pop_importbids(...)``
  with ``return_study`` on, so the console rebuilds the same study.
  ``pop_importbids.m`` always returns ``STUDY`` and ``ALLEEG``. The direct
  call still returns the datasets unless ``return_study=True``. The history
  also keeps the dialog's ``recordings`` selection, along with task, session,
  run, and subject.
- ``pop_importbids(..., subjects=[1, 2])`` uses the same 1-based participant
  indices as the import dialog (``pop_importbids.m`` lines 163 and 281).
  Replaying the recorded ``'subjects', [1 2]`` command selects those
  participants. An index outside that list raises ``pop_importbids`` and
  names the valid range.
- ``pop_exportbids`` writes each study dataset under its own subject. The
  label comes from ``EEG.subject``, then the matching ``STUDY.datasetinfo``
  subject (``pop_exportbids.m`` lines 167-172). A dataset with neither set
  raises ``pop_exportbids: no subject info found...``. One dataset with no
  subject still uses the ``subject`` argument. ``createids`` assigns
  ``sub-001``, ``sub-002``, and so on to the unique subjects
  (``bids_export.m`` lines 605-608), so two sessions of one subject keep one
  id and ``participants.tsv`` has one row per subject. Session and run are
  taken from the dataset when they are set. The export dialog records a
  command that replays in ``eegprep-console`` with the dataset list, task,
  license, changes, and createids. EEGLAB leaves this command empty
  (``pop_exportbids.m`` lines 247-252).
- The BIDS import field "Import only the following participant indices"
  takes 1-based indices into ``participants.tsv`` order, or sorted ``sub-*``
  folder names when that file is missing (``pop_importbids.m`` lines 163 and
  281). The indices are mapped to subject labels before the file filter. A
  non-numeric or out-of-range value raises ``pop_importbids``. The history
  command records the numbers, for example ``'subjects', [1 2]``.
- ``pop_rejspec`` multitaper estimation stops each series on its own ``pmtm``
  test, ``sum(abs(S-S1))/nfft > tol``. A non-finite change stops that series,
  so one NaN sample returns instead of looping. Finite tutorial epochs are
  unchanged: ±30 dB and ±35 dB still mark nothing.
- Reject marked epochs records ``eeg_rejsuperpose`` only after the new-dataset
  dialog is accepted. Cancelling that dialog leaves the command history as it
  was.
- ``vararg2str`` continuous-array history uses MATLAB ``num2str``
  (``vararg2str.m:196-242``).
- ``pop_rejspec`` multitaper power is ``10*log10`` (dB). ``pop_rejspec.m:310``
  uses ``10*log`` (natural log). On ``eeglab_data_epochs_ica.set``, 15–30 Hz
  at ±35 dB marks no data epochs; EEGLAB marks epoch 37. At ±30 dB, EEGPrep
  marks no data epochs and no component epochs; EEGLAB marks data epoch 37
  and components 13, 14, 15, 26, and 79.
- ``pop_rejcont`` applies frequency limits to the real FFT bin frequencies.
  ``pop_rejcont.m:211`` stretches those bins with ``linspace`` from 0 to
  Nyquist. On ``eeglab_data.set``, 20–40 Hz at 10 dB keeps 27720 points in
  8 regions; EEGLAB keeps 28008 points in 9.
- ``pop_rejcont`` accepts a ``[min max]`` threshold and a cell of thresholds
  or frequency ranges (``pop_rejcont.m:172-177`` and ``pop_rejcont.m:239``).
  A single limit is still expanded to ``[-100 limit]``. On
  ``eeglab_data.set``, ``[-50 10]`` keeps 27720 points in 8 regions and
  ``[-5 10]`` keeps 3432 points in 17 regions.
- ``pop_rejcont`` history leaves ``eegplot`` out of the commented call
  (``pop_rejcont.m:370``). Accepting the marked regions records a separate
  ``pop_select( EEG, 'nopoint',[...]);`` entry (``pop_select.m:749``).
- A two-number threshold typed into the ``pop_rejcont`` dialog, such as
  ``-50 10``, stays ``[min max]``. ``pop_rejcont.m:147`` reads that field
  with ``str2double``, which becomes NaN and rejects nothing. On
  ``eeglab_data.set``, ``-50 10`` still keeps 27720 points in 8 regions.
- ``pop_rejcont`` with ``correct`` ``blank`` writes the zeroed samples back
  onto the dataset. A ``float32`` recording is copied by the floating-point
  conversion, so the zeros have to be stored explicitly. On
  ``eeglab_data.set``, 20-40 Hz at 10 dB with the default taper zeroes
  1312 columns.
- ``pop_eegthresh`` history uses the ``vararg2str`` form at
  ``pop_eegthresh.m:251-252``, with contiguous ranges collapsed. The
  reject flag is the value that was applied. ``pop_eegthresh.m:252``
  always writes 0, so replaying a scripted ``reject=1`` call in EEGLAB
  only marks the epochs. Dialog times use MATLAB ``num2str``:
  five significant digits below 1, and ``floor(log10(|x|))+5`` digits
  otherwise. The sample epoch file shows an end time of 1.9922;
  12.34567 is 12.3457 and 123.456789 is 123.4568.
- Update Marks shows the warning that epochs marked for rejection have been
  noted (``eeg_rejmacro.m:119-121``).
- ``pop_rejcont`` reports the Add trails field, the function name, and an
  example value when that field is not one number of seconds.
- Reject marked epochs history is ``pop_rejepoch( EEG, [1:6 9] ,0)``
  (``pop_rejepoch.m:101``).
- Reject marked epochs, for data and for ICA, stores the result with
  ``pop_newset`` (``eeglab.m:663-664`` and ``eeglab.m:676-677``).
- The reject-by-spectra dialog opens on FFT (``pop_rejspec.m:142-149``).
- Unchecking normalization in channel rejection relabels the threshold as
  an absolute limit (``pop_rejchan.m:86-89``).
- Multitaper spectra adapt until ``pmtm``'s convergence test on the full
  ``nfft``, with no iteration cap (``pop_rejspec.m:307-326``).
- ``pop_rejepoch(..., confirm=-1)`` follows ``pop_rejepoch.m:96-97`` and
  allows an empty dataset. Any other confirm value still names
  ``pop_rejepoch`` and tells you to clear a mark.
- ``pop_autorej`` reports which dialog field was not numeric, and shows a
  valid example, instead of a bare float conversion error.
- ``eeg_regepochs`` stores ``urevent`` as an array and inserts every regular
  event. ``pop_epoch`` drops an epoch whose window passes the end of the
  data and warns ``event N out of data boundary``.
- ``pop_eegthresh`` now opens the component dialog on components 1:5
  (``pop_eegthresh.m:120``) instead of every component.
- Reject marked epochs now builds ``rejglobal`` with ``eeg_rejsuperpose``
  before removing trials (``eeglab.m:663-677``). Rejecting every epoch raises
  a ``pop_rejepoch`` error that says how to clear a mark, instead of an empty
  dataset from ``pop_select``.
- Export marks to ICA reject now superposes data marks and copies
  ``rejglobal`` onto ``icarejmanual`` (``eeglab.m:661-662``). Export marks to
  data reject copies ICA marks onto ``rejmanual`` (``eeglab.m:674-675``).
- ``pop_rejchan`` now rejects probability and kurtosis channels with the
  two-sided mask from ``rejkurt.m:131`` and ``jointprob.m:133``, after the
  10% trimmed normalization at ``rejkurt.m:109-114``. The dialog offers
  Probability, Kurtosis, and Spectrum (``pop_rejchan.m:96``) and enables the
  frequency range only for Spectrum (``pop_rejchan.m:99-100``). History uses
  the ``vararg2str`` form at ``pop_rejchan.m:246``.
- ``pop_rejspec`` multitaper spectra follow ``pop_rejspec.m:307-326`` for
  the seven adaptive Slepian tapers and the ``eegthresh`` bin limits, and
  convert power with ``10*log10`` rather than the natural log at
  ``pop_rejspec.m:310``. On ``eeglab_data_epochs_ica.set`` that marks no
  epochs at ±30 dB or ±35 dB.
- ``pop_rejcont`` now rejects the trial-mean spectrum of overlapping windows
  and merges them with the rule at ``pop_rejcont.m:211-316``. On
  ``eeglab_data.set`` the 20-40 Hz, 10 dB settings remove 8 regions and keep
  27720 points. The scroll view uses the ``eegplot.m:457-473`` spacing, and
  the dialog has no Help button (``pop_rejcont.m:140``).
- Channel-mode automatic epoch rejection uses ``rejkurt``. ``pop_autorej.m:224``
  uses ``icarejkurt``, so on ``eeglab_data_epochs_ica.set`` EEGLAB keeps 75
  epochs and EEGPrep keeps 74.
- Clicking or dragging in the scroll window marks data again. The plot
  viewport was asked for ``mapToScene``, which only ``QGraphicsView``
  provides, so every left click raised ``AttributeError``. Continuous
  drags mark the 1-based sample range ``round(x * srate) + 1``
  (``eegplot.m`` lines 2110-2144). An epoched click marks or clears that
  whole epoch on mouse-down, including a one-sample epoch, and a drag does
  not spread to other epochs (``eegplot.m`` lines 2130-2135). Channel
  inspection, epoch inspection,
  and ICA component inspection share this path.
- Rejecting marked continuous data records ``EEG = eeg_eegrej( EEG, ...);``
  from the reduced two-column regions (``eeg_eegrej.m`` builds that command
  after ``combineregions``) and asks what to do with the new dataset through
  ``pop_newset``. Cancel keeps the original dataset and does not record the
  rejection. An empty mark does nothing (``eeg_eegrej.m`` lines 59-60).
- Tools > Inspect/reject data by eye shows the instruction dialog from
  ``pop_eegplot.m`` lines 180-186 before the browser opens. That menu
  calls ``pop_eegplot(EEG, 1)`` (``eeglab.m`` line 644). Cancel returns
  without opening the browser. Closing the dialog also stays on the
  current dataset; EEGLAB continues unless the button text is ``Cancel``.
  Plot > Channel data (scroll) passes ``reject`` explicitly
  (``eeglab.m`` line 683) and does not show the dialog.
- Rejecting marked epochs asks "Are you sure, you want to reject the
  labeled trials ?" (``pop_rejepoch.m`` lines 78-84) only when Reject is
  removing epochs. Update Marks does not ask. NO, or closing the dialog,
  leaves the epochs in place. YES removes them and opens ``pop_newset``.
  Clearing every mark and pressing Reject stores the manual marks as
  unmarked and does not add a rejection command. The history command uses
  ``pop_rejepoch( EEG, [..] ,0);`` (``pop_rejepoch.m`` line 101).
- File > Export opens the ``pop_export``, ``pop_writeeeg``, and ``pop_exportbids``
  dialogs (``pop_export.m`` lines 77-115, ``pop_writeeeg.m`` lines 52-80,
  ``pop_exportbids.m`` lines 48-83). ``pop_expica`` and ``pop_expevents`` stay
  file pickers, matching ``pop_expica.m`` lines 61-69 and ``pop_expevents.m``
  lines 67-73.
- File > BIDS tools opens the task, participant, and event dialogs from
  ``pop_taskinfo.m``, ``pop_participantinfo.m`` line 160, and ``pop_eventinfo.m``
  line 59. Values are stored on ``EEG.etc.bids`` so ``pop_saveset`` keeps them.
  An empty subject code raises an error from ``pop_participantinfo``.
- Importing a BIDS dataset replaces a missing ``chaninfo.plotrad`` or
  ``chaninfo.shrink`` with an empty array, so File > Save no longer fails while
  ``savemat`` is writing those fields. The import dialog matches
  ``pop_importbids.m`` lines 104-164 and loads an in-memory study. It does not
  copy datasets into ``derivatives/eeglab`` (``pop_importbids.m`` lines 839 and
  892 do).
- ``pop_writeeeg`` keeps every EDF and BDF sample and every event annotation.
  EEGLAB ``writeeeg.m`` line 139 shortens a block longer than 1000 samples to
  one second (``round(srate)`` samples per record). Biosig then pads the file
  to that record boundary and fills the extra samples with the digital minimum,
  which reads back as the channel physical minimum. A 2112-sample recording at
  200 Hz is 11 seconds from EEGLAB and stays 2112 samples in EEGPrep.
  ``pop_writeeeg.m`` line 97 records the command without a ``LASTCOM`` prefix.
  Epoched data is concatenated after the same confirmation as lines 53-57.
- ``pop_export`` writes ``%.Nf`` decimal places (``pop_export.m`` line 179).
  The dialog label still says "significant digits" (line 93). ICA columns use
  component numbers (lines 187-191). A non-transposed time row is labeled
  `` Time`` (line 217). ``pop_expica`` writes ``save -ascii`` text
  (``pop_expica.m`` line 79) with each value formatted ``%16.7e``.
  ``pop_expevents`` follows ``eeg_eventtable.m`` lines 123-159: a ``number``
  column, first-seen field order, and ``%f`` numeric cells.
- ``pop_exportbids`` writes ``README``, ``*_eeg.json``, ``*_electrodes.tsv``,
  and ``*_coordsystem.json``. The electrodes and coordinate-system names drop
  ``task-`` and ``desc-`` (``bids_writeelectrodefile.m`` lines 32-49 and
  272-316). Event tables use ``onset``, ``duration``, ``sample``, and ``value``
  (``bids_writeeventfile.m`` lines 128-197), and a numeric type ``1`` is written
  as ``1``. ``validate_bids`` reports a missing README, ``*_eeg.json``,
  electrodes file, or coordsystem file. It runs the official bids-validator
  when ``bids-validator``, ``npx``, or ``deno`` is on PATH, and otherwise says
  the check was sidecars only and how to install the validator. It does not add
  ``eegprep_bids_validation.json`` to the dataset. The File menu exports a study
  only (``pop_exportbids.m`` line 33). ``pop_exportbids(EEG, folder)`` still
  writes one dataset from memory.
- BIDS import uses the ``value`` column as the EEG event type when that column
  exists (``bids_importeventfile.m`` line 14, ``pop_importbids.m`` line 181).
  ``trial_type`` is still used when ``value`` is absent. A ``nan`` event type
  is stored as an empty string. The import dialog's task, session, run, and
  recording selectors are the popup and listboxes from ``pop_importbids.m``
  lines 112-118, filled by scanning the folder. Empty session, run, and
  recording rows are omitted (lines 128-142).
- The export dialog's "Number of significant digits" field is passed through
  as ``%.Nf`` decimal places, which is what ``pop_export.m`` line 179 does
  with the string from line 93. The function default remains 7 decimal places
  (line 129). The dialog default remains 4 (line 94) and is the edit field's
  value, so pressing Ok without changing that field writes 4 decimal places.
- ``newtimef`` and ``timefreq`` keep every requested output time, including
  times that round onto the same sample. ``timefreq.m`` ``gettimes`` drops
  duplicates only when the index list is shorter than its unique values, so
  that drop never runs. A 128-sample epoch from -500 ms to 492 ms at 128 Hz,
  cycles ``[3, 0.8]``, and ``padratio`` 1 returns 111 times.
- ``timefreq`` spaces that time grid the way MATLAB ``linspace`` does:
  ``start + i * (stop - start) / (n - 1)``, and the last point is the end
  limit. Spacing the step first moved 26 of those 111 times by one unit in
  the last place, and five of them (including one in the baseline) crossed a
  half-sample, so every ERSP row picked up a constant dB offset.
- ``plotfresp`` accepts the symmetric and antisymmetric FIR filters that
  ``fir_filterdcpadded`` already runs. ``plotfresp.m`` line 67 compares
  coefficients with exact reversal, so a windowed-sinc highpass whose taps
  are linear-phase but not bit-identical raised "Onepass-zerophase filtering
  is only allowed for linear-phase FIR filters" and the default frequency-
  response plot never opened.
- ``eeg_interp`` limits a time range with inclusive ``EEG.times`` in
  milliseconds, on continuous and epoched data. A full-range list equal to
  ``[xmin, xmax]`` no longer restores the second-to-last sample. Pinned
  ``eeg_interp.m`` line 322 restores ``floor(t*srate)`` and stops short of
  the last sample; this follows the inclusive window in eeglab pull 980.
- ``pop_interp`` keeps a scripted ``t_range``. Pinned ``pop_interp.m`` line 61
  sets ``t_range = ''`` and interpolates the whole recording; the help text
  and a regression test record that difference.
- ``pop_interp`` channel indices are 1-based. Channel 1 is the first channel.
  Replayed history keeps those indices.
- ``runica`` extended kurtosis draws ``fix(rand*datalength)`` on the
  ``RandomState(5489)`` stream and divides by ``m2`` (``runica.m`` lines
  890-903). With ``posact`` off, component order uses sphered-channel energy
  (``runica.m`` lines 1472 and 1519).
- ``newcrossf`` defaults the frequency range to ``[0, 50]`` Hz
  (``newcrossf.m`` lines 396-397). A log scale still lifts a zero lower edge
  to 3 Hz (line 461). The previous default followed ``timefreq`` up to
  Nyquist, so a 128 Hz epoch with ``padratio`` 1 ran 6-64 Hz instead of 6-50 Hz.
- The cross-coherence and time-frequency dialogs show epoch limits with
  MATLAB ``int2str`` rounding, halves away from zero (``pop_newcrossf.m``
  line 87, ``pop_newtimef.m`` line 172), so 0.9996 s is ``1000`` ms. When
  the frequency field is empty and the epoch starts before zero,
  ``pop_newtimef`` sets ``winsize`` the same way (``pop_newtimef.m`` line
  292): ``xmin`` -0.1 s at 256 Hz is 26 samples.
- Sample indices that follow MATLAB ``round`` send halves away from zero.
  ``[25 800]`` ms at 100 Hz with ``xmin`` 0 starts at sample 3
  (``pop_newtimef.m`` lines 305-306; ``pop_newcrossf.m`` line 188 uses the
  same ``round``). The same tie is used for time-warp frames
  (``newtimef.m`` lines 630 and 635), the ``envtopo`` contribution window
  (lines 418-419), ``dftfilt`` tapers (lines 70-71), ``spectopo`` percent
  and window length (lines 320 and 895), scroll-window bounds and click
  latencies (``eegplot.m`` lines 1318-1319, 1403, and 2028-2029), rejection
  regions (``eegrej.m`` line 82), the kurtosis and joint-probability trim
  (``rejkurt.m`` line 110, ``jointprob.m`` line 115), scalp-map and epoch
  dialogs (``pop_topoplot.m`` lines 104 and 391, ``pop_headplot.m`` line
  159, ``pop_epoch.m`` line 122), and epoch latencies (``eeg_checkset.m``
  line 652).
- ``pop_newtimef`` takes the EEGLAB sample window (``pop_newtimef.m`` lines
  305-306) and passes those requested time limits into the decomposition.
  ``pop_newcrossf`` uses that same inclusive end. Pinned ``pop_newcrossf.m``
  line 189 omits the +1, so EEGLAB drops the last sample of a
  whole-millisecond limit; the help text and a regression test record that
  difference. The previous slice dropped the sample after that limit and evaluated
  the decomposition on the kept sample times, so an explicit baseline did
  not move the result.
- ``pop_fileio_brainvision_mat`` channel locations include empty ``ref`` and
  ``type`` on every channel. MATLAB ``eeg_checkset`` fills a missing
  ``chanlocs.ref`` with ``''`` (``eeg_checkset.m`` lines 1105-1117) and the
  checked structure also carries empty ``type``. An unknown channel variable
  is still appended once.
- ``pop_fileio_brainvision_mat`` reads a BrainVision Analyzer MATLAB export
  the way bva-io ``pop_loadbva.m`` lines 46-139 does: one variable per channel,
  ``SampleRate``, the ``t`` axis in milliseconds, and ``Markers``. Channel
  locations are converted with ``convertlocs(..., 'sphbesa2all')`` and the
  BESA angles are then removed (lines 101-103), leaving theta, radius, X/Y/Z,
  and spherical coordinates. A marker with an empty ``Description`` is
  skipped (lines 112-113). A name such as ``xFp1`` matches channel ``Fp1``.
  An unknown channel variable is appended once. ``pop_loadbva.m`` lines 61-65
  each call ``chanstruct(end+1)``, so EEGLAB appends four records (one named
  record and three empty ones). A MATLAB v7.3 file or any other ``.mat``
  layout raises an error instead of returning a one-channel guess. The
  history command remains ``pop_fileio_brainvision_mat``. The dataset
  filename is left empty so a later resave cannot overwrite the ``.mat``
  export.
- File > Clear dataset(s) and Edit > Delete dataset(s) from memory ask for
  confirmation in the same dialog as EEGLAB ``pop_delset.m`` lines 56-70
  ("Delete dataset -- pop_delset()", "Dataset(s) to delete:"). Cancel deletes
  nothing. With only one dataset loaded, the menu refuses and points at
  File > Clear study / Clear all. EEGLAB raises "Restart EEGLAB" for that
  case (``pop_delset.m`` lines 52-54) and also blocks a direct call.
  ``pop_delset(ALLEEG, index)`` still deletes the last dataset. A dataset
  number that is not loaded raises ``ValueError`` from ``pop_delset``
  (previously ``IndexError``). EEGLAB reports that case as
  ``Error: no such dataset``.
- ``pop_saveset`` checks that the ``.set`` file exists before it marks the
  dataset saved. If the parent folder cannot be created or the file is not
  written, it raises an error that names ``pop_saveset`` and the path.
  Creating a missing parent folder was already the behavior.
- Edit > Event values and Edit > Channel locations treat the number field as
  a goto. Enter shows that event or channel, and a number outside the list
  is clamped to the ends, as in ``pop_editeventvals.m`` lines 153-160 and
  571. An edit is applied to the record on screen before the arrows or a
  typed number leave it, and Ok records every edit. EEGLAB writes each field
  from its edit callback before navigation. A number that is still out of
  range raises an error from ``pop_editeventvals`` or ``pop_chanedit``
  instead of ``list index out of range``.
- ``pop_saveset`` resave writes only an existing ``.set`` file. File > Resave
  on a dataset imported from ``.vhdr``, ``.edf``, or another raw file opens
  Save as instead of replacing that recording. Any filename extension is
  replaced with ``.set`` (``pop_saveset.m`` lines 126-127). File > Save
  current dataset as asks before replacing that ``.set`` file. Typing
  ``test.edf`` writes ``test.set`` and does not ask about ``test.edf``.
  EEGLAB prompts
  when the filename is empty (lines 81-96). Its resave path then copies
  ``EEG.filename`` unchanged (line 192), which would overwrite a raw source
  if an importer had stored one. Resave here does not do that.
- ``pop_rmbase`` returns an empty dataset unchanged and logs a warning.
  That includes a zero-epoch result, such as epoching that selected no
  events, and continuous data with no samples. EEGLAB prints the same
  condition and returns (``pop_rmbase.m`` lines 69-71).
- ``eeg_eegrej`` inserts a numeric ``-99`` boundary when ``option_boundary99``
  is on and the first event type is numeric. ``eeg_boundarytype`` looks only
  at that first type. A text first type inserts ``boundary``. A later text
  type leaves the inserted ``-99`` in place until the event-consistency
  check turns it into ``'-99'`` (``eeg_eegrej.m`` line 144,
  ``eeg_checkset.m`` lines 395-400).
- ``pop_select`` with ``sorttrial='off'`` keeps the first copy of a repeated
  trial. ``trial=[5, 4, 3, 2, 1, 5]`` stays ``[5, 4, 3, 2, 1]``. EEGLAB
  ``unique_bc`` keeps the last copy, ``[4, 3, 2, 1, 5]``
  (``pop_select.m`` lines 282-286).
- ``pop_select`` names an epoched ``notime`` span that misses both epoch
  edges, and ``pop_epoch`` names an ``eventindices`` value past the last
  event. Both are the same failure EEGLAB reports for those inputs.
- A dataset saved with ``trials=0`` loads the empty epoch stack from a
  0-byte ``.fdt``. The sidecar reader used to treat that trial count as 1
  and expect ``nbchan * pnts`` samples.
- ``pop_rmbase`` ignores a NaN in the baseline and, on continuous data,
  still removes the baseline inside each boundary segment. EEGLAB uses
  ``nan_mean`` and those segments only when the first event type is text
  (``pop_rmbase.m`` lines 176-198, ``rmbase.m`` lines 108-110). A numeric
  first type, no events, or epoched data takes ``mean()`` at
  ``pop_rmbase.m`` line 202, so one baseline NaN fills the channel and
  boundaries are ignored.
- ``eeg_checkset`` turns a mix of numeric and text event types into text
  with ``num2str``. The event-consistency check inspects every event
  (``eeg_checkset.m`` line 395). The full check decides from the first
  100 (lines 1236-1242). ``65`` becomes ``'65'``, ``pi`` becomes
  ``'3.1416'``, and ``1e20`` becomes ``'1e+20'``. An all-numeric list
  stays numeric.
- ``pop_clean_rawdata`` skips ASR and warns when calibration is shorter than
  one 0.5 s RMS window, or yields fewer than two of those windows. The
  44-sample, 128 Hz crop in issue 524 stays identical, including its two
  NaNs, when the other cleaning stages are off. ``clean_asr`` still raises
  ``ASRCalibrationError``. EEGLAB prints that calibration error and returns
  before window removal; a requested window stage still runs here.
- ``pop_clean_rawdata`` still removes bursts when only two or three 0.5 s
  RMS windows are available. Eleven fit widths round to zero on the
  21-channel, 89-sample, 128 Hz crop in issue 526. EEGLAB indexes that
  width and returns the recording unchanged. The valid widths are still
  used here.
- ``pop_clean_rawdata`` with ``Distance='riemannian'`` warns and leaves the
  burst stage unchanged when Riemannian calibration is shorter than one
  0.1 s RMS window or yields fewer than two windows. ``clean_asr`` still
  raises ``ASRCalibrationError``. EEGLAB calls that path outside the
  ``try``/``catch`` in ``clean_artifacts.m`` (lines 267-268). Those checks run
  before covariance or eigendecomposition, so a constant recording raises the
  same error. Short-calibration errors name the function, the sample counts,
  and whether to record more data or set ``BurstCriterion`` off.
- ``asr_process`` and ``asr_process_r`` refresh the reconstruction on the
  last sample of a chunk when the length modulo the step is 1. A
  29-channel, 257-sample recording at 128 Hz is processed as 289 samples
  with a step of 32, and that last sample was previously left raw.
- ``asr_calibrate`` leaves non-finite samples in the caller's array.
  ``clean_asr`` reflects that recording, and ``asr_process`` replaces
  non-finite samples with 0 only after prepending the carry. A NaN in the
  last half-window was reflected as twice the final sample. On channels
  24-30 of the sample file, samples 158-413 at 128 Hz, with crop channel 1
  samples 245-246 set to NaN, that lookahead was 133.655 and the cleaned
  data differed by 11.593.

Version 0.5.0
=============

- This version, 0.5.0, is shared with ``eegprep-lean``.
- ``dir(eegprep)`` lists every public name, and ``eegprep.help()`` prints
  those names with a one-line summary taken from the docstring. Describing a
  name does not import its module. ``help`` is not in ``__all__``, so
  ``from eegprep import *`` does not replace the builtin.
- A plain ``pip install eegprep`` no longer includes mne, pybids, oct2py, and
  picard. It includes NumPy, SciPy, Matplotlib, psutil, pyedflib, and h5py, so
  loading, cleaning, plotting, and EDF, BDF, and GDF export need no extra.
  Those four libraries are the ``mne``, ``pybids``, ``eeglab``, and ``picard``
  extras. ``pip install "eegprep[full]"`` restores the previous install.
  A missing library raises ``ImportError`` naming the library, the function,
  and ``pip install "eegprep[<extra>]"``, and it mentions ``eegprep[full]``.
  When another path does not need that library, the message says what to do
  instead. The minimal preprocessing image
  installs the ``pybids`` and ``mne`` extras so it can read a BIDS dataset
  and EDF or BDF files. ``eegprep-lean`` stays a separate package.
- ``eeg_eegrej`` sets ``trials`` from the data shape, then runs the full
  event-consistency check. That keeps events when ``trials`` was 0.
  EEGLAB checks the bounds first and drops them. The full check also
  snaps ``xmax`` onto the sample grid when it no longer matches ``pnts``.
  It then drops a trailing event whose latency is more than half a sample
  past the data. Events come back as a list of dicts. A structured array is converted;
  any other event value raises. Events before half a sample are removed, events
  are sorted, and an empty duration becomes 0 once any event has one. A NaN
  duration is kept.
- ``eeg_checkset`` rounds ``srate * (xmax - xmin) + 1`` half away from
  zero before deciding whether ``xmax`` is on the sample grid. A count of
  100.5 frames snaps ``xmax``; Python's half-to-even round would leave it.
- ``clean_windows`` keeps the proportional ``xmax`` from ``eeg_eegrej`` and
  snaps it to ``xmin + (pnts - 1) / srate`` only when that value is off the
  sample grid, including ``xmax = xmin - 1/srate`` when every sample is
  removed. A previous ``clean_sample_mask`` whose count of ones differs from
  the new mask is left unchanged. The power fit raises ``ValueError`` only
  when a rounded search width is 1 or less; 11 to 22 windows still fit.
  ``clean_asr`` then calibrates on the whole recording.
- ``pop_saveset`` writes an event ``urevent`` one-based only when that
  event has the field. A later event without ``urevent`` no longer raises
  ``KeyError``, and a later ``urevent`` is still shifted when the first
  event has no such field. ``None`` and ``[]`` are written empty instead
  of raising ``TypeError``. Structured event records are shifted the same
  way, and the in-memory events stay unchanged.
- ``clean_channels`` always draws ``round(subset_size * nbchan)`` channels,
  including when ``subset_size`` is at least 1. A count larger than the
  channel total raises ``ValueError``. The noise filter uses
  time-domain filtering for a short FIR, so a few NaNs no longer blank the
  whole channel and drop it from the correlation.
- ``eeg_eegrej`` merges overlapping, reversed, and fractional removal
  intervals before shifting events, and leaves a one-sample gap as two
  intervals. Removing every sample sets ``xmax`` to ``-1/srate``.
- ``pop_eegfiltnew`` recomputes the transition width after the default filter
  order is rounded up to an even length, matching EEGLAB. A 1 Hz high-pass
  at 128 Hz uses order 424 and a transition of about 0.996 Hz.
- ``pop_loadset`` reads v7 ``.set`` files without dropping singleton
  dimensions. ``data`` stays ``nbchan`` by ``pnts`` by ``trials``, and
  ``icaweights``, ``icasphere``, ``icawinv``, and ``icaact`` keep a
  length-1 component or channel axis, including a 1-by-1 sphere.
  Multi-dimensional cells and struct arrays keep their MATLAB shape, and
  empty arrays keep the dimensions stored in the file, so ``[]`` stays
  0-by-0. Continuous ``icaact`` is ``ncomp`` by ``pnts`` and is the
  column-major product in the data array's dtype. ``eeg_runica``,
  ``eeg_picard``, and ``eeg_amica`` return that same continuous shape.
  ``pop_saveset`` writes
  ``icaact`` as ``[]`` and leaves the in-memory activations in place.
  ``pop_loadset`` recomputes them from the decomposition and the data.
  ``pop_subcomp`` projects those stored activations. Event and channel
  indices load as integers, and integral ``run`` and ``session`` labels
  stay integers. MATLAB ``load`` keeps those dimensions, and
  ``eeg_checkset`` requires ``size(EEG.data, 1) == EEG.nbchan``.
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
- ``statcond`` ANOVA F and p values are float64. The sums of squares follow
  the double-precision accumulation in EEGLAB ``anova1_cell.m``,
  ``anova1rm_cell.m``, ``anova2_cell.m``, and ``anova2rm_cell.m`` after
  EEGLAB #977. Parametric p-values follow ``myfpval.m`` lines 11-19: the
  complement of the incomplete beta when ``F < df2/df1``, and the upper tail
  in its symmetric form otherwise (``x = df2/(df2+df1*F)``, parameters
  ``(df2/2, df1/2)``). That branch matches ``myfpval``'s choice. F resampling
  arrays are float64 when ``arraycomp`` is on, because ``statcond.m`` lines
  384 and 421 replace the single preallocation at line 261 with the double
  ANOVA output, and float32 when ``arraycomp`` is off, because lines 398-399
  and 435-436 assign into that single array. ``concatdata.m`` stores the
  draws as single and ``ttest_cell`` keeps that class, so t-test resampling
  arrays and their confidence intervals stay float32 for both ``arraycomp``
  modes. A confidence interval follows the surrogate class. Supplied stats
  and resampling arrays keep the class they were given when ``naccu`` matches
  the supplied length. A supplied statistic is accepted when its shape matches
  the resampling array without the last axis after trailing length-1 dimensions
  are removed. A transpose or a flattened column is rejected. p-values stay float64.
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
- ``runica``, ``runicalowmem``, ``dftfilt3``, ``make_timewarp``,
  ``newtimef``, ``pop_biosig``, ``pop_fileio``, and ``pop_editoptions`` accept
  EEGLAB positional name/value pairs as well as keywords. ``make_timewarp`` also
  accepts ``baselineLatency``, ``eventConditions``, ``maxSTDForAbsolute``,
  and ``maxSTDForRelative``. ``pop_editoptions("option_storedisk", 1)``
  updates that option. ``dftfilt3`` still rejects ``type``; EEGLAB's error
  text lists it, but ``dftfilt3.m`` does not accept it. An omitted ``runica``
  ``rndreset`` stays off (``RandomState(5489)``). ``runica.m`` line 205 still
  sets ``DEFAULT_RESETRANDOMSEED = true``, but the help at lines 48-50 says
  the default has been ``'off'`` since 2015. ``newtimef`` accepts every
  name in the ``newtimef.m`` ``finputcheck`` list (lines 513-587). Mode
  ``ignore`` (line 587) forwards ``ffttaper``, ``wavelet``, and
  ``timestretch`` to ``timefreq`` (line 1188). A forwarded ``wavelet``
  replaces ``cycles`` inside ``timefreq`` because lines 1186-1188 pass
  ``g.cycles`` first and the last value wins. EEGLAB does not update
  ``newtimef``'s own ``g.cycles``, so the FFT scale at lines 1207-1209
  (``2/0.375/winsize`` when ``g.cycles(1)==0``, squared for power; the
  default ``winsize`` is line 453) follows the original ``cycles``. A
  ``wavelet`` of 0 therefore runs an unscaled FFT, and ``cycles`` 0 with a
  wavelet applies that FFT scale to the wavelet. EEGPrep scales the
  decomposition that ran. Any other name, including ``overlap``, is an
  undefined ``timefreq`` argument, reported with the caller's spelling.
  ``freqrange``, ``erspboot``, and
  ``itcboot`` are undefined arguments. ``rng`` is an EEGPrep-only name
  that seeds the bootstrap. A scalar ``freqs`` keeps its first value
  when ``maxfreq`` replaces ``freqs(2)``. ``ntimesout`` sets the time
  count when ``timesout`` is not a vector (lines 1176-1182). ``itctype``
  replaces ``type`` unless it is ``phasecoher`` (line 591). The default
  frequency range is ``[0, 50]``; ``maxfreq`` above Nyquist is capped and
  written into ``freqs`` (lines 704-709). Integer-valued MATLAB doubles such
  as ``runica(..., 'pca', 2)`` are left for the shared boundary converter.
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
