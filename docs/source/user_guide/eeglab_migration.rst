.. _eeglab_migration:

========================
EEGLAB Migration Notes
========================

EEGPrep is a Python port, not a MATLAB compatibility layer. Use these notes
when translating EEGLAB habits into EEGPrep scripts and GUI workflows.

What Carries Over
=================

.. list-table::
   :header-rows: 1

   * - EEGLAB habit
     - EEGPrep equivalent
   * - ``pop_loadset``, ``pop_saveset``, ``pop_resample``, ``pop_epoch``
     - Same public names, Python arguments, and replayable command strings.
   * - ``EEG``, ``ALLEEG``, ``CURRENTSET``, ``LASTCOM``, ``ALLCOM``
     - Same workspace names inside ``eegprep-console``.
   * - GUI menu action then command history
     - GUI actions update ``LASTCOM`` and ``ALLCOM`` in the shared session.
   * - ``chanlocs``, ``event``, ``urevent``, ``epoch``, ICA fields
     - Same dictionary field names.
   * - Bundled plugins
     - clean_rawdata, FIRFilt, ICLabel, DIPFIT, and EEG-BIDS are in-package
       Python ports or explicit standalone boundaries.
   * - STUDY
     - Implemented as EEGPrep-owned dictionaries and JSON-safe measure caches.

What Changes
============

.. list-table::
   :header-rows: 1

   * - Topic
     - Migration rule
   * - Arrays
     - Use Python/NumPy zero-based indexing when slicing data.
   * - User-facing selectors
     - Keep channel, component, dataset, and epoch numbers one-based in GUI
       dialogs and command strings unless a function documents otherwise.
   * - Globals
     - EEGPrep functions take explicit inputs. The shared console namespace is
       session state, not a hidden global dependency.
   * - Runtime EEGLAB checkout
     - Installed EEGPrep does not read from ``src/eegprep/eeglab``.
   * - Optional toolboxes
     - MATLAB-only LIMO, FieldTrip STUDY source statistics, MRI BEM creation,
       AFNI atlas clipping, and unavailable ICLabel network artifacts fail
       clearly instead of being faked.
   * - Scripts
     - Assign return values explicitly outside ``eegprep-console``.

Command Translation Examples
============================

.. list-table::
   :header-rows: 1

   * - Migration intent
     - EEGPrep Python
   * - Load a dataset
     - ``EEG = pop_loadset("sample_data/eeglab_data.set")``
   * - Filter
     - ``EEG = pop_eegfiltnew(EEG, locutoff=1, hicutoff=40, plotfreqz=False)``
   * - Resample
     - ``EEG = pop_resample(EEG, 64)``
   * - Average reference
     - ``EEG = pop_reref(EEG, [])``
   * - Epoch
     - ``EEG = pop_epoch(EEG, ["square"], [-1, 2])``
   * - Run ICA
     - ``EEG = pop_runica(EEG, icatype="picard", gui=False)``
   * - Classify ICs
     - ``EEG = pop_iclabel(EEG, "default")``
   * - Create a STUDY
     - ``STUDY, ALLEEG, com = pop_study(None, ALLEEG, name="My study", return_com=True)``

History Replay
==============

In MATLAB EEGLAB, menu commands appear in the MATLAB command window. In
EEGPrep, use ``LASTCOM`` and ``ALLCOM``:

.. code-block:: python

   EEG, LASTCOM = pop_resample(EEG, 64, return_com=True)
   ALLCOM.append(LASTCOM)

Inside ``eegprep-console``, a bare call can update the current dataset:

.. code-block:: python

   pop_resample(EEG, 64)

In normal Python, assign the return value:

.. code-block:: python

   EEG = pop_resample(EEG, 64)

Indexing Boundaries
===================

EEGPrep intentionally keeps both conventions visible:

* Python arrays are zero-based: ``EEG["data"][0, :]`` is the first channel.
* GUI channel/component/dataset selectors are one-based.
* Event ``latency`` fields are sample positions in the EEGLAB-facing event
  model.
* ``icachansind`` is normalized after loading so internal indexing is
  consistent, then converted back when saving.

MATLAB Structure Transport
==========================

When passing lists of dictionaries through the MATLAB bridge, numeric and text
values in the same field retain their individual types regardless of record
order. For example, ``42.0`` remains a MATLAB double when another record holds
``"boundary"``; it does not become ``"42.0"``. EEGLAB's ``eeg_checkset`` may
subsequently normalize mixed event types to strings as part of event validation.

Known Differences from EEGLAB
=============================

These are intentional. EEGPrep does not reproduce the EEGLAB result.

.. list-table::
   :header-rows: 1

   * - Topic
     - What EEGPrep does
   * - STUDY ``pop_clust`` algorithm ``kmeans``
     - EEGLAB calls ``kmeans(data, k, 'replicates', 10, 'emptyaction', 'drop')``.
       Cluster numbers follow MATLAB's random stream and are not reproduced.
       EEGPrep uses a seeded NumPy initialization and Lloyd updates.
   * - STUDY clustering dialog
     - EEGLAB's dialog calls ``kmeans_st(real(clustdata), clus_num, 150)``
       with random restarts. The command-line path calls ``kmeanscluster()``.
       EEGPrep's dialog uses the ``kmeanscluster`` route.
   * - ASR calibration on a short recording
     - ``clean_asr`` still removes high-power calibration windows when
       ``fit_eeg_distribution`` does not return a positive scale. The scale is
       then the median and MAD of each channel's window RMS. EEGLAB's fit
       errors once only a few windows are available, and ``clean_asr`` then
       calibrates on every sample.
   * - ASR on a very short recording
     - With calibration reference selection off, ``clean_asr`` still returns
       cleaned data when only two or three RMS windows are available.
       EEGLAB's calibration fit indexes a zero window width and errors.
   * - One ASR calibration window
     - ``clean_asr`` raises ``ValueError`` when fewer than two calibration
       RMS windows are available, and when the calibration is shorter than
       one window. The error names the window count, the sample count, and
       the window length. EEGLAB raises an error that asks for different
       ``BurstCriterionRefTolerances``. Neither call returns cleaned data.
   * - ASR at a low sampling rate
     - ``clean_asr`` raises an error naming the sampling rate when the default
       spectral breakpoints are not monotonic. ``pop_clean_rawdata`` warns and
       leaves the recording unchanged. EEGLAB's ``clean_asr`` reports that
       ``yulewalk`` was not found, and ``clean_artifacts`` prints that message
       and returns the recording.
   * - NaNs during re-referencing
     - ``pop_reref`` leaves a NaN on a channel only when that sample or the
       reference subtracted from it is NaN. EEGLAB multiplies by a reference
       matrix that contains explicit zeros, so ``0*NaN`` marks every channel
       at that time.
   * - BDF ``bdfeventmode``
     - ``pop_biosig`` decodes modes 1-8 and 99. EEGLAB passes the mode to
       ``sopen`` as ``BDF:[n]``. The ``sopen`` regexp matches ``BDF:n`` only,
       so the brackets are ignored and EEGLAB always decodes mode 4.
   * - BDF ``blockrange`` event types
     - ``pop_biosig`` keeps each trigger code on the latency of its own edge
       when ``blockrange`` starts after an earlier event.
       ``biosig2eeglabevent`` writes the latency on ``event(count)`` and the
       type on ``event(index)``. After events before the window are skipped,
       ``count`` and ``index`` diverge, so a kept event receives another
       event's type or an empty type that EEGLAB returns as NaN. The
       latencies match.
   * - BDF trigger codes
     - ``pop_biosig`` stores the integer Status code, including codes above
       255. With no ``blockrange``, EEGLAB's ``importannot`` (default on)
       maps codes above 255 through BioSig ``eventcodes.txt``, so ``0x105``
       becomes ``artifact:Failing Electrode``. ``eeg_checkset`` then stores
       every event type as a string.
   * - BioSemi Status channel
     - ``pop_biosig`` removes ``Status`` only when ``rmeventchan`` is
       ``on``. ``rmeventchan="off"`` keeps the channel both when triggers are
       imported and when ``importevent="off"``. ``importevent="off"`` leaves
       the events already stored by the file reader. EEGLAB's ``sopen``
       deletes the Status column from ``ReRefMx`` (``sopen.m`` lines
       1037-1038) for every BDF file before the samples are read, so the
       returned data never contains ``Status``.
   * - BDF event modes 3 and 5
     - ``pop_biosig`` keeps each trigger code on the rising or falling edge
       that produced it. ``bdf2biosig_events.m`` lines 88-89 and 104-105 build
       ``POS`` in time order and ``TYP`` as all rises followed by all falls,
       so those two arrays no longer correspond.
   * - EGI ``epoc`` pauses
     - ``pop_readegi`` renames every ``epoc`` event to ``boundary`` when no
       ``tim0`` event is present. EEGLAB's ``for index = indepoc`` walks a
       column vector once. A single pause is renamed. Two or more pauses make
       ``EEG.event(index).type`` fail, the surrounding ``try`` discards the
       error, and every pause stays ``epoc``.
   * - EGI ``epoc`` with ``tim0``
     - ``pop_readegi`` reshapes a continuous recording when both codes are
       present, the sample count divides evenly by the number of ``epoc``
       events plus one, and the ``epoc`` latencies are exactly ``k * pnts``
       for ``k = 1 .. n``. That latency is the last sample of epoch ``k``.
       The ``epoc`` events are then removed; the pauses become the trial
       boundaries, and the other events stay. If the pauses are spaced
       differently, or the length does not divide, the recording stays
       continuous and every event stays. EEGLAB deletes ``EEG.event(1)``
       with the channel-loop index and sets ``trials`` to that epoch count.
       ``eeg_checkset`` then sees two-dimensional data and sets ``trials``
       back to 1, so the samples stay continuous and only the first event
       is removed. When the length does not divide evenly, EEGLAB leaves
       the events in place.
   * - Epoched event sort
     - ``eeg_checkset(..., 'eventconsistency')`` sorts by epoch, then by
       latency. EEGLAB asks ``pop_editeventvals`` to sort on epoch and then
       latency, but that function takes the epoch order from the unsorted
       events and applies it to the latency-sorted list. Events A (epoch 2,
       latency 75), B (1, 15), C (1, 5), D (3, 145) come out C, B, A, D here
       and B, A, C, D in EEGLAB. Events X (2, 85), Y (2, 72), Z (1, 8) come
       out Z, Y, X here and X, Z, Y in EEGLAB.
   * - Numeric boundary types
     - Consecutive equal-latency boundaries are merged whether ``type`` is
       the string ``boundary`` or the number ``-99`` with
       ``option_boundary99`` on. EEGLAB skips that merge when the first
       event type is numeric, so two ``-99`` boundaries with durations 1
       and 2 stay separate. EEGPrep keeps one boundary whose duration is 3.

Plugin Migration
================

Bundled plugins are importable from ``eegprep``:

.. code-block:: python

   from eegprep import pop_clean_rawdata, pop_eegfiltnew, pop_iclabel, pop_dipfit_settings

External EEGLAB MATLAB plugins are not loaded by path. EEGPrep extensions are
normal Python packages discovered through the ``eegprep.extensions`` entry
point. See :ref:`plugins` and :ref:`extensions` for the package format.

When to Stay in MNE or MATLAB
=============================

Use MNE-Python when your analysis depends on MNE-native object workflows,
forward models, reports, or statistics. Use MATLAB EEGLAB when you need an
unported MATLAB-only plugin or toolbox today. Use EEGPrep when you want
familiar preprocessing and review workflows in Python with standalone package
behavior.
