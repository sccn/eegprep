.. _preprocessing_pipeline:

======================
Preprocessing Pipeline
======================

This page describes a practical EEGLAB-style preprocessing order in EEGPrep.
It is not a mandatory recipe for every experiment. Adapt thresholds and review
steps to your data, task, and lab standards.

Recommended Order
=================

.. raw:: html

   <div class="eegprep-path">
     <p>Load data and verify channel locations, sampling rate, events, and data
     shape.</p>
     <p>Select or remove channels that should not enter EEG cleaning.</p>
     <p>Filter or clean continuous data before epoching.</p>
     <p>Resample when lower sampling rate is appropriate for the analysis.</p>
     <p>Run ICA on cleaned continuous or appropriate epoched data.</p>
     <p>Classify and inspect components with ICLabel and property dashboards.</p>
     <p>Epoch, reject, baseline correct, and save reviewed datasets.</p>
   </div>

Start From Sample Data
======================

.. code-block:: python

   from pathlib import Path
   from eegprep import pop_loadset

   EEG = pop_loadset(Path("sample_data") / "eeglab_data.set")
   print(EEG["data"].shape, EEG["srate"], len(EEG["event"]))

Select Data and Channels
========================

Use ``pop_select`` for user-facing channel, point, time, trial, and event
selection:

.. code-block:: python

   from eegprep import pop_select

   EEG, com = pop_select(EEG, channel=[1, 2, 3, 4, 5], return_com=True)

GUI path: ``Edit > Select data``. Channel/component values in GUI dialogs are
EEGLAB-facing one-based values.

Filter
======

The usual FIR filtering entry point is ``pop_eegfiltnew``:

.. code-block:: python

   from eegprep import pop_eegfiltnew

   EEG, com = pop_eegfiltnew(
       EEG,
       locutoff=1.0,
       hicutoff=40.0,
       plotfreqz=False,
       return_com=True,
   )

GUI path: ``Tools > Filter the data``. The wrapper uses bundled FIRFilt-style
Hamming-window defaults, respects continuous-data boundary events, clears stale
ICA activations, and returns a replayable history command.

Use FIRFilt helpers for custom filter design:

.. code-block:: python

   from eegprep import pop_firws, pop_firwsord, pop_kaiserbeta

   beta = pop_kaiserbeta(0.001)
   order = pop_firwsord("kaiser", EEG["srate"], 2, 0.001)
   EEG, com = pop_firws(
       EEG,
       fcutoff=[1, 40],
       ftype="bandpass",
       wtype="kaiser",
       warg=beta,
       forder=order,
       return_com=True,
   )

Clean Continuous Data
=====================

Use ``pop_clean_rawdata`` when you want the EEGLAB-style GUI wrapper and
history string:

.. code-block:: python

   from eegprep import pop_clean_rawdata

   EEG, com = pop_clean_rawdata(
       EEG,
       FlatlineCriterion=5,
       ChannelCriterion=0.8,
       LineNoiseCriterion=4,
       Highpass=(0.25, 0.75),
       BurstCriterion=20,
       WindowCriterion=0.25,
       return_com=True,
   )

Use ``clean_artifacts`` when you need lower-level state outputs:

.. code-block:: python

   from eegprep import clean_artifacts

   clean_eeg, highpass_state, burst_state, removed_channels = clean_artifacts(
       EEG,
       FlatlineCriterion=5,
       ChannelCriterion=0.8,
       LineNoiseCriterion=4,
       Highpass=(0.25, 0.75),
       BurstCriterion=20,
       WindowCriterion=0.25,
   )

``clean_artifacts`` returns four values. Scripts that only need the cleaned EEG
should use the first value or call ``pop_clean_rawdata``.

Riemannian ASR Notes
====================

EEGPrep supports the Riemannian ASR calibration variant through
``clean_asr(EEG, useriemannian="calib")`` and
``clean_artifacts(EEG, Distance="Riemannian")``. Full Riemannian ASR
processing is not ported, so direct full-process requests fail clearly instead
of silently substituting MATLAB-only behavior.

Review Rejected Segments
========================

``vis_artifacts(clean_eeg, original_eeg)`` opens an EEG browser with rejected
sample intervals highlighted from ``clean_sample_mask``. Use
``show=False`` when scripting diagnostics:

.. code-block:: python

   from eegprep import vis_artifacts

   diagnostics = vis_artifacts(clean_eeg, EEG, show=False)
   print(diagnostics["rejected_fraction"])

Resample
========

Resample after early cleaning/filtering when a lower sampling rate is
appropriate:

.. code-block:: python

   from eegprep import pop_resample

   EEG, com = pop_resample(EEG, 64, return_com=True)

GUI path: ``Tools > Change sampling rate``. Continuous data with boundary
events is resampled by segment. Event latencies and the time vector are updated.

Re-Reference and Interpolate
============================

Average reference:

.. code-block:: python

   from eegprep import pop_reref

   EEG, com = pop_reref(EEG, [], return_com=True)

Interpolate known bad channels after channel removal or rejection:

.. code-block:: python

   from eegprep import pop_interp

   EEG, com = pop_interp(EEG, [0, 4, 9], return_com=True)

Use ``Tools > Re-reference the data`` and ``Tools > Interpolate electrodes``
from the GUI.

The programmatic ``bad_elec`` list for ``pop_interp`` uses Python zero-based
indices when integers are supplied. GUI channel selection remains one-based and
label-driven.

Run ICA and ICLabel
===================

Run ICA:

.. code-block:: python

   from eegprep import pop_runica

   EEG, com = pop_runica(EEG, icatype="picard", gui=False, return_com=True)

Label and review components:

.. code-block:: python

   from eegprep import eeg_icalabelstat, pop_iclabel, pop_viewprops

   EEG, com = pop_iclabel(EEG, "default", return_com=True)
   stats = eeg_icalabelstat(EEG, threshold=0.9, verbose=False)
   figures = pop_viewprops(EEG, typecomp=0, chanorcomp=[1, 2, 3])

See :ref:`ica_rejection` before removing components.

Epoch and Baseline
==================

Use ``pop_epoch`` and ``pop_rmbase`` for ERP-style workflows:

.. code-block:: python

   from eegprep import pop_epoch, pop_rmbase

   EEG, com_epoch = pop_epoch(EEG, ["square"], [-1, 2], return_com=True)
   EEG, com_base = pop_rmbase(EEG, [-200, 0], return_com=True)

GUI paths: ``Tools > Extract epochs`` and ``Tools > Remove epoch baseline``.

Save
====

Save reviewed datasets with ``pop_saveset``:

.. code-block:: python

   from pathlib import Path
   from eegprep import pop_saveset

   pop_saveset(EEG, Path("sample_data") / "eeglab_data_preprocessed.set")

For large datasets, review :ref:`large_dataset_storage` before deciding whether to keep data
in memory, memory-map sidecar files, or retrieve offloaded datasets.

Quality Control Checklist
=========================

* Confirm ``EEG["data"].shape`` after each major transform.
* Inspect events after filtering/resampling and before epoching.
* Review rejected channels/windows visually before final removal.
* Inspect ICLabel probabilities, scalp maps, spectra, and activity before
  removing components.
* Keep ``LASTCOM`` and ``ALLCOM`` from GUI/console runs so the workflow can be
  replayed in scripts.
