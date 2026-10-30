.. _preprocessing_pipeline:

======================
Preprocessing Pipeline
======================

This page describes a practical preprocessing order in EEGPrep.
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
EEGLAB-facing one-based values. ``pop_select`` then checks event consistency:
a leading boundary whose duration is less than one sample is removed, a leading
boundary latency in (0, 1) is set to 0.5, events with an epoch number outside
the dataset are removed, and the remaining events are sorted by epoch, then
latency. Once any event has a duration, an empty or missing duration on
another event is 0. A NaN duration is kept. Consecutive boundary-list entries
that share a latency, taken in event order before that sort, chain their
durations into the last entry of the run. A later boundary at the same latency
stays separate when a different latency stands between the two entries.
Boundary latencies 20, 20, 30, and 20 with durations 4, 1, 8, and 2 sort to
latency 20 duration 5, latency 20 duration 2, and latency 30 duration 8.

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
ICA activations, and returns a replayable history command. ``minphase=True``
converts the FIR to minimum phase with the real cepstrum before causal filtering.
``usefftfilt=True`` filters each segment by overlap-add, so a missing sample
stays inside the FFT block that contains it.

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

Use ``pop_clean_rawdata`` when you want the interactive wrapper and history
string:

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

Sampling Rates
==============

ASR calibration designs its spectral-shaping filter for the dataset's sampling
rate, so ordinary clinical rates, including non-integer rates such as 258 Hz,
need no resampling first. A rate whose default frequency breakpoints are not
monotonic, such as 32 Hz, cannot be calibrated. ``clean_asr`` raises an error
that names the rate. ``clean_artifacts`` and ``pop_clean_rawdata`` warn and
leave that recording unchanged.

A calibration shorter than one 0.5 s RMS window, or one that yields fewer than
two such windows, cannot set thresholds either. ``clean_asr`` raises
``ASRCalibrationError`` and names the sample or window counts. ``clean_artifacts``
and ``pop_clean_rawdata`` warn, leave the burst stage unchanged, and still run
window removal when it was requested. On a 44-sample recording at 128 Hz the
burst stage needs 64 samples, so with the other stages off the samples stay
identical, including NaNs. EEGLAB prints the calibration error and returns
before window removal.

Riemannian ASR Notes
====================

``clean_asr(EEG, useriemannian=True)`` and
``clean_artifacts(EEG, Distance="Riemannian")`` run Riemannian ASR.
Calibration takes the geometric median of block covariances and sets
component thresholds from the nonlinear eigenspace of the mixing matrix.
Processing uses one sample covariance per chunk. A later chunk averages that
covariance with the previous chunk along the affine-invariant geodesic.

``clean_asr(EEG, useriemannian="calib")`` keeps the smaller variant: a
Riemannian covariance during calibration, then the Euclidean ASR processor.
Passing an unknown ``useriemannian`` value raises ``ValueError``.

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
A missing sample stays missing where the anti-aliasing filter reaches it,
including the exact-zero delay pads in MATLAB's resampler. Outputs no tap
reaches stay finite.

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

Time-Frequency and ERP Images
=============================

After epoching or component review, use EEGPrep's standalone time-frequency
wrappers for the common EEGLAB plotting workflows:

.. code-block:: python

   from eegprep import pop_newcrossf, pop_newtimef, pop_spectopo

   ersp, com_timef = pop_newtimef(EEG, typeproc=1, num=1, return_com=True)
   cross, com_crossf = pop_newcrossf(EEG, typeproc=1, num1=1, num2=2, return_com=True)
   spectra, com_spectopo = pop_spectopo(EEG, 1, timerange=[], return_com=True)

``pop_newtimef`` plots one channel or component at a time. Set ``typeproc=1`` for
a channel and ``typeproc=0`` for an ICA component; ``num`` is the 1-based channel
or component index. ``tlimits`` is the epoch window in milliseconds and ``cycles``
selects the decomposition: ``[0]`` uses a short-time FFT, while ``[n]`` (optionally
``[n factor]``) uses an ``n``-cycle Morlet wavelet that narrows toward higher
frequencies. Called with no index, ``pop_newtimef`` opens a dialog to collect
these values; the ``Plot`` menu exposes the same dialog under *Channel
time-frequency* and *Component time-frequency*.

Reading the figure
------------------

The figure stacks two images that share a time axis:

- **ERSP** (top) -- event-related spectral perturbation, the trial-averaged power
  change from the baseline window, in dB.
- **ITC** (bottom) -- inter-trial coherence, from 0 (phase varies across trials)
  to 1 (perfect phase locking).

Each image carries marginal panels: a rotated plot to its left (the baseline
power spectrum beside ERSP; the frequency-averaged coherence beside ITC) and a
horizontal plot below it (the ERSP minimum/maximum envelope; the ERP trace). A
dashed line marks time zero, colorbars sit on the right, and -- when channel
locations are available -- a scalp-map inset in the upper left shows the
channel's head position or the component's interpolated map, captioned with the
channel label or ``IC n``.

Baseline and significance
-------------------------

Pass a ``baseline`` window in milliseconds to normalize power, and an ``alpha``
to test each time-frequency point against a bootstrap null drawn from the
baseline:

.. code-block:: python

   result = pop_newtimef(EEG, 1, 1, [-1000, 1000], [3, 0.5],
                         baseline=[-1000, 0], alpha=0.05)

Non-significant points are drawn at the colormap's green midpoint so the
surviving effects stand out; add ``mcorrect="fdr"`` to control the false
discovery rate across the image. Because this is a permutation test, roughly
``alpha`` of the effect-free baseline points survive by chance -- expected
scatter, not a real effect.

Display options
---------------

- ``pcontour="on"`` keeps every point at its measured value and outlines the
  significant regions with a contour instead of masking them.
- The ITC image is colored by the sign of the coherence phase by default; pass
  ``plotphase="off"`` for magnitude only, or ``plotphaseonly="on"`` to show the
  phase angle in degrees.
- ``erspmax`` sets a symmetric ERSP color scale (``[-erspmax, erspmax]``) and
  ``itcmax`` the ITC image maximum; leave them unset to auto-scale.

Two-condition comparisons (contrasting two datasets in a single figure) are not
yet part of the standalone wrapper.

Legacy ``pop_timef`` and ``pop_crossf`` calls route through the standalone
``pop_newtimef`` and ``pop_newcrossf`` implementations.
ERP-image workflows use ``pop_erpimage`` for channel or component images of epoched data
(see `Epoch and Baseline`_ below), with EEGLAB's arguments:

.. code-block:: python

   from eegprep import pop_erpimage

   # Channel Cz of epoched data, sorted by reaction-time latency, smoothed over 10 trials.
   result, com = pop_erpimage(EEG, 1, "Cz", None, "", 10, 1, ["rt"], None, "latency", return_com=True)

``pop_erpimage`` draws the ERP trace, a colorbar and a scalp map by default,
and accepts every :func:`~eegprep.erpimage` option, such as phase sorting or inter-trial coherence.
See :ref:`gui_tutorials` for the dialog.

GUI paths live under the ``Plot`` menu when a dataset is loaded. EEGPrep
returns replayable Python history commands for these wrappers.

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
