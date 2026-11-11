.. _plugins:

===============================
Bundled Plugins and Extensions
===============================

EEGPrep ships Python ports of several EEGLAB plugin workflows. They are bundled
with the package, exposed in the GUI, available from ``eegprep-console``, and
documented in the API reference. External extensions use a separate Python
package registry described in :ref:`extensions`.

Bundled Plugin Inventory
========================

.. list-table::
   :header-rows: 1

   * - Plugin
     - Main entry points
     - Implemented workflow
   * - clean_rawdata
     - ``pop_clean_rawdata``, ``clean_artifacts``, ``clean_asr``,
       ``vis_artifacts``
     - Drift, flatline, noisy-channel, ASR, window cleaning, and visual artifact
       review for continuous data.
   * - FIRFilt
     - ``pop_eegfiltnew``, ``pop_firws``, ``pop_firpm``, ``pop_firma``,
       ``pop_xfirws``
     - FIR filtering, filter-order helpers, response reports, and
       boundary-aware filtering.
   * - ICLabel
     - ``pop_iclabel``, ``pop_icflag``, ``eeg_icalabelstat``,
       ``pop_viewprops``, ``pop_prop_extended``
     - Default ICLabel network classification, threshold flagging, component
       dashboards, and label statistics.
   * - DIPFIT
     - ``pop_dipfit_settings``, ``pop_dipfit_gridsearch``,
       ``pop_dipfit_nonlinear``, ``pop_multifit``, ``pop_dipplot``,
       ``pop_leadfield``
     - Standalone spherical source-localization workflows for ICA components
       and explicit limits for unavailable external backends.
   * - EEG-BIDS
     - ``pop_importbids``, ``pop_exportbids``, ``pop_load_frombids``,
       ``bids_preproc``, ``bids_list_eeg_files``
     - BIDS import/export, metadata loading, and batch preprocessing helpers.

Inspect Installed Plugin Metadata
=================================

From normal Python or ``eegprep-console``:

.. code-block:: python

   import eegprep

   for plugin in eegprep.bundled_plugins():
       print(plugin["plugin"], plugin["funcname"], plugin["menu"])

   status, names, records = eegprep.plugin_status("ICLabel", exactmatch=True)
   print(status, names)

   inventory = eegprep.plugin_menu(show=False)
   print(eegprep.format_plugin_menu())

The Extension Manager dialog uses the same metadata and adds curated catalog
records when available.

clean_rawdata and ASR
=====================

Use the ``pop_*`` wrapper when you want GUI behavior or command history:

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

Use ``clean_artifacts`` when you need lower-level tuple outputs:

.. code-block:: python

   from eegprep import clean_artifacts

   clean_eeg, highpass_state, burst_state, removed_channels = clean_artifacts(EEG)

Channel correlation cleaning uses the spatial reconstruction method when the
dataset has usable ``X``, ``Y``, and ``Z`` channel coordinates. If channel
locations are empty, ``pop_clean_rawdata`` automatically uses the bundled
location-free correlation method instead. The RANSAC subset size is
``round(subset_size * nbchan)`` channels, and a larger count raises
``ValueError``. The line-noise filter is a short FIR, so a NaN stays in a
neighborhood of that channel instead of removing the channel from every
correlation.

``clean_windows`` keeps the removal duration from ``eeg_eegrej`` and sets
``xmax`` to ``xmin + (pnts - 1) / srate`` only when that duration is off the
sample grid. A recording whose fit widths round to 1 or less raises
``ValueError``, and ``clean_asr`` calibrates on the whole recording instead.
Eleven to twenty-two windows still fit.

.. _cleaning_inputs:

Copies, dtypes, and measured criteria
--------------------------------------

Copy a dataset with ``copy.deepcopy`` before calling a function that
replaces its samples when the original recording must be preserved.

.. list-table::
   :header-rows: 1
   :widths: 24 22 28 26

   * - Function
     - Same object when nothing is removed
     - Caller's ``data``
     - Caller's ``etc``
   * - ``clean_drifts``
     - yes
     - replaced; float32 becomes float64
     - gains ``clean_drifts_kernel``
   * - ``clean_flatlines``
     - yes
     - unchanged
     - unchanged
   * - ``clean_channels``
     - yes
     - replaced; float32 becomes float64
     - unchanged
   * - ``clean_channels_nolocs``
     - yes, the dataset inside the returned pair
     - replaced; float32 becomes float64
     - unchanged
   * - ``clean_windows``
     - no; the result is always a new dataset
     - unchanged
     - unchanged
   * - ``pop_eegfiltnew``
     - no
     - unchanged
     - unchanged
   * - ``pop_select``
     - no
     - unchanged
     - unchanged

Removing one flat channel returns a new dataset. The dataset passed in keeps
its channel count and its previous ``etc``. The result stores
``etc['clean_channel_mask']``, true for each original channel that remains.
``clean_channels`` and ``clean_channels_nolocs`` also return a new dataset
when they delete channels. They replace the caller's samples with float64
before that copy, so the caller keeps the original channel count and its
previous ``etc`` while holding those float64 samples.

``clean_windows`` processes the copy in float64 and writes the returned
samples in the input floating-point dtype. Its
``etc['clean_sample_mask']`` marks retained samples. ``pop_eegfiltnew``
filters in float64 and restores float32 when the input samples were float32.
``pop_select`` keeps the input dtype. ``clean_drifts`` leaves the returned
samples in float64. ``etc`` must already be a dict; ``eeg_checkset``
provides one.

``clean_channel_mask`` records original channels that remain in the dataset.
Pass ``return_diagnostics=True`` to ``clean_channels``,
``clean_channels_nolocs``, ``clean_flatlines``, or ``clean_windows`` for the
measurements already computed during that call. Omitting
``return_diagnostics`` keeps the current return value. Channel indices and
sample offsets in the report are 0-based positions in the input array.

``clean_channels`` reports noise scores, noise z-scores, noise flags, the
correlation of each usable channel with its RANSAC estimate in each window,
and those windows' sample offsets. The RANSAC subsets come from
``numpy.random.RandomState(seed)``. ``seed`` defaults to 5489, which is the
value previously fixed in the function, and the draws stay in the same
order. The report's ``seed`` is the value used.

``clean_channels_nolocs`` reports the sorted absolute pairwise correlations
and the row indices kept after dropping the most-correlated fraction. A
channel is flagged when every retained row is below ``min_corr``. The
boolean array returned beside the dataset is that detection result. It is
all true when every channel is flagged and then kept.

``clean_windows`` reports window offsets, the window length in samples, and
the RMS z-score of each channel in each window. ``clean_flatlines`` reports
which channels exceeded the flat-line duration.

``detected_channels`` is the criterion. ``removed_channels`` in the report is
what the call deleted. When every channel is flagged and none is removed,
``removal_suppressed`` gives the reason and ``clean_channel_mask`` is left
as it was, including staying absent. A prior mask is returned as
``prior_clean_channel_mask`` or ``prior_clean_sample_mask``.

.. code-block:: python

   import copy

   from eegprep import clean_channels

   original = copy.deepcopy(EEG)
   cleaned, diagnostics = clean_channels(EEG, seed=5489, return_diagnostics=True)
   diagnostics["noise_zscores"]
   diagnostics["correlations"]
   diagnostics["seed"]

FIRFilt
=======

``pop_eegfiltnew`` is the usual filtering entry point. When the filter order
is left at its default, the transition width is recomputed from that even
order (``3.3 / order * srate``) so the reported cutoff matches the coefficients:

.. code-block:: python

   EEG, com = pop_eegfiltnew(EEG, locutoff=1, hicutoff=40, plotfreqz=False, return_com=True)

Use the lower-level FIRFilt helpers for custom order/window work:

.. code-block:: python

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

ICLabel
=======

The standalone Python engine ships the default ICLabel network. ``lite`` and
``beta`` network requests require MATLAB or Octave with an EEGLAB ICLabel
checkout that provides those artifacts.

.. code-block:: python

   EEG, com = pop_iclabel(EEG, "default", return_com=True)
   stats = eeg_icalabelstat(EEG, threshold=0.9, verbose=False)
   EEG, com = pop_icflag(EEG, return_com=True)

Review components visually with ``pop_viewprops`` before removing them.

DIPFIT
======

DIPFIT helpers support standalone spherical workflows. They store model fields
under ``EEG["dipfit"]["model"]`` and keep command strings replayable:

.. code-block:: python

   EEG, com = pop_dipfit_settings(EEG, model="standardBESA", return_com=True)
   EEG, com = pop_multifit(EEG, [1, 2, 3], threshold=40, return_com=True)

Unavailable MRI/BEM, AFNI, LORETA, and STUDY-level source-statistics workflows
raise explicit errors.

EEG-BIDS
========

Use ``pop_load_frombids`` for one EEG recording inside a BIDS dataset and
``bids_preproc`` for batch workflows:

.. code-block:: python

   EEG, report = pop_load_frombids(
       "sub-01/eeg/sub-01_task-rest_eeg.set",
       return_report=True,
   )

See :ref:`bids_workflow` for BIDS dataset layout and batch notes.

External Extensions
===================

External EEGPrep extensions are Python packages, not MATLAB plugin folders.
They declare an ``eegprep.extensions`` entry point and return an
``ExtensionSpec``. The Extension Manager shows installed and curated extension
metadata but does not install, update, remove, unzip, or execute package-manager
commands for the user.

Continue with :ref:`extensions` for package authoring and :ref:`extension_curation`
for catalog/trust policy.
