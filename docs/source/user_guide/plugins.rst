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

FIRFilt
=======

``pop_eegfiltnew`` is the usual filtering entry point:

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
