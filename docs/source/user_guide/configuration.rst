.. _configuration:

=============
Configuration
=============

EEGPrep keeps global options intentionally small. Use explicit function
arguments for preprocessing choices, and use ``EEG_OPTIONS`` only for session,
storage, compatibility, and GUI behavior that really is global.

``EEG_OPTIONS`` is a dictionary, not a configuration object:

.. code-block:: python

   from eegprep import EEG_OPTIONS

   print(dict(EEG_OPTIONS))
   EEG_OPTIONS["option_allmenus"] = 1
   EEG_OPTIONS["option_savetwofiles"] = 1

Save the previous values when a script changes global options:

.. code-block:: python

   from eegprep import EEG_OPTIONS

   old_options = dict(EEG_OPTIONS)
   try:
       EEG_OPTIONS["option_memmapdata"] = 1
       # Load or save datasets here.
   finally:
       EEG_OPTIONS.clear()
       EEG_OPTIONS.update(old_options)

Common Global Options
=====================

.. list-table::
   :header-rows: 1

   * - Key
     - Effect
   * - ``option_allmenus``
     - Show advanced and legacy menu entries in the main GUI.
   * - ``option_storedisk``
     - Keep saved non-current ``ALLEEG`` datasets on disk to reduce memory use.
   * - ``option_savetwofiles``
     - Make ``pop_saveset`` write a ``.set`` header plus ``.fdt`` sidecar by
       default.
   * - ``option_memmapdata``
     - Memory-map two-file ``.fdt`` data when loading with ``pop_loadset``.
   * - ``option_boundary99``
     - Treat numeric ``-99`` events as boundary events for ERPLAB-style data.
   * - ``option_computeica``
     - Precompute ICA activations where supported.
   * - ``option_native_dialogs``
     - Use experimental native OS file panels instead of Qt dialogs.
   * - ``option_scaleicarms``
     - Scale ICA component activations to RMS microvolt during checkset paths.
   * - ``option_cachesize``
     - STUDY cache size in MB for implemented STUDY measure workflows.

For storage details, see :ref:`large_dataset_storage`.

Set Options From GUI or Console
===============================

Use the Options menu in the GUI, or call ``pop_editoptions`` from
``eegprep-console``:

.. code-block:: python

   from eegprep import EEG_OPTIONS, pop_editoptions

   com = pop_editoptions(option_allmenus=1, option_savetwofiles=1)
   print(com)
   print(EEG_OPTIONS["option_allmenus"])

In a shared GUI/console session, the menu refresh uses the same
``EEG_OPTIONS`` dictionary.

Preprocessing Parameters
========================

Most analysis choices are ordinary function arguments. This keeps scripts
readable and prevents hidden global state from changing scientific results.

.. code-block:: python

   from pathlib import Path

   from eegprep import pop_clean_rawdata, pop_eegfiltnew, pop_loadset, pop_resample

   EEG = pop_loadset(Path("sample_data") / "eeglab_data.set")

   EEG, filter_com = pop_eegfiltnew(
       EEG,
       locutoff=1,
       hicutoff=40,
       plotfreqz=False,
       return_com=True,
   )
   EEG, resample_com = pop_resample(EEG, 64, return_com=True)
   EEG, clean_com = pop_clean_rawdata(
       EEG,
       FlatlineCriterion=5,
       ChannelCriterion=0.8,
       LineNoiseCriterion=4,
       Highpass=(0.25, 0.75),
       BurstCriterion=20,
       WindowCriterion=0.25,
       return_com=True,
   )

   history = [filter_com, resample_com, clean_com]

The same rule applies to ICA, ICLabel, and BIDS preprocessing:

.. code-block:: python

   from eegprep import bids_preproc, pop_iclabel, pop_runica

   EEG, ica_com = pop_runica(EEG, icatype="picard", gui=False, return_com=True)
   EEG, label_com = pop_iclabel(EEG, "default", return_com=True)

   bids_preproc(
       "/path/to/bids-root",
       OutputDir="/path/to/bids-root/derivatives/eegprep",
       Subjects=["01"],
       Tasks=["rest"],
       WithInterp=True,
       WithICA=False,
       NumJobs=1,
   )

Reusable Presets
================

Represent lab presets as plain dictionaries and pass them explicitly:

.. code-block:: python

   CLEANING_PRESETS = {
       "resting": {
           "FlatlineCriterion": 5,
           "ChannelCriterion": 0.8,
           "LineNoiseCriterion": 4,
           "Highpass": (0.25, 0.75),
           "BurstCriterion": 20,
           "WindowCriterion": 0.25,
       },
       "erp": {
           "FlatlineCriterion": 5,
           "ChannelCriterion": 0.85,
           "LineNoiseCriterion": 4,
           "Highpass": (0.1, 0.5),
           "BurstCriterion": 20,
           "WindowCriterion": 0.25,
       },
   }

   EEG, com = pop_clean_rawdata(EEG, **CLEANING_PRESETS["resting"], return_com=True)

Keep these preset dictionaries in your lab package, notebook, or analysis repo
so parameter changes are version-controlled with the study.

When to Use Which Surface
=========================

.. list-table::
   :header-rows: 1

   * - Need
     - Use
   * - Change menu visibility or storage behavior
     - ``EEG_OPTIONS`` or ``pop_editoptions``.
   * - Change filtering, cleaning, resampling, ICA, ICLabel, or BIDS behavior
     - Explicit function arguments.
   * - Record replayable commands
     - ``return_com=True`` and keep the returned command strings.
   * - Share settings across scripts
     - A normal Python dictionary or module constant in your analysis code.
   * - Use the same state model as the GUI
     - ``EEGPrepSession``; see :ref:`gui_console_session`.

Troubleshooting
===============

Option Did Not Affect a Processing Step
---------------------------------------

Check whether the function actually reads ``EEG_OPTIONS``. Most processing
functions are configured through explicit keyword arguments.

Memory-Mapped Loading Did Not Happen
------------------------------------

``option_memmapdata`` applies to two-file ``.set``/``.fdt`` datasets. Save with
``savemode="twofiles"`` or set ``option_savetwofiles`` before saving, then load
again with ``option_memmapdata`` enabled.

Menu Entries Did Not Change
---------------------------

After changing ``option_allmenus`` in a running GUI, reopen or refresh the main
window so menu status is rebuilt from the current option dictionary.
