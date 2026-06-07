.. _quickstart:

===========
Quick Start
===========

This quick start uses the checked-in tutorial data under ``sample_data/``. It
shows the same first workflow in normal Python and in the shared GUI plus
``eegprep-console`` session.

Sample Data
===========

The repository includes small tutorial datasets named after EEGLAB's
``sample_data`` convention:

.. list-table::
   :header-rows: 1

   * - File
     - Use
   * - ``sample_data/eeglab_data.set``
     - Continuous 32-channel tutorial data with events.
   * - ``sample_data/eeglab_data_epochs_ica.set``
     - Epoched tutorial data with ICA fields.
   * - ``sample_data/eeglab_data_with_ica_tmp.set``
     - Continuous tutorial data with ICA fields.
   * - ``sample_data/eeglab_data_hdf5.set``
     - HDF5-backed EEGLAB ``.set`` load path.

Five-Minute Python Workflow
===========================

Run this from the repository root after installing EEGPrep or syncing the
source checkout.

.. code-block:: python

   from pathlib import Path

   from eegprep import pop_eegfiltnew, pop_loadset, pop_resample, pop_saveset

   input_file = Path("sample_data") / "eeglab_data.set"
   output_file = Path("sample_data") / "eeglab_data_quickstart.set"

   EEG = pop_loadset(input_file)
   print(EEG["setname"], EEG["nbchan"], EEG["pnts"], EEG["srate"])

   EEG, filter_com = pop_eegfiltnew(
       EEG,
       locutoff=1.0,
       hicutoff=40.0,
       plotfreqz=False,
       return_com=True,
   )
   EEG, resample_com = pop_resample(EEG, 64, return_com=True)
   pop_saveset(EEG, output_file)

   print(filter_com)
   print(resample_com)

The important pattern is ``return_com=True``. It gives you the updated dataset
and the history command that the GUI or console would record.

GUI Plus Console Workflow
=========================

Launch the shared GUI/console session:

.. code-block:: bash

   uv run eegprep-console --full

Then:

.. raw:: html

   <div class="eegprep-path">
     <p>Choose <strong>File > Load existing dataset</strong> and open
     <code>sample_data/eeglab_data.set</code>.</p>
     <p>In the console, inspect <code>EEG["nbchan"]</code>,
     <code>EEG["srate"]</code>, <code>CURRENTSET</code>, and
     <code>LASTCOM</code>.</p>
     <p>Choose <strong>Tools > Filter the data</strong> or run
     <code>pop_eegfiltnew(EEG, locutoff=1, hicutoff=40)</code> from the
     console.</p>
     <p>Choose <strong>Tools > Change sampling rate</strong> or run
     <code>pop_resample(EEG, 64)</code>.</p>
     <p>Choose <strong>Plot > Channel data (scroll)</strong> to inspect the
     current dataset in EEGBrowser.</p>
   </div>

The GUI and console share the same ``EEGPrepSession``. A GUI action updates the
console's ``EEG``, ``ALLEEG``, ``CURRENTSET``, ``LASTCOM``, and ``ALLCOM``.
Console ``pop_*`` calls update the GUI when they use the console wrappers.

Inspect the EEG Structure
=========================

EEGPrep datasets are dictionaries:

.. code-block:: python

   print(EEG.keys())
   print(EEG["data"].shape)
   print(EEG["event"][0])
   print([chan["labels"] for chan in EEG["chanlocs"][:5]])

Continuous data is usually ``(nbchan, pnts)``. Epoched data is usually
``(nbchan, pnts, trials)``.

Load Epoched ICA Data
=====================

Use the ICA sample when you want to review ICLabel/component workflows without
waiting for a decomposition:

.. code-block:: python

   from pathlib import Path
   from eegprep import eeg_icalabelstat, pop_iclabel, pop_loadset, pop_viewprops

   EEG = pop_loadset(Path("sample_data") / "eeglab_data_epochs_ica.set")
   EEG, com = pop_iclabel(EEG, "default", return_com=True)
   stats = eeg_icalabelstat(EEG, threshold=0.9, verbose=False)
   figures = pop_viewprops(EEG, typecomp=0, chanorcomp=[1], plot=False)

   print(com)
   print(stats["counts"])

``typecomp=0`` means component mode, matching EEGLAB's component property
dialogs. Component numbers are user-facing one-based values.

Where to Go Next
================

* Read :ref:`concepts` before writing longer scripts.
* Use :ref:`gui_console_session` to understand switching between the GUI and
  console.
* Use :ref:`gui_tutorials` to repeat the workflow from menus.
* Use :ref:`interactive_console` for console launch and history details.
* Use :ref:`scripting_workflows` to turn history into reproducible scripts.
* Use :ref:`mne_integration` for issue #22's MNE examples.
