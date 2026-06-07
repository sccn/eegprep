.. _bids_workflow:

=============
BIDS Workflow
=============

EEGPrep can load BIDS EEG files, export EEGPrep datasets into a small
BIDS-style folder, and run the bundled BIDS preprocessing helper across a BIDS
root. This page uses the checked-in ``sample_data`` files where EEGPrep can
create a tutorial folder locally, and it uses ``/path/to/bids-root`` whenever
you must provide a real lab BIDS dataset.

The repository does not ship a complete BIDS study. For a real analysis, start
with a BIDS-valid dataset from your lab and validate it with the BIDS Validator
before batch preprocessing.

BIDS EEG Layout
===============

A typical EEG BIDS dataset stores one raw data file plus sidecars for each
recording:

.. code-block:: text

   dataset/
   |-- dataset_description.json
   |-- participants.tsv
   `-- sub-01/
       `-- eeg/
           |-- sub-01_task-rest_run-01_eeg.set
           |-- sub-01_task-rest_run-01_eeg.fdt
           |-- sub-01_task-rest_run-01_channels.tsv
           |-- sub-01_task-rest_run-01_events.tsv
           `-- sub-01_task-rest_run-01_eeg.json

EEGPrep's BIDS loader supports EEGLAB ``.set``/``.fdt``, BrainVision
``.vhdr``, EDF, and BDF EEG files. Sidecars can replace or merge metadata,
events, and channel information into the returned EEG dictionary.

Create a Small Tutorial Export
==============================

Use ``pop_exportbids`` when you want a local folder to inspect the path layout
without downloading a public BIDS dataset:

.. code-block:: python

   from pathlib import Path

   from eegprep import bids_list_eeg_files, pop_exportbids, pop_load_frombids, pop_loadset

   EEG = pop_loadset(Path("sample_data") / "eeglab_data.set")
   export_root = Path("tutorial_outputs") / "bids_demo"
   export_root.mkdir(parents=True, exist_ok=True)

   bids_root = pop_exportbids(EEG, export_root, subject="01", task="tutorial")
   eeg_files = bids_list_eeg_files(bids_root)
   print(eeg_files)

   EEG = pop_load_frombids(eeg_files[0])
   print(EEG["setname"], EEG["nbchan"], EEG["pnts"], EEG["srate"])

This export is useful for learning EEGPrep's BIDS file handling. It is not a
replacement for a curated experimental BIDS dataset with full study metadata.

List Files in a Real BIDS Root
==============================

Use ``bids_list_eeg_files`` to discover supported EEG recordings:

.. code-block:: python

   from eegprep import bids_list_eeg_files

   files = bids_list_eeg_files(
       "/path/to/bids-root",
       subjects=["01", "02"],
       tasks=["rest"],
   )

   for filename in files:
       print(filename)

The function returns file paths as strings. Query values use raw BIDS
identifiers such as ``"01"`` and ``"rest"``, not prefixed values such as
``"sub-01"`` or ``"task-rest"``.

Load One BIDS EEG File
======================

``pop_load_frombids`` takes a concrete EEG data-file path, not a root/query
combination:

.. code-block:: python

   from eegprep import pop_load_frombids

   EEG, report = pop_load_frombids(
       "/path/to/bids-root/sub-01/eeg/sub-01_task-rest_run-01_eeg.set",
       bidsmetadata=True,
       bidschanloc=True,
       bidsevent="replace",
       return_report=True,
   )

   print(report["ImporterUsed"])
   print(EEG["nbchan"], EEG["pnts"], EEG["srate"])

Common loading options:

.. list-table::
   :header-rows: 1

   * - Option
     - Use
   * - ``bidsmetadata=True``
     - Apply metadata from BIDS sidecars to the EEG dictionary.
   * - ``bidschanloc=True``
     - Apply BIDS channel metadata and coordinates when available.
   * - ``bidsevent="replace"``
     - Replace file events with events from the BIDS sidecar. Use ``"merge"``,
       ``"append"``, or ``False`` when your dataset needs a different policy.
   * - ``eventtype="trial_type"``
     - Choose the event sidecar column to use as the EEG event type.
   * - ``infer_locations=None``
     - Infer locations only when no channel locations are present. Pass
       ``True``, ``False``, or a montage filename for explicit behavior.

Preprocess a BIDS Dataset
=========================

``bids_preproc`` can process one EEG file or every supported file under a BIDS
root. Use capitalized option names for the current EEGPrep API:

.. code-block:: python

   from eegprep import bids_preproc

   bids_preproc(
       "/path/to/bids-root",
       OutputDir="/path/to/bids-root/derivatives/eegprep",
       Subjects=["01", "02"],
       Tasks=["rest"],
       SamplingRate=128,
       Highpass=(0.25, 0.75),
       ChannelCriterion=0.8,
       LineNoiseCriterion=4.0,
       BurstCriterion=20,
       WindowCriterion=0.25,
       WithInterp=True,
       WithICA=False,
       WithICLabel=False,
       SkipIfPresent=True,
       NumJobs=1,
   )

Important processing options:

.. list-table::
   :header-rows: 1

   * - Option
     - Use
   * - ``OutputDir``
     - Destination for derivative files. The default is
       ``{root}/derivatives/eegprep``.
   * - ``Subjects`` / ``Sessions`` / ``Runs`` / ``Tasks``
     - Filter which BIDS files are processed.
   * - ``SamplingRate``
     - Resample final data when set.
   * - ``Highpass``
     - Initial drift-removal transition band, or ``"off"``.
   * - ``FlatlineCriterion``, ``ChannelCriterion``, ``LineNoiseCriterion``
     - Channel cleaning criteria.
   * - ``BurstCriterion`` and ``WindowCriterion``
     - ASR repair/rejection and final window rejection criteria.
   * - ``WithInterp``
     - Reinterpolate removed channels after cleaning.
   * - ``WithICA`` / ``ICAAlgorithm`` / ``WithPicard``
     - Run ICA after cleaning. Use this only when the data length and runtime
       budget are appropriate.
   * - ``WithICLabel``
     - Classify components after ICA.
   * - ``ReturnData``
     - Return final EEG dictionaries instead of only writing derivatives. This
       can use substantial memory for large studies.
   * - ``NumJobs`` / ``ReservePerJob``
     - Control multiprocessing. Use an ``if __name__ == "__main__":`` guard in
       scripts that launch multiple processes.

For exploratory runs, start with one subject, ``NumJobs=1``, and
``ReturnData=False``. Then inspect the derivative files and reports before
expanding to the full dataset.

Batch Script Pattern
====================

Keep paths and options explicit in lab scripts:

.. code-block:: python

   from pathlib import Path

   from eegprep import bids_list_eeg_files, bids_preproc

   bids_root = Path("/path/to/bids-root")
   output_root = bids_root / "derivatives" / "eegprep"

   files = bids_list_eeg_files(str(bids_root), subjects=["01"], tasks=["rest"])
   if not files:
       raise RuntimeError("No matching BIDS EEG files were found.")

   bids_preproc(
       str(bids_root),
       OutputDir=str(output_root),
       Subjects=["01"],
       Tasks=["rest"],
       WithInterp=True,
       WithICA=False,
       SkipIfPresent=True,
   )

Load Derivatives
================

Derivative files written as EEGLAB ``.set`` files can be loaded directly with
``pop_loadset``. When the derivative folder also contains BIDS sidecars, use
``bids_list_eeg_files`` plus ``pop_load_frombids``:

.. code-block:: python

   from eegprep import bids_list_eeg_files, pop_load_frombids

   derivative_files = bids_list_eeg_files("/path/to/bids-root/derivatives/eegprep")
   EEG = pop_load_frombids(derivative_files[0])

GUI and Console Workflow
========================

The GUI exposes BIDS import/export menu actions through the bundled EEG-BIDS
plugin menus. In a shared session, use:

.. code-block:: bash

   uv run eegprep-console --full

Then choose ``File > Import data > From BIDS folder structure`` or
``File > Export > To BIDS folder structure``. Successful menu actions update
``EEG``, ``ALLEEG``, ``CURRENTSET``, ``LASTCOM``, and ``ALLCOM`` in the shared
session just like other loading and saving workflows.

Troubleshooting
===============

No Files Found
--------------

If ``bids_list_eeg_files`` returns an empty list:

* pass the BIDS root or a path inside that BIDS tree;
* use raw identifiers such as ``"01"`` rather than ``"sub-01"``;
* check that EEG data files end in ``.set``, ``.vhdr``, ``.edf``, or ``.bdf``;
* confirm that files use the BIDS ``suffix-eeg`` naming pattern.

Wrong Events or Labels
----------------------

Use ``return_report=True`` with ``pop_load_frombids`` and inspect the report.
Then adjust ``bidsevent`` or ``eventtype``. For example, set
``bidsevent=False`` to keep events stored in the raw EEG file, or set
``eventtype="value"`` when your event sidecar stores condition labels in a
non-default column.

Batch Job Uses Too Much Memory
------------------------------

Start with ``NumJobs=1`` and ``ReturnData=False``. Add ``ReservePerJob`` only
after a serial run shows the peak RAM needed for one dataset. Avoid enabling ICA
or ICLabel until the cleaning-only derivative workflow is correct.

Stale Derivative Results
------------------------

``SkipIfPresent=True`` skips files that already have matching derivatives. When
changing preprocessing parameters, either remove the old derivative folder, set
``SkipIfPresent=False``, or enable ``UseHashes=True`` so parameter changes are
reflected in intermediate file names.
