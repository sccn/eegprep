.. _advanced_topics:

===============
Advanced Topics
===============

This page collects patterns for experienced EEGPrep users who are building lab
pipelines, batch scripts, or small extensions. The examples use EEGPrep's
current dictionary-based EEG data model and explicit ``pop_*`` calls.

Custom Pipelines
================

A reliable custom pipeline is just a Python function that accepts an EEG
dictionary, returns an EEG dictionary, and records command strings when you want
to reproduce the workflow later.

.. code-block:: python

   from pathlib import Path

   from eegprep import pop_eegfiltnew, pop_loadset, pop_resample, pop_saveset

   def tutorial_pipeline(EEG, *, output_rate=64):
       history = []

       EEG, com = pop_eegfiltnew(
           EEG,
           locutoff=1,
           hicutoff=40,
           plotfreqz=False,
           return_com=True,
       )
       history.append(com)

       EEG, com = pop_resample(EEG, output_rate, return_com=True)
       history.append(com)

       return EEG, history

   EEG = pop_loadset(Path("sample_data") / "eeglab_data.set")
   EEG, history = tutorial_pipeline(EEG)
   Path("tutorial_outputs").mkdir(exist_ok=True)
   pop_saveset(EEG, Path("tutorial_outputs") / "eeglab_data_pipeline.set")

Use dictionary fields, not object attributes:

.. code-block:: python

   print(EEG["nbchan"], EEG["pnts"], EEG["srate"])
   print(EEG["data"].shape)
   print(EEG.get("history", ""))

Conditional Steps
=================

Inspect the current EEG dictionary and make explicit decisions:

.. code-block:: python

   from eegprep import pop_clean_rawdata, pop_iclabel, pop_runica

   def clean_and_optionally_label(EEG, *, run_ica=False):
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
       history = [clean_com]

       if run_ica and EEG["pnts"] > 30 * EEG["nbchan"]:
           EEG, ica_com = pop_runica(EEG, icatype="picard", gui=False, return_com=True)
           EEG, label_com = pop_iclabel(EEG, "default", return_com=True)
           history.extend([ica_com, label_com])

       return EEG, history

This style keeps scientific choices visible in code review and lab notebooks.

Custom Marking Functions
========================

If a lab-specific quality-control step marks samples or channels, write into
clear EEG dictionary fields and keep NumPy indexing internal:

.. code-block:: python

   import numpy as np

   def mark_large_sample_artifacts(EEG, *, z_threshold=6):
       data = np.asarray(EEG["data"])
       channel_scale = np.std(data, axis=1, keepdims=True)
       channel_scale[channel_scale == 0] = 1
       z = np.abs(data / channel_scale)
       bad_samples = np.flatnonzero(z.max(axis=0) > z_threshold)

       reject = EEG.setdefault("reject", {})
       reject["manual_bad_samples"] = (bad_samples + 1).tolist()
       return EEG

The stored sample numbers are one-based because they are user-facing EEG
metadata. Direct array slicing remains zero-based Python.

Batch Processing
================

For small local batches, use ``pathlib`` and keep outputs outside the immutable
sample files:

.. code-block:: python

   from pathlib import Path

   from eegprep import pop_loadset, pop_saveset

   input_files = sorted(Path("sample_data").glob("eeglab_data*.set"))
   output_dir = Path("tutorial_outputs") / "batch"
   output_dir.mkdir(parents=True, exist_ok=True)

   for input_file in input_files:
       EEG = pop_loadset(input_file)
       EEG, history = tutorial_pipeline(EEG)
       EEG["etc"] = {**EEG.get("etc", {}), "eegprep_history": history}
       pop_saveset(EEG, output_dir / input_file.name)

For BIDS studies, prefer ``bids_preproc`` so file discovery, derivative paths,
and sidecar handling stay consistent. See :ref:`bids_workflow`.

Multiprocessing Pattern
=======================

Python multiprocessing requires a main guard. Keep the worker function
top-level so it can be imported by child processes:

.. code-block:: python

   from concurrent.futures import ProcessPoolExecutor
   from pathlib import Path

   from eegprep import pop_loadset, pop_saveset

   def process_one(path_text):
       path = Path(path_text)
       EEG = pop_loadset(path)
       EEG, history = tutorial_pipeline(EEG)
       output = Path("tutorial_outputs") / "parallel" / path.name
       output.parent.mkdir(parents=True, exist_ok=True)
       pop_saveset(EEG, output)
       return path.name, history

   if __name__ == "__main__":
       files = [str(path) for path in Path("sample_data").glob("eeglab_data*.set")]
       with ProcessPoolExecutor(max_workers=2) as pool:
           for name, history in pool.map(process_one, files):
               print(name, history[-1] if history else "")

Avoid sending a live ``EEGPrepSession`` or Qt GUI object into worker processes.
Use filenames and returned results at process boundaries.

MNE Interoperability
====================

Use MNE when you need MNE-specific algorithms, then convert back to an EEGPrep
dictionary for EEGPrep GUI review or ``pop_*`` workflows:

.. code-block:: python

   from pathlib import Path

   from eegprep import eeg_eeg2mne, eeg_mne2eeg, pop_loadset

   EEG = pop_loadset(Path("sample_data") / "eeglab_data.set")
   raw = eeg_eeg2mne(EEG)
   raw.filter(l_freq=1, h_freq=40)
   EEG = eeg_mne2eeg(raw)

For full examples, see :ref:`mne_integration`.

Memory and Storage
==================

Large projects should use the storage options instead of keeping every dataset
resident:

.. code-block:: python

   from pathlib import Path

   from eegprep import EEG_OPTIONS, pop_loadset, pop_saveset

   old_options = dict(EEG_OPTIONS)
   try:
       EEG_OPTIONS["option_savetwofiles"] = 1
       EEG_OPTIONS["option_memmapdata"] = 1
       EEG = pop_loadset("sample_data/eeglab_data.set")
       output_dir = Path("tutorial_outputs") / "storage"
       output_dir.mkdir(parents=True, exist_ok=True)
       output_file = output_dir / "subject01.set"
       pop_saveset(EEG, output_file)
       EEG = pop_loadset(output_file)
   finally:
       EEG_OPTIONS.clear()
       EEG_OPTIONS.update(old_options)

See :ref:`large_dataset_storage` for ``option_storedisk`` and two-file storage
details.

Extending EEGPrep
=================

Use the extension SDK for reusable lab tools instead of patching EEGPrep
internals. A good extension:

* registers an ``eegprep.extensions`` entry point;
* exposes user-facing ``pop_*`` functions with ``return_com=True`` support;
* packages Markdown help resources;
* uses ``EEGPrepSession`` helpers when adding GUI actions;
* documents the workflow in Sphinx docs and packaged help.

See :ref:`extensions` and :ref:`extension_curation` for the extension contract.

Testing Custom Workflows
========================

Use the checked-in sample files for smoke tests and keep assertions tied to EEG
dictionary behavior:

.. code-block:: python

   from pathlib import Path

   from eegprep import pop_loadset

   def test_tutorial_pipeline_preserves_eeg_shape():
       EEG = pop_loadset(Path("sample_data") / "eeglab_data.set")
       output, history = tutorial_pipeline(EEG)

       assert output["nbchan"] == EEG["nbchan"]
       assert output["pnts"] < EEG["pnts"]
       assert history

For GUI/console synchronization changes, add coverage near
``tests/test_console_workspace.py`` so ``EEG``, ``ALLEEG``, ``CURRENTSET``,
``LASTCOM``, ``ALLCOM``, ``STUDY``, and ``CURRENTSTUDY`` stay synchronized.

Math Backends
=============

Floating-point results from iterative algorithms such as ICA and ASR can vary
across numerical-library and threading configurations. Inspect the current
process with either output format:

.. code-block:: console

   eegprep software_info
   eegprep software_info --json

The report distinguishes the BLAS and LAPACK libraries used to build NumPy
from libraries currently visible to ``threadpoolctl``. This distinction matters
on platforms such as macOS, where Apple Accelerate can appear in NumPy's build
configuration without appearing in the loaded-library list. Effective thread
counts and explicitly configured thread environment variables are reported when
available.

CLI manifests store the same facts under ``software.math_backend_info``. This
metadata helps compare environments; it does not predict or guarantee a
particular numerical tolerance. Backend inspection is best-effort, and any
collection error is recorded without interrupting the processing command.
