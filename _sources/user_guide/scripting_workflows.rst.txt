.. _scripting_workflows:

===================
Scripting Workflows
===================

EEGPrep scripts should be ordinary Python: explicit imports, explicit
assignments, and clear paths. Use ``return_com=True`` whenever a command should
be replayable from session history.

From Menu History to Script
===========================

When you run a GUI action or a ``pop_*`` call in ``eegprep-console``, inspect
``LASTCOM`` and ``ALLCOM``:

.. code-block:: python

   LASTCOM
   ALLCOM[-5:]

Move the same calls into a script with explicit assignment:

.. code-block:: python

   from pathlib import Path

   from eegprep import pop_eegfiltnew, pop_loadset, pop_resample, pop_saveset

   EEG = pop_loadset(Path("sample_data") / "eeglab_data.set")
   EEG, com_filter = pop_eegfiltnew(EEG, locutoff=1, hicutoff=40, plotfreqz=False, return_com=True)
   EEG, com_resample = pop_resample(EEG, 64, return_com=True)
   pop_saveset(EEG, Path("sample_data") / "eeglab_data_scripted.set")

   history = [com_filter, com_resample]

Normal Python does not auto-store bare ``pop_*`` calls. This is intentional:
auto-storage belongs to ``eegprep-console`` because it has a shared
``EEGPrepSession`` to update.

Batch Over Sample Data
======================

Use ``pathlib`` and keep outputs outside immutable tutorial files:

.. code-block:: python

   from pathlib import Path

   from eegprep import pop_eegfiltnew, pop_loadset, pop_resample, pop_saveset

   inputs = [
       Path("sample_data") / "eeglab_data.set",
       Path("sample_data") / "eeglab_data_with_ica_tmp.set",
   ]
   output_dir = Path("sample_data") / "processed"
   output_dir.mkdir(exist_ok=True)

   all_history = {}
   for input_file in inputs:
       EEG = pop_loadset(input_file)
       EEG, filter_com = pop_eegfiltnew(EEG, locutoff=1, hicutoff=40, plotfreqz=False, return_com=True)
       EEG, resample_com = pop_resample(EEG, 64, return_com=True)
       pop_saveset(EEG, output_dir / input_file.name)
       all_history[input_file.name] = [filter_com, resample_com]

Do not mutate ``ALLEEG`` by hand in scripts unless you are deliberately
building a session-like workflow. Use ``EEGPrepSession`` when you need
the same dataset storage model as the GUI and console.

Use an EEGPrepSession
=====================

``EEGPrepSession`` is useful when a script needs the same state model as the
GUI:

.. code-block:: python

   from pathlib import Path
   from eegprep import EEGPrepSession, pop_loadset, pop_newset, pop_resample

   session = EEGPrepSession()
   EEG = pop_loadset(Path("sample_data") / "eeglab_data.set")
   ALLEEG, EEG, CURRENTSET, com = pop_newset(session.ALLEEG, EEG, session.CURRENTSET, setname="tutorial")
   session.ALLEEG = ALLEEG
   session.store_current(EEG, set_index=CURRENTSET)
   session.add_history(com)

   EEG, com = pop_resample(session.EEG, 64, return_com=True)
   session.store_current(EEG)
   session.add_history(com)

This mirrors what ``eegprep-console`` does for you automatically.

Script Component Review
=======================

Use the ICA sample to script ICLabel review:

.. code-block:: python

   from pathlib import Path

   import numpy as np
   from eegprep import eeg_icalabelstat, pop_icflag, pop_iclabel, pop_loadset, pop_subcomp

   EEG = pop_loadset(Path("sample_data") / "eeglab_data_epochs_ica.set")
   EEG, com_label = pop_iclabel(EEG, "default", return_com=True)
   stats = eeg_icalabelstat(EEG, threshold=0.9, verbose=False)

   # Flag components with high eye or muscle probability, then remove flagged components.
   EEG, com_flag = pop_icflag(
       EEG,
       threshold=np.array(
           [
               [0, 0],        # Brain
               [0.8, 1.0],    # Muscle
               [0.8, 1.0],    # Eye
               [0, 0],        # Heart
               [0, 0],        # Line noise
               [0, 0],        # Channel noise
               [0, 0],        # Other
           ]
       ),
       return_com=True,
   )
   EEG, com_remove = pop_subcomp(EEG, [], return_com=True)

The class order follows ICLabel: Brain, Muscle, Eye, Heart, Line Noise,
Channel Noise, Other.

Work With STUDY Scripts
=======================

Create a small STUDY from loaded datasets:

.. code-block:: python

   from pathlib import Path

   from eegprep import pop_loadset, pop_precomp, pop_study, std_readerp

   ALLEEG = [
       pop_loadset(Path("sample_data") / "eeglab_data_epochs_ica.set"),
       pop_loadset(Path("sample_data") / "eeglab_data_epochs_ica.set"),
   ]
   for index, EEG in enumerate(ALLEEG, start=1):
       EEG["subject"] = f"S{index:02d}"
       EEG["condition"] = "tutorial"

   STUDY, ALLEEG, com_study = pop_study(None, ALLEEG, name="Tutorial study", return_com=True)
   STUDY, ALLEEG, com_precomp = pop_precomp(STUDY, ALLEEG, "channels", erp="on", return_com=True)
   erpdata, erptimes = std_readerp(STUDY, ALLEEG, channels=[1])[:2]

STUDY dataset and trial selectors return EEGLAB-facing one-based indices. Use
Python zero-based indices only when directly slicing arrays.

Guardrails
==========

* Keep tutorial and lab scripts explicit: assign returned ``EEG`` or ``STUDY``.
* Use ``return_com=True`` for history-relevant steps.
* Keep user-facing component, channel, and dataset selectors one-based in
  command strings.
* Do not rely on the vendored ``src/eegprep/eeglab`` tree at runtime.
* Prefer packaged resources, project-relative paths, and ``pathlib.Path``.
