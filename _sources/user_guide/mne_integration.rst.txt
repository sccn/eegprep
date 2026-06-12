.. _mne_integration:

======================
MNE-Python Integration
======================

This page addresses the MNE examples requested in GitHub issue #22. EEGPrep is
not a replacement for MNE-Python. The intended workflow is to use EEGPrep when
you want EEGPrep preprocessing, ``pop_*`` history, GUI review, and standalone
plugin ports, then convert to MNE when your downstream analysis belongs in MNE.

Install
=======

MNE-Python is a core dependency in current EEGPrep source installs. If you are
building a custom environment, make sure it is present:

.. code-block:: bash

   uv add mne

EEGPrep's conversion helpers use MNE's EEGLAB import/export path internally.
Keep MNE and EEGPrep in the same environment.

EEGPrep to MNE Raw
==================

Continuous EEGPrep data converts to an MNE ``Raw`` object:

.. code-block:: python

   from pathlib import Path

   from eegprep import eeg_eeg2mne, pop_loadset

   EEG = pop_loadset(Path("sample_data") / "eeglab_data.set")
   raw = eeg_eeg2mne(EEG)

   print(raw)
   print(raw.info["sfreq"])

Use this path when you want to inspect, plot, or analyze an EEGPrep-processed
dataset with MNE functions.

MNE Raw to EEGPrep
==================

You can also start from an MNE object and move into EEGPrep:

.. code-block:: python

   import numpy as np
   import mne

   from eegprep import eeg_mne2eeg, pop_eegfiltnew

   sfreq = 128.0
   data = 1e-6 * np.random.default_rng(0).normal(size=(4, int(sfreq * 10)))
   info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq=sfreq, ch_types="eeg")
   raw = mne.io.RawArray(data, info)
   raw.set_annotations(mne.Annotations(onset=[1.0, 5.0], duration=[0.0, 0.0], description=["stim", "stim"]))

   EEG = eeg_mne2eeg(raw)
   EEG, com = pop_eegfiltnew(EEG, locutoff=1, hicutoff=40, plotfreqz=False, return_com=True)

   print(EEG["data"].shape)
   print(EEG["event"])
   print(com)

Annotations become EEGPrep event records with EEGLAB-facing sample latencies.

MNE Epochs to EEGPrep
=====================

``eeg_mne2eeg`` accepts MNE Epochs objects through MNE's EEGLAB exporter. Use
``eeg_mne2eeg_epochs`` when you specifically need the ICA-aware epochs
conversion helper; that helper expects both an MNE ``Epochs`` object and an MNE
``ICA`` object.

.. code-block:: python

   import numpy as np
   import mne

   from eegprep import eeg_mne2eeg

   sfreq = 128.0
   info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq=sfreq, ch_types="eeg")
   data = 1e-6 * np.random.default_rng(1).normal(size=(3, 4, 128))
   events = np.array([[0, 0, 1], [128, 0, 1], [256, 0, 2]])
   epochs = mne.EpochsArray(data, info, events=events, event_id={"target": 1, "standard": 2}, tmin=-0.2)

   EEG = eeg_mne2eeg(epochs)
   print(EEG["data"].shape)
   print(EEG["trials"])

Round-Trip Pattern
==================

A practical workflow is:

.. raw:: html

   <div class="eegprep-path">
     <p>Use MNE to read a hardware-specific file format when MNE has the best
     reader for your data.</p>
     <p>Convert the MNE object to EEGPrep with <code>eeg_mne2eeg</code>.</p>
     <p>Run EEGPrep preprocessing, GUI review, ICLabel, or STUDY setup in
     EEGPrep.</p>
     <p>Convert back with <code>eeg_eeg2mne</code> for MNE source analysis,
     statistics, reports, or pipelines that require MNE objects.</p>
   </div>

Boundary Notes
==============

* EEGPrep stores data in EEG dictionaries. MNE stores data in ``Raw`` and
  ``Epochs`` objects with an ``Info`` structure.
* MNE channel data is usually in volts. EEGLAB ``.set`` readers and writers may
  display EEG data in microvolt-oriented workflows. Inspect units before
  comparing absolute amplitudes.
* EEGPrep event latencies are EEGLAB-facing sample positions. MNE annotations
  are onset times in seconds and events arrays are sample-index rows.
* Conversion helpers use temporary EEGLAB files internally, so keep output paths
  writable and avoid using them inside tight real-time loops.
* For BIDS projects, use :ref:`bids_workflow` when you want EEGPrep's BIDS
  metadata handling and use MNE-BIDS/MNE when your downstream analysis is
  MNE-native.
