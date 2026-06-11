.. _eeglab_migration:

========================
EEGLAB Migration Notes
========================

EEGPrep is a Python port, not a MATLAB compatibility layer. Use these notes
when translating EEGLAB habits into EEGPrep scripts and GUI workflows.

What Carries Over
=================

.. list-table::
   :header-rows: 1

   * - EEGLAB habit
     - EEGPrep equivalent
   * - ``pop_loadset``, ``pop_saveset``, ``pop_resample``, ``pop_epoch``
     - Same public names, Python arguments, and replayable command strings.
   * - ``EEG``, ``ALLEEG``, ``CURRENTSET``, ``LASTCOM``, ``ALLCOM``
     - Same workspace names inside ``eegprep-console``.
   * - GUI menu action then command history
     - GUI actions update ``LASTCOM`` and ``ALLCOM`` in the shared session.
   * - ``chanlocs``, ``event``, ``urevent``, ``epoch``, ICA fields
     - Same dictionary field names.
   * - Bundled plugins
     - clean_rawdata, FIRFilt, ICLabel, DIPFIT, and EEG-BIDS are in-package
       Python ports or explicit standalone boundaries.
   * - STUDY
     - Implemented as EEGPrep-owned dictionaries and JSON-safe measure caches.

What Changes
============

.. list-table::
   :header-rows: 1

   * - Topic
     - Migration rule
   * - Arrays
     - Use Python/NumPy zero-based indexing when slicing data.
   * - User-facing selectors
     - Keep channel, component, dataset, and epoch numbers one-based in GUI
       dialogs and command strings unless a function documents otherwise.
   * - Globals
     - EEGPrep functions take explicit inputs. The shared console namespace is
       session state, not a hidden global dependency.
   * - Runtime EEGLAB checkout
     - Installed EEGPrep does not read from ``src/eegprep/eeglab``.
   * - Optional toolboxes
     - MATLAB-only LIMO, FieldTrip STUDY source statistics, MRI BEM creation,
       AFNI atlas clipping, and unavailable ICLabel network artifacts fail
       clearly instead of being faked.
   * - Scripts
     - Assign return values explicitly outside ``eegprep-console``.

Command Translation Examples
============================

.. list-table::
   :header-rows: 1

   * - Migration intent
     - EEGPrep Python
   * - Load a dataset
     - ``EEG = pop_loadset("sample_data/eeglab_data.set")``
   * - Filter
     - ``EEG = pop_eegfiltnew(EEG, locutoff=1, hicutoff=40, plotfreqz=False)``
   * - Resample
     - ``EEG = pop_resample(EEG, 64)``
   * - Average reference
     - ``EEG = pop_reref(EEG, [])``
   * - Epoch
     - ``EEG = pop_epoch(EEG, ["square"], [-1, 2])``
   * - Run ICA
     - ``EEG = pop_runica(EEG, icatype="picard", gui=False)``
   * - Classify ICs
     - ``EEG = pop_iclabel(EEG, "default")``
   * - Create a STUDY
     - ``STUDY, ALLEEG, com = pop_study(None, ALLEEG, name="My study", return_com=True)``

History Replay
==============

In MATLAB EEGLAB, menu commands appear in the MATLAB command window. In
EEGPrep, use ``LASTCOM`` and ``ALLCOM``:

.. code-block:: python

   EEG, LASTCOM = pop_resample(EEG, 64, return_com=True)
   ALLCOM.append(LASTCOM)

Inside ``eegprep-console``, a bare call can update the current dataset:

.. code-block:: python

   pop_resample(EEG, 64)

In normal Python, assign the return value:

.. code-block:: python

   EEG = pop_resample(EEG, 64)

Indexing Boundaries
===================

EEGPrep intentionally keeps both conventions visible:

* Python arrays are zero-based: ``EEG["data"][0, :]`` is the first channel.
* GUI channel/component/dataset selectors are one-based.
* Event ``latency`` fields are sample positions in the EEGLAB-facing event
  model.
* ``icachansind`` is normalized after loading so internal indexing is
  consistent, then converted back when saving.

Plugin Migration
================

Bundled plugins are importable from ``eegprep``:

.. code-block:: python

   from eegprep import pop_clean_rawdata, pop_eegfiltnew, pop_iclabel, pop_dipfit_settings

External EEGLAB MATLAB plugins are not loaded by path. EEGPrep extensions are
normal Python packages discovered through the ``eegprep.extensions`` entry
point. See :ref:`plugins` and :ref:`extensions` for the package format.

When to Stay in MNE or MATLAB
=============================

Use MNE-Python when your analysis depends on MNE-native object workflows,
forward models, reports, or statistics. Use MATLAB EEGLAB when you need an
unported MATLAB-only plugin or toolbox today. Use EEGPrep when you want
familiar preprocessing and review workflows in Python with standalone package
behavior.
