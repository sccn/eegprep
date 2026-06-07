.. _large_dataset_storage:

=====================
Large-Dataset Storage
=====================

EEGPrep keeps normal EEG dictionaries as the public API while supporting
EEGLAB-style large-dataset workflows through explicit Python storage handles.
The runtime does not depend on EEGLAB's MATLAB ``@memmapdata`` or ``@mmo``
classes.

Two-File ``.set`` / ``.fdt`` Datasets
=====================================

``pop_saveset`` saves one-file ``.set`` files by default. To write a MATLAB
EEGLAB-style header plus float32 data sidecar, pass ``savemode="twofiles"`` or
set ``EEG_OPTIONS["option_savetwofiles"] = 1``:

.. code-block:: python

   from eegprep import EEG_OPTIONS, pop_loadset, pop_saveset

   pop_saveset(EEG, "subject01.set", savemode="twofiles")

   EEG_OPTIONS["option_savetwofiles"] = 1
   pop_saveset(EEG, "subject02.set")

The ``.fdt`` sidecar uses EEGLAB's channel-fast float32 layout. Continuous data
round-trips as ``(nbchan, pnts)`` and epoched data round-trips as
``(nbchan, pnts, trials)``.

Memory-Mapped Data
==================

When ``EEG_OPTIONS["option_memmapdata"] = 1``, ``pop_loadset`` loads two-file
datasets through a NumPy-compatible ``MemmapData`` handle instead of copying
the full sidecar into memory:

.. code-block:: python

   EEG_OPTIONS["option_memmapdata"] = 1
   EEG = pop_loadset("subject01.set")

   first_channel = EEG["data"][0, :]
   EEG["data"][0, 0] = 0
   EEG["data"].flush()

Single-file ``.set`` datasets still load as in-memory NumPy arrays because no
separate data file exists to map. Mutating a ``MemmapData`` value writes to the
``.fdt`` sidecar; use normal EEGPrep save/history workflows when the dataset
metadata should be marked clean.

Storedisk Sessions
==================

``EEG_OPTIONS["option_storedisk"] = 1`` keeps the current selected dataset
resident and evicts saved non-current datasets from ``ALLEEG``. Evicted
datasets hold an ``OffloadedData`` handle with their saved ``.set`` path and
shape metadata. Accessing samples through that handle raises a clear error;
retrieve the dataset first:

.. code-block:: python

   EEG_OPTIONS["option_storedisk"] = 1
   ALLEEG, EEG, CURRENTSET = eeg_store(ALLEEG, EEG, 0)
   EEG, ALLEEG, CURRENTSET = eeg_retrieve(ALLEEG, 1)

The GUI, ``EEGPrepSession``, and ``eegprep-console`` use the same
``eeg_store``/``eeg_retrieve`` path, so ``EEG``, ``ALLEEG``, ``CURRENTSET``,
history, and dataset menus stay synchronized. Unsaved resident datasets cannot
be offloaded; save them first or keep ``option_storedisk`` disabled.

Current Limitations
===================

``pop_loadset`` supports full dataset loading for Phase 5. EEGLAB channel-only
and ``loadmode="info"`` paths fail clearly instead of pretending data is
available. Derived caches such as ``icaact`` are not managed by a separate
lazy-storage layer, and EEGPrep does not provide multi-process write
coordination for shared ``.fdt`` files.
