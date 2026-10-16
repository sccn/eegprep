.. _large_dataset_storage:

=====================
Large-Dataset Storage
=====================

EEGPrep keeps normal EEG dictionaries as the public API while supporting
large-dataset workflows through explicit Python storage handles.
The runtime does not depend on EEGLAB's MATLAB ``@memmapdata`` or ``@mmo``
classes.

Two-File ``.set`` / ``.fdt`` Datasets
=====================================

``pop_saveset`` saves one-file ``.set`` files by default. To write a two-file
``.set`` header plus float32 data sidecar, pass ``savemode="twofiles"`` or set
``EEG_OPTIONS["option_savetwofiles"] = 1``:

.. code-block:: python

   from eegprep import EEG_OPTIONS, pop_loadset, pop_saveset

   pop_saveset(EEG, "subject01.set", savemode="twofiles")

   EEG_OPTIONS["option_savetwofiles"] = 1
   pop_saveset(EEG, "subject02.set")

The ``.fdt`` sidecar uses EEGLAB's channel-fast float32 layout. Continuous data
round-trips as ``(nbchan, pnts)`` and epoched data round-trips as
``(nbchan, pnts, trials)``. Length-1 channel, sample, and trial axes are kept
when the array is stored inside the ``.set`` file as well. ICA weight, sphere,
and inverse matrices keep a length-1 component or channel axis.

When an existing two-file dataset is saved with ``savemode="resave"``,
EEGPrep keeps writing the same ``.fdt`` sidecar. A plain save without
``savemode`` follows the current ``option_savetwofiles`` setting; if that
option is disabled, the data is saved inline in the ``.set`` file.

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
separate data file exists to map. A numeric array stored in the ``.set`` file
is the dataset even when ``datfile`` names a missing or stale ``.fdt``. The
sidecar is read only when ``data`` is a filename, or when ``data`` is
``in set file`` and ``datfile`` names the sidecar. Mutating a ``MemmapData``
value writes to the ``.fdt`` sidecar; use normal EEGPrep save/history workflows
when the dataset metadata should be marked clean.

``MemmapData`` uses zero-based NumPy indices against the logical
``(channels, samples[, trials])`` shape. Ordinary item assignment follows
NumPy: an index past the end raises, and assigning an empty array does not
delete. ``matlab_assign`` is the EEGLAB subscript operation. An index past
the end grows that axis and fills the new samples with zeros. An empty value
deletes the indexed samples. A key with fewer entries than the array rank
folds the trailing dimensions together before the deletion, so a
channels-by-samples-by-trials array indexed with two subscripts is treated
as channels by samples-times-trials. A boolean mask is converted with
``find``: only ``True`` positions are removed, and an all-false mask leaves
the data unchanged. ``delete(indices, axis=...)`` addresses one axis of the
full array and does not fold; it uses the same ``find`` rule for masks.
Copies made with ``copy.copy`` or ``copy.deepcopy`` share the sidecar until
one copy is written, then use a private copy-on-write sidecar so processing
a copied EEG does not mutate its source. ``resize`` retains the overlapping
region and fills new samples with zeros by default. A read-only handle is
copied on the first write. ``debug=True`` makes ``matlab_assign`` print
whether that write reused the file.

EEGPrep preprocessing keeps disk-backed input disk-backed through continuous
rejection, epoch extraction, baseline removal, FIR filtering, rereferencing,
selection, and resampling. Each result gets its own writable temporary mapping,
while the original dataset and sidecar remain unchanged.

For direct construction, ``mmo`` is the compact EEGLAB-compatible entry point:

.. code-block:: python

   from eegprep import mmo

   data = mmo("subject01.fdt", (64, 30000), writable=True)
   blank = mmo(None, (64, 30000))

Pass ``transposed=True`` for files physically stored as
``(samples, trials, channels)``. Indexing still uses the logical channels-first
shape; storage orientation never changes the public shape or index order.

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

Calling ``eeg_retrieve(ALLEEG, 0)`` represents the no-current-dataset state: it
returns an empty EEG dataset and ``CURRENTSET == 0`` without deleting or
renumbering ``ALLEEG``.

The GUI, ``EEGPrepSession``, and ``eegprep-console`` use the same
``eeg_store``/``eeg_retrieve`` path, so ``EEG``, ``ALLEEG``, ``CURRENTSET``,
history, and dataset menus stay synchronized. Unsaved resident datasets cannot
be offloaded; save them first or keep ``option_storedisk`` disabled.

Selective Loading
=================

``pop_loadset(path, loadmode="info")`` loads metadata without loading sample
data. ``EEG["data"]`` is the sidecar filename when that file exists, or
``"in set file"`` when the samples live in the ``.set``. If the sidecar is
missing and the ``.set`` already contains a numeric array, that array is kept.
Passing an integer or sequence as ``loadmode`` loads those 1-based channels
and clears ICA fields that no longer describe the selected channel matrix.

Derived caches such as ``icaact`` are not managed by a separate lazy-storage
layer, and EEGPrep does not provide multi-process write coordination for shared
``.fdt`` files.
