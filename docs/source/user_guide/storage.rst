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

``pop_saveset`` saves new datasets as one-file ``.set`` files by default. To write a two-file
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
and inverse matrices keep a length-1 component or channel axis. A
multi-dimensional cell or struct array keeps its MATLAB shape, and an empty
array keeps the dimensions stored in the file. ``pop_saveset`` stores
``icaact`` as ``[]`` and keeps the in-memory activations. Continuous
``icaact`` reloads as ``(ncomp, pnts)`` and is recomputed in the data
array's dtype. ``pop_subcomp`` subtracts components with those stored
activations. Event and channel indices load as integers, and integral
``run`` and ``session`` labels stay integers.

When an existing two-file dataset is saved with ``savemode="resave"``,
EEGPrep keeps writing the same ``.fdt`` sidecar. Resave requires a ``.set``
filename. A dataset imported from a raw file (``.vhdr``, ``.edf``, and the
other readers) keeps that source name for display, but resave will not write
the dataset over it. Pass a ``.set`` path, or use File > Save current dataset
as. Any filename extension is replaced with ``.set`` before writing, so
``my.data`` and ``x.edf`` are stored as ``my.set`` and ``x.set``. The original
file is left unchanged. File > Save current dataset as asks before replacing
the ``.set`` file that will be written. If the parent folder cannot be created or the
``.set`` file is not written, ``pop_saveset`` raises an error and does not
mark the dataset saved. A missing parent folder is created when that is
possible. Save as also preserves a loaded dataset's two-file layout, naming
both files after the new target. Use ``savemode="onefile"`` to embed samples
explicitly. New datasets follow ``option_savetwofiles``.

MATLAB v7.3 datasets load through the same API, preserving complete Unicode
labels and event types, nested metadata, and empty events. Companion references
in the header must be bare filenames without directories, drives, or path
separators. The named files must sit beside the header, but may be symlinks
to data elsewhere, as in DataLad/git-annex OpenNeuro checkouts. Keep the
``.set`` and ``.fdt`` names together when moving a dataset. BrainVision
headers apply the same rule to their sample and marker files.

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

Importing MFF and Neuroscan recordings
======================================

For EGI MFF, select the ``.mff`` folder itself. Install the reader and XML
support with ``pip install "eegprep[mne]"``. For ``.eeg`` and ``.EEG`` files,
``pop_fileio`` first looks for a sibling ``.vhdr`` or ``.ahdr`` header and
uses the BrainVision reader when one exists. Without a sibling header it
uses the native Neuroscan epoch reader, which does not require MNE.
Each sweep retains trial, accept, correct, response, and reaction-time events,
including rejected sweeps.

The F119 audit's File-IO comparison reports 1,193 trials and 5,960 events.
The same recording contains 1,200 sweeps, including seven marked rejected.
FieldTrip's ``read_ns_hdr`` uses the accepted-count field at byte 364 as
``nsweeps``; its reader then reads that many leading sweeps. EEGPrep uses the
total-sweep field at byte 362 and retains all complete sweeps and their
acceptance metadata. This source-based difference is regression-tested;
MATLAB comparison remains a human verification step.
