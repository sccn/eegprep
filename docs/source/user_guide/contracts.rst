.. _contracts:

=========================
EEG and Session Contracts
=========================

EEGPrep follows EEGLAB's public data model while keeping Python internals
explicit and testable. These are the shared guarantees you can rely on across
functions, whether you are scripting against EEGPrep or extending it. For an
introduction to the field names and shapes themselves, read :ref:`concepts`
first.

EEG Dictionaries
================

Stored EEG dictionaries are normalized through ``eeg_checkset`` or an
``EEGPrepSession`` storage helper before other code relies on shape or type
invariants. If you build a dataset dictionary by hand, pass it through
``eeg_checkset`` before handing it to other functions.

``event`` entries keep EEGLAB-facing ``latency`` values and, when available,
``urevent`` pointers back to ``urevent`` entries. ``urevent`` is the
original-event table; functions that create, delete, or reorder events state
whether they preserve, extend, or rebuild it.

ICA fields are cleared or recomputed consistently when the data, channel order,
or channel count changes, so a stale decomposition is never left attached to
data it no longer describes.

Session Selection
=================

``EEGPrepSession.CURRENTSET`` is always a Python ``list[int]`` of EEGLAB-facing
1-based dataset indices. The console presents it in EEGLAB's shape: ``0`` when
the selection is empty, ``n`` for a single dataset, and ``[n, ...]`` for a
multi-dataset selection. Selection order is preserved, and duplicate dataset
indices are invalid.

Read selection state through ``EEGPrepSession.selected_dataset_indices()`` when
you need the current dataset vector, for example in group-level or STUDY code.

History and Menu Inventory
==========================

User-facing ``pop_*`` functions support ``return_com=True`` and return a history
command that converts to valid ``eegprep-console`` input. GUI and console code
appends each successful command once, through ``EEGPrepSession.add_history`` or
a storage helper, so history never double-records an action.

GUI actions that produce a new EEG dataset — resampling, filtering, cleaning,
epoching, selecting data, rereferencing, interpolation, component removal —
commit through ``pop_newset``, so you choose whether to overwrite the current
dataset or keep the result as a new one. Actions that only update metadata,
marks, history, ICA fields, or STUDY state may store directly, matching
EEGLAB's callback behavior.

``eegprep-console`` and the GUI share one session. GUI command echoes show
replayable Python input before progress messages or warnings from the same
action. ``eegh`` presents history newest-first like EEGLAB, while
``EEGPrepSession.ALLCOM`` stays chronological internally.

EEGPrep does not emulate EEGLAB's one-dataset-in-memory ``option_storedisk``
behavior. Saved non-current datasets are represented by explicit offloaded disk
handles and rehydrated through the shared storage path via ``eeg_store`` and
``eeg_retrieve``. Unsaved resident datasets stay resident, or fail clearly,
until you save them.

Standalone Runtime Boundary
===========================

Runtime package code never reads, imports, or shells out to
``src/eegprep/eeglab``; that tree is a development parity reference only. GUI
Help buttons and Help-menu topics resolve to packaged Markdown files under
``src/eegprep/resources/help`` rather than falling back to the vendored EEGLAB
tree or Python docstrings.

Menu placeholders are machine-readable: each placeholder action carries either a
target epic phase or an explicit exclusion reason for workflows that cannot be
packaged in EEGPrep.

.. note::

   Extending EEGPrep? When you add an implemented GUI-reachable ``pop_*`` or
   ``eeg_*`` action, add its help resource under
   ``src/eegprep/resources/help`` and extend the menu and help-resource
   inventory tests. See :ref:`development` for the test layout.
