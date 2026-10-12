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

Channel locations can be normalized independently with
``eeg_checkchanlocs``. It fills the standard location fields, converts
compatible coordinate systems, moves fiducials and ``IGNORE`` entries to
``EEG["chaninfo"]["nodatchans"]``, and normalizes the recorded nose direction
to ``+X`` when the coordinates allow it.

Use ``eeg_getdatact`` to extract channels, ICA activations, component-removed
data, or component backprojections without mutating the dataset. Its selector
arguments are 1-based at the public API boundary, while stored
``icachansind`` values remain 0-based. The default result is always
``(signals, samples, trials)``; ``reshape="2d"`` concatenates trials in
column-major order for EEGLAB-compatible numerical workflows.

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
When one command is applied to a multi-dataset selection, each selected EEG
history records it with the ``% multiple datasets command:`` marker used by
EEGLAB, while the session records the command once in ``ALLCOM``.

EEGPrep does not emulate EEGLAB's one-dataset-in-memory ``option_storedisk``
behavior. Saved non-current datasets are represented by explicit offloaded disk
handles and rehydrated through the shared storage path via ``eeg_store`` and
``eeg_retrieve``. Unsaved resident datasets stay resident, or fail clearly,
until you save them.

Asynchronous Results
====================

An awaited computation, such as ICLabel in a browser runtime,
can finish after the GUI or the console has changed the session.
Code that stores such a result captures ``token = session.dataset_state_token()`` before awaiting
and calls ``session.require_dataset_state_unchanged(token, function_name)`` before storing.
The second call raises ``RuntimeError`` naming ``function_name`` when the result would overwrite data it was not computed from,
and says whether the dataset selection changed or a selected dataset was modified.
The GUI ICLabel action and the console wrapper for any coroutine ``pop_*`` function follow this contract;
functions that return a result without storing it, such as ``iclabel_async``, need no check.

The token is a ``DatasetStateToken`` holding ``CURRENTSET``
and a BLAKE2b digest of the current ``EEG`` and the selected ``ALLEEG`` entries, each object hashed once.
It compares content, not identity or change notifications,
so it detects edits made through any alias, including in-place NumPy writes.
Capturing a token reads the selected data once, in fixed-size chunks and without copying the arrays;
an awaited call costs two such passes, because ``require_dataset_state_unchanged`` captures a second token to compare.
Dictionaries, lists, tuples, NumPy arrays, and memory-mapped data are hashed by content.
Any other value, such as a SciPy sparse matrix stored in a dataset field, is hashed by its ``repr``,
so an in-place edit that does not change the ``repr`` is not detected:
replace such objects instead of editing them in place.

The digest skips the top-level ``history`` and ``saved`` fields,
because recording a command or marking the dataset saved does not change the data a result was computed from.
Nested fields with those names are hashed, and unselected datasets are ignored.
Setting the ``saved`` flag alone therefore keeps an in-flight result,
but a Save that assigns a new ``filename`` or ``filepath``, such as the first Save of a dataset,
changes hashed fields and discards it.
An edit that is reverted before the check is not reported, because the content is the same again.

Under Emscripten, ``MenuActionDispatcher.dispatch_gui`` returns the ``dispatch_gui_async`` coroutine for ICLabel instead of running it.
The main window schedules it on the running asyncio event loop, and shows a warning when no loop is running.

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
