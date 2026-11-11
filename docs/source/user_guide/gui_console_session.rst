.. _gui_console_session:

========================
GUI and Console Together
========================

.. note::

   Processing a selection of multiple datasets first asks whether to overwrite
   those datasets. Cancel leaves the selection and history unchanged. Appending
   datasets opens Dataset info so you can name or save the merged result. An
   empty processing result is rejected before replacing the current dataset.

   Saved session history scripts use MATLAB commands, including dataset storage,
   concrete selection indices, and ``eeglab redraw``. These scripts are for
   EEGLAB; File > History scripts > Run script executes Python ``.py`` files.
   The console translates displayed commands to Python while preserving the
   MATLAB form in ``ALLCOM`` and dataset history.

   Replaying MATLAB vectors such as ``[1:32]`` or ``[1:2:7]`` expands inclusive
   colon ranges in Python. Channel selections still cross the MATLAB 1-based
   to Python 0-based boundary. Dataset selection and copy commands also handle
   Python's extra returned history value without changing the saved commands.


EEGPrep is designed for a mixed interactive workflow. You can load and review
data from the GUI, inspect the live workspace in ``eegprep-console``, run a
Python command, then return to the GUI without manually copying datasets between
tools.

The important idea is simple: the GUI and console are two views of one
``EEGPrepSession``.

Calling ``gui()`` or ``eeglab()`` from the console opens another window on its
existing session. It keeps scripted ``pop_editoptions(..., persist=False)``
changes without reloading saved preferences.

When to Use This Workflow
=========================

Use ``eegprep-console --full`` when you want to:

* learn what each menu action does by reading ``LASTCOM``;
* inspect ``EEG`` fields immediately after a dialog closes;
* try a Python command before committing it to a script;
* compare multiple loaded datasets in ``ALLEEG``;
* review STUDY state while using Study menu actions;
* keep a complete ``ALLCOM`` history for a notebook, script, or lab record.

Use ``eegprep-gui --full`` only when you want a GUI-only session and do not
need live Python inspection.

Daily Workflow
==============

Start the shared session:

.. code-block:: bash

   uv run eegprep-console --full

Then move between the GUI and console:

.. raw:: html

   <div class="eegprep-path">
     <p>Load <code>sample_data/eeglab_data.set</code> with
     <strong>File > Load existing dataset</strong>.</p>
     <p>In the console, inspect <code>EEG["data"].shape</code>,
     <code>CURRENTSET</code>, and <code>LASTCOM</code>.</p>
     <p>Run <code>pop_resample(EEG, 64)</code> in the console.</p>
     <p>Return to the GUI and open <strong>Plot > Channel data (scroll)</strong>;
     the browser sees the resampled current dataset.</p>
     <p>Open <strong>Tools > Filter the data</strong> from the GUI, press OK,
     then inspect <code>LASTCOM</code> and <code>ALLCOM[-3:]</code>.</p>
     <p>Copy the useful commands into a script with explicit assignment.</p>
   </div>

What Stays Synchronized
=======================

.. list-table::
   :header-rows: 1

   * - Name
     - What it means
     - How it updates
   * - ``EEG``
     - Current dataset dictionary.
     - GUI actions and console ``pop_*`` wrappers store the new current
       dataset through the session.
   * - ``ALLEEG``
     - Loaded dataset list.
     - Load, save-as-new, delete, clone, and dataset-selection actions go
       through session storage helpers.
   * - ``CURRENTSET``
     - Current dataset selection, exposed as ``0``, a scalar, or a list in the
       console.
     - Dataset menu changes and console storage calls update the same value.
   * - ``LASTCOM``
     - Most recent replayable command string.
     - Successful GUI actions and history-aware ``pop_*`` calls replace it.
   * - ``ALLCOM``
     - Chronological session command history.
     - The session appends commands once, so GUI and console history stay in
       the same order.
   * - ``STUDY`` and ``CURRENTSTUDY``
     - Current group-analysis state.
     - Study menu actions and study ``pop_*`` calls update the shared session.
   * - ``session.PLUGINLIST``
     - Bundled and installed extension inventory.
     - Extension Manager and plugin registry calls store the inventory on the
       shared session object.

How EEGPrep Implements It
=========================

``EEGPrepSession`` is the single source of truth for interactive state. The main
window, menus, dialogs, help actions, dataset menus, Study workflows, Extension
Manager, and ``eegprep-console`` all read from and write to this session.

GUI actions should update state through session helpers such as
``store_current()``, ``add_history()``, ``set_study()``, and
``notify_changed()``. They should not mutate a GUI-only copy of ``EEG`` that
the console cannot see.

Wrap menu operations in ``session.gui_action(action)`` so errors can identify
the active function through ``active_gui_action()`` and console output stays
associated with the operation. The context restores the previous action when
the operation exits, including on errors.

Long-running GUI actions use the same session boundary. For example,
GUI-launched ICA opens the EEGLAB-like ``pop_runica`` options dialog on the main
thread, runs the ICA computation behind a progress dialog, then stores the
updated dataset and history only after the worker finishes successfully. While
the worker is running, progress messages are buffered safely for
``eegprep-console`` so the replayable command remains visible before related
output.

``eegprep-console`` wraps registered ``pop_*`` functions. When a bare call such
as ``pop_resample(EEG, 64)`` returns a dataset and command string, the wrapper
stores the returned dataset, updates ``LASTCOM`` and ``ALLCOM``, and tells the
GUI to refresh. This auto-store behavior belongs to ``eegprep-console`` because
it has a session to update.

Normal Python scripts keep normal Python semantics:

.. code-block:: python

   EEG, com = pop_resample(EEG, 64, return_com=True)

Use explicit assignment in scripts, notebooks, tests, and batch jobs.

Run Registered Menus Without Dialogs
====================================

Automation and parity checks can run a real registered menu workflow against
the same session without opening Qt dialogs. Pass the exact menu label, its
registered ``pop_*`` function, and explicit parameters:

.. code-block:: python

   from eegprep import EEGPrepSession, eeglab_execmenu

   session = EEGPrepSession()
   eeglab_execmenu(
       "From ASCII/float file or MATLAB array",
       "pop_importdata",
       {"data": data, "dataformat": "array", "srate": 100},
       session=session,
   )
   eeglab_execmenu(
       "Change sampling rate",
       "pop_resample",
       [64],
       session=session,
   )

``eeglab_execmenu`` verifies that the label and function identify the same
declarative menu item, calls only an explicit set of noninteractive workflows,
and updates ``EEG``, ``ALLEEG``, ``CURRENTSET``, ``LASTCOM``, and ``ALLCOM`` as
one session operation. It never evaluates callback or history strings. For a
deterministic replay, retain the label/function/parameter triples and dispatch
them in the same order into a fresh ``EEGPrepSession``.

History You Can Reuse
=====================

After a GUI action:

.. code-block:: python

   print(LASTCOM)
   for command in ALLCOM[-5:]:
       print(command)

The recorded commands are meant to be readable and scriptable. When you move
them out of the console, keep explicit return values:

.. code-block:: python

   EEG, com = pop_eegfiltnew(EEG, locutoff=1, hicutoff=40, return_com=True)
   EEG, com = pop_resample(EEG, 64, return_com=True)

Practical Rules
===============

The data and ICA rejection menus keep calculations in the open dialog until
you choose **Close (keep marks)**. Closing stores the marks and calculation
commands in the shared session, where you can inspect ``EEG['reject']`` and
``LASTCOM`` from the console. Cancel discards the dialog's edits.

The probability and kurtosis **PLOT** buttons calculate marks and open the
same browser as **Scroll Data** or **Scroll Acts.**, with the calculated marks
ready for review. EEGLAB's **PLOT** opens ``rejstatepoch``, an interactive
statistics viewer that has not been ported to EEGPrep.

* Prefer ``eegprep-console --full`` while learning or reviewing data.
* Use the GUI for inspection-heavy actions and the console for quick checks,
  summaries, and repeatable commands.
* Do not edit ``ALLEEG`` by hand in the console unless you are deliberately
  managing session state yourself.
* Use ``return_com=True`` when a command should appear in a reproducible script.
* Remember that GUI selectors are user-facing, while direct NumPy slicing is
  zero-based Python.
* Save reviewed datasets explicitly; synchronized state is live session state,
  not a substitute for saving files.

Troubleshooting State
=====================

If the GUI and console do not show what you expect, inspect:

.. code-block:: python

   CURRENTSET
   EEG.get("setname")
   EEG["data"].shape
   LASTCOM
   ALLCOM[-3:]
   CURRENTSTUDY

If a menu item is disabled, the current dataset or STUDY probably lacks the
fields that workflow needs. For example, component-review menus need ICA fields,
and Study plotting menus need an active ``STUDY`` with precomputed measures.

Importer options and command history
====================================

File > Import data opens an options dialog after selecting a recording.
ASCII/array import includes its file selector in the options form. Set the
sampling rate, channel count, epoch length, start time, dataset name, subject,
condition, session, group, reference metadata, channel locations, and optional
ICA matrices there. ``From other dataset`` selects channel or ICA information
from the datasets in the shared session. Array input in the form accepts a
Python numeric literal; scripts can pass NumPy arrays directly. ASCII data
with more rows than columns are transposed unless ``nbchan`` specifies the
orientation. Samples are float32; absent channel locations and names remain
empty. A sampling rate of zero uses 1 Hz, which is also recorded in history.

BIOSIG import exposes channels, a time range in seconds, reference channels,
BDF event decoding, EDF+ annotations, and memory mapping. Channel and reference
indices are 1-based; references index the loaded channel list after removing
the Status channel. Marker events and EDF+ annotations have independent
switches. The Python readers use calibrated continuous data; MATLAB-only
BIOSIG mexSLOAD, overflow, and uncalibrated-reader switches are absent.
BrainVision uses inclusive 1-based sample ranges and channel indices. CNT
exposes precision, start time, duration, and keyboard-response import. EGI
Simple Binary exposes frame/segment selection, format version, and locations.

File > Import event info includes column names, header lines, time units,
append/overwrite, event indices, alignment, and automatic rate adjustment.
Reserved field names are lowercase: ``type``, ``latency``, and ``duration``.
For a capitalized header, enter the lowercase fields in column order and set
header lines to 1. A time unit of 1 means seconds; ``NaN`` means sample
positions. Channel-event import exposes channel indices, the preprocessing
comparison, edge direction, transition length, duration, and deletion options.
Duration is enabled for leading edges. Both coincident edges retain their
own event values, matching the pinned EEGLAB source.

File > Import epoch info requires field names and the number of header lines.
Optional latency, duration, and type fields determine the events generated for
each epoch. The GUI defaults to seconds; the direct API retains its samples
(default ``1 / EEG['srate']``) convention. Accepted imports update ``EEG``,
``ALLEEG``, ``CURRENTSET``, and history through the shared session. Cancel
leaves them unchanged. Use ``return_com=True`` from a script to obtain the
same command, or ``gui=True`` to open the options form.

Presentation import selects a LOG file, then offers event-type, latency, and
duration columns. The GUI defaults to ``Event Type``, ``Time``, and no duration;
the direct API defaults to ``Code`` and ``Time``. History records the choices
and timestamp units so the console can replay them.
Undefined numeric options use Python's ``float('nan')`` in recorded commands.

ERPLAB import takes an event list (numeric code, quoted label, event index)
and a bin-definition file. It adds labels and simple/contrast bin membership
to existing events while retaining their types and latencies. Use
``pop_importerplab(EEG, "events.txt", "bins.txt", True)`` to also import
non-contrast bins. Ordinary type/latency tables belong in ``pop_importevent``.
Cancel in either form preserves the shared dataset and history.

Stream 9 audit dispositions
---------------------------

F8 was resolved at EEGLAB commit ``8ac485f65``: import menu labels and order
are unchanged by these fixes. F31 records an EEGLAB Cancel error in ERPLAB;
EEGPrep cancellation preserves the session without that error. F32 was an
unverified last-folder observation. EEGPrep passes an empty initial folder
to Qt, leaving last-folder behavior to the native picker; no custom
last-folder policy is introduced. Tests cover these dispositions and the
import options. Fresh MATLAB numerical and side-by-side GUI verification
remain a human follow-up. BIDS metadata issue #538 belongs to Stream 10.
