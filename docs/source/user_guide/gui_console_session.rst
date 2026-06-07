.. _gui_console_session:

========================
GUI and Console Together
========================

EEGPrep is designed for a mixed interactive workflow. You can load and review
data from the GUI, inspect the live workspace in ``eegprep-console``, run a
Python command, then return to the GUI without manually copying datasets between
tools.

The important idea is simple: the GUI and console are two views of one
``EEGPrepSession``.

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
   * - ``PLUGINLIST``
     - Bundled and installed extension inventory.
     - Extension Manager and plugin registry calls read the same runtime
       registry.

How EEGPrep Implements It
=========================

``EEGPrepSession`` is the single source of truth for interactive state. The main
window, menus, dialogs, help actions, dataset menus, Study workflows, Extension
Manager, and ``eegprep-console`` all read from and write to this session.

GUI actions should update state through session helpers such as
``store_current()``, ``add_history()``, ``set_study()``, and
``notify_changed()``. They should not mutate a GUI-only copy of ``EEG`` that
the console cannot see.

``eegprep-console`` wraps registered ``pop_*`` functions. When a bare call such
as ``pop_resample(EEG, 64)`` returns a dataset and command string, the wrapper
stores the returned dataset, updates ``LASTCOM`` and ``ALLCOM``, and tells the
GUI to refresh. This auto-store behavior belongs to ``eegprep-console`` because
it has a session to update.

Normal Python scripts keep normal Python semantics:

.. code-block:: python

   EEG, com = pop_resample(EEG, 64, return_com=True)

Use explicit assignment in scripts, notebooks, tests, and batch jobs.

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

