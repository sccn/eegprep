.. _interactive_console:

===================
Interactive Console
===================

EEGPrep can run the main GUI and a Python console against one shared session.
This is the recommended workflow when you want to move back and forth between
menu-driven actions and Python commands.

Launch it with:

.. code-block:: bash

   uv run eegprep-console --full

EEGPrep uses IPython for this console because it provides a mature Qt input
hook, tab completion, command history, and rich interactive inspection. If
IPython is not installed, the command fails with an install hint instead of
falling back to a different console.

The console uses Qt file dialogs by default because native macOS file pickers
can close immediately when opened from IPython's Qt input hook. To opt into
native file pickers, launch:

.. code-block:: bash

   uv run eegprep-console --full --native-file-dialogs

The console starts with familiar workspace names already defined:

.. code-block:: python

   EEG
   ALLEEG
   CURRENTSET
   ALLCOM
   LASTCOM
   STUDY
   CURRENTSTUDY

Actions taken in the GUI update these names in the console. Commands run in the
console update the same GUI session.

For the full mixed workflow and implementation model, see
:ref:`gui_console_session`.

Sample Data Walkthrough
=======================

Start the console and load the tutorial dataset from the GUI with
``File > Load existing dataset``. Select ``sample_data/eeglab_data.set``. Then
inspect the shared state:

.. code-block:: python

   EEG["setname"]
   EEG["data"].shape
   CURRENTSET
   LASTCOM

Now run a preprocessing step from the console:

.. code-block:: python

   pop_resample(EEG, 64)

The current dataset is stored back into the session, the GUI refreshes, and the
returned command is appended to ``ALLCOM``. The console-local ``eegprep`` object
wraps ``pop_*`` functions the same way, so this also updates the shared session:

.. code-block:: python

   eegprep.pop_reref(EEG, [])

Assignment-style calls also work:

.. code-block:: python

   EEG, LASTCOM = pop_reref(EEG, [])

This console behavior is specific to ``eegprep-console``. Normal Python imports
keep standard Python semantics, where returned values must be assigned manually.

History Replay
==============

Use ``LASTCOM`` for the most recent menu or console command and ``ALLCOM`` for
the ordered session history:

.. code-block:: python

   print(LASTCOM)
   for command in ALLCOM[-5:]:
       print(command)

When you move commands into a script, keep the explicit assignment:

.. code-block:: python

   EEG, com = pop_resample(EEG, 64, return_com=True)

EEGLAB history and GUI commands use one-based channel indices. When those
commands are replayed, ``eegprep-console`` translates literal ``bad_elec``
values for ``pop_interp`` and literal ``icachansind`` values for
``pop_editset`` to Python's zero-based indices. Dynamic expressions are left
unchanged, so write their results in zero-based form.

GUI Help, STUDY, and Extensions
===============================

Help and admin menu actions use the same shared session. For example, loading
or creating a STUDY from the GUI updates ``STUDY`` and ``CURRENTSTUDY`` in the
console, while retrieving a dataset from the Datasets menu returns
``CURRENTSTUDY`` to ``0``. Dialog Help buttons and Help-menu topics open
packaged EEGPrep Markdown resources; missing help is treated as a packaging
error rather than falling back to the vendored EEGLAB reference tree.

The same session also tracks Extension Manager metadata. Choose
``File > Manage EEGPrep extensions`` in the GUI, then inspect the mirrored
inventory from the console:

.. code-block:: python

   PLUGINLIST = session.PLUGINLIST
   [plugin["plugin"] for plugin in PLUGINLIST]

Headless scripts can use the same public API without opening Qt:

.. code-block:: python

   import eegprep

   session = eegprep.EEGPrepSession()
   plugins = eegprep.plugin_menu(session=session, show=False)
   eegprep.plugin_status("ICLabel", exactmatch=True)

The inventory combines installed registry records with curated catalog metadata
when available. Install and update guidance is displayed as copyable package
manager commands; EEGPrep does not execute those commands or manage extension
package installation for you.
