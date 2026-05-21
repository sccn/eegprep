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

The console starts with EEGLAB-style workspace names already defined:

.. code-block:: python

   EEG
   ALLEEG
   CURRENTSET
   ALLCOM
   LASTCOM
   STUDY
   CURRENTSTUDY

Actions taken in the GUI update these names in the console. Commands run in the
console update the same GUI session. For example:

.. code-block:: python

   pop_reref(EEG, [])

updates the current dataset, refreshes the GUI, and appends the returned command
to ``ALLCOM``. The console-local ``eegprep`` object wraps ``pop_*`` functions
the same way, so this also updates the shared session:

.. code-block:: python

   eegprep.pop_reref(EEG, [])

Assignment-style calls also work:

.. code-block:: python

   EEG, LASTCOM = pop_reref(EEG, [])

This console behavior is specific to ``eegprep-console``. Normal Python imports
keep standard Python semantics, where returned values must be assigned manually.
