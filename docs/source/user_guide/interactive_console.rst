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
to ``ALLCOM``. Assignment-style calls also work:

.. code-block:: python

   EEG, LASTCOM = pop_reref(EEG, [])

This console behavior is specific to ``eegprep-console``. Normal Python imports
keep standard Python semantics, where returned values must be assigned manually.
