.. _gui_support:

===========
GUI Support
===========

EEGPREP's GUI layer is optional and is designed to keep EEGLAB workflows
familiar while keeping the Python API testable and scriptable.

Installation
============

Install the optional GUI dependencies when you need desktop dialogs:

.. code-block:: bash

   pip install eegprep[gui]

Event Latency Adjustment
========================

``pop_adjustevents`` ports EEGLAB's "Adjust event latencies" workflow. Event
latencies remain EEGLAB-style 1-based floating sample positions.

Use the command-line path for reproducible scripts:

.. code-block:: python

   EEG, com = eegprep.pop_adjustevents(EEG, addsamples=2.5)
   EEG, com = eegprep.pop_adjustevents(EEG, addms=-10, eventtypes=["stim"])

Use the GUI path when interactive selection is preferred:

.. code-block:: python

   EEG, com = eegprep.pop_adjustevents(EEG, gui=True)

The GUI path collects options, then calls the same tested computation path as
the script API.
