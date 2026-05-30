.. _gui_help_menus:

===================
GUI and Help Menus
===================

EEGPrep's main window follows the EEGLAB menu layout while using standalone
Python implementations at runtime. Launch it with:

.. code-block:: bash

   uv run eegprep-gui --full

Use ``eegprep-console`` when you want the same GUI session and an interactive
Python workspace:

.. code-block:: bash

   uv run eegprep-console --full

Menu State
==========

The File and Help menus are available at startup. Edit, Tools, Plot, Study,
and Datasets become available when the shared session contains a compatible
dataset or STUDY. The Datasets menu reflects ``ALLEEG`` and ``CURRENTSET`` and
can select one or more loaded datasets.

Actions that are not implemented yet are marked as coming soon and are backed
by machine-readable placeholder metadata. EEGBrowser/eegplot-style scrolling
workflows are explicitly excluded from the current parity scope rather than
being treated as ordinary pending work.

Help Resources
==============

Dialog Help buttons and Help-menu topics open packaged Markdown resources from
``eegprep.resources.help``. Installed EEGPrep does not read from the vendored
``src/eegprep/eeglab`` reference tree for help text.

When adding a new GUI-reachable function or Help-menu topic, add a matching
``src/eegprep/resources/help/<function_name>.md`` file and cover it in tests.
Missing help resources fail clearly so incomplete runtime help cannot silently
fall back to EEGLAB source files or Python docstrings.

Admin and File Workflows
========================

Implemented admin and File-menu surfaces include dataset load/save, import and
export wrappers, BIDS import/export and metadata helpers, dataset deletion,
preferences, history save/run, STUDY load/save, update links, issue links, and
the read-only EEGPrep extension inventory.

Successful GUI actions update the shared ``EEGPrepSession`` and append history
once. In ``eegprep-console``, the corresponding ``EEG``, ``ALLEEG``,
``CURRENTSET``, ``LASTCOM``, ``ALLCOM``, ``STUDY``, and ``CURRENTSTUDY`` names
refresh from the same session.

Menu Inventory Parity
=====================

Menu inventory parity is checked structurally before visual screenshots. Export
EEGPrep inventories with:

.. code-block:: bash

   uv run --no-sync python tools/visual_parity/export_eegprep_menu_inventory.py \
      --state continuous \
      --output .visual-parity/menu_inventory/eegprep_continuous.json

Compare against an EEGLAB inventory with:

.. code-block:: bash

   uv run --no-sync python tools/visual_parity/menu_inventory.py \
      --reference .visual-parity/menu_inventory/eeglab_continuous.json \
      --candidate .visual-parity/menu_inventory/eegprep_continuous.json \
      --report .visual-parity/menu_inventory/continuous_report.md

Use inventory reports for labels, order, separators, checked state, and enabled
state. Use screenshots for main-window and dialog visual QA after the menu tree
is structurally correct.
