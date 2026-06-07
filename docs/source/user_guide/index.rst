.. _user_guide:

==========
User Guide
==========

The EEGPrep user guide is organized like a working EEGLAB manual: first the
workspace and data structures, then GUI workflows, console history, scripting,
plugins, group analysis, migration notes, and troubleshooting.

.. raw:: html

   <div class="eegprep-callout">
     <strong>Recommended path:</strong> run the quick start with
     <code>sample_data/eeglab_data.set</code>, read the Concepts Guide, then
     repeat the same workflow once through the GUI and once from
     <code>eegprep-console</code>.
   </div>

Getting Started
===============

.. toctree::
   :maxdepth: 1

   installation
   quickstart
   concepts
   eeglab_migration

GUI, Console, and Scripts
=========================

.. toctree::
   :maxdepth: 1

   gui_tutorials
   interactive_console
   scripting_workflows
   mne_integration

Preprocessing and Review
========================

.. toctree::
   :maxdepth: 1

   preprocessing_pipeline
   ica_rejection
   eegbrowser
   bids_workflow
   storage

Group and Plugin Workflows
==========================

.. toctree::
   :maxdepth: 1

   study_workflows
   plugins
   extensions
   extension_curation

Admin, Help, and Development
============================

.. toctree::
   :maxdepth: 1

   gui_help_menus
   configuration
   visual_parity
   advanced_topics

Common Entry Points
===================

.. list-table::
   :header-rows: 1

   * - Task
     - GUI path
     - Console or Python call
   * - Load a tutorial dataset
     - ``File > Load existing dataset``
     - ``EEG = pop_loadset("sample_data/eeglab_data.set")``
   * - Inspect channel data
     - ``Plot > Channel data (scroll)``
     - ``eegplot(EEG)`` or ``pop_eegplot(EEG)``
   * - Filter data
     - ``Tools > Filter the data``
     - ``pop_eegfiltnew(EEG, locutoff=1, hicutoff=40)``
   * - Resample data
     - ``Tools > Change sampling rate``
     - ``pop_resample(EEG, 64)``
   * - Run clean_rawdata
     - ``Tools > Reject data using Clean Rawdata and ASR``
     - ``pop_clean_rawdata(EEG, BurstCriterion=20)``
   * - Run ICA
     - ``Tools > Decompose data by ICA``
     - ``pop_runica(EEG, icatype="picard", gui=False)``
   * - Label components
     - ``Tools > Classify components using ICLabel``
     - ``pop_iclabel(EEG, "default")``
   * - Create a STUDY
     - ``File > Create study > Using all loaded datasets``
     - ``pop_study(None, ALLEEG, name="My study")``
   * - Manage extensions
     - ``File > Manage EEGPrep extensions``
     - ``plugin_menu(show=False)``

Help Coverage
=============

Every implemented GUI Help button and ``pophelp`` topic uses EEGPrep-owned
Markdown resources packaged under ``src/eegprep/resources/help``. Missing help
is a packaging error; runtime code does not fall back to the vendored EEGLAB
reference tree.
