=======
EEGPrep
=======

.. raw:: html

   <div class="eegprep-hero">
     <div class="eegprep-kicker">Standalone Python manual for EEGLAB users</div>
     <p>EEGPrep ports core EEGLAB preprocessing ideas, names, data structures,
     GUI workflows, command history, and bundled plugin behavior into a Python
     package that can run without a MATLAB or EEGLAB checkout.</p>
   </div>

Use this manual as a working path, not only as an API index. Start with the
sample datasets in ``sample_data/``, move between the Qt GUI and
``eegprep-console``, then reuse the recorded ``pop_*`` commands in scripts.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Start and Load Data
      :link: user_guide/quickstart
      :link-type: doc

      Install EEGPrep, launch the GUI/console, load ``eeglab_data.set``, inspect
      the EEG structure, and save the result.

   .. grid-item-card:: Learn the Data Model
      :link: user_guide/concepts
      :link-type: doc

      Understand ``EEG``, ``ALLEEG``, events, epochs, channel locations, ICA
      fields, STUDY, and history replay.

   .. grid-item-card:: Follow GUI Workflows
      :link: user_guide/gui_tutorials
      :link-type: doc

      Run EEGLAB-style menus for filtering, rejection, ICA, ICLabel, EEGBrowser,
      STUDY, and DIPFIT while tracking console state.

   .. grid-item-card:: Script the Same Steps
      :link: user_guide/scripting_workflows
      :link-type: doc

      Convert menu history into reusable Python scripts and batch workflows.

Manual
======

.. toctree::
   :maxdepth: 2
   :caption: Workflows

   user_guide/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   faq
   glossary
   references

.. toctree::
   :maxdepth: 2
   :caption: Project

   contributing
   development
   changelog

Five-Minute Script
==================

Load the checked-in EEGLAB tutorial dataset, run two common preprocessing steps,
and keep the replayable command strings.

.. code-block:: python

   from pathlib import Path
   from eegprep import pop_eegfiltnew, pop_loadset, pop_resample, pop_saveset

   sample = Path("sample_data") / "eeglab_data.set"
   EEG = pop_loadset(sample)

   EEG, filter_com = pop_eegfiltnew(EEG, locutoff=1.0, hicutoff=40.0, return_com=True)
   EEG, resample_com = pop_resample(EEG, 64, return_com=True)
   pop_saveset(EEG, Path("sample_data") / "eeglab_data_preprocessed.set")

   print(filter_com)
   print(resample_com)

Where EEGLAB Users Should Go First
==================================

.. list-table::
   :header-rows: 1

   * - If you know EEGLAB as...
     - Start here in EEGPrep
   * - ``EEG`` / ``ALLEEG`` / ``CURRENTSET``
     - :doc:`user_guide/concepts`
   * - GUI menus plus MATLAB command history
     - :doc:`user_guide/gui_tutorials` and :doc:`user_guide/interactive_console`
   * - ``pop_*`` scripts
     - :doc:`user_guide/scripting_workflows` and :doc:`api/pop_functions`
   * - clean_rawdata, FIRFilt, ICLabel, DIPFIT, EEG-BIDS
     - :doc:`user_guide/plugins`
   * - EEGBrowser and visual rejection
     - :doc:`user_guide/eegbrowser`
   * - STUDY workflows
     - :doc:`user_guide/study_workflows`
   * - MNE-Python interop
     - :doc:`user_guide/mne_integration`

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
