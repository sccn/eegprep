=======
EEGPrep
=======

.. raw:: html

   <div class="eegprep-hero">
     <div class="eegprep-kicker">EEGLAB workflows, modern Python tooling</div>
     <p>EEGPrep is a modernized Python port of core
     <a href="https://sccn.ucsd.edu/eeglab/">EEGLAB</a> preprocessing
     workflows for EEG researchers. It preserves familiar concepts such as
     <code>EEG</code>, <code>ALLEEG</code>, <code>pop_*</code> commands,
     EEGBrowser review, ICA/component workflows, STUDY analyses, and command
     history, while making those workflows easier to run, review, automate,
     and share from Python.</p>
   </div>

Use this manual to move from a first dataset to a reproducible workflow. Start
with the sample data, try the same steps in the GUI and ``eegprep-console``,
then turn the recorded ``pop_*`` history into scripts, pipelines, or
agent-assisted analyses.

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

   .. grid-item-card:: Work in GUI and Console
      :link: user_guide/gui_console_session
      :link-type: doc

      Keep the main window and ``eegprep-console`` side by side: run menu
      actions, inspect ``EEG`` and history, then continue from Python.

   .. grid-item-card:: Follow GUI Workflows
      :link: user_guide/gui_tutorials
      :link-type: doc

      Run menu workflows for filtering, rejection, ICA, ICLabel, EEGBrowser,
      STUDY, and DIPFIT while tracking the same session from the console.

   .. grid-item-card:: Script the Same Steps
      :link: user_guide/scripting_workflows
      :link-type: doc

      Convert menu history into reusable Python scripts and batch workflows.

   .. grid-item-card:: Work With Research Agents
      :link: user_guide/agent_cli
      :link-type: doc

      Use the EEGPrep CLI, JSON outputs, manifests, pipeline plans, and bundled
      agent skill guidance for AI-assisted EEG research projects.

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
   references

.. toctree::
   :maxdepth: 2
   :caption: Project

   contributing
   development
   releasing
   changelog

Five-Minute Script
==================

Load the checked-in tutorial dataset, run two common preprocessing steps, and
keep the replayable command strings.

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
     - :doc:`user_guide/gui_console_session`, :doc:`user_guide/gui_tutorials`,
       and :doc:`user_guide/interactive_console`
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

Core Maintainers
================

- `Arnaud Delorme <https://arnauddelorme.com/>`_, UCSD, CA, USA
- `Suraj Ranganath <https://surajr.dev/>`_, UCSD, CA, USA
- Christian Kothe, Intheon, CA, USA
- Bruno Aristimunha Pinto, Inria, France

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
