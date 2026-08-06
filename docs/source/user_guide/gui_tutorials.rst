.. _gui_tutorials:

=============
GUI Tutorials
=============

EEGPrep's GUI is built for practical EEG review: use menus and dialogs for
visual decisions, keep ``eegprep-console`` beside it for inspection and
repeatable commands, and save the resulting datasets when review is complete.

Launch
======

Use the full menu set while learning:

.. code-block:: bash

   uv run eegprep-console --full

Use the GUI-only launcher when you do not need a console:

.. code-block:: bash

   uv run eegprep-gui --full

The console workflow is recommended because it shows ``LASTCOM`` and ``ALLCOM``
after each menu action. See :ref:`gui_console_session` for the shared-session
model and implementation details.

.. note::

   Plotting commands such as ``Plot > Channel spectra and maps`` and
   ``Plot > Channel ERPs > With scalp maps`` open a figure window on any
   interactive backend, whether you trigger them from a Plot menu or call the
   matching ``pop_*`` function in ``eegprep-console``. On non-interactive
   backends (for example ``Agg`` in headless runs) the figure is built and
   returned without opening a window.

   When scripting, pass ``plot='off'`` to any plotting ``pop_*`` function to
   build and return the figure without popping a window, for example
   ``fig = pop_spectopo(EEG, 1, [], freqs=[10], plot='off')['figure']`` to save
   or embed it. The default ``plot='on'`` displays it.

Load, Inspect, and Save a Dataset
=================================

.. raw:: html

   <div class="eegprep-path">
     <p>Open <strong>File > Load existing dataset</strong>.</p>
     <p>Select <code>sample_data/eeglab_data.set</code>.</p>
     <p>In the console, run <code>EEG["data"].shape</code>,
     <code>len(ALLEEG)</code>, <code>CURRENTSET</code>, and
     <code>LASTCOM</code>.</p>
     <p>Choose <strong>Plot > Channel data (scroll)</strong> to inspect the
     current dataset in EEGBrowser.</p>
     <p>Choose <strong>File > Save current dataset as</strong> when you want a
     new <code>.set</code> file.</p>
   </div>

Equivalent console calls:

.. code-block:: python

   from pathlib import Path
   from eegprep import pop_loadset, pop_saveset

   EEG = pop_loadset(Path("sample_data") / "eeglab_data.set")
   pop_saveset(EEG, Path("sample_data") / "eeglab_data_reviewed.set")

Filter and Resample
===================

Use these steps for a small, inspectable preprocessing pass on the tutorial
dataset:

.. list-table::
   :header-rows: 1

   * - GUI action
     - Console equivalent
   * - ``Tools > Filter the data``
     - ``pop_eegfiltnew(EEG, locutoff=1, hicutoff=40)``
   * - ``Tools > Change sampling rate``
     - ``pop_resample(EEG, 64)``
   * - ``Plot > Channel spectra and maps``
     - ``pop_spectopo(EEG, 1, [])``

Filtering with ``pop_eegfiltnew`` uses the bundled FIRFilt-style
Hamming-window FIR defaults. Boundary events are respected when continuous data
contains breaks. Resampling updates event latencies and clears stale ICA
activations.

Clean Continuous Data
=====================

The bundled clean_rawdata workflow is available through the extension menu
surface and from Python:

.. code-block:: python

   from eegprep import pop_clean_rawdata

   EEG, com = pop_clean_rawdata(
       EEG,
       FlatlineCriterion=5,
       ChannelCriterion=0.8,
       LineNoiseCriterion=4,
       Highpass=(0.25, 0.75),
       BurstCriterion=20,
       WindowCriterion=0.25,
       return_com=True,
   )

Run this on continuous data before epoching. If the dialog offers a rejected
data browser, accepting the browser keeps marks and history synchronized with
the shared session.

Run ICA and Review ICLabel
==========================

For a fast review tutorial, load the ICA sample first:

.. code-block:: python

   EEG = pop_loadset("sample_data/eeglab_data_epochs_ica.set")

Then use the GUI:

.. raw:: html

   <div class="eegprep-path">
     <p>Choose <strong>Tools > Decompose data by ICA</strong> for datasets that
     do not already contain ICA fields. The sample above already has ICA.</p>
     <p>Choose <strong>Tools > Classify components using ICLabel</strong>.</p>
     <p>Choose <strong>Plot > Component properties</strong> to open the
     ICLabel-aware property dashboard.</p>
     <p>Toggle component rejection marks, press OK, and inspect
     <code>EEG["reject"]["gcompreject"]</code> in the console.</p>
     <p>Choose <strong>Tools > Remove components from data</strong> only after
     reviewing labels, scalp maps, spectra, activity, and any DIPFIT fields.</p>
   </div>

Equivalent console calls:

.. code-block:: python

   from eegprep import eeg_icalabelstat, pop_icflag, pop_iclabel, pop_subcomp, pop_viewprops

   EEG, com = pop_iclabel(EEG, "default", return_com=True)
   stats = eeg_icalabelstat(EEG, threshold=0.9, verbose=False)
   figures = pop_viewprops(EEG, typecomp=0, chanorcomp=[1, 2, 3])
   EEG, com = pop_icflag(EEG, return_com=True)
   EEG, com = pop_subcomp(EEG, [], return_com=True)

Use the empty component list in ``pop_subcomp(EEG, [])`` to remove components
currently marked in ``EEG.reject.gcompreject``.

Use EEGBrowser for Rejection
============================

``Plot > Channel data (scroll)`` opens EEGBrowser for inspection. Rejection
workflows open the same browser in a marking mode. Continuous marks can become
sample-removal intervals through ``eeg_eegrej``; epoch marks can update
``EEG.reject.rejmanual`` and then be removed with ``pop_rejepoch``.

See :ref:`eegbrowser` for browser modes, mark fields, and performance behavior.

Create a STUDY
==============

With multiple datasets loaded in ``ALLEEG``:

.. raw:: html

   <div class="eegprep-path">
     <p>Choose <strong>File > Create study > Using all loaded datasets</strong>.</p>
     <p>Inspect <code>STUDY</code> and <code>CURRENTSTUDY</code> in the console.</p>
     <p>Choose <strong>Study > Precompute channel measures</strong> for ERP,
     spectrum, ERSP, or ITC caches.</p>
     <p>Choose <strong>Study > Plot channel measures</strong> to review cached
     measures.</p>
     <p>Choose <strong>File > Save current study as</strong> for a durable
     <code>.study</code> file.</p>
   </div>

Equivalent console call:

.. code-block:: python

   STUDY, ALLEEG, com = pop_study(None, ALLEEG, name="Tutorial study", return_com=True)

See :ref:`study_workflows` for design variables, cached measures, clustering,
PAC, and explicit optional-backend boundaries.

DIPFIT Source Workflow
======================

Dataset-level DIPFIT helpers are bundled for standalone spherical workflows.
Use them after ICA:

.. code-block:: python

   from eegprep import pop_dipfit_gridsearch, pop_dipfit_nonlinear, pop_dipfit_settings, pop_dipplot

   EEG, com = pop_dipfit_settings(EEG, model="standardBESA", return_com=True)
   EEG, com = pop_dipfit_gridsearch(
       EEG,
       [1],
       [-40, -20, 0, 20, 40],
       [-40, -20, 0, 20, 40],
       [20, 40, 60],
       40,
       return_com=True,
   )
   EEG, com = pop_dipfit_nonlinear(EEG, component=1, return_com=True)
   figures, com = pop_dipplot(EEG, [1], summary="on", return_com=True)

MRI-derived BEM creation, AFNI atlas clipping, LORETA analysis, and STUDY-level
source statistics remain explicit backend boundaries. EEGPrep raises clear
errors for unavailable external workflows instead of producing placeholder
results.

Troubleshooting GUI State
=========================

If a menu item is disabled, check the current state:

.. code-block:: python

   CURRENTSET
   CURRENTSTUDY
   EEG.get("icaweights")
   EEG.get("trials")

Use ``--full`` to show legacy and advanced menu entries. Use ``--no-plugins``
to diagnose extension startup problems with only core EEGPrep menus.
