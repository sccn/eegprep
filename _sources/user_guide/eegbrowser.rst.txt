.. _eegbrowser:

====================
EEGBrowser Workflows
====================

EEGPrep includes a scrolling browser for inspecting continuous, epoched,
component, spectral, and overlay data. The browser is available from the Plot
menu, rejection workflows, Python APIs, and ``eegprep-console``.

Opening The Browser
===================

From the GUI, load or select a dataset and use the Plot menu entries for
channel data, component activity, or visual rejection. The rejection dialogs can
also open browser-backed review windows before updating marks or removing data.

With the tutorial data:

.. raw:: html

   <div class="eegprep-path">
     <p>Load <code>sample_data/eeglab_data.set</code>.</p>
     <p>Choose <strong>Plot > Channel data (scroll)</strong>.</p>
     <p>Use the browser to review channels, events, scaling, and visible time
     ranges before making rejection decisions.</p>
     <p>Run <code>LASTCOM</code> in <code>eegprep-console</code> to confirm
     which browser action was recorded.</p>
   </div>

From Python or ``eegprep-console``:

.. code-block:: python

   from eegprep import eegplot, pop_eegplot

   window = eegplot(EEG)
   EEG, com = pop_eegplot(EEG, return_com=True)

Use ``show=False`` when testing or scripting normalization without opening Qt:

.. code-block:: python

   model = eegplot(EEG, winlength=5, dispchans=16, show=False)

Browser Modes
=============

.. list-table::
   :header-rows: 1

   * - Workflow
     - GUI entry
     - Programmatic entry
   * - Channel inspection
     - ``Plot > Channel data (scroll)``
     - ``eegplot(EEG)`` or ``pop_eegplot(EEG)``
   * - Component inspection
     - ``Plot > Component activations (scroll)``
     - ``pop_eegplot(EEG, icacomp=0, reject=0)``
   * - Epoched data rejection
     - ``Tools > Reject data epochs > Reject by inspection``
     - ``pop_eegplot(EEG, reject=1)``
   * - ICA rejection marks
     - ``Tools > Reject data using ICA > Reject by inspection``
     - ``pop_eegplot(EEG, icacomp=0, reject=1)``

Marks And Rejection
===================

Dragging across the browser marks a time range. On continuous data, accepted
marks can be stored in ``EEG.reject.rejmanualwinrej`` for later review or
converted through ``eeg_eegrej`` to remove samples. On epoched data, accepted
marks update ``EEG.reject.rejmanual`` and ``EEG.reject.rejmanualE`` or remove
marked epochs through ``pop_rejepoch``.

Component mode uses ICA activations and writes ICA-prefixed rejection fields
where applicable. Event overlays use EEGLAB one-based event latencies; Python
array indexing remains zero-based internally.

Accepting Browser Results
=========================

In ``eegprep-console``, browser accept callbacks keep GUI state, ``EEG``,
``ALLEEG``, and history synchronized. If a browser action creates a new dataset
instead of mutating the current one, the session stores that dataset through the
same dataset helpers used by menu actions.

In normal Python, call the lower-level rejection helpers explicitly after
reviewing the browser model or marks.

Performance
===========

The browser draws only the visible time range and applies min/max decimation to
large traces, preserving endpoints and extrema while keeping UI redraw work
bounded by the displayed pixel width.
