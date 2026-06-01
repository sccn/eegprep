.. _eegbrowser:

====================
EEGBrowser Workflows
====================

EEGPrep includes an EEGLAB-style scrolling browser for inspecting continuous,
epoched, component, spectral, and overlay data. The browser is available from
the Plot menu, rejection workflows, Python APIs, and ``eegprep-console``.

Opening The Browser
===================

From the GUI, load or select a dataset and use the Plot menu entries for
channel data, component activity, or visual rejection. The rejection dialogs can
also open browser-backed review windows before updating marks or removing data.

From Python or ``eegprep-console``:

.. code-block:: python

   from eegprep import eegplot, pop_eegplot

   window = eegplot(EEG)
   EEG, com = pop_eegplot(EEG, return_com=True)

Use ``show=False`` when testing or scripting normalization without opening Qt:

.. code-block:: python

   model = eegplot(EEG, winlength=5, dispchans=16, show=False)

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

Performance
===========

The browser draws only the visible time range and applies min/max decimation to
large traces, preserving endpoints and extrema while keeping UI redraw work
bounded by the displayed pixel width.
