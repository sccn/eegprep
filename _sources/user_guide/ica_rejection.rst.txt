.. _ica_rejection:

=================================================
ICA, ICLabel, Rejection, and Visual Diagnostics
=================================================

EEGPrep follows the EEGLAB component-review workflow: compute ICA, label
components, inspect labels and diagnostic displays, flag likely artifacts, then
remove flagged components when review is complete.

Use the Sample ICA Dataset
==========================

The fastest way to learn the workflow is to load the ICA tutorial data:

.. code-block:: python

   from pathlib import Path
   from eegprep import pop_loadset

   EEG = pop_loadset(Path("sample_data") / "eeglab_data_epochs_ica.set")

The GUI equivalent is ``File > Load existing dataset``.

Run ICA When Needed
===================

For datasets without ICA fields, use ``Tools > Decompose data by ICA`` or call:

.. code-block:: python

   from eegprep import pop_runica

   EEG, com = pop_runica(EEG, icatype="picard", gui=False, return_com=True)

The default GUI offers runica, robust runica, AMICA, and Picard choices.
Standalone AMICA requires an AMICA executable configured outside the Python
package.

ICLabel Workflow
================

Run ICLabel after ICA:

.. code-block:: python

   from eegprep import eeg_icalabelstat, pop_icflag, pop_iclabel, pop_subcomp, pop_viewprops

   EEG, label_com = pop_iclabel(EEG, "default", return_com=True)
   stats = eeg_icalabelstat(EEG, threshold=0.9, verbose=False)
   figures = pop_viewprops(EEG, typecomp=0, chanorcomp=[1, 2, 3])
   EEG, flag_com = pop_icflag(EEG, return_com=True)
   EEG, remove_com = pop_subcomp(EEG, [], return_com=True)

``pop_iclabel`` stores class probabilities in
``EEG["etc"]["ic_classification"]["ICLabel"]["classifications"]``. The class
order is Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, and Other.

GUI Review
==========

.. raw:: html

   <div class="eegprep-path">
     <p>Choose <strong>Tools > Classify components using ICLabel</strong>.</p>
     <p>Choose <strong>Plot > Component properties</strong>.</p>
     <p>Review scalp maps, activity, spectra, ICLabel probabilities, and any
     stored DIPFIT projection.</p>
     <p>Toggle component rejection marks and press OK.</p>
     <p>Inspect <code>EEG["reject"]["gcompreject"]</code>,
     <code>LASTCOM</code>, and <code>ALLCOM[-1]</code> from
     <code>eegprep-console</code>.</p>
     <p>Choose <strong>Tools > Remove components from data</strong> only after
     review.</p>
   </div>

Dashboard reject toggles write pending marks to ``EEG.reject.gcompreject`` only
when the user presses OK, so GUI review and ``eegprep-console`` history remain
synchronized.

Label Statistics
================

``eeg_icalabelstat`` mirrors EEGLAB's threshold-count summary and returns the
same information in a Python dictionary:

.. code-block:: python

   stats = eeg_icalabelstat(EEG, threshold=0.9, verbose=False)
   print(stats["counts"])
   print(stats["component_indices"])

Returned values include per-class counts above threshold, one-based component
indices, mean probabilities, dominant-class counts, and rejected/kept tallies
from ``EEG.reject.gcompreject``.

Flagging Rules
==============

Use ``pop_icflag`` defaults for an EEGLAB-like first pass, or pass a 7-by-2
threshold matrix in ICLabel class order:

.. code-block:: python

   import numpy as np
   from eegprep import pop_icflag

   EEG, com = pop_icflag(
       EEG,
       threshold=np.array(
           [
               [0, 0],
               [0.8, 1.0],
               [0.8, 1.0],
               [0, 0],
               [0, 0],
               [0, 0],
               [0, 0],
           ]
       ),
       return_com=True,
   )

Review flags visually before removing components. Automatic labels are a guide,
not a replacement for expert inspection.

Visual Diagnostics
==================

``pop_viewprops(EEG, typecomp=0)`` opens EEGPrep's native component-property
browser. With ICLabel classifications present, component mode opens the
``pop_prop_extended`` dashboard with scalp map, activity browser, ERP/image
summary, spectrum, class probabilities, component accept/reject controls, and
DIPFIT projections when localized dipoles are already stored in ``EEG.dipfit``.

The activity browser uses EEGLAB-facing one-based component indices and
preserves event display state through the ``scroll_event`` option.

Network Availability
====================

The standalone Python engine ships the default ICLabel network
(``netICL.mat``). EEGLAB ``lite`` and ``beta`` network artifacts are not
bundled in the Python package; requesting them with ``engine=None`` raises a
clear limitation. They can still be requested through ``engine="matlab"`` or
``engine="octave"`` when that runtime has an EEGLAB ICLabel checkout with those
artifacts.
