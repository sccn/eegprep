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

The standalone Python engine ships the gate-selected weight-only int8 ICLabel
network as ``iclabel.onnx`` and classifies components through `onnxruntime`;
install the ``iclabel`` extra (``eegprep[iclabel]``) to run it. Feature
extraction, float32 input normalization, four-way augmentation, and output
softmax remain unchanged. On the frozen 217-component, subject-disjoint
evaluation set recorded in ``tools/iclabel/evaluation_manifest.json``, the
shipped artifact agreed with the preserved float32 teacher on all 217 top-1
labels and all 217 keep-or-reject decisions under the existing
``pop_icflag`` thresholds (100% and 100%). The calibrated int8 candidate also
passed the fixed gate (98.1567% top-1 and 100% keep-or-reject), but the smaller
weight-only candidate is shipped. These are measured parity results on the
frozen set, not a general accuracy claim. The reproducible float32 reference,
both int8 candidates, and the complete report are retained under
``tools/iclabel/``.

EEGLAB ``lite`` and ``beta`` network artifacts are not bundled in the Python
package; requesting them with ``engine=None`` raises a clear limitation. They
can still be requested through ``engine="matlab"`` or ``engine="octave"``
when that runtime has an EEGLAB ICLabel checkout with those artifacts.

Why Not 4 Bits
--------------

A 4-bit artifact was built and measured rather than assumed. It works, and it is
still not what ships. ``tools/iclabel/quantize_iclabel_int4.py`` regenerates
``iclabel_int4_block32.onnx``: asymmetric uint4 Conv weights with the scale
blocked in groups of 32 input channels, which is the finest-grained scheme a
portable ONNX graph can express for a network that is entirely ``Conv``. It
measures 1,931,262 bytes against 2,932,897 for the shipped int8 artifact, and it
clears the same frozen gate at 100% top-1 and 100% keep-or-reject agreement.

Three measurements argue against promoting it:

* **The size target is already met.** Every one of the network's 2,903,817
  parameters is a Conv weight, so int8 achieves a clean 3.96x reduction with
  nothing left over to dilute it. Dropping to 4 bits saves a further 1.0 MB of
  artifact and 0.74 MB of wheel (7.82 MB to 7.08 MB), on top of the 8.2 MB that
  int8 already removed.
* **It costs compatibility.** int4 tensors and blocked ``DequantizeLinear`` are
  opset 21 constructs. onnxruntime 1.18 refuses the graph with
  ``MLDataType for: tensor(uint4) is not currently registered``, so shipping it
  would raise the floor declared in ``pyproject.toml`` from 1.18 to 1.19.
* **It buys no speed.** Weight-only 4-bit dequantizes to float before the same
  float ``Conv``. Measured under ONNX Runtime Web, 868 augmented rows took
  3,214 ms at int4 against 3,043 ms at float32.

It also perturbs the probabilities more: a maximum absolute change of 4.5e-2
against the float32 teacher, versus 1.3e-2 for int8, on the same components.
That bought no measured label accuracy, and it consumes more of the margin that
separates the top two classes.

The finding worth keeping is that ONNX Runtime Web can execute this graph today.
``tools/iclabel/check_iclabel_ort_web.mjs`` demonstrates it against the pinned
``onnxruntime-web`` build. If ICLabel ever needs to be materially smaller in the
browser, the path is open and measured; it is simply not worth its compatibility
cost while int8 already sits under the target.
