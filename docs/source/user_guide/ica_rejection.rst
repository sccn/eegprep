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

AMICA input uses a channels-by-frames matrix. EEGPrep writes its binary input
as little-endian float32 values, with all channels of each frame together,
matching MATLAB's ``runamica15``. Versions affected by `issue #435
<https://github.com/sccn/eegprep/issues/435>`_ wrote entire channels consecutively,
scrambling the samples read by AMICA. Recompute affected AMICA decompositions
and downstream component labels and rejection decisions from the original data.

pAMICA runs AMICA in Python with PyTorch, so it needs no executable.
Install it with ``pip install "eegprep[pamica]"``, then call:

.. code-block:: python

   EEG = pop_runica(EEG, icatype="pamica", gui=False, max_iter=2000)

The decomposition is computed by the ``pamica`` package, through its ``pamica.AMICA(backend='torch').fit`` pipeline, and not by EEGLAB's AMICA binary.
It runs in float64 on the CPU with one thread and ``seed=5489``.
Reproducibility is pamica's promise: the same seed, data, settings, and pamica and torch versions give bit-identical weights in float64 on the CPU with one thread, on one machine.
EEGPrep's tests check this on one host, by repeating a run and by replaying a run from its record.
Another platform, another torch version, a GPU, or more threads is not covered by that check.
The settings that reproduce a run are the ones recorded in ``EEG["etc"]["pamica"]``.
A setting you do not pass takes eegprep's ``runamica15`` default, such as ``lrate=0.05`` and ``do_newton=True``.
These defaults follow eegprep's own table (``_AMICA_DEFAULTS``), which differs from EEGLAB's ``runamica15.m`` in places.
The exception is ``block_size``, which is pAMICA's own 131072, because the 128 of ``runamica15`` is the block each thread of the Fortran program processes.
Pass ``pcakeep`` to keep fewer components than channels.
``icaweights @ icasphere`` is the unmixing pAMICA applies.
``pop_runica`` sorts components by variance by default, as it does for the other engines; ``reorder="off"`` keeps pAMICA's order, which ``eeg_pamica`` called directly keeps by default.
``EEG["etc"]["pamica"]`` records the pamica and torch versions, the pamica ``backend`` class and ``pipeline`` that produced the decomposition, the device, the seed, the settings used, the rank that ran, the iteration count, the stop reason, and ``did_converge``.
``did_converge`` is false when pAMICA used all ``max_iter`` iterations without meeting a convergence check.
If pAMICA ends on a non-finite value, ``pop_runica`` raises ``PamicaFitError`` (from ``eegprep.functions.popfunc.eeg_pamica``) and leaves the ICA fields unset.
The ICA dialog does not list pAMICA yet, so choose it from the command line or the console.

EEGPrep's test suite also compares a pAMICA decomposition of ``sample_data/eeglab_data.set`` with one from EEGLAB's ``runamica15`` binary, and prints the Hungarian-matched correlations of the unmixing rows and their Amari distance.
That comparison is informational.
The binary seeds its random start from the clock, so the numbers depend on that start: two runs of the same code gave a minimum matched correlation of 0.9966 and of 0.5339.

Picard accepts EEGLAB's PCA option through ``options``. A positive value is the
requested component count; a negative value subtracts dimensions from the
selected channel count. For example, ``options={"pca": -1}`` computes one
fewer component than channels, which is useful after average referencing. The
initial unmixing matrix is sized to the reduced decomposition while remaining
deterministic.

Picard starts from an identity matrix by default. To choose a reproducible
random starting point, call ``eeg_picard(EEG, random_state=42)`` or pass
``options={"seed": 42}`` to ``pop_runica`` with ``icatype="picard"``.
An explicit ``w_init`` matrix takes precedence over the seed. Passing
``random_state=None`` explicitly uses NumPy's global random state, as in
Python Picard. Different starts can converge to equivalent ICA solutions.

``eeg_picard`` and ``eeg_runica`` then scale each component so the RMS of its
mixing column is 1, matching EEGLAB ``eeg_checkset`` with
``option_scaleicarms`` on. The unscaled weights and sphere are stored in
``EEG["etc"]["icaweights_beforerms"]`` and
``EEG["etc"]["icasphere_beforerms"]``. ICLabel uses those scaled activations
and topographies, so label probabilities change if the scale is skipped.

ICLabel Workflow
================

Run ICLabel after ICA. ICLabel runs on Open Neural Network Exchange (ONNX) Runtime, which the ``iclabel`` extra
installs (``uv add "eegprep[iclabel]"``):

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

The standalone Python engine ships one ICLabel network, the EEGLAB ``default``
network, in two files, and classifies components with ``onnxruntime``. Install
the ``iclabel`` extra (``eegprep[iclabel]``) to run it. Feature extraction, the
cast to float32, four-way augmentation, and the output softmax are the same for
both files.

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - File
     - Size
     - What loads it
   * - ``iclabel.onnx``
     - 11.6 MB
     - ``iclabel``, ``pop_iclabel``, and ``pop_iclabel_async`` on every native
       platform. The weights are full-precision float32, so probabilities match
       the PyTorch network of earlier releases to about 1e-6.
   * - ``iclabel_int8.onnx``
     - 2.9 MB
     - The browser. The host page gives these bytes to ONNX Runtime Web, which
       ``iclabel_async`` awaits under Pyodide. Only the convolution weights are
       stored as 8-bit integers (weight-only quantization) to keep the download
       small.

EEGLAB ``lite`` and ``beta`` network artifacts are not bundled in the Python
package; requesting them with ``engine=None`` raises a clear limitation. They
can still be requested through ``engine="matlab"`` or ``engine="octave"``
when that runtime has an EEGLAB ICLabel checkout with those artifacts.

.. _iclabel_measured_agreement:

Measured Agreement of the Two Files
-----------------------------------

The full-precision file matches the PyTorch network built from the same EEGLAB
weights to float32 kernel noise. On the 217 evaluation components described
below, the raw network outputs differ from PyTorch by at most 2.1e-06 and the
averaged class probabilities by at most 1.2e-06, with the same top-1 class and
keep-or-reject decision for every component
(``tests/test_iclabel_quantization.py`` checks this against PyTorch whenever
PyTorch is installed).

Quantization changes the class probabilities of the 8-bit file slightly. To
measure how much, the full-precision (float32) network classified a frozen set
of 217 independent components, one recording from each of seven subjects of the
public OpenNeuro dataset ds002680, and its probabilities were stored once as
reference values (``tools/iclabel/evaluation_teacher_probabilities.npy``). The
committed manifest ``tools/iclabel/evaluation_manifest.json`` records the
recordings and the ICA settings. The 8-bit file passed all three acceptance
criteria:

.. list-table::
   :header-rows: 1
   :widths: 46 18 18 18

   * - Criterion
     - Required
     - Measured
     - Passed
   * - Components whose top-1 class matches
     - at least 0.95
     - 1.0 (217 of 217)
     - yes
   * - Components with the same keep-or-reject decision under the default
       ``pop_icflag`` thresholds
     - at least 0.99
     - 1.0 (217 of 217)
     - yes
   * - Largest absolute difference in any class probability
     - at most 0.015
     - 0.01346
     - yes

The mean absolute probability difference was 0.00080. "Keep-or-reject" asks
whether ``pop_icflag`` would flag the component for removal. A variant that
also quantized the activations (calibrated int8) was evaluated during development
and exceeded the 0.015 probability criterion, so it was not shipped.

The complete report is ``tools/iclabel/parity_report.json``. It records the
SHA-256 of ``iclabel_int8.onnx``, which the release workflow checks against the
packaged file. Continuous integration also rebuilds both networks from
``netICL.mat``, requires the full-precision export to match the shipped
``iclabel.onnx`` to 1e-5 and the 8-bit file to produce network outputs
identical to the shipped ``iclabel_int8.onnx`` on the evaluation components.

These are measured agreement results on one frozen set, not an accuracy claim.
The set is dominated by one class: the full-precision network labels 193 of the
217 components Other (and 18 Brain, 5 Eye, 1 Muscle), and Heart, Line Noise, and
Channel Noise do not occur, so the agreement figures say nothing about those
classes. Native results follow releases up to 0.4.0, which classified with
PyTorch, to about 1e-6; results from the browser's 8-bit file can differ from
them by up to about 0.0135 (the largest difference measured on this set).

Developers who change ``netICL.mat`` or the quantization rebuild the packaged
networks with the ``torch`` extra; the procedure is in :ref:`releasing`.
