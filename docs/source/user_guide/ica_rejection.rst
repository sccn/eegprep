.. _ica_rejection:

=============================================
ICA, ICLabel, Rejection, and Visual Diagnostics
=============================================

EEGPrep follows the EEGLAB component-review workflow: compute ICA, label
components, inspect the labels and diagnostic displays, flag likely artifacts,
then remove flagged components when the review is complete.

ICLabel Workflow
================

Run ICA before calling ICLabel:

.. code-block:: python

    from eegprep import eeg_icalabelstat, pop_icflag, pop_iclabel, pop_subcomp, pop_viewprops

    EEG = pop_iclabel(EEG, "default")
    stats = eeg_icalabelstat(EEG, threshold=0.9)
    figures = pop_viewprops(EEG, typecomp=0, chanorcomp=[1, 2, 3])
    EEG = pop_icflag(EEG)
    EEG = pop_subcomp(EEG, [])

`pop_iclabel` stores class probabilities in
`EEG.etc.ic_classification.ICLabel.classifications`. The class order is Brain,
Muscle, Eye, Heart, Line Noise, Channel Noise, and Other.

The standalone Python engine ships the default ICLabel network (`netICL.mat`).
The EEGLAB `lite` and `beta` network artifacts are not bundled in the Python
package; requesting them with `engine=None` raises a clear limitation. They can
still be requested through `engine="matlab"` or `engine="octave"` when that
runtime has an EEGLAB ICLabel checkout with those artifacts.

Label Statistics
================

`eeg_icalabelstat` mirrors EEGLAB's threshold-count summary and returns the same
information in a Python dictionary:

.. code-block:: python

    stats = eeg_icalabelstat(EEG, threshold=0.9, verbose=False)
    print(stats["counts"])
    print(stats["component_indices"])

The returned values include per-class counts above threshold, 1-based component
indices, mean probabilities, dominant-class counts, and rejected/kept tallies
from `EEG.reject.gcompreject`.

Visual Diagnostics
==================

`pop_viewprops(EEG, typecomp=0)` opens EEGPrep's native component-property
browser. With ICLabel classifications present, component mode opens the
`pop_prop_extended` dashboard with scalp map, activity browser, ERP/image
summary, spectrum, class probabilities, component accept/reject controls, and
DIPFIT projections when localized dipoles are already stored in `EEG.dipfit`.

The activity browser uses EEGLAB-facing 1-based component indices and preserves
event display state through the `scroll_event` option. Dashboard reject toggles
write pending marks to `EEG.reject.gcompreject` only when the user presses OK,
so GUI review and `eegprep-console` history remain synchronized.
