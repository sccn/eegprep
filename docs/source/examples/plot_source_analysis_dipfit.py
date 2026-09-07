"""Equivalent-dipole source localization of ICA components with EEGPrep DIPFIT.

Mirrors EEGLAB's "Source analysis" tutorial workflow: choose a head model,
coarse-fit component dipoles on a grid, refine them, then read the resulting
``EEG.dipfit.model`` entries.

EEGPrep's bundled DIPFIT backend is a single-sphere analytic leadfield, so
positions are approximate and will not match FieldTrip BEM coordinates.
"""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np

from eegprep import (
    pop_dipfit_gridsearch,
    pop_dipfit_nonlinear,
    pop_dipfit_settings,
    pop_dipplot,
    pop_loadset,
    pop_multifit,
)

SAMPLE = Path(__file__).resolve().parents[3] / "sample_data" / "eeglab_data_epochs_ica.set"

# 1. Load a dataset that already carries channel locations and an ICA decomposition.
EEG = pop_loadset(str(SAMPLE))
print(f"channels={EEG['nbchan']} trials={EEG['trials']} components={EEG['icaweights'].shape[0]}")

# 2. Head model and settings (Tools > Source localization using DIPFIT > Head model and settings).
EEG, com = pop_dipfit_settings(EEG, model="standardBESA", return_com=True)
print(com)
print(f"coordformat={EEG['dipfit']['coordformat']} hdmfile={EEG['dipfit']['hdmfile']}")

# 3. Coarse fit on a grid, then fine (nonlinear) fit of component 1.
EEG, com = pop_dipfit_gridsearch(EEG, [1, 2, 3], gui=False, return_com=True)
print(com[:96])
EEG, com = pop_dipfit_nonlinear(EEG, 1, gui=False, return_com=True)
print(com)

# 4. Autofit components 1-3 and drop fits above 40% residual variance.
EEG, com = pop_multifit(EEG, [1, 2, 3], threshold=40, gui=False, return_com=True)
print(com)

for index, model in enumerate(EEG["dipfit"]["model"][:3], start=1):
    pos = np.atleast_2d(np.asarray(model["posxyz"], dtype=float))
    rv = float(np.asarray(model["rv"], dtype=float).ravel()[0])
    coords = "; ".join(f"({x:6.1f}, {y:6.1f}, {z:6.1f})" for x, y, z in pos)
    print(f"IC{index}: posxyz={coords} rv={100 * rv:.1f}%")

# 5. Dipole plot over the packaged MNI MRI slices (Component dipole plot).
_, com = pop_dipplot(EEG, [1, 2, 3], gui=False, cornermri="on", return_com=True)
print(com)
