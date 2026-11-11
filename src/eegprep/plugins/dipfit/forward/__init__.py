"""FieldTrip forward-model entry points used by DIPFIT.

Concentric-sphere EEG leadfields live here. Boundary-element models plug into
``ft_compute_leadfield`` by head-model type.
"""

from eegprep.plugins.dipfit.forward.eeg_leadfield1 import eeg_leadfield1
from eegprep.plugins.dipfit.forward.eeg_leadfield4 import eeg_leadfield4, eeg_leadfield4_prepare
from eegprep.plugins.dipfit.forward.ft_compute_leadfield import ft_compute_leadfield
from eegprep.plugins.dipfit.forward.ft_headmodeltype import ft_headmodeltype, normalize_headmodel
from eegprep.plugins.dipfit.forward.ft_inside_headmodel import ft_inside_headmodel
from eegprep.plugins.dipfit.forward.ft_prepare_vol_sens import ft_prepare_vol_sens
from eegprep.plugins.dipfit.forward.standard_besa import standard_besa_headmodel


__all__ = [
    "eeg_leadfield1",
    "eeg_leadfield4",
    "eeg_leadfield4_prepare",
    "ft_compute_leadfield",
    "ft_headmodeltype",
    "ft_inside_headmodel",
    "ft_prepare_vol_sens",
    "normalize_headmodel",
    "standard_besa_headmodel",
]
