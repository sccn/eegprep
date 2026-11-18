"""FieldTrip-named EEG forward helpers used by DIPFIT."""

from eegprep.plugins.dipfit.fieldtrip.bem import eeg_infinite_dipole, eeg_leadfieldb
from eegprep.plugins.dipfit.forward.ft_compute_leadfield import ft_compute_leadfield
from eegprep.plugins.dipfit.forward.ft_headmodeltype import ft_headmodeltype
from eegprep.plugins.dipfit.forward.ft_prepare_vol_sens import ft_prepare_vol_sens

__all__ = [
    "eeg_infinite_dipole",
    "eeg_leadfieldb",
    "ft_compute_leadfield",
    "ft_headmodeltype",
    "ft_prepare_vol_sens",
]
