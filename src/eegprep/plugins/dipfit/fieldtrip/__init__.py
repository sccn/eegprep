"""FieldTrip-named EEG forward helpers used by DIPFIT."""

from eegprep.plugins.dipfit.fieldtrip.bem import eeg_infinite_dipole, eeg_leadfieldb, ft_prepare_vol_sens
from eegprep.plugins.dipfit.fieldtrip.ft_compute_leadfield import ft_compute_leadfield, ft_headmodeltype

__all__ = [
    "eeg_infinite_dipole",
    "eeg_leadfieldb",
    "ft_compute_leadfield",
    "ft_headmodeltype",
    "ft_prepare_vol_sens",
]
