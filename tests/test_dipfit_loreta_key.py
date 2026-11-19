"""Distributed source leadfields are outside the supported dipole fitting workflow."""

import pytest

from eegprep.plugins.dipfit.pop_dipfit_settings import pop_dipfit_settings
from eegprep.plugins.dipfit.pop_leadfield import pop_leadfield
from tests.fixtures import create_test_eeg_with_ica


@pytest.mark.parametrize("model", ["standardBESA", "standardBEM"])
def test_loreta_key_is_explicitly_unsupported(model):
    eeg = pop_dipfit_settings(create_test_eeg_with_ica(n_channels=4, n_samples=32, n_components=1), model=model)
    with pytest.raises(NotImplementedError, match="pop_leadfield"):
        pop_leadfield(eeg, sourcemodel="loreta-key")
    assert "sourcemodel" not in eeg["dipfit"]
