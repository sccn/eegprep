from __future__ import annotations

import numpy as np

from eegprep.plugins.clean_rawdata.private.channel_removal import update_clean_channel_mask


def test_update_clean_channel_mask_composites_original_channel_mask():
    eeg = {"etc": {"clean_channel_mask": np.array([True, False, True, True])}}
    removed_channels = np.array([False, True, False])

    update_clean_channel_mask(eeg, removed_channels)

    np.testing.assert_array_equal(eeg["etc"]["clean_channel_mask"], [True, False, False, True])
