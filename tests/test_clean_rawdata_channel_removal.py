from __future__ import annotations

import numpy as np

from eegprep.plugins.clean_rawdata.private.channel_removal import update_clean_channel_mask


def test_update_clean_channel_mask_composites_original_channel_mask():
    eeg = {"etc": {"clean_channel_mask": np.array([True, False, True, True])}}
    removed_channels = np.array([False, True, False])

    update_clean_channel_mask(eeg, removed_channels)

    np.testing.assert_array_equal(eeg["etc"]["clean_channel_mask"], [True, False, False, True])


def test_update_clean_channel_mask_resets_mismatched_existing_mask():
    eeg = {"etc": {"clean_channel_mask": np.array([True, False, True, True])}}
    removed_channels = np.array([False, True])

    update_clean_channel_mask(eeg, removed_channels)

    np.testing.assert_array_equal(eeg["etc"]["clean_channel_mask"], [True, False])


def test_update_clean_channel_mask_creates_zero_and_all_removal_masks():
    no_removal = {"etc": {}}
    all_removed = {}

    update_clean_channel_mask(no_removal, np.zeros(3, dtype=bool))
    update_clean_channel_mask(all_removed, np.ones(3, dtype=bool))

    np.testing.assert_array_equal(no_removal["etc"]["clean_channel_mask"], [True, True, True])
    np.testing.assert_array_equal(all_removed["etc"]["clean_channel_mask"], [False, False, False])
