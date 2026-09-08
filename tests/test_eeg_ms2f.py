"""Parity tests for eeg_ms2f - epoch latency (ms) to nearest epoch frame.

Ports EEGLAB eeg_ms2f.m: ``outf = 1 + round((pnts-1)*(ms/1000 - xmin)/(xmax - xmin))``
with an out-of-range error. Frame numbers are 1-based to match EEGLAB. Expected
values are closed-form and mirror tests/matlab/test_eeg_ms2f.m.
"""

from __future__ import annotations

import pytest

from eegprep.functions.miscfunc.eeg_ms2f import eeg_ms2f

pytestmark = pytest.mark.parity


def _eeg(xmin, xmax, pnts):
    return {"xmin": xmin, "xmax": xmax, "pnts": pnts}


def test_first_frame_at_xmin():
    # ms at xmin -> first frame (1-based).
    assert eeg_ms2f(_eeg(0, 1, 1001), 0) == 1


def test_last_frame_at_xmax():
    # ms at xmax -> last frame == pnts.
    assert eeg_ms2f(_eeg(0, 1, 1001), 1000) == 1001


def test_midpoint():
    # 500 ms -> 0.5 s -> frame 1 + 1000*0.5 = 501.
    assert eeg_ms2f(_eeg(0, 1, 1001), 500) == 501


def test_rounds_to_nearest():
    # 499.4 ms -> 1 + round(499.4) = 500.
    assert eeg_ms2f(_eeg(0, 1, 1001), 499.4) == 500


def test_epoch_center_negative_xmin():
    # Epoched data xmin<0: ms=0 -> centre frame.
    assert eeg_ms2f(_eeg(-1, 1, 2001), 0) == 1001


def test_below_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        eeg_ms2f(_eeg(0, 1, 1001), -1)


def test_above_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        eeg_ms2f(_eeg(0, 1, 1001), 2000)
