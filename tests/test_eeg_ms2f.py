"""Parity tests for eeg_ms2f - epoch latency (ms) to nearest epoch frame.

Ports EEGLAB eeg_ms2f.m: ``outf = 1 + round((pnts-1)*(ms/1000 - xmin)/(xmax - xmin))``
with an out-of-range error. Frame numbers are 1-based to match EEGLAB. Expected
values are closed-form and mirror tests/matlab/test_eeg_ms2f.m.
"""

from __future__ import annotations

import numpy as np
import pytest

from eegprep.functions.miscfunc.eeg_ms2f import eeg_ms2f
from tests.eeglab_tests import assert_matlab_near, eeglab_test

pytestmark = pytest.mark.parity


def _eeg(xmin, xmax, pnts):
    return {"xmin": xmin, "xmax": xmax, "pnts": pnts}


def test_epoch_center_negative_xmin():
    # Epoched data xmin<0: ms=0 -> centre frame.
    assert eeg_ms2f(_eeg(-1, 1, 2001), 0) == 1001


def test_above_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        eeg_ms2f(_eeg(0, 1, 1001), 2000)


@eeglab_test("unittesting_miscfunc/eeg_ms2f/miscfunc_eeg_ms2f_wrapperTest.m", "test_pass_center")
@eeglab_test("unittesting_miscfunc/eeg_ms2f/miscfunc_eeg_ms2f_wrapperTest.m", "test_pass_exact")
@eeglab_test("unittesting_miscfunc/eeg_ms2f/miscfunc_eeg_ms2f_wrapperTest.m", "test_pass_rounded")
def test_reference_eeg_ms2f_original_dataset_and_latencies(eeglab_backend):
    for latency, expected in ((500.0, 2.0), (1000.0, 2.0), (1653.0, 3.0)):
        eeg = eeglab_backend("eeg_emptyset")
        eeg.update(nbchan=2.0, pnts=3.0, trials=3.0, srate=1.0, xmin=0.0, xmax=2.0)
        eeg["data"] = np.zeros((2, 3, 3))
        eeg["data"][0, :, :] = [[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [1.0, 1.0, 2.0]]
        eeg["data"][1, :, :] = [[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        assert_matlab_near(eeglab_backend("eeg_ms2f", eeg, latency), [[expected]])
