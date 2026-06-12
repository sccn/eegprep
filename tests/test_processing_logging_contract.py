import logging

import numpy as np
import pytest

from eegprep.functions.popfunc.eeg_eegrej import _combine_regions
from eegprep.functions.popfunc.eeg_lat2point import eeg_lat2point
from eegprep.functions.popfunc.pop_select import pop_select


def _minimal_eeg():
    return {
        "data": np.zeros((2, 20), dtype=np.float32),
        "nbchan": 2,
        "pnts": 20,
        "trials": 1,
        "srate": 100,
        "xmin": 0,
        "xmax": 0.19,
        "times": np.arange(20),
        "chanlocs": [{"labels": "Cz"}, {"labels": "Pz"}],
        "event": [],
        "urevent": [],
        "epoch": [],
        "history": "",
        "icaact": np.array([]),
        "icaweights": np.array([]),
        "icasphere": np.array([]),
        "icawinv": np.array([]),
        "icachansind": np.array([], dtype=int),
        "chaninfo": {},
        "reject": {},
    }


def test_pop_select_warnings_use_logging_not_stdout(capsys, caplog):
    caplog.set_level(logging.WARNING)

    with pytest.raises(ValueError, match="Channels not found"):
        pop_select(_minimal_eeg(), channel=["Missing"])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert "channels not found" in caplog.text


def test_latency_range_warning_uses_logging_not_stdout(capsys, caplog):
    caplog.set_level(logging.WARNING)

    newlat, flag = eeg_lat2point([2], [1], 1, [0, 0], outrange=1)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert flag == 1
    np.testing.assert_array_equal(newlat, np.array([1.0]))
    assert "Points out of range detected" in caplog.text


def test_eegrej_overlap_warning_uses_logging_not_stdout(capsys, caplog):
    caplog.set_level(logging.WARNING)

    combined = _combine_regions(np.array([[1, 3], [3, 5], [10, 12]]))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    np.testing.assert_array_equal(combined, np.array([[1, 5], [10, 12]]))
    assert "Overlapping regions detected" in caplog.text
