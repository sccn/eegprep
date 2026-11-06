"""MATLAB linspace spacing."""

from __future__ import annotations

import numpy as np

from eegprep.functions.miscfunc.linspace import linspace


def test_linspace_matches_matlab_values():
    got = linspace(0.0, 1.0, 7)
    expected = np.array(
        [
            float.fromhex("0x0.0p+0"),
            float.fromhex("0x1.5555555555555p-3"),
            float.fromhex("0x1.5555555555555p-2"),
            float.fromhex("0x1.0000000000000p-1"),
            float.fromhex("0x1.5555555555555p-1"),
            float.fromhex("0x1.aaaaaaaaaaaabp-1"),
            float.fromhex("0x1.0000000000000p+0"),
        ]
    )
    np.testing.assert_array_equal(got, expected)
    assert got[5] != np.linspace(0.0, 1.0, 7)[5]
    np.testing.assert_array_equal(linspace(1.0, 2.0, 1), np.array([2.0]))


def test_linspace_keeps_matlab_half_sample_ties():
    start = float.fromhex("-0x1.b19b366cd9b36p+8")
    stop = float.fromhex("0x1.a99b366cd9b36p+8")
    got = linspace(start, stop, 111)
    numpy_grid = np.linspace(start, stop, 111)
    ties = {
        20: "-0x1.1562c58b162c5p+8",
        72: "0x1.0193264c99328p+7",
        77: "0x1.4faf5ebd7af5cp+7",
        79: "0x1.6eeddbb76edd8p+7",
        106: "0x1.8a5cb972e5cbap+8",
    }
    for index, hex_value in ties.items():
        assert got[index].hex() == hex_value
        assert got[index] != numpy_grid[index]
    assert got[0] == start
    assert got[-1] == stop
