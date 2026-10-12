import unittest
from pathlib import PureWindowsPath

import numpy as np

from eegprep.functions.popfunc._pop_utils import (
    format_history_value,
    parse_key_value_args,
    parse_numeric_sequence,
)


class PopUtilsTests(unittest.TestCase):
    def test_parse_key_value_args_decodes_bytes_and_lowercases_keys(self):
        options = parse_key_value_args((b"Channel", [1], "Force", "on"), {"Explicit": 2})

        self.assertEqual(options, {"Explicit": 2, "channel": [1], "force": "on"})

    def test_parse_numeric_sequence_handles_eeglab_colon_ranges(self):
        self.assertEqual(parse_numeric_sequence("1:3", dtype=int), [1, 2, 3])
        self.assertEqual(parse_numeric_sequence("5:-2:1", dtype=int), [5, 3, 1])
        self.assertEqual(parse_numeric_sequence("[1, 2.5 4]", dtype=float), [1.0, 2.5, 4.0])
        self.assertEqual(parse_numeric_sequence("[1 2; 3 4]", dtype=int), [1, 2, 3, 4])
        self.assertEqual(parse_numeric_sequence(["1:2", 4], dtype=int), [1, 2, 4])

        parsed = parse_numeric_sequence("nan Inf -Inf", dtype=float)
        self.assertTrue(np.isnan(parsed[0]))
        self.assertEqual(parsed[1:], [np.inf, -np.inf])

    def test_format_history_value_defaults_to_eeglab_like_literals(self):
        self.assertEqual(format_history_value("F'z"), "'F''z'")
        self.assertEqual(format_history_value([1, 2.0, np.float64(3.0)]), "[1 2 3]")
        self.assertEqual(format_history_value(np.array([[1, 2], [3, 4]])), "[1 2; 3 4]")
        self.assertEqual(format_history_value(["Fz", "Cz"]), "{'Fz' 'Cz'}")
        self.assertEqual(format_history_value([-np.inf, np.inf]), "[-Inf Inf]")
        self.assertEqual(
            format_history_value(PureWindowsPath("sample_data/eeglab_data.set")), "'sample_data/eeglab_data.set'"
        )


if __name__ == "__main__":
    unittest.main()
