import unittest
from pathlib import PureWindowsPath

import numpy as np

from eegprep.functions.popfunc._pop_utils import (
    format_history_value,
    is_empty_value,
    is_on,
    parse_key_value_args,
    parse_numeric_sequence,
    parse_text_tokens,
)


class PopUtilsTests(unittest.TestCase):
    def test_parse_key_value_args_decodes_bytes_and_lowercases_keys(self):
        options = parse_key_value_args((b"Channel", [1], "Force", "on"), {"Explicit": 2})

        self.assertEqual(options, {"Explicit": 2, "channel": [1], "force": "on"})

    def test_parse_key_value_args_can_lowercase_kwargs(self):
        options = parse_key_value_args((), {"Force": "on"}, lowercase_kwargs=True)

        self.assertEqual(options, {"force": "on"})

    def test_parse_key_value_args_rejects_invalid_pairs(self):
        with self.assertRaisesRegex(ValueError, "pairs"):
            parse_key_value_args(("channel",))
        with self.assertRaisesRegex(ValueError, "Keys"):
            parse_key_value_args((1, "value"))

    def test_parse_text_tokens_preserves_or_parses_ints(self):
        text = "{'Fz' \"Cz\" 3}"

        self.assertEqual(parse_text_tokens(text), ["Fz", "Cz", "3"])
        self.assertEqual(parse_text_tokens(text, parse_ints=True), ["Fz", "Cz", 3])

    def test_parse_numeric_sequence_handles_eeglab_colon_ranges(self):
        self.assertEqual(parse_numeric_sequence("1:3", dtype=int), [1, 2, 3])
        self.assertEqual(parse_numeric_sequence("5:-2:1", dtype=int), [5, 3, 1])
        self.assertEqual(parse_numeric_sequence("[1, 2.5 4]", dtype=float), [1.0, 2.5, 4.0])
        self.assertEqual(parse_numeric_sequence("[1 2; 3 4]", dtype=int), [1, 2, 3, 4])
        self.assertEqual(parse_numeric_sequence(["1:2", 4], dtype=int), [1, 2, 4])

        parsed = parse_numeric_sequence("nan Inf -Inf", dtype=float)
        self.assertTrue(np.isnan(parsed[0]))
        self.assertEqual(parsed[1:], [np.inf, -np.inf])

    def test_is_empty_value_matches_gui_dialog_empty_literals(self):
        self.assertTrue(is_empty_value(None))
        self.assertTrue(is_empty_value(""))
        self.assertTrue(is_empty_value("[]"))
        self.assertTrue(is_empty_value("{}"))
        self.assertTrue(is_empty_value(np.array([])))
        self.assertTrue(is_empty_value([]))
        self.assertFalse(is_empty_value("0"))
        self.assertFalse(is_empty_value([0]))

    def test_is_on_normalizes_eeglab_on_off_values(self):
        self.assertTrue(is_on("on"))
        self.assertTrue(is_on("1"))
        self.assertTrue(is_on(True))
        self.assertTrue(is_on([1, 0]))
        self.assertTrue(is_on(np.array([1, 0])))
        self.assertFalse(is_on("off"))
        self.assertFalse(is_on("0"))
        self.assertFalse(is_on(False))
        self.assertFalse(is_on([0, 1]))
        self.assertFalse(is_on(np.array([0, 1])))

    def test_format_history_value_defaults_to_eeglab_like_literals(self):
        self.assertEqual(format_history_value("F'z"), "'F''z'")
        self.assertEqual(format_history_value([1, 2.0, np.float64(3.0)]), "[1 2 3]")
        self.assertEqual(format_history_value(np.array([[1, 2], [3, 4]])), "[1 2; 3 4]")
        self.assertEqual(format_history_value(["Fz", "Cz"]), "{'Fz' 'Cz'}")
        self.assertEqual(format_history_value([-np.inf, np.inf]), "[-Inf Inf]")
        self.assertEqual(
            format_history_value(PureWindowsPath("sample_data/eeglab_data.set")), "'sample_data/eeglab_data.set'"
        )

    def test_format_history_value_supports_pop_specific_options(self):
        self.assertEqual(format_history_value(True, bool_style="onoff"), "'on'")
        self.assertEqual(format_history_value(["Fz", 1], cell_for_sequence="any_strings"), "{'Fz' 1}")
        self.assertEqual(format_history_value(["Fz", "Cz"], cell_for_sequence=None), "['Fz' 'Cz']")
        self.assertEqual(format_history_value(None, none_as_empty=True), "[]")


if __name__ == "__main__":
    unittest.main()
