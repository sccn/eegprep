import unittest

import numpy as np

from eegprep.functions.popfunc._pop_utils import (
    format_history_value,
    parse_key_value_args,
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

    def test_format_history_value_defaults_to_eeglab_like_literals(self):
        self.assertEqual(format_history_value("F'z"), "'F''z'")
        self.assertEqual(format_history_value([1, 2.0, np.float64(3.0)]), "[1 2 3]")
        self.assertEqual(format_history_value(np.array([[1, 2], [3, 4]])), "[1 2; 3 4]")
        self.assertEqual(format_history_value(["Fz", "Cz"]), "{'Fz' 'Cz'}")
        self.assertEqual(format_history_value([-np.inf, np.inf]), "[-Inf Inf]")

    def test_format_history_value_supports_pop_specific_options(self):
        self.assertEqual(format_history_value(True, bool_style="onoff"), "'on'")
        self.assertEqual(format_history_value(["Fz", 1], cell_for_sequence="any_strings"), "{'Fz' 1}")
        self.assertEqual(format_history_value(["Fz", "Cz"], cell_for_sequence=None), "['Fz' 'Cz']")
        self.assertEqual(format_history_value(None, none_as_empty=True), "[]")


if __name__ == "__main__":
    unittest.main()
