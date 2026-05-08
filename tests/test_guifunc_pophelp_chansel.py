import unittest
import importlib
from unittest.mock import patch

from eegprep.functions.guifunc.pophelp import pophelp_text
pophelp_module = importlib.import_module("eegprep.functions.guifunc.pophelp")
from eegprep.functions.popfunc.pop_chansel import (
    pop_chansel_display_values,
    pop_chansel_selected_string,
)


class PopHelpAndChanSelTests(unittest.TestCase):
    def test_pophelp_reads_eeglab_source_and_appends_called_function(self):
        text, source_path = pophelp_text("pop_reref")

        self.assertIn("POP_REREF - Convert an EEG dataset", text)
        self.assertIn("The 'pop' function above calls the eponymous Matlab function below", text)
        self.assertIn("REREF - convert common reference EEG data", text)
        self.assertTrue(source_path.endswith("pop_reref.m"))

    def test_pophelp_accepts_pophelp_expression(self):
        text, source_path = pophelp_text("pophelp('pop_reref')")

        self.assertIn("POP_REREF", text)
        self.assertTrue(source_path.endswith("pop_reref.m"))

    def test_pophelp_falls_back_to_python_sources_without_eeglab_tree(self):
        with patch.object(pophelp_module, "EEGLAB_ROOTS", ()):
            text, source_path = pophelp_text("pop_reref")

        self.assertIn("average or common-reference data", text)
        self.assertIn("Re-reference channel-major EEG data", text)
        self.assertTrue(source_path.endswith("pop_reref.py"))

    def test_pop_chansel_display_values_match_withindex_format(self):
        values = pop_chansel_display_values(["Fp1", "Cz", "Pz"], withindex="on")

        self.assertEqual(values, ["1  -  Fp1", "2  -  Cz", "3  -  Pz"])

    def test_pop_chansel_selected_string_matches_eeglab_output(self):
        selected = pop_chansel_selected_string(["Fp1", "Cz", "Pz"], ["Fp1", "Pz"])

        self.assertEqual(selected, "Fp1 Pz")

    def test_pop_chansel_selected_string_matches_default_withindex_off_output(self):
        selected = pop_chansel_selected_string(["Fp1", "Cz", "Pz"], ["Cz"])

        self.assertEqual(selected, "Cz")

    def test_pop_chansel_quotes_labels_with_spaces(self):
        selected = pop_chansel_selected_string(["Left mastoid", "Cz"], ["Left mastoid"])

        self.assertEqual(selected, "'Left mastoid'")

    def test_pop_chansel_selects_1_based_numeric_indices(self):
        selected = pop_chansel_selected_string(["Fp1", "Cz", "Pz"], [1, 3])

        self.assertEqual(selected, "Fp1 Pz")

    def test_pop_chansel_raises_for_missing_selected_label(self):
        with self.assertRaisesRegex(ValueError, "Cannot find 'Pz'"):
            pop_chansel_selected_string(["Fp1", "Cz"], ["Pz"])


if __name__ == "__main__":
    unittest.main()
