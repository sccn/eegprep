import unittest
import importlib
from pathlib import Path
from unittest.mock import patch

from eegprep.functions.guifunc.pophelp import pophelp_text
pophelp_module = importlib.import_module("eegprep.functions.guifunc.pophelp")
from eegprep.functions.popfunc.pop_interp import pop_interp_dialog_spec
from eegprep.functions.popfunc.pop_chansel import (
    pop_chansel_display_values,
    pop_chansel_selected_string,
)
from eegprep.functions.popfunc.pop_reref import pop_reref_dialog_spec


class PopHelpAndChanSelTests(unittest.TestCase):
    def test_pophelp_reads_packaged_markdown_and_appends_called_function(self):
        text, source_path = pophelp_text("pop_reref")

        self.assertIn("POP_REREF - Convert an EEG dataset", text)
        self.assertIn("The 'pop' function above calls the lower-level function below", text)
        self.assertIn("REREF - convert common reference EEG data", text)
        self.assertIn("resources/help", Path(source_path).as_posix())
        self.assertTrue(source_path.endswith("pop_reref.md"))

    def test_pophelp_accepts_pophelp_expression(self):
        text, source_path = pophelp_text("pophelp('pop_reref')")

        self.assertIn("POP_REREF", text)
        self.assertIn("resources/help", Path(source_path).as_posix())
        self.assertTrue(source_path.endswith("pop_reref.md"))

    def test_pophelp_reads_pop_interp_packaged_resource(self):
        text, source_path = pophelp_text("pop_interp")

        self.assertIn("POP_INTERP - interpolate data channels", text)
        self.assertIn("resources/help", Path(source_path).as_posix())
        self.assertTrue(source_path.endswith("pop_interp.md"))

    def test_pophelp_reads_reref_packaged_resource(self):
        text, source_path = pophelp_text("reref")

        self.assertIn("REREF - convert common reference EEG data", text)
        self.assertIn("resources/help", Path(source_path).as_posix())
        self.assertTrue(source_path.endswith("reref.md"))

    def test_dialog_help_targets_have_packaged_resources(self):
        interp_eeg = {"data": [], "trials": 1, "chanlocs": [], "chaninfo": {}, "epoch": []}
        specs = (pop_reref_dialog_spec(), pop_interp_dialog_spec(interp_eeg))

        for spec in specs:
            with self.subTest(spec=spec.function_name):
                text, source_path = pophelp_text(spec.help_text)
                self.assertIn(spec.function_name.upper(), text)
                self.assertIn("resources/help", Path(source_path).as_posix())

    def test_pophelp_requires_packaged_resource(self):
        with patch.object(pophelp_module, "HELP_ROOT", Path("/missing")):
            with self.assertRaisesRegex(FileNotFoundError, "Missing packaged EEGPrep help resource"):
                pophelp_text("pop_reref")

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
