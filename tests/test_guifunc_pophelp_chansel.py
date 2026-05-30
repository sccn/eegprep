import unittest
from importlib import resources
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from eegprep.functions.guifunc.eeglab_menu import eeglab_menus, menu_actions
from eegprep.functions.guifunc.menu_actions import action_kind
from eegprep.functions.guifunc.pophelp import pophelp_text
from eegprep.functions.popfunc.pop_interp import pop_interp_dialog_spec
from eegprep.functions.popfunc.pop_chansel import (
    pop_chansel_display_values,
    pop_chansel_selected_string,
)
from eegprep.functions.popfunc.pop_reref import pop_reref_dialog_spec


REPO_ROOT = Path(__file__).resolve().parents[1]


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

    def test_help_resources_are_packaged_importlib_resources(self):
        help_files = resources.files("eegprep.resources.help")

        self.assertTrue(help_files.joinpath("eegprep.md").is_file())
        self.assertTrue(help_files.joinpath("eeg_helpadmin.md").is_file())
        self.assertIn("EEGPrep", help_files.joinpath("eegprep.md").read_text(encoding="utf-8"))

    def test_help_resources_are_declared_as_package_data(self):
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        package_data = pyproject["tool"]["setuptools"]["package-data"]["eegprep"]
        package_root = REPO_ROOT / "src/eegprep"
        packaged = {
            path.relative_to(package_root).as_posix()
            for pattern in package_data
            for path in package_root.glob(pattern)
            if path.is_file()
        }

        self.assertIn("resources/help/eegprep.md", packaged)
        self.assertIn("resources/help/eeg_helpadmin.md", packaged)
        self.assertIn("resources/help/pop_reref.md", packaged)

    def test_implemented_menu_help_actions_have_packaged_resources(self):
        full_menu_actions = menu_actions(eeglab_menus(all_menus=True, include_plugins=True))

        help_targets = set()
        for action in full_menu_actions:
            base = action.partition(":")[0]
            if action.startswith("help:"):
                help_targets.add(action.partition(":")[2])
            elif base.startswith(("pop_", "eeg_")) and action_kind(action) == "implemented":
                help_targets.add(base)

        self.assertIn("eeg_helpstudy", help_targets)
        self.assertIn("pop_study", help_targets)
        self.assertIn("pop_adjustevents", help_targets)
        for target in sorted(help_targets):
            with self.subTest(target=target):
                text, source_path = pophelp_text(target)
                self.assertIn(target.upper(), text.upper())
                self.assertTrue(source_path.endswith(f"{target}.md"))

    def test_pophelp_requires_packaged_resource(self):
        with self.assertRaisesRegex(FileNotFoundError, "Missing packaged EEGPrep help resource"):
            pophelp_text("missing_resource")

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
