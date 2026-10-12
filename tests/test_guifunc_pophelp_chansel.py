import unittest
from pathlib import Path

import tomllib

import numpy as np
import pytest

from eegprep.functions.guifunc.eeglab_menu import eeglab_menus, menu_actions
from eegprep.functions.guifunc.menu_actions import action_kind
from eegprep.functions.guifunc.pophelp import pophelp_text
from eegprep.functions.popfunc.pop_chansel import (
    pop_chansel_selected_string,
)
from tests.eeglab_tests import eeglab_test
from tests.eeglab_tests.gui import close_reference_gui


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.gui
@eeglab_test("unittesting_guifunc/pophelp/guifunc_pophelp_wrapperTest.m", "test_test_pophelp")
def test_reference_pophelp(eeglab_backend, request):
    matlab = request.config.getoption("--eeglab-backend") == "matlab"
    if not matlab:
        from PySide6.QtWidgets import QApplication

        # Keep the real Python application and returned dialogs alive until
        # the source closes them; MATLAB's pophelp itself has no output.
        _application = QApplication.instance() or QApplication([])
    for function in ("pop_editoptions", "pop_editoptions.m"):
        window = eeglab_backend("pophelp", function, nargout=0 if matlab else 1)
        close_reference_gui(eeglab_backend, request, window=window)


@eeglab_test("unittesting_adminfunc/eeg_helphelp/adminfunc_eeg_helphelp_wrapperTest.m", "test_pass_general")
def test_reference_eeg_helphelp(eeglab_backend, request):
    if request.config.getoption("--eeglab-backend") == "matlab":
        # This one help script executes `help eeg_helphelp`; the other five
        # category-help scripts are comment-only, not pophelp_text contracts.
        eeglab_backend("eegprep_test_run_script", "eeg_helphelp", np.empty((1, 0), dtype=object), nargout=0)
    else:
        eeglab_backend("eeg_helphelp", nargout=0)


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

    def test_pop_chansel_quotes_labels_with_spaces(self):
        selected = pop_chansel_selected_string(["Left mastoid", "Cz"], ["Left mastoid"])

        self.assertEqual(selected, "'Left mastoid'")

    def test_pop_chansel_selects_1_based_numeric_indices(self):
        selected = pop_chansel_selected_string(["Fp1", "Cz", "Pz"], [1, 3])

        self.assertEqual(selected, "Fp1 Pz")


if __name__ == "__main__":
    unittest.main()
