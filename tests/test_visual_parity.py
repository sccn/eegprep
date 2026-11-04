import base64
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

from tools.visual_parity.capture import CaptureResult, capture_case
from tools.visual_parity.config import load_manifest
from tools.visual_parity.menu_inventory import compare_menu_trees
from tools.visual_parity.visual_capture import (
    _capture_case_handlers,
)


ONE_PIXEL_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="


class VisualParityConfigTests(unittest.TestCase):
    def test_eegprep_visual_capture_registry_covers_manifest_cases(self):
        handlers = _capture_case_handlers()
        cases = load_manifest()

        for case_id, case in cases.items():
            eegprep_target = case.targets.get("eegprep")
            if eegprep_target is None:
                continue
            if "tools.visual_parity.visual_capture" in eegprep_target.command:
                with self.subTest(case_id=case_id):
                    self.assertIn(case_id, handlers)


class VisualParityCaptureTests(unittest.TestCase):
    def test_capture_command_receives_output_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            manifest_path = tmp_path / "cases.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "id": "demo",
                                "targets": {
                                    "eegprep": {
                                        "type": "command",
                                        "command": [
                                            sys.executable,
                                            "-c",
                                            (
                                                "import base64, os; "
                                                "open(os.environ['EEGPREP_VISUAL_OUTPUT'], 'wb').write("
                                                f"base64.b64decode('{ONE_PIXEL_PNG}'))"
                                            ),
                                        ],
                                    }
                                },
                            }
                        ]
                    }
                )
            )

            case = load_manifest(manifest_path)["demo"]
            results = capture_case(case, "eegprep", output_dir=tmp_path)

            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].ok)
            self.assertTrue((tmp_path / "demo" / "eegprep.png").exists())

    def test_matlab_figure_capture_uses_interactive_desktop_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            case = load_manifest()["main_window"]
            captured_command = []

            def fake_run_subprocess(target_name, output_path, command, env, timeout_seconds):
                captured_command.extend(command)
                output_path.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
                return CaptureResult(target_name, output_path, command, 0)

            with (
                mock.patch("tools.visual_parity.capture.shutil.which", return_value="/usr/common/bin/matlab"),
                mock.patch("tools.visual_parity.capture._run_subprocess", side_effect=fake_run_subprocess),
            ):
                results = capture_case(case, "eeglab", output_dir=tmp_path)

            self.assertTrue(results[0].ok)
            self.assertIn("-nosplash", captured_command)
            self.assertIn("-nodesktop", captured_command)
            self.assertIn("-r", captured_command)
            self.assertNotIn("-batch", captured_command)
            script_text = next((tmp_path / "main_window").glob("*.m")).read_text()
            self.assertIn("'Units', 'pixels'", script_text)

    def test_matlab_capture_honors_eeglab_root_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            eeglab_root = tmp_path / "reference-eeglab"
            case = load_manifest()["main_window"]

            def fake_run_subprocess(target_name, output_path, command, env, timeout_seconds):
                output_path.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
                return CaptureResult(target_name, output_path, command, 0)

            with (
                mock.patch.dict("tools.visual_parity.capture.os.environ", {"EEGPREP_EEGLAB_ROOT": str(eeglab_root)}),
                mock.patch("tools.visual_parity.capture.shutil.which", return_value="/usr/common/bin/matlab"),
                mock.patch("tools.visual_parity.capture._run_subprocess", side_effect=fake_run_subprocess),
            ):
                results = capture_case(case, "eeglab", output_dir=tmp_path)

            self.assertTrue(results[0].ok)
            script_text = next((tmp_path / "main_window").glob("*.m")).read_text()
            self.assertIn(f"eeglab_root = '{eeglab_root.resolve().as_posix()}';", script_text)


class MenuInventoryTests(unittest.TestCase):
    def test_compare_menu_trees_reports_label_and_enabled_differences(self):
        reference = [
            {
                "label": "File",
                "enabled": "on",
                "children": [{"label": "Load existing dataset", "enabled": "on"}],
            }
        ]
        candidate = [
            {
                "label": "File",
                "enabled": True,
                "children": [{"label": "Load dataset", "enabled": False}],
            }
        ]

        differences = compare_menu_trees(reference, candidate)

        self.assertEqual(len(differences), 2)
        self.assertIn("label mismatch", differences[0])
        self.assertIn("enabled mismatch", differences[1])

    def test_compare_menu_trees_reports_checked_differences(self):
        reference = [{"label": "Datasets", "children": [{"label": "Dataset 1:demo", "checked": "on"}]}]
        candidate = [{"label": "Datasets", "children": [{"label": "Dataset 1:demo", "checked": False}]}]

        differences = compare_menu_trees(reference, candidate)

        self.assertEqual(len(differences), 1)
        self.assertIn("checked mismatch", differences[0])


if __name__ == "__main__":
    unittest.main()
