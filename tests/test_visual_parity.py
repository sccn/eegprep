import base64
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

from tools.visual_parity.capture import CaptureResult, capture_case
from tools.visual_parity.compare import compare_images, write_report
from tools.visual_parity.config import load_manifest
from tools.visual_parity.menu_inventory import compare_menu_trees


ONE_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class VisualParityConfigTests(unittest.TestCase):
    def test_load_manifest_parses_cases(self):
        cases = load_manifest()

        self.assertIn("main_window", cases)
        self.assertEqual(cases["main_window"].window_size, (1100, 750))
        self.assertIn("eeglab", cases["main_window"].targets)
        self.assertIn("adjust_events_dialog", cases)
        self.assertEqual(cases["adjust_events_dialog"].targets["eeglab"].type, "matlab_dialog")
        self.assertIn("eegprep.functions.guifunc.visual_capture", cases["adjust_events_dialog"].targets["eegprep"].command)
        self.assertIn("reref_dialog", cases)
        self.assertEqual(cases["reref_dialog"].targets["eeglab"].action, "pop_reref")


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

    def test_matlab_dialog_capture_generates_pop_adjustevents_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            case = load_manifest()["adjust_events_dialog"]
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
            self.assertNotIn("-batch", captured_command)
            script_text = next((tmp_path / "adjust_events_dialog").glob("*.m")).read_text()
            self.assertIn("pop_adjustevents(EEG)", script_text)
            self.assertIn("capture_pop_adjustevents_dialog", script_text)


class VisualParityCompareTests(unittest.TestCase):
    def test_identical_images_have_zero_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            left = tmp_path / "left.png"
            right = tmp_path / "right.png"
            left.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
            right.write_bytes(base64.b64decode(ONE_PIXEL_PNG))

            result = compare_images(
                left,
                right,
                diff_path=tmp_path / "diff.png",
                side_by_side_path=tmp_path / "side_by_side.png",
            )
            write_report("demo", result, tmp_path / "report.md", tmp_path / "diff.png", tmp_path / "side_by_side.png")

            self.assertFalse(result.size_mismatch)
            self.assertEqual(result.mean_abs_delta, 0.0)
            self.assertTrue((tmp_path / "diff.png").exists())
            self.assertIn("VLM Review Prompt", (tmp_path / "report.md").read_text())


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


if __name__ == "__main__":
    unittest.main()
