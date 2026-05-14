import base64
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

from tools.visual_parity.capture import CaptureResult, _main_window_menu_state, capture_case
from tools.visual_parity.compare import compare_images, write_report
from tools.visual_parity.config import load_manifest
from tools.visual_parity.export_eegprep_menu_inventory import export_inventory
from tools.visual_parity.menu_inventory import compare_menu_trees
from eegprep.functions.guifunc.visual_capture import _main_window_menu_state as _eegprep_main_window_menu_state


ONE_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class VisualParityConfigTests(unittest.TestCase):
    def test_load_manifest_parses_cases(self):
        cases = load_manifest()

        self.assertIn("main_window", cases)
        self.assertEqual(cases["main_window"].window_size, (520, 380))
        self.assertIn("eeglab", cases["main_window"].targets)
        self.assertIn("eegprep.functions.guifunc.visual_capture", cases["main_window"].targets["eegprep"].command)
        self.assertIn("adjust_events_dialog", cases)
        self.assertEqual(cases["adjust_events_dialog"].targets["eeglab"].type, "matlab_dialog")
        self.assertIn("eegprep.functions.guifunc.visual_capture", cases["adjust_events_dialog"].targets["eegprep"].command)
        self.assertIn("reref_dialog", cases)
        self.assertEqual(cases["reref_dialog"].targets["eeglab"].action, "pop_reref")
        self.assertEqual(cases["reref_dialog_channel_ref"].targets["eeglab"].action, "pop_reref:channels")
        self.assertEqual(cases["reref_dialog_huber_ref"].targets["eeglab"].action, "pop_reref:huber")
        self.assertIn("pop_interp_dialog", cases)
        self.assertEqual(cases["pop_interp_dialog"].targets["eeglab"].action, "pop_interp:continuous")
        self.assertEqual(cases["pop_interp_epoched_dialog"].targets["eeglab"].action, "pop_interp:epoched")
        self.assertEqual(cases["pop_select_dialog"].targets["eeglab"].action, "pop_select")
        self.assertEqual(cases["pop_resample_dialog"].targets["eeglab"].action, "pop_resample")
        self.assertEqual(cases["pop_runica_dialog"].targets["eeglab"].action, "pop_runica")
        self.assertEqual(cases["pop_iclabel_dialog"].targets["eeglab"].action, "pop_iclabel")
        self.assertEqual(cases["pop_clean_rawdata_dialog"].targets["eeglab"].action, "pop_clean_rawdata")
        self.assertIn("pop_chansel_dialog", cases)
        self.assertEqual(cases["pop_chansel_dialog"].targets["eeglab"].action, "pop_chansel")
        self.assertEqual(cases["pop_interp_dataset_index_dialog"].targets["eeglab"].action, "inputdlg2:dataset_index")
        self.assertEqual(cases["pop_reref_help_dialog"].targets["eeglab"].action, "pophelp:pop_reref")


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

    def test_matlab_figure_capture_generates_open_menu_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            case = load_manifest()["file_menu"]

            def fake_run_subprocess(target_name, output_path, command, env, timeout_seconds):
                output_path.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
                return CaptureResult(target_name, output_path, command, 0)

            with (
                mock.patch("tools.visual_parity.capture.shutil.which", return_value="/usr/common/bin/matlab"),
                mock.patch("tools.visual_parity.capture._run_subprocess", side_effect=fake_run_subprocess),
            ):
                results = capture_case(case, "eeglab", output_dir=tmp_path)

            self.assertTrue(results[0].ok)
            script_text = next((tmp_path / "file_menu").glob("*.m")).read_text()
            self.assertIn("menu_label = 'File';", script_text)
            self.assertIn("add_viewprops_menu_if_present(eeglab_root", script_text)
            self.assertIn("open_figure_menu(fig, menu_label);", script_text)
            self.assertIn("write_figure_screen_capture(fig, output_file);", script_text)

    def test_matlab_figure_capture_uses_study_state_for_study_menu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            case = load_manifest()["study_menu"]

            def fake_run_subprocess(target_name, output_path, command, env, timeout_seconds):
                output_path.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
                return CaptureResult(target_name, output_path, command, 0)

            with (
                mock.patch("tools.visual_parity.capture.shutil.which", return_value="/usr/common/bin/matlab"),
                mock.patch("tools.visual_parity.capture._run_subprocess", side_effect=fake_run_subprocess),
            ):
                results = capture_case(case, "eeglab", output_dir=tmp_path)

            self.assertTrue(results[0].ok)
            script_text = next((tmp_path / "study_menu").glob("*.m")).read_text()
            self.assertIn("menu_label = 'Study';", script_text)
            self.assertIn("main_window_state = 'study';", script_text)

    def test_open_menu_default_state_uses_study_only_for_study_menu(self):
        self.assertEqual(_main_window_menu_state("Study"), "study")
        self.assertEqual(_main_window_menu_state("File"), "continuous")
        self.assertEqual(_main_window_menu_state("", ""), "")
        self.assertEqual(_main_window_menu_state("Study", "multiple"), "multiple")
        self.assertEqual(_eegprep_main_window_menu_state("Study", "startup"), "study")
        self.assertEqual(_eegprep_main_window_menu_state("File", "startup"), "continuous")
        self.assertEqual(_eegprep_main_window_menu_state("Study", "multiple"), "multiple")

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

    def test_matlab_dialog_capture_generates_pop_chansel_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            case = load_manifest()["pop_chansel_dialog"]
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
            script_text = next((tmp_path / "pop_chansel_dialog").glob("*.m")).read_text()
            self.assertIn("pop_chansel({'Fp1', 'Fp2', 'Cz', 'Oz'}, 'withindex', 'on')", script_text)
            self.assertIn("capture_pop_chansel_dialog", script_text)

    def test_matlab_dialog_capture_generates_pop_reref_variant_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            case = load_manifest()["reref_dialog_channel_ref"]
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
            script_text = next((tmp_path / "reref_dialog_channel_ref").glob("*.m")).read_text()
            self.assertIn("capture_variant = 'channels';", script_text)
            self.assertIn("apply_pop_reref_variant", script_text)

    def test_matlab_dialog_capture_generates_pop_interp_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            case = load_manifest()["pop_interp_epoched_dialog"]
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
            script_text = next((tmp_path / "pop_interp_epoched_dialog").glob("*.m")).read_text()
            self.assertIn("capture_variant = 'epoched';", script_text)
            self.assertIn("[EEG, com] = pop_interp(EEG);", script_text)
            self.assertIn("EEG.epoch = struct", script_text)

    def test_matlab_dialog_capture_generates_simple_pop_function_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            case = load_manifest()["pop_resample_dialog"]

            def fake_run_subprocess(target_name, output_path, command, env, timeout_seconds):
                output_path.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
                return CaptureResult(target_name, output_path, command, 0)

            with (
                mock.patch("tools.visual_parity.capture.shutil.which", return_value="/usr/common/bin/matlab"),
                mock.patch("tools.visual_parity.capture._run_subprocess", side_effect=fake_run_subprocess),
            ):
                results = capture_case(case, "eeglab", output_dir=tmp_path)

            self.assertTrue(results[0].ok)
            script_text = next((tmp_path / "pop_resample_dialog").glob("*.m")).read_text()
            self.assertIn("action = 'pop_resample';", script_text)
            self.assertIn("[EEG, com] = pop_resample(EEG);", script_text)
            self.assertIn("capture_simple_pop_dialog", script_text)
            self.assertIn("inputgui_override_dir =", script_text)
            self.assertIn("addpath(inputgui_override_dir, '-begin');", script_text)
            self.assertIn("write_figure_capture(fig, output_file);", script_text)
            override_text = next((tmp_path / "pop_resample_dialog" / "inputgui_plot_override").glob("inputgui.m")).read_text()
            self.assertIn("args{6} = 'plot';", override_text)
            self.assertIn("args = [args {'mode' 'plot'}];", override_text)

    def test_matlab_dialog_capture_generates_pophelp_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            case = load_manifest()["pop_interp_help_dialog"]
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
            script_text = next((tmp_path / "pop_interp_help_dialog").glob("*.m")).read_text()
            self.assertIn("function_name = 'pop_interp';", script_text)
            self.assertIn("pophelp(function_name);", script_text)
            self.assertIn("write_pophelp_text_capture", script_text)


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
    def test_export_eegprep_menu_inventory_writes_main_window_tree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "menu.json"

            export_inventory(output, all_menus=True, state="continuous")

            payload = json.loads(output.read_text())
            self.assertEqual(
                [item["label"] for item in payload["menus"]],
                ["File", "Edit", "Tools", "Plot", "Study", "Datasets", "Help"],
            )

    def test_export_eegprep_menu_inventory_includes_demo_dataset_menu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = pathlib.Path(tmpdir) / "menu.json"

            export_inventory(output, state="multiple")

            payload = json.loads(output.read_text())
            datasets = next(item for item in payload["menus"] if item["label"] == "Datasets")
            self.assertEqual(
                [item["label"] for item in datasets["children"]],
                ["Dataset 1:menu one", "Dataset 2:menu two", "Select multiple datasets"],
            )
            self.assertEqual([item["checked"] for item in datasets["children"][:2]], [True, True])

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

    def test_compare_menu_trees_accepts_matlab_single_child_objects(self):
        reference = [{"label": "File", "children": {"label": "Import data", "enabled": "on"}}]
        candidate = [{"label": "File", "children": [{"label": "Import data", "enabled": True}]}]

        self.assertEqual(compare_menu_trees(reference, candidate), [])

    def test_compare_menu_trees_reports_checked_differences(self):
        reference = [{"label": "Datasets", "children": [{"label": "Dataset 1:demo", "checked": "on"}]}]
        candidate = [{"label": "Datasets", "children": [{"label": "Dataset 1:demo", "checked": False}]}]

        differences = compare_menu_trees(reference, candidate)

        self.assertEqual(len(differences), 1)
        self.assertIn("checked mismatch", differences[0])


if __name__ == "__main__":
    unittest.main()
