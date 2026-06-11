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


ONE_PIXEL_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="


class VisualParityConfigTests(unittest.TestCase):
    def test_load_manifest_parses_cases(self):
        cases = load_manifest()

        self.assertIn("main_window", cases)
        self.assertEqual(cases["main_window"].window_size, (520, 380))
        self.assertIn("eeglab", cases["main_window"].targets)
        self.assertIn("eegprep.functions.guifunc.visual_capture", cases["main_window"].targets["eegprep"].command)
        self.assertIn("adjust_events_dialog", cases)
        self.assertEqual(cases["adjust_events_dialog"].targets["eeglab"].type, "matlab_dialog")
        self.assertIn(
            "eegprep.functions.guifunc.visual_capture", cases["adjust_events_dialog"].targets["eegprep"].command
        )
        self.assertIn("reref_dialog", cases)
        self.assertEqual(cases["reref_dialog"].targets["eeglab"].action, "pop_reref")
        self.assertEqual(cases["reref_dialog_channel_ref"].targets["eeglab"].action, "pop_reref:channels")
        self.assertEqual(cases["reref_dialog_huber_ref"].targets["eeglab"].action, "pop_reref:huber")
        for case_id in (
            "eegbrowser_continuous",
            "eegbrowser_continuous_marked",
            "eegbrowser_epoched",
            "eegbrowser_epoched_marked",
            "eegbrowser_events",
            "eegbrowser_grid_off",
            "eegbrowser_labels",
            "eegbrowser_component_activity",
            "eegbrowser_data2_overlay",
            "eegbrowser_pop_eegplot_reject_data",
            "eegbrowser_rejcont_continuous",
            "eegbrowser_rejection_epochs",
        ):
            with self.subTest(case_id=case_id):
                self.assertEqual(cases[case_id].targets["eeglab"].type, "matlab_figure")
                self.assertIn("eegprep.functions.guifunc.visual_capture", cases[case_id].targets["eegprep"].command)
        self.assertIn("pop_interp_dialog", cases)
        self.assertEqual(cases["pop_interp_dialog"].targets["eeglab"].action, "pop_interp:continuous")
        self.assertEqual(cases["pop_interp_epoched_dialog"].targets["eeglab"].action, "pop_interp:epoched")
        self.assertEqual(cases["pop_select_dialog"].targets["eeglab"].action, "pop_select")
        self.assertEqual(cases["pop_resample_dialog"].targets["eeglab"].action, "pop_resample")
        self.assertEqual(cases["pop_newset_dialog"].targets["eeglab"].action, "pop_newset")
        self.assertEqual(cases["pop_epoch_dialog"].targets["eeglab"].action, "pop_epoch")
        self.assertEqual(cases["pop_rmbase_dialog"].targets["eeglab"].action, "pop_rmbase")
        self.assertEqual(cases["pop_eegfilt_dialog"].targets["eeglab"].action, "pop_eegfilt")
        self.assertEqual(cases["pop_eegfiltnew_dialog"].targets["eeglab"].action, "pop_eegfiltnew")
        self.assertEqual(cases["pop_firws_dialog"].targets["eeglab"].action, "pop_firws")
        self.assertEqual(cases["pop_firpm_dialog"].targets["eeglab"].action, "pop_firpm")
        self.assertEqual(cases["pop_firma_dialog"].targets["eeglab"].action, "pop_firma")
        self.assertEqual(cases["pop_kaiserbeta_dialog"].targets["eeglab"].action, "pop_kaiserbeta")
        self.assertEqual(cases["pop_firwsord_dialog"].targets["eeglab"].action, "pop_firwsord")
        self.assertEqual(cases["pop_firpmord_dialog"].targets["eeglab"].action, "pop_firpmord")
        self.assertEqual(cases["pop_xfirws_dialog"].targets["eeglab"].action, "pop_xfirws")
        self.assertEqual(cases["pop_spectopo_channels_dialog"].targets["eeglab"].action, "pop_spectopo:channels")
        self.assertEqual(cases["pop_spectopo_components_dialog"].targets["eeglab"].action, "pop_spectopo:components")
        self.assertEqual(cases["pop_prop_channels_dialog"].targets["eeglab"].action, "pop_prop:channels")
        self.assertEqual(cases["pop_prop_components_dialog"].targets["eeglab"].action, "pop_prop:components")
        self.assertEqual(cases["pop_timtopo_dialog"].targets["eeglab"].action, "pop_timtopo")
        self.assertEqual(cases["pop_plottopo_dialog"].targets["eeglab"].action, "pop_plottopo")
        self.assertEqual(cases["pop_headplot_erp_dialog"].targets["eeglab"].action, "pop_headplot:erp")
        self.assertEqual(cases["pop_headplot_components_dialog"].targets["eeglab"].action, "pop_headplot:components")
        self.assertEqual(cases["coregister_dialog"].targets["eeglab"].action, "coregister")
        self.assertEqual(cases["pop_plotdata_dialog"].targets["eeglab"].action, "pop_plotdata")
        self.assertEqual(cases["pop_erpimage_channels_dialog"].targets["eeglab"].action, "pop_erpimage:channels")
        self.assertEqual(cases["pop_erpimage_components_dialog"].targets["eeglab"].action, "pop_erpimage:components")
        self.assertEqual(cases["pop_envtopo_dialog"].targets["eeglab"].action, "pop_envtopo")
        self.assertEqual(cases["pop_comperp_channels_dialog"].targets["eeglab"].action, "pop_comperp:channels")
        self.assertEqual(cases["pop_comperp_components_dialog"].targets["eeglab"].action, "pop_comperp:components")
        self.assertEqual(cases["pop_newtimef_channels_dialog"].targets["eeglab"].action, "pop_newtimef:channels")
        self.assertEqual(cases["pop_newtimef_components_dialog"].targets["eeglab"].action, "pop_newtimef:components")
        self.assertEqual(cases["pop_newcrossf_channels_dialog"].targets["eeglab"].action, "pop_newcrossf:channels")
        self.assertEqual(cases["pop_newcrossf_components_dialog"].targets["eeglab"].action, "pop_newcrossf:components")
        self.assertEqual(cases["pop_signalstat_channels_dialog"].targets["eeglab"].action, "pop_signalstat:channels")
        self.assertEqual(
            cases["pop_signalstat_components_dialog"].targets["eeglab"].action, "pop_signalstat:components"
        )
        self.assertEqual(cases["pop_eventstat_dialog"].targets["eeglab"].action, "pop_eventstat")
        self.assertEqual(cases["pop_runica_dialog"].targets["eeglab"].action, "pop_runica")
        self.assertEqual(cases["pop_iclabel_dialog"].targets["eeglab"].action, "pop_iclabel")
        self.assertEqual(cases["pop_icflag_dialog"].targets["eeglab"].action, "pop_icflag")
        self.assertEqual(cases["pop_subcomp_dialog"].targets["eeglab"].action, "pop_subcomp")
        self.assertEqual(cases["pop_eegthresh_dialog"].targets["eeglab"].action, "pop_eegthresh")
        self.assertEqual(cases["pop_jointprob_dialog"].targets["eeglab"].action, "pop_jointprob")
        self.assertEqual(cases["pop_rejchan_dialog"].targets["eeglab"].action, "pop_rejchan")
        self.assertEqual(cases["pop_rejcont_dialog"].targets["eeglab"].action, "pop_rejcont")
        self.assertEqual(cases["pop_rejkurt_dialog"].targets["eeglab"].action, "pop_rejkurt")
        self.assertEqual(cases["pop_rejspec_dialog"].targets["eeglab"].action, "pop_rejspec")
        self.assertEqual(cases["pop_rejtrend_dialog"].targets["eeglab"].action, "pop_rejtrend")
        self.assertEqual(cases["pop_selectcomps_dialog"].targets["eeglab"].action, "pop_selectcomps")
        self.assertEqual(cases["pop_viewprops_dialog"].targets["eeglab"].action, "pop_viewprops")
        self.assertEqual(
            cases["iclabel_pop_prop_extended_dashboard"].targets["eeglab"].action,
            "iclabel_pop_prop_extended",
        )
        self.assertEqual(cases["pop_dipfit_settings_dialog"].targets["eeglab"].action, "pop_dipfit_settings")
        self.assertEqual(cases["pop_dipfit_gridsearch_dialog"].targets["eeglab"].action, "pop_dipfit_gridsearch")
        self.assertEqual(cases["pop_dipfit_nonlinear_dialog"].targets["eeglab"].action, "pop_dipfit_nonlinear")
        self.assertEqual(cases["pop_dipplot_dialog"].targets["eeglab"].action, "pop_dipplot")
        self.assertEqual(cases["pop_multifit_dialog"].targets["eeglab"].action, "pop_multifit")
        self.assertEqual(cases["pop_leadfield_dialog"].targets["eeglab"].action, "pop_leadfield")
        self.assertEqual(cases["pop_dipfit_loreta_dialog"].targets["eeglab"].action, "pop_dipfit_loreta")
        self.assertIn("eegprep", cases["pop_dipfit_headmodel_dialog"].targets)
        self.assertEqual(cases["pop_clean_rawdata_dialog"].targets["eeglab"].action, "pop_clean_rawdata")
        self.assertEqual(cases["pop_editeventfield_dialog"].targets["eeglab"].action, "pop_editeventfield")
        self.assertEqual(cases["pop_editeventvals_dialog"].targets["eeglab"].action, "pop_editeventvals")
        self.assertEqual(cases["pop_selectevent_dialog"].targets["eeglab"].action, "pop_selectevent")
        self.assertEqual(cases["pop_rmdat_dialog"].targets["eeglab"].action, "pop_rmdat")
        self.assertEqual(cases["pop_chanedit_dialog"].targets["eeglab"].action, "pop_chanedit")
        self.assertEqual(cases["pop_copyset_dialog"].targets["eeglab"].action, "pop_copyset")
        self.assertEqual(cases["pop_mergeset_dialog"].targets["eeglab"].action, "pop_mergeset")
        self.assertIn("pop_chansel_dialog", cases)
        self.assertEqual(cases["pop_chansel_dialog"].targets["eeglab"].action, "pop_chansel")
        self.assertEqual(cases["select_multiple_datasets_dialog"].targets["eeglab"].action, "select_multiple_datasets")
        self.assertEqual(cases["pop_interp_dataset_index_dialog"].targets["eeglab"].action, "inputdlg2:dataset_index")
        self.assertEqual(cases["pop_reref_help_dialog"].targets["eeglab"].action, "pophelp:pop_reref")

    def test_eegbrowser_epoched_cases_compare_raw_matrix_captures(self):
        cases = load_manifest()

        for case_id in ("eegbrowser_epoched", "eegbrowser_epoched_marked", "eegbrowser_rejection_epochs"):
            with self.subTest(case_id=case_id):
                matlab_command = cases[case_id].targets["eeglab"].matlab_command
                self.assertIn("data = zeros(8,250,3)", matlab_command)
                self.assertIn("eegplot(data,", matlab_command)
                self.assertNotIn("eegplot(EEG", matlab_command)


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
            self.assertIn("capture_simple_pop_dialog", script_text)
            self.assertIn("inputgui_override_dir =", script_text)

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
            self.assertIn("variant = 'epoched';", script_text)
            self.assertIn("[EEG, com] = pop_interp(EEG);", script_text)
            self.assertIn("EEG.epoch = struct", script_text)
            self.assertIn("capture_simple_pop_dialog", script_text)

    def test_matlab_dialog_capture_generates_simple_pop_function_script(self):
        cases = [
            ("pop_resample_dialog", "pop_resample"),
            ("pop_epoch_dialog", "pop_epoch"),
            ("pop_rmbase_dialog", "pop_rmbase"),
            ("pop_editeventfield_dialog", "pop_editeventfield"),
            ("pop_selectevent_dialog", "pop_selectevent"),
            ("pop_rmdat_dialog", "pop_rmdat"),
            ("pop_chanedit_dialog", "pop_chanedit"),
        ]
        for case_id, action in cases:
            with self.subTest(case_id=case_id), tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = pathlib.Path(tmpdir)
                case = load_manifest()[case_id]

                def fake_run_subprocess(target_name, output_path, command, env, timeout_seconds):
                    output_path.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
                    return CaptureResult(target_name, output_path, command, 0)

                with (
                    mock.patch("tools.visual_parity.capture.shutil.which", return_value="/usr/common/bin/matlab"),
                    mock.patch("tools.visual_parity.capture._run_subprocess", side_effect=fake_run_subprocess),
                ):
                    results = capture_case(case, "eeglab", output_dir=tmp_path)

                self.assertTrue(results[0].ok)
                script_text = next((tmp_path / case_id).glob("*.m")).read_text()
                self.assertIn(f"action = '{action}';", script_text)
                self.assertIn(f"[EEG, com] = {action}(EEG);", script_text)
                self.assertIn("capture_simple_pop_dialog", script_text)
                self.assertIn("inputgui_override_dir =", script_text)
                self.assertIn("addpath(inputgui_override_dir, '-begin');", script_text)
                self.assertIn("write_figure_capture(fig, output_file);", script_text)
                override_text = next((tmp_path / case_id / "inputgui_plot_override").glob("inputgui.m")).read_text()
                self.assertIn("args{6} = 'plot';", override_text)
                self.assertIn("args = [args {'mode' 'plot'}];", override_text)

    def test_matlab_dialog_capture_generates_firfilt_helper_scripts(self):
        cases = [
            ("pop_kaiserbeta_dialog", "pop_kaiserbeta", "pop_kaiserbeta;"),
            ("pop_firwsord_dialog", "pop_firwsord", "pop_firwsord;"),
            (
                "pop_firpmord_dialog",
                "pop_firpmord",
                "pop_firpmord([0.2 0.3], [0 1]);",
            ),
            ("pop_xfirws_dialog", "pop_xfirws", "pop_xfirws;"),
        ]
        for case_id, action, call in cases:
            with self.subTest(case_id=case_id), tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = pathlib.Path(tmpdir)
                case = load_manifest()[case_id]

                def fake_run_subprocess(target_name, output_path, command, env, timeout_seconds):
                    output_path.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
                    return CaptureResult(target_name, output_path, command, 0)

                with (
                    mock.patch("tools.visual_parity.capture.shutil.which", return_value="/usr/common/bin/matlab"),
                    mock.patch("tools.visual_parity.capture._run_subprocess", side_effect=fake_run_subprocess),
                ):
                    results = capture_case(case, "eeglab", output_dir=tmp_path)

                self.assertTrue(results[0].ok)
                script_text = next((tmp_path / case_id).glob("*.m")).read_text()
                self.assertIn(f"action = '{action}';", script_text)
                self.assertIn(call, script_text)
                self.assertNotIn(f"[EEG, com] = {action}(EEG);", script_text)
                self.assertIn("capture_simple_pop_dialog", script_text)

    def test_matlab_dialog_capture_generates_pop_newset_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            case = load_manifest()["pop_newset_dialog"]

            def fake_run_subprocess(target_name, output_path, command, env, timeout_seconds):
                output_path.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
                return CaptureResult(target_name, output_path, command, 0)

            with (
                mock.patch("tools.visual_parity.capture.shutil.which", return_value="/usr/common/bin/matlab"),
                mock.patch("tools.visual_parity.capture._run_subprocess", side_effect=fake_run_subprocess),
            ):
                results = capture_case(case, "eeglab", output_dir=tmp_path)

            self.assertTrue(results[0].ok)
            script_text = next((tmp_path / "pop_newset_dialog").glob("*.m")).read_text()
            self.assertIn("action = 'pop_newset';", script_text)
            self.assertIn("ALLEEG = EEG;", script_text)
            self.assertIn("CURRENTSET = 1;", script_text)
            self.assertIn("[ALLEEG, EEG, CURRENTSET, com] = pop_newset(ALLEEG, EEG, CURRENTSET);", script_text)
            self.assertIn("capture_simple_pop_dialog", script_text)

    def test_matlab_dialog_capture_generates_pop_editeventvals_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            case = load_manifest()["pop_editeventvals_dialog"]

            def fake_run_subprocess(target_name, output_path, command, env, timeout_seconds):
                output_path.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
                return CaptureResult(target_name, output_path, command, 0)

            with (
                mock.patch("tools.visual_parity.capture.shutil.which", return_value="/usr/common/bin/matlab"),
                mock.patch("tools.visual_parity.capture._run_subprocess", side_effect=fake_run_subprocess),
            ):
                results = capture_case(case, "eeglab", output_dir=tmp_path)

            self.assertTrue(results[0].ok)
            script_text = next((tmp_path / "pop_editeventvals_dialog").glob("*.m")).read_text()
            self.assertIn("inputgui(geometry, uilist", script_text)
            self.assertIn("pop_editeventvals('goto', 0)", script_text)

    def test_matlab_dialog_capture_generates_dataset_pop_function_scripts(self):
        cases = [("pop_copyset_dialog", "pop_copyset", "pop_copyset(ALLEEG, 1)")]
        for case_id, action, call in cases:
            with self.subTest(case_id=case_id), tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = pathlib.Path(tmpdir)
                case = load_manifest()[case_id]

                def fake_run_subprocess(target_name, output_path, command, env, timeout_seconds):
                    output_path.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
                    return CaptureResult(target_name, output_path, command, 0)

                with (
                    mock.patch("tools.visual_parity.capture.shutil.which", return_value="/usr/common/bin/matlab"),
                    mock.patch("tools.visual_parity.capture._run_subprocess", side_effect=fake_run_subprocess),
                ):
                    results = capture_case(case, "eeglab", output_dir=tmp_path)

                self.assertTrue(results[0].ok)
                script_text = next((tmp_path / case_id).glob("*.m")).read_text()
                self.assertIn(f"action = '{action}';", script_text)
                self.assertIn(call, script_text)

    def test_matlab_dialog_capture_generates_plot_variant_scripts(self):
        cases = [
            ("pop_spectopo_channels_dialog", "pop_spectopo", "com = pop_spectopo(EEG, 1);"),
            ("pop_spectopo_components_dialog", "pop_spectopo", "com = pop_spectopo(EEG, 0);"),
            ("pop_prop_channels_dialog", "pop_prop", "com = pop_prop(EEG, 1);"),
            ("pop_prop_components_dialog", "pop_prop", "com = pop_prop(EEG, 0);"),
            ("pop_timtopo_dialog", "pop_timtopo", "com = pop_timtopo(EEG);"),
            ("pop_plottopo_dialog", "pop_plottopo", "com = pop_plottopo(EEG);"),
            ("pop_headplot_erp_dialog", "pop_headplot", "com = pop_headplot(EEG, 1);"),
            ("pop_headplot_components_dialog", "pop_headplot", "com = pop_headplot(EEG, 0);"),
            ("coregister_dialog", "coregister", "coregister(EEG.chanlocs"),
            ("pop_plotdata_dialog", "pop_plotdata", "com = pop_plotdata(EEG);"),
            ("pop_erpimage_channels_dialog", "pop_erpimage", "com = pop_erpimage(EEG, 1);"),
            ("pop_erpimage_components_dialog", "pop_erpimage", "com = pop_erpimage(EEG, 0);"),
            ("pop_envtopo_dialog", "pop_envtopo", "com = pop_envtopo(EEG);"),
            ("pop_comperp_channels_dialog", "pop_comperp", "pop_comperp(ALLEEG, 1);"),
            ("pop_comperp_components_dialog", "pop_comperp", "pop_comperp(ALLEEG, 0);"),
            ("pop_newtimef_channels_dialog", "pop_newtimef", "com = pop_newtimef(EEG, 1);"),
            ("pop_newtimef_components_dialog", "pop_newtimef", "com = pop_newtimef(EEG, 0);"),
            ("pop_newcrossf_channels_dialog", "pop_newcrossf", "com = pop_newcrossf(EEG, 1);"),
            ("pop_newcrossf_components_dialog", "pop_newcrossf", "com = pop_newcrossf(EEG, 0);"),
            ("pop_signalstat_channels_dialog", "pop_signalstat", "com = pop_signalstat(EEG, 1);"),
            ("pop_signalstat_components_dialog", "pop_signalstat", "com = pop_signalstat(EEG, 0);"),
            ("pop_eventstat_dialog", "pop_eventstat", "com = pop_eventstat(EEG);"),
        ]
        for case_id, action, expected_call in cases:
            with self.subTest(case_id=case_id), tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = pathlib.Path(tmpdir)
                case = load_manifest()[case_id]

                def fake_run_subprocess(target_name, output_path, command, env, timeout_seconds):
                    output_path.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
                    return CaptureResult(target_name, output_path, command, 0)

                with (
                    mock.patch("tools.visual_parity.capture.shutil.which", return_value="/usr/common/bin/matlab"),
                    mock.patch("tools.visual_parity.capture._run_subprocess", side_effect=fake_run_subprocess),
                ):
                    results = capture_case(case, "eeglab", output_dir=tmp_path)

                self.assertTrue(results[0].ok)
                script_text = next((tmp_path / case_id).glob("*.m")).read_text()
                self.assertIn(f"action = '{action}';", script_text)
                self.assertIn(expected_call, script_text)
                self.assertIn("capture_simple_pop_dialog", script_text)

    def test_matlab_dialog_capture_generates_pop_mergeset_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            case = load_manifest()["pop_mergeset_dialog"]

            def fake_run_subprocess(target_name, output_path, command, env, timeout_seconds):
                output_path.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
                return CaptureResult(target_name, output_path, command, 0)

            with (
                mock.patch("tools.visual_parity.capture.shutil.which", return_value="/usr/common/bin/matlab"),
                mock.patch("tools.visual_parity.capture._run_subprocess", side_effect=fake_run_subprocess),
            ):
                results = capture_case(case, "eeglab", output_dir=tmp_path)

            self.assertTrue(results[0].ok)
            script_text = next((tmp_path / "pop_mergeset_dialog").glob("*.m")).read_text()
            self.assertIn("Merge datasets -- pop_mergeset()", script_text)
            self.assertIn("Dataset indices to merge", script_text)

    def test_matlab_dialog_capture_generates_select_multiple_dataset_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            case = load_manifest()["select_multiple_datasets_dialog"]

            def fake_run_subprocess(target_name, output_path, command, env, timeout_seconds):
                output_path.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
                return CaptureResult(target_name, output_path, command, 0)

            with (
                mock.patch("tools.visual_parity.capture.shutil.which", return_value="/usr/common/bin/matlab"),
                mock.patch("tools.visual_parity.capture._run_subprocess", side_effect=fake_run_subprocess),
            ):
                results = capture_case(case, "eeglab", output_dir=tmp_path)

            self.assertTrue(results[0].ok)
            script_text = next((tmp_path / "select_multiple_datasets_dialog").glob("*.m")).read_text()
            self.assertIn("Dataset 1:menu one", script_text)
            self.assertIn("'Tag', 'listboxvals'", script_text)

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
