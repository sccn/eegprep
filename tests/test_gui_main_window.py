import os
import logging
import sys
import unittest
from unittest import mock

import numpy as np
import pytest

from eegprep.functions.guifunc.eeglab_menu import eeglab_menus, menu_actions
from eegprep.functions.guifunc.menu_actions import (
    MenuActionDispatcher,
    action_kind,
)
from eegprep.functions.guifunc.menu_placeholders import is_placeholder_action, placeholder_message
from eegprep.functions.guifunc.menu_spec import menu_enabled
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset


def _labels(items):
    return [item.label for item in items]


def _child(menu, label):
    items = menu.children if hasattr(menu, "children") else menu
    for item in items:
        if item.label == label:
            return item
    raise AssertionError(f"missing menu item {label!r}")


def _qt_actions(actions):
    collected = []
    for action in actions:
        if action.isSeparator():
            continue
        collected.append(action)
        if action.menu() is not None:
            collected.extend(_qt_actions(action.menu().actions()))
    return collected


def _fake_qt_widgets(*, open_file="", save_file="", directory="", double_value=1.0):
    class QFileDialog:
        @staticmethod
        def getOpenFileName(*_args, **_kwargs):
            return open_file, ""

        @staticmethod
        def getOpenFileNames(*_args, **_kwargs):
            return ([open_file] if open_file else []), ""

        @staticmethod
        def getSaveFileName(*_args, **_kwargs):
            return save_file, ""

        @staticmethod
        def getExistingDirectory(*_args, **_kwargs):
            return directory

    class QInputDialog:
        @staticmethod
        def getDouble(*_args, **_kwargs):
            return double_value, True

        @staticmethod
        def getInt(*_args, **_kwargs):
            return 1, True

        @staticmethod
        def getMultiLineText(*_args, **_kwargs):
            return "TaskName=eeg", True

    class QMessageBox:
        Yes = 1
        No = 2
        Cancel = 3

        @staticmethod
        def question(*_args, **_kwargs):
            return QMessageBox.Yes

        @staticmethod
        def warning(*_args, **_kwargs):
            return None

        @staticmethod
        def information(*_args, **_kwargs):
            return None

    return type(
        "FakeQtWidgets", (), {"QFileDialog": QFileDialog, "QInputDialog": QInputDialog, "QMessageBox": QMessageBox}
    )


def _demo_eeg(*, epoched=False, chanlocs=True, ica=True):
    data = np.zeros((2, 20, 2), dtype=np.float32) if epoched else np.zeros((2, 40), dtype=np.float32)
    eeg = {
        "setname": "demo",
        "filename": "demo.set",
        "filepath": "/tmp",
        "data": data,
        "nbchan": 2,
        "pnts": 20 if epoched else 40,
        "trials": 2 if epoched else 1,
        "srate": 100,
        "xmin": -0.1 if epoched else 0,
        "xmax": 0.09 if epoched else 0.39,
        "times": np.arange(20 if epoched else 40),
        "event": [{"type": "stim", "latency": 10}],
        "urevent": [],
        "epoch": [],
        "history": "",
        "chaninfo": {},
        "reject": {},
        "ref": "common",
        "icaact": np.array([]),
        "icawinv": np.eye(2) if ica else np.array([]),
        "icasphere": np.eye(2) if ica else np.array([]),
        "icaweights": np.eye(2) if ica else np.array([]),
        "icachansind": np.arange(2) if ica else np.array([]),
    }
    if chanlocs:
        eeg["chanlocs"] = [
            {"labels": "Cz", "theta": 0.0, "radius": 0.0, "ref": "common"},
            {"labels": "Pz", "theta": 180.0, "radius": 0.25, "ref": "common"},
        ]
    else:
        eeg["chanlocs"] = [{"labels": "Cz"}, {"labels": "Pz"}]
    return eeg


class MainMenuSpecTests(unittest.TestCase):
    def test_default_menu_matches_eeglab_top_level_and_hides_legacy_items(self):
        menus = eeglab_menus(all_menus=False)

        self.assertEqual(_labels(menus), ["File", "Edit", "Tools", "Plot", "Study", "Datasets", "Help"])
        edit_labels = _labels(_child(tuple(menus), "Edit").children)
        file_labels = _labels(_child(tuple(menus), "File").children)
        tools_labels = _labels(_child(tuple(menus), "Tools").children)

        self.assertIn("BIDS tools", file_labels)
        self.assertNotIn("Adjust event latencies", edit_labels)
        self.assertIn('(Expand tool choices via "File > Preferences")', tools_labels)
        self.assertNotIn("Automatic channel rejection", tools_labels)
        self.assertIn("Reject data using Clean Rawdata and ASR", tools_labels)
        self.assertIn("Classify components using ICLabel", tools_labels)
        self.assertIn("Source localization using DIPFIT", tools_labels)

    def test_all_menus_mode_reveals_legacy_items_and_hides_expand_prompt(self):
        menus = eeglab_menus(all_menus=True)
        edit_labels = _labels(_child(tuple(menus), "Edit").children)
        tools_labels = _labels(_child(tuple(menus), "Tools").children)

        self.assertIn("Adjust event latencies", edit_labels)
        self.assertIn("Automatic channel rejection", tools_labels)
        self.assertIn("Reject data epochs", tools_labels)
        self.assertNotIn('(Expand tool choices via "File > Preferences")', tools_labels)

    def test_firfilt_plugin_items_precede_legacy_filter(self):
        tools = _child(eeglab_menus(all_menus=True), "Tools")
        filter_menu = _child(tools.children, "Filter the data")

        self.assertEqual(
            _labels(filter_menu.children)[:5],
            [
                "Basic FIR filter (new, default)",
                "Windowed sinc FIR filter",
                "Parks-McClellan (equiripple) FIR filter",
                "Moving average FIR filter",
                "Basic FIR filter (legacy)",
            ],
        )

    def test_eeg_bids_plugin_items_match_file_menu_locations(self):
        file_menu = _child(eeglab_menus(all_menus=False), "File")
        import_menu = _child(file_menu.children, "Import data")
        import_functions = _child(import_menu.children, "Using EEGPrep functions and plugins")
        export_menu = _child(file_menu.children, "Export")

        self.assertIn("From BIDS folder structure", _labels(import_functions.children))
        self.assertIn("Import Magstim/EGI .mff file", _labels(import_functions.children))
        self.assertIn("To BIDS folder structure", _labels(export_menu.children))
        self.assertEqual(_labels(file_menu.children)[4], "BIDS tools")
        self.assertIn("Manage EEGPrep extensions", _labels(file_menu.children))

    def test_help_menu_uses_eegprep_branding_and_eeglab_style_actions(self):
        help_menu = _child(eeglab_menus(all_menus=False), "Help")
        help_labels = _labels(help_menu.children)

        self.assertIn("About EEGPrep", help_labels)
        self.assertIn("Check for EEGPrep updates", help_labels)
        self.assertIn("EEGPrep menus", help_labels)
        self.assertIn("EEGPrep tutorial", help_labels)
        self.assertIn("Email the EEGPrep team", help_labels)
        self.assertIn("Report an EEGPrep issue", help_labels)
        self.assertNotIn("EEGLAB tutorial", help_labels)
        self.assertEqual(_child(help_menu.children, "About EEGPrep").action, "help:eegprep")
        self.assertEqual(_child(help_menu.children, "Check for EEGPrep updates").action, "updates")
        self.assertEqual(_child(help_menu.children, "About EEGPrep help").action, "help:eeg_helphelp")
        self.assertEqual(_child(help_menu.children, "EEGPrep menus").action, "help:eeg_helpmenu")
        self.assertEqual(_child(help_menu.children, "EEGPrep tutorial").action, "tutorial")
        self.assertEqual(_child(help_menu.children, "Email the EEGPrep team").action, "mailto:eeglab@sccn.ucsd.edu")
        self.assertEqual(_child(help_menu.children, "Report an EEGPrep issue").action, "issues")

    def test_help_functions_submenu_uses_eeglab_pophelp_topics(self):
        functions_menu = _child(_child(eeglab_menus(all_menus=False), "Help").children, "EEGPrep functions")

        self.assertEqual(
            {item.label: item.action for item in functions_menu.children},
            {
                "Admin. functions": "help:eeg_helpadmin",
                "Interactive pop_ functions": "help:eeg_helppop",
                "Signal processing functions": "help:eeg_helpsigproc",
                "Group data (STUDY) functions": "help:eeg_helpstudy",
                "Time-frequency functions": "help:eeg_helptimefreq",
                "Statistical functions": "help:eeg_helpstatistics",
                "Graphic interface builder functions": "help:eeg_helpgui",
                "Misc. command line functions": "help:eeg_helpmisc",
            },
        )

    def test_help_menu_pophelp_actions_have_packaged_topics(self):
        help_menu = _child(eeglab_menus(all_menus=False), "Help")
        help_actions = [action for action in menu_actions((help_menu,)) if action.startswith("help:")]
        help_topics = {action.split(":", 1)[1] for action in help_actions}

        self.assertEqual(
            help_topics,
            {
                "eegprep",
                "eeg_helphelp",
                "eeg_helpmenu",
                "eeg_helpadmin",
                "eeg_helppop",
                "eeg_helpsigproc",
                "eeg_helpstudy",
                "eeg_helptimefreq",
                "eeg_helpstatistics",
                "eeg_helpgui",
                "eeg_helpmisc",
            },
        )

    def test_viewprops_plugin_items_match_plot_menu_locations(self):
        plot_menu = _child(eeglab_menus(all_menus=False), "Plot")

        self.assertEqual(
            _labels(plot_menu.children)[-2:],
            ["View extended channel properties", "View extended component properties"],
        )

    def test_menu_enabled_matches_startup_and_dataset_rules(self):
        menus = eeglab_menus(all_menus=True)
        file_menu = _child(menus, "File")
        edit_menu = _child(menus, "Edit")
        tools_menu = _child(menus, "Tools")
        plot_menu = _child(menus, "Plot")
        channel_locations = _child(plot_menu.children, "Channel locations")

        self.assertTrue(menu_enabled(file_menu, {"startup"}))
        self.assertFalse(menu_enabled(edit_menu, {"startup"}))
        self.assertFalse(menu_enabled(tools_menu, {"startup"}))
        self.assertTrue(menu_enabled(tools_menu, {"continuous_dataset"}))
        self.assertFalse(menu_enabled(channel_locations, {"continuous_dataset", "chanloc_absent"}))

    def test_startup_top_level_enabled_states_match_eegprep_ux(self):
        enabled_by_label = {menu.label: menu_enabled(menu, {"startup"}) for menu in eeglab_menus(all_menus=False)}

        self.assertEqual(
            enabled_by_label,
            {
                "File": True,
                "Edit": False,
                "Tools": False,
                "Plot": False,
                "Study": False,
                "Datasets": False,
                "Help": True,
            },
        )

    def test_main_window_stylesheet_makes_in_window_disabled_menus_discernible(self):
        from eegprep.functions.guifunc.main_window import _main_window_stylesheet

        stylesheet = _main_window_stylesheet()

        self.assertIn("QMenuBar::item:disabled", stylesheet)
        self.assertIn("color: #64708f", stylesheet)
        self.assertIn("background: transparent", stylesheet)

    def test_all_menu_actions_are_classified(self):
        actions = menu_actions(eeglab_menus(all_menus=True))

        self.assertIn("pop_reref", actions)
        self.assertEqual(action_kind("pop_reref"), "implemented")
        self.assertEqual(action_kind("pop_select"), "implemented")
        self.assertEqual(action_kind("pop_resample"), "implemented")
        self.assertEqual(action_kind("pop_epoch"), "implemented")
        self.assertEqual(action_kind("pop_eegfilt"), "implemented")
        self.assertEqual(action_kind("pop_eegfiltnew"), "implemented")
        self.assertEqual(action_kind("pop_firws"), "implemented")
        self.assertEqual(action_kind("pop_firpm"), "implemented")
        self.assertEqual(action_kind("pop_firma"), "implemented")
        self.assertEqual(action_kind("pop_clean_rawdata"), "implemented")
        self.assertEqual(action_kind("pop_runica"), "implemented")
        self.assertEqual(action_kind("pop_iclabel"), "implemented")
        self.assertEqual(action_kind("pop_icflag"), "implemented")
        self.assertEqual(action_kind("pop_subcomp"), "implemented")
        self.assertEqual(action_kind("pop_exportbids"), "implemented")
        self.assertEqual(action_kind("select_multiple_datasets"), "implemented")
        self.assertEqual(action_kind("topoplot:labels"), "implemented")
        self.assertTrue(all(action_kind(action) in {"implemented", "placeholder"} for action in actions))
        self.assertTrue(
            all(action_kind(action) == "implemented" or is_placeholder_action(action) for action in actions)
        )

    def test_file_menu_actions_are_implemented_or_explicit_placeholders(self):
        file_menu = _child(eeglab_menus(all_menus=True), "File")
        file_actions = menu_actions((file_menu,))

        self.assertIn("pop_importdata", file_actions)
        self.assertIn("pop_exportbids", file_actions)
        self.assertIn("pop_saveh:dataset", file_actions)
        self.assertEqual(
            [action for action in sorted(file_actions) if action_kind(action) != "implemented"],
            [],
        )
        self.assertEqual(action_kind("pop_fileio_brainvision_mat"), "implemented")


class EEGPrepSessionTests(unittest.TestCase):
    def test_session_reports_startup_without_data(self):
        self.assertEqual(EEGPrepSession().menu_statuses(), {"startup"})

    def test_session_uses_one_based_dataset_indices(self):
        session = EEGPrepSession()
        index = session.store_current(_demo_eeg(), new=True, command="EEG = demo;")

        self.assertEqual(index, 1)
        self.assertEqual(session.CURRENTSET, [1])
        self.assertEqual(session.dataset_summaries()[0][1], "Dataset 1:demo")
        self.assertEqual(session.ALLCOM, ["EEG = demo;"])

    def test_session_stores_multiple_selected_datasets_back_to_same_indices(self):
        session = EEGPrepSession()
        first = _demo_eeg()
        second = _demo_eeg()
        second["setname"] = "second"
        session.store_current(first, new=True)
        session.store_current(second, new=True)
        session.retrieve([1, 2])

        edited = [dict(item, ref="average") for item in session.EEG]
        stored = session.store_current(edited, command="EEG = pop_reref(EEG);")

        self.assertEqual(stored, [1, 2])
        self.assertEqual(session.CURRENTSET, [1, 2])
        self.assertEqual([item["ref"] for item in session.ALLEEG], ["average", "average"])

    def test_session_delete_current_selects_remaining_dataset(self):
        session = EEGPrepSession()
        first = _demo_eeg()
        second = _demo_eeg()
        second["setname"] = "second"
        session.store_current(first, new=True)
        session.store_current(second, new=True)
        session.retrieve(1)

        session.delete_current()

        self.assertEqual(session.CURRENTSET, [1])
        self.assertEqual(session.EEG["setname"], "second")
        self.assertEqual(session.menu_statuses(), {"continuous_dataset"})

    def test_session_reports_dataset_status_edges(self):
        session = EEGPrepSession()
        session.EEG = _demo_eeg(chanlocs=False, ica=False)
        self.assertEqual(session.menu_statuses(), {"continuous_dataset", "chanloc_absent", "ica_absent"})

        session.EEG = _demo_eeg(epoched=True)
        self.assertEqual(session.menu_statuses(), {"epoched_dataset"})

        session.EEG = [_demo_eeg(), _demo_eeg()]
        session.CURRENTSET = [1, 2]
        self.assertEqual(session.menu_statuses(), {"multiple_datasets"})

        session.STUDY = {"name": "study"}
        session.CURRENTSTUDY = 1
        self.assertEqual(session.menu_statuses(), {"study"})

    def test_session_treats_nonzero_xmin_single_trial_data_as_continuous(self):
        session = EEGPrepSession()
        session.EEG = _demo_eeg()
        session.EEG["xmin"] = 1.5
        session.EEG["trials"] = 1

        self.assertEqual(session.menu_statuses(), {"continuous_dataset"})

    def test_session_dataset_summaries_include_empty_dataset_structs(self):
        session = EEGPrepSession()
        session.ALLEEG = [eeg_emptyset(), {}, _demo_eeg()]
        session.CURRENTSET = [1]

        self.assertEqual(
            session.dataset_summaries(),
            [
                (1, "Dataset 1:(no dataset name)", True),
                (3, "Dataset 3:demo", False),
            ],
        )

    def test_main_window_summary_handles_empty_numpy_metadata_values(self):
        from eegprep.functions.guifunc.main_window import _channel_location_state, _reference_state

        eeg = _demo_eeg()
        eeg["ref"] = np.array([])
        for chanloc in eeg["chanlocs"]:
            chanloc["ref"] = np.array([])
            chanloc["theta"] = np.array([])

        self.assertEqual(_reference_state(eeg), "unknown")
        self.assertEqual(_channel_location_state(eeg), "No (labels only)")


class MenuActionDispatcherTests(unittest.TestCase):
    def test_placeholder_message_is_user_facing(self):
        message = placeholder_message("pop_selectcomps")

        self.assertIn("not yet available in EEGPrep", message)
        self.assertIn("https://github.com/sccn/eegprep/issues", message)
        self.assertNotIn("TODO", message)

    def test_gui_dispatch_shows_warning_for_action_errors(self):
        dispatcher = MenuActionDispatcher(EEGPrepSession())

        with (
            mock.patch.object(dispatcher, "dispatch", side_effect=ValueError("bad input")),
            mock.patch.object(dispatcher, "_warn") as warn,
        ):
            dispatcher.dispatch_gui("pop_adjustevents", parent="window")

        warn.assert_called_once_with("window", "bad input")

    def test_show_help_missing_resource_raises_clear_error_not_coming_soon(self):
        dispatcher = MenuActionDispatcher(EEGPrepSession())

        with (
            mock.patch(
                "eegprep.functions.guifunc.menu_actions.pophelp",
                side_effect=FileNotFoundError("missing packaged help"),
            ),
            mock.patch.object(dispatcher, "show_coming_soon") as coming_soon,
            self.assertRaisesRegex(FileNotFoundError, "missing packaged help"),
        ):
            dispatcher.dispatch("help:missing")

        coming_soon.assert_not_called()

    def test_show_help_uses_packaged_pophelp_for_help_topics(self):
        dispatcher = MenuActionDispatcher(EEGPrepSession())

        with mock.patch("eegprep.functions.guifunc.menu_actions.pophelp") as help_dialog:
            dispatcher.dispatch("help:eeg_helpadmin")

        help_dialog.assert_called_once_with("eeg_helpadmin", parent=None)

    def test_bare_help_action_defaults_to_eegprep_topic(self):
        dispatcher = MenuActionDispatcher(EEGPrepSession())

        with mock.patch("eegprep.functions.guifunc.menu_actions.pophelp") as help_dialog:
            dispatcher.dispatch("help")

        help_dialog.assert_called_once_with("eegprep", parent=None)

    def test_help_and_admin_link_actions_do_not_mutate_session_history(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        original_currentset = list(session.CURRENTSET)
        original_history = list(session.ALLCOM)
        dispatcher = MenuActionDispatcher(session)

        with (
            mock.patch("eegprep.functions.guifunc.menu_actions.pophelp"),
            mock.patch("eegprep.functions.guifunc.menu_actions.webbrowser.open"),
        ):
            for action in (
                "help:eeg_helpadmin",
                "help:eeg_helpmenu",
                "tutorial",
                "mailto:eeglab@sccn.ucsd.edu",
                "updates",
                "issues",
                "license",
            ):
                dispatcher.dispatch(action)

        self.assertEqual(session.CURRENTSET, original_currentset)
        self.assertEqual(session.ALLCOM, original_history)
        self.assertEqual(session.EEG["setname"], "demo")

    def test_tutorial_mailto_updates_and_issue_actions_open_expected_targets(self):
        dispatcher = MenuActionDispatcher(EEGPrepSession())

        with mock.patch("eegprep.functions.guifunc.menu_actions.webbrowser.open") as open_url:
            dispatcher.dispatch("tutorial")
            dispatcher.dispatch("mailto:eeglab@sccn.ucsd.edu")
            dispatcher.dispatch("updates")
            dispatcher.dispatch("issues")

        self.assertEqual(
            [call.args[0] for call in open_url.call_args_list],
            [
                "https://sccn.github.io/eegprep/user_guide/quickstart.html",
                "mailto:eeglab@sccn.ucsd.edu",
                "https://github.com/sccn/eegprep/releases",
                "https://github.com/sccn/eegprep/issues",
            ],
        )

    def test_dispatch_gui_reraises_headless_errors_and_logs_traceback(self):
        dispatcher = MenuActionDispatcher(EEGPrepSession())

        with (
            mock.patch.object(dispatcher, "dispatch", side_effect=RuntimeError("boom")),
            self.assertLogs("eegprep.functions.guifunc.menu_actions", level=logging.ERROR) as logs,
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            dispatcher.dispatch_gui("pop_reref")

        self.assertIn("EEGPrep GUI menu action failed: pop_reref", "\n".join(logs.output))

    def test_retrieve_dataset_menu_action_clears_study_mode(self):
        session = EEGPrepSession()
        first = _demo_eeg()
        second = _demo_eeg()
        second["setname"] = "second"
        session.store_current(first, new=True)
        session.store_current(second, new=True)
        session.STUDY = {"name": "study"}
        session.CURRENTSTUDY = 1
        echoed = []
        session.add_command_echo_listener(echoed.append)
        dispatcher = MenuActionDispatcher(session)

        dispatcher.dispatch("retrieve_dataset:2")

        self.assertEqual(
            echoed,
            ["CURRENTSTUDY = 0;[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'retrieve', 2);"],
        )
        self.assertEqual(session.CURRENTSTUDY, 0)
        self.assertEqual(session.CURRENTSET, [2])
        self.assertEqual(session.EEG["setname"], "second")
        self.assertEqual(session.menu_statuses(), {"continuous_dataset"})
        self.assertIn("CURRENTSTUDY = 0;", session.ALLCOM[-1])

    def test_select_study_set_menu_action_restores_study_mode(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        session.STUDY = {"name": "study", "datasetinfo": [], "design": []}
        session.CURRENTSTUDY = 0
        echoed = []
        session.add_command_echo_listener(echoed.append)
        dispatcher = MenuActionDispatcher(session)

        dispatcher.dispatch("select_study_set")

        self.assertEqual(session.CURRENTSTUDY, 1)
        self.assertEqual(echoed, ["CURRENTSTUDY = 1"])
        self.assertEqual(session.ALLCOM[-1], "CURRENTSTUDY = 1")
        self.assertEqual(session.menu_statuses(), {"study"})

    def test_multiple_dataset_reref_preserves_selection(self):
        session = EEGPrepSession()
        first = _demo_eeg()
        second = _demo_eeg()
        second["setname"] = "second"
        session.store_current(first, new=True)
        session.store_current(second, new=True)
        session.retrieve([1, 2])
        dispatcher = MenuActionDispatcher(session)
        reref_output = [dict(item, ref="average") for item in session.EEG]

        with mock.patch(
            "eegprep.functions.popfunc.pop_reref.pop_reref",
            return_value=(reref_output, "EEG = pop_reref(EEG);"),
        ) as reref:
            dispatcher.dispatch("pop_reref")

        reref.assert_called_once()
        self.assertIsInstance(reref.call_args.args[0], list)
        self.assertEqual(session.CURRENTSET, [1, 2])
        self.assertEqual([item["ref"] for item in session.EEG], ["average", "average"])
        self.assertEqual([item["ref"] for item in session.ALLEEG], ["average", "average"])

    def test_resave_updates_single_dataset_metadata_and_saved_state(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        session.EEG["saved"] = "no"
        session.ALLEEG[0]["saved"] = "no"
        dispatcher = MenuActionDispatcher(session)

        with mock.patch("eegprep.functions.popfunc.pop_saveset.pop_saveset") as saveset:
            dispatcher.dispatch("pop_saveset:resave")

        saveset.assert_called_once_with(mock.ANY, os.path.normpath("/tmp/demo.set"))
        self.assertEqual(session.EEG["filename"], "demo.set")
        self.assertEqual(session.EEG["filepath"], os.path.normpath("/tmp"))
        self.assertEqual(session.EEG["saved"], "yes")
        self.assertEqual(session.ALLEEG[0]["saved"], "yes")

    def test_resave_multiple_datasets_does_not_collapse_selection(self):
        session = EEGPrepSession()
        first = _demo_eeg()
        first["setname"] = "first"
        first["filename"] = "first.set"
        second = _demo_eeg()
        second["setname"] = "second"
        second["filename"] = "second.set"
        session.store_current(first, new=True)
        session.store_current(second, new=True)
        session.retrieve([1, 2])
        for eeg in session.EEG:
            eeg["saved"] = "no"
        for eeg in session.ALLEEG:
            eeg["saved"] = "no"
        dispatcher = MenuActionDispatcher(session)

        with mock.patch("eegprep.functions.popfunc.pop_saveset.pop_saveset") as saveset:
            dispatcher.dispatch("pop_saveset:resave")

        self.assertEqual(
            [call.args[1] for call in saveset.call_args_list],
            [os.path.normpath("/tmp/first.set"), os.path.normpath("/tmp/second.set")],
        )
        self.assertEqual(session.CURRENTSET, [1, 2])
        self.assertEqual([item["setname"] for item in session.EEG], ["first", "second"])
        self.assertEqual([item["saved"] for item in session.EEG], ["yes", "yes"])
        self.assertEqual([item["saved"] for item in session.ALLEEG], ["yes", "yes"])

    def test_new_main_window_pop_actions_dispatch_to_real_wrappers(self):
        action_specs = [
            ("pop_comments", "eegprep.functions.popfunc.pop_comments.pop_comments", "commented"),
            ("pop_chanedit", "eegprep.functions.popfunc.pop_chanedit.pop_chanedit", "chanedited"),
            ("pop_editset", "eegprep.functions.popfunc.pop_editset.pop_editset", "edited"),
            ("pop_editeventfield", "eegprep.functions.popfunc.pop_editeventfield.pop_editeventfield", "eventfields"),
            ("pop_editeventvals", "eegprep.functions.popfunc.pop_editeventvals.pop_editeventvals", "eventvals"),
            ("pop_select", "eegprep.functions.popfunc.pop_select.pop_select", "selected"),
            ("pop_selectevent", "eegprep.functions.popfunc.pop_selectevent.pop_selectevent", "selectedevent"),
            ("pop_resample", "eegprep.functions.popfunc.pop_resample.pop_resample", "resampled"),
            ("pop_rmbase", "eegprep.functions.popfunc.pop_rmbase.pop_rmbase", "baseline"),
            ("pop_rmdat", "eegprep.functions.popfunc.pop_rmdat.pop_rmdat", "rmdat"),
            ("pop_epoch", "eegprep.functions.popfunc.pop_epoch.pop_epoch", "epoched"),
            ("pop_eegfilt", "eegprep.functions.popfunc.pop_eegfilt.pop_eegfilt", "legacy_filtered"),
            ("pop_eegfiltnew", "eegprep.plugins.firfilt.pop_eegfiltnew.pop_eegfiltnew", "filtered"),
            ("pop_firws", "eegprep.plugins.firfilt.pop_firws.pop_firws", "firws"),
            ("pop_firpm", "eegprep.plugins.firfilt.pop_firpm.pop_firpm", "firpm"),
            ("pop_firma", "eegprep.plugins.firfilt.pop_firma.pop_firma", "firma"),
            ("pop_clean_rawdata", "eegprep.plugins.clean_rawdata.pop_clean_rawdata.pop_clean_rawdata", "cleaned"),
            ("pop_runica", "eegprep.functions.popfunc.pop_runica.pop_runica", "ica"),
            ("pop_iclabel", "eegprep.plugins.ICLabel.pop_iclabel.pop_iclabel", "labeled"),
        ]

        for action, patch_target, setname in action_specs:
            with self.subTest(action=action):
                session = EEGPrepSession()
                session.store_current(_demo_eeg(), new=True)
                dispatcher = MenuActionDispatcher(session)
                output = dict(session.EEG, setname=setname)

                with mock.patch(patch_target, return_value=(output, f"EEG = {action}(EEG);")) as pop_func:
                    dispatcher.dispatch(action)

                if action == "pop_comments":
                    pop_func.assert_called_once_with(mock.ANY, "Comments of dataset: demo", return_com=True)
                else:
                    pop_func.assert_called_once_with(mock.ANY, return_com=True)
                self.assertEqual(session.EEG["setname"], setname)
                self.assertEqual(session.ALLEEG[0]["setname"], setname)
                self.assertEqual(session.ALLCOM[-1], f"EEG = {action}(EEG);")

    def test_topoplot_menu_actions_record_history_without_replacing_dataset(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        dispatcher = MenuActionDispatcher(session)
        original_eeg = session.EEG

        with mock.patch(
            "eegprep.functions.popfunc.pop_topoplot.plot_channel_locations",
            return_value=("figure", "topoplot([], EEG['chanlocs'], style='blank', electrodes='labelpoint')"),
        ) as locations:
            dispatcher.dispatch("topoplot:labels")

        locations.assert_called_once_with(original_eeg, mode="labels", return_com=True)
        self.assertIs(session.EEG, original_eeg)
        self.assertIs(session.ALLEEG[0], original_eeg)
        self.assertEqual(
            session.ALLCOM[-1],
            "topoplot([], EEG['chanlocs'], style='blank', electrodes='labelpoint')",
        )

    def test_pop_topoplot_menu_actions_record_history_without_replacing_dataset(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(epoched=True), new=True)
        dispatcher = MenuActionDispatcher(session)
        original_eeg = session.EEG

        with mock.patch(
            "eegprep.functions.popfunc.pop_topoplot.pop_topoplot",
            return_value=(["figure"], "pop_topoplot(EEG, typeplot=1, items=[0])"),
        ) as topoplot_func:
            dispatcher.dispatch("pop_topoplot:erp")

        topoplot_func.assert_called_once_with(original_eeg, typeplot=1, return_com=True)
        self.assertIs(session.EEG, original_eeg)
        self.assertIs(session.ALLEEG[0], original_eeg)
        self.assertEqual(session.ALLCOM[-1], "pop_topoplot(EEG, typeplot=1, items=[0])")

    def test_copyset_menu_updates_alleeg_eeg_currentset_and_history(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        dispatcher = MenuActionDispatcher(session)
        copied = dict(session.EEG, setname="copied")

        with mock.patch(
            "eegprep.functions.popfunc.pop_copyset.pop_copyset",
            return_value=(
                [session.EEG, copied],
                copied,
                2,
                "[ALLEEG EEG CURRENTSET LASTCOM] = pop_copyset(ALLEEG, 1, 2);",
            ),
        ) as copyset:
            dispatcher.dispatch("pop_copyset")

        copyset.assert_called_once_with([session.ALLEEG[0]], 1, gui=True, return_com=True)
        self.assertEqual(session.CURRENTSET, [2])
        self.assertEqual(session.EEG["setname"], "copied")
        self.assertEqual(session.ALLEEG[1]["setname"], "copied")
        self.assertEqual(session.LASTCOM, "[ALLEEG EEG CURRENTSET LASTCOM] = pop_copyset(ALLEEG, 1, 2);")

    def test_select_multiple_datasets_menu_preserves_ordered_session_selection(self):
        session = EEGPrepSession()
        first = _demo_eeg()
        first["setname"] = "first"
        second = _demo_eeg()
        second["setname"] = "second"
        session.store_current(first, new=True)
        session.store_current(second, new=True)
        dispatcher = MenuActionDispatcher(session)

        with mock.patch(
            "eegprep.functions.guifunc.select_multiple_datasets.select_multiple_datasets",
            side_effect=lambda session_arg, **_kwargs: (
                session_arg.retrieve([2, 1]),
                "[ALLEEG EEG CURRENTSET LASTCOM] = pop_newset(ALLEEG, EEG, CURRENTSET, 'retrieve', [2 1]);",
            ),
        ):
            dispatcher.dispatch("select_multiple_datasets")

        self.assertEqual(session.CURRENTSET, [2, 1])
        self.assertEqual([item["setname"] for item in session.EEG], ["second", "first"])
        self.assertIn("pop_newset", session.ALLCOM[-1])

    def test_mergeset_menu_stores_merged_dataset_as_new_dataset(self):
        session = EEGPrepSession()
        first = _demo_eeg()
        second = _demo_eeg()
        second["setname"] = "second"
        session.store_current(first, new=True)
        session.store_current(second, new=True)
        session.retrieve([1, 2])
        dispatcher = MenuActionDispatcher(session)
        merged = dict(first, setname="merged")

        with mock.patch(
            "eegprep.functions.popfunc.pop_mergeset.pop_mergeset",
            return_value=(merged, "EEG = pop_mergeset( ALLEEG, [1 2], 0);"),
        ) as mergeset:
            dispatcher.dispatch("pop_mergeset")

        mergeset.assert_called_once()
        self.assertEqual(len(mergeset.call_args.args[0]), 2)
        self.assertEqual(mergeset.call_args.args[1], [1, 2])
        self.assertEqual(mergeset.call_args.kwargs, {"gui": True, "return_com": True})
        self.assertEqual(session.CURRENTSET, [3])
        self.assertEqual(session.EEG["setname"], "merged")
        self.assertEqual(session.ALLEEG[2]["setname"], "merged")

    def test_pop_interp_dispatch_uses_generic_gui_command_echo(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        dispatcher = MenuActionDispatcher(session)
        command = "EEG = pop_interp(EEG, [1], 'spherical');"
        echoed = []
        output = dict(session.EEG, setname="interpolated")

        def fake_pop_interp(eeg, *, alleeg, return_com):
            self.assertIs(eeg, session.EEG)
            self.assertIs(alleeg, session.ALLEEG)
            self.assertTrue(return_com)
            return output, command

        session.add_command_echo_listener(echoed.append)
        with mock.patch("eegprep.functions.popfunc.pop_interp.pop_interp", side_effect=fake_pop_interp):
            dispatcher.dispatch("pop_interp")

        self.assertEqual(echoed, [command])
        self.assertEqual(session.EEG["setname"], "interpolated")
        self.assertEqual(session.ALLCOM[-1], command)

    def test_file_menu_importdata_dispatch_stores_new_dataset(self):
        session = EEGPrepSession()
        dispatcher = MenuActionDispatcher(session)
        imported = _demo_eeg()
        imported["setname"] = "imported"
        qt_widgets = _fake_qt_widgets(open_file="/tmp/data.tsv", double_value=250.0)

        with (
            mock.patch("eegprep.functions.guifunc.menu_actions._require_qt_widgets", return_value=qt_widgets),
            mock.patch(
                "eegprep.functions.popfunc.pop_importdata.pop_importdata",
                return_value=(imported, "EEG = pop_importdata('data', '/tmp/data.tsv');"),
            ) as importdata,
        ):
            dispatcher.dispatch("pop_importdata")

        importdata.assert_called_once()
        self.assertEqual(session.EEG["setname"], "imported")
        self.assertEqual(session.CURRENTSET, [1])
        self.assertEqual(session.ALLCOM[-1], "EEG = pop_importdata('data', '/tmp/data.tsv');")

    def test_file_menu_import_uses_native_file_dialog_by_default(self):
        captured = {}

        class QFileDialog:
            class Option:
                DontUseNativeDialog = 4

            @staticmethod
            def getOpenFileName(*args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs
                return "", ""

        qt_widgets = type("FakeQtWidgets", (), {"QFileDialog": QFileDialog})
        dispatcher = MenuActionDispatcher(EEGPrepSession())

        with mock.patch("eegprep.functions.guifunc.menu_actions._require_qt_widgets", return_value=qt_widgets):
            filename = dispatcher._open_import_filename("pop_fileio", None)

        self.assertEqual(filename, "")
        self.assertEqual(captured["args"][1], "Import data")
        self.assertEqual(captured["kwargs"], {})

    def test_file_menu_import_can_use_stable_qt_file_dialog(self):
        captured = {}

        class QFileDialog:
            class Option:
                DontUseNativeDialog = 4

            @staticmethod
            def getOpenFileName(*args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs
                return "", ""

        qt_widgets = type("FakeQtWidgets", (), {"QFileDialog": QFileDialog})
        dispatcher = MenuActionDispatcher(EEGPrepSession(), native_file_dialogs=False)

        with mock.patch("eegprep.functions.guifunc.menu_actions._require_qt_widgets", return_value=qt_widgets):
            filename = dispatcher._open_import_filename("pop_fileio", None)

        self.assertEqual(filename, "")
        self.assertEqual(captured["args"][1], "Import data")
        self.assertEqual(captured["kwargs"], {"options": QFileDialog.Option.DontUseNativeDialog})

    def test_file_menu_export_dispatch_records_history_without_changing_dataset(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        dispatcher = MenuActionDispatcher(session)
        qt_widgets = _fake_qt_widgets(save_file="/tmp/export.tsv")

        with (
            mock.patch("eegprep.functions.guifunc.menu_actions._require_qt_widgets", return_value=qt_widgets),
            mock.patch(
                "eegprep.functions.popfunc.pop_export.pop_export",
                return_value="LASTCOM = pop_export(EEG, '/tmp/export.tsv');",
            ) as export,
        ):
            dispatcher.dispatch("pop_export")

        export.assert_called_once()
        self.assertEqual(session.EEG["setname"], "demo")
        self.assertEqual(session.ALLCOM[-1], "LASTCOM = pop_export(EEG, '/tmp/export.tsv');")

    def test_file_menu_study_and_history_actions_update_session(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        dispatcher = MenuActionDispatcher(session)
        qt_widgets = _fake_qt_widgets(save_file="/tmp/history.m")
        study = {"name": "study", "datasetinfo": [{"index": 1, "setname": "demo"}], "design": []}

        with (
            mock.patch("eegprep.functions.guifunc.menu_actions._require_qt_widgets", return_value=qt_widgets),
            mock.patch(
                "eegprep.functions.studyfunc.pop_study.pop_study",
                return_value=(study, session.ALLEEG, "STUDY, ALLEEG = pop_study(STUDY, ALLEEG);"),
            ) as pop_study,
            mock.patch(
                "eegprep.functions.popfunc.pop_saveh.pop_saveh",
                return_value="pop_saveh(ALLCOM, 'history.m', '/tmp');",
            ) as saveh,
        ):
            dispatcher.dispatch("pop_study")
            dispatcher.dispatch("pop_saveh:session")

        pop_study.assert_called_once_with(None, mock.ANY, gui=True, return_com=True)
        saveh.assert_called_once()
        self.assertEqual(session.CURRENTSTUDY, 1)
        self.assertEqual(session.STUDY["datasetinfo"][0]["setname"], "demo")
        self.assertEqual(session.ALLCOM[-1], "pop_saveh(ALLCOM, 'history.m', '/tmp');")

    def test_file_menu_savestudy_uses_all_loaded_datasets(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        session.store_current(dict(_demo_eeg(), setname="second"), new=True)
        session.retrieve(1)
        session.STUDY = {
            "name": "study",
            "datasetinfo": [{"index": 1, "setname": "demo"}, {"index": 2, "setname": "second"}],
            "design": [],
        }
        dispatcher = MenuActionDispatcher(session)
        qt_widgets = _fake_qt_widgets(save_file="/tmp/study.study")
        saved = dict(session.STUDY, saved="yes")

        with (
            mock.patch("eegprep.functions.guifunc.menu_actions._require_qt_widgets", return_value=qt_widgets),
            mock.patch(
                "eegprep.functions.studyfunc.pop_savestudy.pop_savestudy",
                return_value=(saved, "STUDY = pop_savestudy(STUDY, ALLEEG, filename='study.study');"),
            ) as pop_savestudy,
        ):
            dispatcher.dispatch("pop_savestudy")

        pop_savestudy.assert_called_once_with(
            mock.ANY, session.ALLEEG, "/tmp/study.study", savemode=None, return_com=True
        )
        self.assertEqual(session.STUDY["saved"], "yes")
        self.assertEqual(session.ALLCOM[-1], "STUDY = pop_savestudy(STUDY, ALLEEG, filename='study.study');")

    def test_file_menu_loadstudy_updates_shared_session(self):
        session = EEGPrepSession()
        dispatcher = MenuActionDispatcher(session)
        qt_widgets = _fake_qt_widgets(open_file="/tmp/study.study")
        eeg = _demo_eeg()
        study = {"name": "loaded study", "datasetinfo": [{"index": 1, "setname": "demo"}], "design": []}

        with (
            mock.patch("eegprep.functions.guifunc.menu_actions._require_qt_widgets", return_value=qt_widgets),
            mock.patch(
                "eegprep.functions.studyfunc.pop_loadstudy.pop_loadstudy",
                return_value=(study, [eeg], "STUDY, ALLEEG = pop_loadstudy(filename='study.study');"),
            ) as pop_loadstudy,
        ):
            dispatcher.dispatch("pop_loadstudy")

        pop_loadstudy.assert_called_once_with("/tmp/study.study", return_com=True)
        self.assertEqual(session.CURRENTSTUDY, 1)
        self.assertEqual(session.STUDY["name"], "loaded study")
        self.assertEqual(session.ALLEEG[0]["setname"], "demo")
        self.assertEqual(session.ALLCOM[-1], "STUDY, ALLEEG = pop_loadstudy(filename='study.study');")

    def test_file_menu_studywizard_uses_browsed_datasets(self):
        session = EEGPrepSession()
        dispatcher = MenuActionDispatcher(session)
        qt_widgets = _fake_qt_widgets(open_file="/tmp/one.set")
        eeg = _demo_eeg()
        study = {"name": "wizard study", "datasetinfo": [{"index": 1, "setname": "demo"}], "design": []}

        with (
            mock.patch("eegprep.functions.guifunc.menu_actions._require_qt_widgets", return_value=qt_widgets),
            mock.patch(
                "eegprep.functions.studyfunc.pop_studywizard.pop_studywizard",
                return_value=(study, [eeg], "STUDY, ALLEEG = pop_studywizard(filenames=['/tmp/one.set']);"),
            ) as pop_studywizard,
        ):
            dispatcher.dispatch("pop_studywizard")

        pop_studywizard.assert_called_once_with(["/tmp/one.set"], return_com=True)
        self.assertEqual(session.CURRENTSTUDY, 1)
        self.assertEqual(session.STUDY["name"], "wizard study")
        self.assertEqual(session.ALLCOM[-1], "STUDY, ALLEEG = pop_studywizard(filenames=['/tmp/one.set']);")

    def test_study_menu_design_action_updates_shared_session(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        study = {"name": "study", "datasetinfo": [{"index": 1, "setname": "demo"}], "design": []}
        session.STUDY = study
        session.CURRENTSTUDY = 1
        dispatcher = MenuActionDispatcher(session)
        edited = dict(session.STUDY, currentdesign=1)

        with mock.patch(
            "eegprep.functions.studyfunc.pop_studydesign.pop_studydesign",
            return_value=(edited, session.ALLEEG, "STUDY = std_makedesign(STUDY, ALLEEG, 1);"),
        ) as pop_studydesign:
            dispatcher.dispatch("pop_studydesign")

        pop_studydesign.assert_called_once_with(study, session.ALLEEG, gui=True, return_com=True)
        self.assertEqual(session.STUDY["currentdesign"], 1)
        self.assertEqual(session.ALLCOM[-1], "STUDY = std_makedesign(STUDY, ALLEEG, 1);")

    def test_file_menu_simple_erp_study_uses_loaded_datasets(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        dispatcher = MenuActionDispatcher(session)
        study = {"name": "Simple ERP STUDY", "datasetinfo": [{"index": 1, "setname": "demo"}], "design": []}

        with mock.patch(
            "eegprep.functions.studyfunc.pop_studyerp.pop_studyerp",
            return_value=(study, session.ALLEEG, "STUDY, ALLEEG = pop_studyerp(ALLEEG);"),
        ) as pop_studyerp:
            dispatcher.dispatch("pop_studyerp")

        pop_studyerp.assert_called_once_with(session.ALLEEG, return_com=True)
        self.assertEqual(session.STUDY["name"], "Simple ERP STUDY")
        self.assertEqual(session.ALLCOM[-1], "STUDY, ALLEEG = pop_studyerp(ALLEEG);")

    def test_file_menu_clear_study_matches_eeglab_clear_all(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        session.STUDY = {"name": "study", "datasetinfo": [{"index": 1, "setname": "demo"}], "design": []}
        session.CURRENTSTUDY = 1
        dispatcher = MenuActionDispatcher(session)

        dispatcher.dispatch("clear_study")

        self.assertEqual(session.ALLEEG, [])
        self.assertEqual(session.CURRENTSET, [])
        self.assertIsNone(session.STUDY)
        self.assertEqual(session.CURRENTSTUDY, 0)
        self.assertEqual(session.ALLCOM[-1], "STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];")

    def test_file_menu_runscript_updates_currentset_from_namespace(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        session.store_current(dict(_demo_eeg(), setname="second"), new=True)
        session.retrieve(1)
        dispatcher = MenuActionDispatcher(session)
        qt_widgets = _fake_qt_widgets(open_file="/tmp/script.py")

        def fake_runscript(_filename, namespace):
            namespace["CURRENTSET"] = 2
            return "LASTCOM = pop_runscript('/tmp/script.py');"

        with (
            mock.patch("eegprep.functions.guifunc.menu_actions._require_qt_widgets", return_value=qt_widgets),
            mock.patch("eegprep.functions.popfunc.pop_runscript.pop_runscript", side_effect=fake_runscript),
        ):
            dispatcher.dispatch("pop_runscript")

        self.assertEqual(session.CURRENTSET, [2])
        self.assertEqual(session.ALLCOM[-1], "LASTCOM = pop_runscript('/tmp/script.py');")


class QtMainWindowTests(unittest.TestCase):
    def test_gui_main_window_startup_branding_size_and_menu_states(self):
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from eegprep.functions.guifunc.main_window import (
            _macos_application_menu_title,
            _macos_process_name,
            build_main_window,
        )

        window = build_main_window(EEGPrepSession(), all_menus=False)
        window.show()
        window.app.processEvents()
        size = window.window.size()
        minimum_size = window.window.minimumSize()
        enabled_by_label = {item["label"]: item["enabled"] for item in window.menu_inventory()}

        self.assertEqual(window.window.windowTitle(), "EEGPrep")
        self.assertEqual(window.app.applicationName(), "EEGPrep")
        self.assertEqual(window.app.applicationDisplayName(), "EEGPrep")
        if sys.platform == "darwin":
            self.assertEqual(_macos_process_name(), "EEGPrep")
            menu_title = _macos_application_menu_title()
            if menu_title is not None:
                self.assertEqual(menu_title, "EEGPrep")
        self.assertEqual((size.width(), size.height()), (520, 380))
        self.assertEqual((minimum_size.width(), minimum_size.height()), (460, 340))
        self.assertEqual(
            enabled_by_label,
            {
                "File": True,
                "Edit": False,
                "Tools": False,
                "Plot": False,
                "Study": False,
                "Datasets": False,
                "Help": True,
            },
        )
        top_level_actions = {action.text(): action for action in window.window.menuBar().actions()}
        self.assertTrue(top_level_actions["File"].menu().isEnabled())
        self.assertFalse(top_level_actions["Edit"].menu().isEnabled())
        self.assertFalse(top_level_actions["Tools"].menu().isEnabled())
        self.assertFalse(top_level_actions["Plot"].menu().isEnabled())
        self.assertFalse(top_level_actions["Study"].menu().isEnabled())
        self.assertFalse(top_level_actions["Datasets"].menu().isEnabled())
        self.assertTrue(top_level_actions["Help"].menu().isEnabled())
        window.window.close()

    def test_gui_main_window_inventory_includes_dynamic_dataset_menu(self):
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from eegprep.functions.guifunc.main_window import build_main_window

        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        window = build_main_window(session, all_menus=False)
        inventory = window.menu_inventory()

        self.assertEqual(
            [item["label"] for item in inventory],
            ["File", "Edit", "Tools", "Plot", "Study", "Datasets", "Help"],
        )
        datasets = next(item for item in inventory if item["label"] == "Datasets")
        self.assertEqual(datasets["children"][0]["label"], "Dataset 1:demo")
        window.window.close()

    def test_gui_main_window_checks_selected_dataset_menu_item(self):
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from eegprep.functions.guifunc.main_window import build_main_window

        session = EEGPrepSession()
        first = _demo_eeg()
        second = _demo_eeg()
        second["setname"] = "second"
        session.store_current(first, new=True)
        session.store_current(second, new=True)
        session.retrieve(2)
        window = build_main_window(session, all_menus=False)
        datasets = next(action.menu() for action in window.window.menuBar().actions() if action.text() == "Datasets")
        dataset_actions = {
            action.text(): action for action in datasets.actions() if action.text().startswith("Dataset")
        }

        self.assertFalse(dataset_actions["Dataset 1:demo"].isChecked())
        self.assertTrue(dataset_actions["Dataset 2:second"].isCheckable())
        self.assertTrue(dataset_actions["Dataset 2:second"].isChecked())
        window.window.close()

    def test_gui_main_window_marks_unimplemented_actions_distinctly(self):
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from eegprep.functions.guifunc.main_window import COMING_SOON_SUFFIX, build_main_window

        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        window = build_main_window(session, all_menus=False)
        actions_by_data = {
            action.data(): action for action in _qt_actions(window.window.menuBar().actions()) if action.data()
        }
        placeholder = actions_by_data["pop_eegplot:data"]
        implemented = actions_by_data["pop_resample"]

        self.assertEqual(placeholder.property("eegprep_label"), "Inspect/reject data by eye")
        self.assertEqual(placeholder.property("eegprep_implementation_state"), "coming_soon")
        self.assertTrue(placeholder.text().endswith(COMING_SOON_SUFFIX))
        self.assertFalse(placeholder.isEnabled())
        self.assertTrue(placeholder.font().italic())
        self.assertEqual(implemented.text(), "Change sampling rate")
        self.assertNotEqual(implemented.property("eegprep_implementation_state"), "coming_soon")
        self.assertTrue(implemented.isEnabled())
        window.window.close()

    def test_gui_main_window_inventory_reports_coming_soon_source_label(self):
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from eegprep.functions.guifunc.main_window import build_main_window

        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        window = build_main_window(session, all_menus=False)
        tools = next(item for item in window.menu_inventory() if item["label"] == "Tools")
        by_source_label = {item["source_label"]: item for item in tools["children"]}

        coming_soon = by_source_label["Inspect/reject data by eye"]
        self.assertEqual(coming_soon["implementation_state"], "coming_soon")
        self.assertFalse(coming_soon["enabled"])
        self.assertIn("coming soon", coming_soon["label"])
        self.assertEqual(by_source_label["Change sampling rate"]["implementation_state"], "implemented")
        self.assertTrue(by_source_label["Change sampling rate"]["enabled"])
        window.window.close()

    def test_gui_main_window_uses_native_menu_request_and_non_native_menu_roles(self):
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtGui

        from eegprep.functions.guifunc.main_window import build_main_window

        window = build_main_window(EEGPrepSession(), all_menus=True)
        menubar = window.window.menuBar()
        actions = _qt_actions(menubar.actions())

        if sys.platform == "darwin" and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            self.assertTrue(menubar.isNativeMenuBar())
            self.assertEqual(menubar.actions()[0].menuRole(), QtGui.QAction.MenuRole.NoRole)
        else:
            self.assertFalse(menubar.isNativeMenuBar())
        self.assertTrue(actions)
        self.assertTrue(all(action.menuRole() == QtGui.QAction.MenuRole.NoRole for action in actions))
        window.window.close()

    def test_gui_main_window_can_force_in_window_menu_bar(self):
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from eegprep.functions.guifunc.main_window import build_main_window

        window = build_main_window(EEGPrepSession(), all_menus=True, native_menu_bar=False)

        self.assertFalse(window.window.menuBar().isNativeMenuBar())
        self.assertEqual(
            [action.text() for action in window.window.menuBar().actions()],
            ["File", "Edit", "Tools", "Plot", "Study", "Datasets", "Help"],
        )
        window.window.close()

    def test_gui_main_window_reapplies_branding_after_menu_action(self):
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from eegprep.functions.guifunc.main_window import build_main_window

        class FakeDispatcher:
            def __init__(self):
                self.actions = []

            def dispatch_gui(self, action_id, parent):
                self.actions.append((action_id, parent))

        window = build_main_window(EEGPrepSession(), all_menus=False)
        dispatcher = FakeDispatcher()
        branding_calls = []
        window.dispatcher = dispatcher
        window._apply_application_branding = lambda: branding_calls.append("branding")

        window._dispatch_menu_action("pop_loadset")

        self.assertEqual(dispatcher.actions, [("pop_loadset", window.window)])
        self.assertEqual(branding_calls, ["branding"])
        window.window.close()
