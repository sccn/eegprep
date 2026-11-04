import ast
import inspect
import os
import sys
import textwrap
import unittest
from unittest import mock

import numpy as np
import pytest

from eegprep.functions.guifunc.eeglab_menu import eeglab_menus
from eegprep.functions.guifunc.menu_actions import (
    IMPLEMENTED_ACTIONS,
    MenuActionDispatcher,
)
from eegprep.functions.guifunc.long_task import LongTaskHandle
from eegprep.functions.guifunc.menu_spec import menu_enabled
from eegprep.functions.guifunc.session import EEGPrepSession


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

    def test_implemented_actions_registry_matches_dispatch_routing(self):
        # IMPLEMENTED_ACTIONS gates whether a menu item is shown enabled. An entry
        # with no dispatch arm would render enabled yet fall through to the
        # show_coming_soon catch-all; a dispatch arm missing from the set would be
        # classified "unknown"/placeholder. Keep the two in lockstep.
        routed = _dispatch_routed_actions()
        self.assertEqual(
            sorted(IMPLEMENTED_ACTIONS - routed),
            [],
            "IMPLEMENTED_ACTIONS entries with no dispatch handler (enabled menu items that no-op)",
        )
        self.assertEqual(
            sorted(routed - IMPLEMENTED_ACTIONS),
            [],
            "dispatch handlers missing from IMPLEMENTED_ACTIONS (classified unknown/placeholder)",
        )


def _dispatch_routed_actions():
    """Base action names routed to a real handler by ``MenuActionDispatcher.dispatch``."""
    import eegprep.functions.guifunc.menu_actions as menu_actions_module

    source = textwrap.dedent(inspect.getsource(MenuActionDispatcher.dispatch))
    module_sets = {name: value for name, value in vars(menu_actions_module).items() if isinstance(value, set)}
    routed: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not (isinstance(left, ast.Name) and left.id in ("base", "action")):
            continue
        operator = node.ops[0]
        comparator = node.comparators[0]
        if isinstance(operator, ast.Eq) and isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
            routed.add(comparator.value)
        elif isinstance(operator, ast.In):
            if isinstance(comparator, (ast.Set, ast.List, ast.Tuple)):
                routed.update(
                    element.value
                    for element in comparator.elts
                    if isinstance(element, ast.Constant) and isinstance(element.value, str)
                )
            elif isinstance(comparator, ast.Name) and comparator.id in module_sets:
                routed.update(module_sets[comparator.id])
    return routed


class EEGPrepSessionTests(unittest.TestCase):
    def test_session_delete_current_keeps_dataset_numbers_and_reuses_slot(self):
        session = EEGPrepSession()
        for name in ("first", "second", "third"):
            eeg = _demo_eeg()
            eeg["setname"] = name
            session.store_current(eeg, new=True)
        session.retrieve(2)

        session.delete_current()

        self.assertEqual(session.ALLEEG[1], {})
        self.assertEqual(session.ALLEEG[2]["setname"], "third")
        self.assertEqual(session.CURRENTSET, [3])
        self.assertEqual([index for index, _label, _selected in session.dataset_summaries()], [1, 3])

        fourth = _demo_eeg()
        fourth["setname"] = "fourth"
        session.store_current(fourth, new=True)

        self.assertEqual(session.CURRENTSET, [2])  # new datasets fill the lowest empty slot
        self.assertEqual(session.ALLEEG[1]["setname"], "fourth")

    def test_session_study_action_keeps_the_selected_dataset(self):
        from eegprep.functions.studyfunc.pop_study import pop_study

        session = EEGPrepSession()
        for index, name in enumerate(("first", "second", "third"), start=1):
            eeg = _demo_eeg()
            eeg["setname"] = name
            eeg["subject"] = f"S0{index}"
            session.store_current(eeg, new=True)
        session.retrieve(2)
        session.delete_current()
        self.assertEqual(session.EEG["setname"], "third")

        study, alleeg = pop_study(None, session.ALLEEG, name="Gaps")
        session.set_study(study, alleeg)

        # A STUDY needs contiguous numbers, so the workspace compacts, but the
        # selection follows the dataset it was on rather than its old number.
        self.assertEqual([eeg["setname"] for eeg in session.ALLEEG], ["first", "third"])
        self.assertEqual(session.EEG["setname"], "third")
        self.assertEqual(session.CURRENTSET, [2])

    def test_session_summary_handles_a_deleted_slot_in_study_mode(self):
        from eegprep.functions.guifunc.main_window import _summary_for_session

        session = EEGPrepSession()
        for name in ("first", "second"):
            eeg = _demo_eeg()
            eeg["setname"] = name
            session.store_current(eeg, new=True)
        session.retrieve(1)
        session.delete_current()
        session.STUDY = {"name": "demo", "datasetinfo": []}
        session.CURRENTSTUDY = 1

        _title, _subtitle, rows = _summary_for_session(session)

        # The emptied slot is not a dataset, so it must not reach the summary helpers.
        self.assertEqual(dict(rows)["Channels per frame"], str(session.ALLEEG[1]["nbchan"]))

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
    def test_gui_dispatch_shows_warning_for_action_errors(self):
        dispatcher = MenuActionDispatcher(EEGPrepSession())

        with (
            mock.patch.object(dispatcher, "dispatch", side_effect=ValueError("bad input")),
            mock.patch.object(dispatcher, "_warn") as warn,
        ):
            dispatcher.dispatch_gui("pop_adjustevents", parent="window")

        warn.assert_called_once_with("window", "bad input")

    def test_merge_datasets_warns_when_only_one_dataset_survives_a_delete(self):
        session = EEGPrepSession()
        for name in ("first", "second"):
            eeg = _demo_eeg()
            eeg["setname"] = name
            session.store_current(eeg, new=True)
        session.retrieve(1)
        session.delete_current()
        dispatcher = MenuActionDispatcher(session)

        with mock.patch.object(dispatcher, "_warn") as warn:
            dispatcher._merge_datasets("window")

        # ALLEEG still has two slots, but one of them is an emptied slot, not a dataset.
        self.assertEqual(len(session.ALLEEG), 2)
        warn.assert_called_once_with("window", "Load at least two datasets before merging")

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

    def test_pop_eegplot_menu_variants_use_eeglab_parity_arguments(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(epoched=True, ica=True), new=True)
        dispatcher = MenuActionDispatcher(session)
        calls = []

        def fake_pop_eegplot(
            eeg,
            *,
            icacomp=1,
            superpose=0,
            reject=1,
            command_callback=None,
            return_com=False,
        ):
            self.assertIs(eeg, session.EEG)
            self.assertTrue(callable(command_callback))
            self.assertTrue(return_com)
            calls.append((icacomp, superpose, reject))
            return eeg, f"pop_eegplot(EEG, {icacomp}, {superpose}, {reject})"

        with mock.patch("eegprep.functions.popfunc.pop_eegplot.pop_eegplot", side_effect=fake_pop_eegplot):
            for action in (
                "pop_eegplot:data",
                "pop_eegplot:channels",
                "pop_eegplot:components",
                "pop_eegplot:reject_data",
                "pop_eegplot:reject_ica",
            ):
                dispatcher.dispatch(action)

        self.assertEqual(calls, [(1, 0, 1), (1, 1, 1), (0, 1, 1), (1, 0, 1), (0, 0, 1)])
        self.assertEqual(
            session.ALLCOM,
            [
                "pop_eegplot(EEG, 1, 0, 1)",
                "pop_eegplot(EEG, 1, 1, 1)",
                "pop_eegplot(EEG, 0, 1, 1)",
                "pop_eegplot(EEG, 1, 0, 1)",
                "pop_eegplot(EEG, 0, 0, 1)",
            ],
        )

    def test_pop_eegplot_accept_updates_original_dataset_after_selection_changes(self):
        session = EEGPrepSession()
        first = _demo_eeg()
        second = _demo_eeg()
        second["setname"] = "second"
        session.store_current(first, new=True)
        session.store_current(second, new=True)
        session.retrieve(1)
        dispatcher = MenuActionDispatcher(session)
        captured = {}
        command = "pop_eegplot(EEG, 1, 0, 1)"

        def fake_pop_eegplot(eeg, *, command_callback=None, return_com=False, **_kwargs):
            captured["callback"] = command_callback
            self.assertEqual(eeg["setname"], "demo")
            self.assertTrue(return_com)
            return eeg, command

        with mock.patch("eegprep.functions.popfunc.pop_eegplot.pop_eegplot", side_effect=fake_pop_eegplot):
            dispatcher.dispatch("pop_eegplot:data")

        session.retrieve(2)
        out = dict(first, setname="accepted first")
        captured["callback"](out, command)

        self.assertEqual(session.CURRENTSET, [1])
        self.assertEqual(session.EEG["setname"], "accepted first")
        self.assertEqual(session.ALLEEG[0]["setname"], "accepted first")
        self.assertEqual(session.ALLEEG[1]["setname"], "second")
        self.assertEqual(session.ALLCOM, [command])

    def test_pop_eegplot_menu_enabled_states_follow_dataset_and_ica_status(self):
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from eegprep.functions.guifunc.main_window import build_main_window

        continuous_no_ica = EEGPrepSession()
        continuous_no_ica.store_current(_demo_eeg(ica=False), new=True)
        window = build_main_window(continuous_no_ica, all_menus=True)
        actions = {action.data(): action for action in _qt_actions(window.window.menuBar().actions()) if action.data()}

        self.assertTrue(actions["pop_eegplot:data"].isEnabled())
        self.assertTrue(actions["pop_eegplot:channels"].isEnabled())
        self.assertFalse(actions["pop_eegplot:components"].isEnabled())
        self.assertFalse(actions["pop_eegplot:reject_data"].isEnabled())
        self.assertFalse(actions["pop_eegplot:reject_ica"].isEnabled())
        window.window.close()

        epoched_no_ica = EEGPrepSession()
        epoched_no_ica.store_current(_demo_eeg(epoched=True, ica=False), new=True)
        window = build_main_window(epoched_no_ica, all_menus=True)
        actions = {action.data(): action for action in _qt_actions(window.window.menuBar().actions()) if action.data()}

        self.assertTrue(actions["pop_eegplot:reject_data"].isEnabled())
        self.assertFalse(actions["pop_eegplot:reject_ica"].isEnabled())
        window.window.close()

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

    def test_cancelled_newset_commit_does_not_pollute_dataset_history(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        original_history = session.EEG.get("history", "")
        original_allcom = list(session.ALLCOM)
        dispatcher = MenuActionDispatcher(session)
        select_command = "EEG = pop_select(EEG, 'point', [1 20]);"

        with (
            mock.patch(
                "eegprep.functions.popfunc.pop_select.pop_select",
                return_value=(session.EEG, select_command),
            ),
            # User cancels the pop_newset dialog, so no dataset is committed.
            mock.patch(
                "eegprep.functions.guifunc.menu_actions.pop_newset",
                return_value=(session.ALLEEG, session.EEG, [1], ""),
            ),
        ):
            dispatcher.dispatch("pop_select", parent=object())

        self.assertEqual(session.EEG.get("history", ""), original_history)
        self.assertNotIn(select_command, str(session.EEG.get("history", "")))
        self.assertEqual(session.ALLCOM, original_allcom)

    def test_headplot_menu_action_commits_spline_file_through_session(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        events = []
        session.add_change_listener(lambda _session: events.append("changed"))
        dispatcher = MenuActionDispatcher(session)
        headplot_command = "pop_headplot(EEG, 1, [0], '', [1 1]);"

        def fake_pop_headplot(eeg, *, typeplot, return_com):
            eeg["splinefile"] = "/tmp/demo.spl"
            return (["figure"], headplot_command)

        with mock.patch("eegprep.functions.popfunc.pop_headplot.pop_headplot", side_effect=fake_pop_headplot):
            dispatcher.dispatch("pop_headplot")

        self.assertEqual(session.EEG["splinefile"], "/tmp/demo.spl")
        self.assertEqual(session.ALLEEG[0]["splinefile"], "/tmp/demo.spl")
        self.assertEqual(session.ALLCOM[-1], headplot_command)
        self.assertTrue(events, "headplot commit must notify session listeners")
        # Committing through store_current records the edit in the dataset
        # history; a history-only path would leave the dataset .history untouched.
        self.assertIn(headplot_command, str(session.EEG["history"]))
        self.assertIn(headplot_command, str(session.ALLEEG[0]["history"]))

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
        newset_actions = {
            "pop_clean_rawdata",
            "pop_eegfilt",
            "pop_eegfiltnew",
            "pop_epoch",
            "pop_firma",
            "pop_firpm",
            "pop_firws",
            "pop_resample",
            "pop_rmbase",
            "pop_rmdat",
            "pop_select",
            "pop_selectevent",
        }
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
                if action in newset_actions:
                    self.assertEqual(session.ALLCOM[-2], f"EEG = {action}(EEG);")
                    self.assertEqual(
                        session.ALLCOM[-1],
                        "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'overwrite', 'on');",
                    )
                    self.assertIn(f"EEG = {action}(EEG);", session.EEG["history"])
                else:
                    self.assertEqual(session.ALLCOM[-1], f"EEG = {action}(EEG);")

    def test_gui_pop_runica_runs_ica_in_long_task_before_committing_result(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        refresh = mock.Mock()
        dispatcher = MenuActionDispatcher(session, refresh=refresh)
        output = dict(session.EEG, setname="ica")
        options = {
            "icatype": "runica",
            "options": {"extended": 1, "interrupt": "on"},
            "reorder": "on",
            "chanind": None,
            "dataset": None,
            "concatenate": "off",
            "concatcond": "off",
        }
        captured = {}
        handle = LongTaskHandle(thread=object(), worker=object(), dialog=object())
        events = []
        original_selection = session.EEG
        session.add_gui_action_listener(lambda event, action: events.append((event, action)))

        def fake_run_long_task(**kwargs):
            captured.update(kwargs)
            return handle

        with (
            mock.patch("eegprep.functions.popfunc.pop_runica.pop_runica_gui_options", return_value=options),
            mock.patch(
                "eegprep.functions.popfunc.pop_runica.pop_runica",
                return_value=(output, "EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'interrupt', 'on');"),
            ) as pop_func,
            mock.patch("eegprep.functions.guifunc.menu_actions.run_long_task", side_effect=fake_run_long_task),
        ):
            dispatcher.dispatch("pop_runica", parent=object())
            self.assertEqual(session.EEG["setname"], "demo")

            result = captured["task"]()
            captured["on_success"](result)
            captured["on_finished"](handle)

        pop_func.assert_called_once_with(original_selection, gui=False, return_com=True, **options)
        self.assertEqual(session.EEG["setname"], "ica")
        self.assertEqual(session.ALLEEG[0]["setname"], "ica")
        self.assertEqual(
            session.ALLCOM[-1],
            "EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'interrupt', 'on');",
        )
        refresh.assert_called_once()
        self.assertEqual(events, [("begin", "pop_runica"), ("end", "pop_runica")])
        self.assertEqual(dispatcher._long_tasks, [])

    def test_gui_transform_action_can_commit_processed_dataset_as_new_set(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        dispatcher = MenuActionDispatcher(session)
        processed = dict(session.EEG, setname="resampled")
        newset_command = (
            "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'setname', 'resampled', 'overwrite', 'off');"
        )

        with (
            mock.patch(
                "eegprep.functions.popfunc.pop_resample.pop_resample",
                return_value=(processed, "EEG = pop_resample( EEG, 64);"),
            ),
            mock.patch(
                "eegprep.functions.guifunc.menu_actions.pop_newset",
                return_value=([session.ALLEEG[0], processed], processed, 2, newset_command),
            ) as newset,
        ):
            dispatcher.dispatch("pop_resample", parent=object())

        newset.assert_called_once()
        self.assertEqual(newset.call_args.args[:3], ([session.ALLEEG[0]], processed, 1))
        self.assertEqual(newset.call_args.args[3:], ("gui", "on", "overwrite", "off"))
        self.assertEqual(len(session.ALLEEG), 2)
        self.assertEqual(session.CURRENTSET, [2])
        self.assertEqual(session.EEG["setname"], "resampled")
        self.assertEqual(session.ALLCOM[-2:], ["EEG = pop_resample( EEG, 64);", newset_command])

    def test_gui_transform_action_overwrites_multiple_selected_datasets(self):
        session = EEGPrepSession()
        first = _demo_eeg()
        second = _demo_eeg()
        second["setname"] = "second"
        session.store_current(first, new=True)
        session.store_current(second, new=True)
        session.retrieve([1, 2])
        dispatcher = MenuActionDispatcher(session)
        original_alleeg = list(session.ALLEEG)
        processed = [
            dict(original_alleeg[0], setname="first resampled"),
            dict(original_alleeg[1], setname="second resampled"),
        ]
        newset_command = "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'overwrite', 'on');"

        with (
            mock.patch(
                "eegprep.functions.popfunc.pop_resample.pop_resample",
                return_value=(processed, "ALLEEG = pop_resample(ALLEEG, 64);"),
            ),
            mock.patch(
                "eegprep.functions.guifunc.menu_actions.pop_newset",
                return_value=(processed, processed, [1, 2], newset_command),
            ) as newset,
        ):
            dispatcher.dispatch("pop_resample", parent=object())

        newset.assert_called_once()
        self.assertEqual(newset.call_args.args[:3], (original_alleeg, processed, [1, 2]))
        self.assertEqual(newset.call_args.args[3:], ("gui", "off", "overwrite", "on"))
        self.assertEqual(session.CURRENTSET, [1, 2])
        self.assertEqual(session.ALLEEG[0]["setname"], "first resampled")
        self.assertEqual(session.ALLEEG[1]["setname"], "second resampled")
        self.assertEqual(session.ALLCOM[-2:], ["ALLEEG = pop_resample(ALLEEG, 64);", newset_command])
        self.assertIn("ALLEEG = pop_resample(ALLEEG, 64);", session.ALLEEG[0]["history"])
        self.assertIn("ALLEEG = pop_resample(ALLEEG, 64);", session.ALLEEG[1]["history"])

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

    def test_file_menu_brainvision_dispatch_uses_pop_loadbv(self):
        session = EEGPrepSession()
        dispatcher = MenuActionDispatcher(session)
        imported = _demo_eeg()
        imported["setname"] = "brainvision"
        command = "EEG = pop_loadbv('/tmp', 'recording.vhdr');"
        qt_widgets = _fake_qt_widgets(open_file="/tmp/recording.vhdr")

        with (
            mock.patch("eegprep.functions.guifunc.menu_actions._require_qt_widgets", return_value=qt_widgets),
            mock.patch(
                "eegprep.functions.popfunc.pop_loadbv.pop_loadbv",
                return_value=(imported, command),
            ) as loadbv,
        ):
            dispatcher.dispatch("pop_fileio_brainvision")

        loadbv.assert_called_once_with("/tmp/recording.vhdr", return_com=True)
        self.assertEqual(session.EEG["setname"], "brainvision")
        self.assertEqual(session.CURRENTSET, [1])
        self.assertEqual(session.ALLCOM[-1], command)

    def test_preferences_dialog_updates_menu_and_file_dialog_options(self):
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        parent = QtWidgets.QWidget()
        original_options = dict(EEG_OPTIONS)
        session = EEGPrepSession()
        dispatcher = MenuActionDispatcher(session)

        def accept_with_both_options(dialog):
            checkboxes = {checkbox.text(): checkbox for checkbox in dialog.findChildren(QtWidgets.QCheckBox)}
            self.assertEqual(
                set(checkboxes),
                {"Show advanced legacy menu items?", "Use Native OS Dialogs"},
            )
            for checkbox in checkboxes.values():
                checkbox.setChecked(True)
            return QtWidgets.QDialog.Accepted

        try:
            EEG_OPTIONS["option_allmenus"] = 0
            EEG_OPTIONS["option_native_dialogs"] = 0
            with (
                mock.patch(
                    "eegprep.functions.guifunc.menu_actions._require_qt_widgets",
                    return_value=QtWidgets,
                ),
                mock.patch.object(QtWidgets.QDialog, "exec", accept_with_both_options),
                mock.patch.object(dispatcher, "_info"),
            ):
                dispatcher._edit_options(parent)

            self.assertEqual(EEG_OPTIONS["option_allmenus"], 1)
            self.assertEqual(EEG_OPTIONS["option_native_dialogs"], 1)
            self.assertEqual(session.ALLCOM, ["LASTCOM = pop_editoptions();"])
        finally:
            EEG_OPTIONS.clear()
            EEG_OPTIONS.update(original_options)
            parent.close()
            app.processEvents()

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
            self.assertIn("ALLCOM", namespace)
            self.assertIn("LASTCOM", namespace)
            self.assertIn("CURRENTSTUDY", namespace)
            namespace["CURRENTSET"] = 2
            namespace["ALLCOM"].append("EEG = script_command(EEG);")
            namespace["LASTCOM"] = "EEG = script_command(EEG);"
            namespace["STUDY"] = {"name": "script study"}
            namespace["CURRENTSTUDY"] = 1
            return "LASTCOM = pop_runscript('/tmp/script.py');"

        with (
            mock.patch("eegprep.functions.guifunc.menu_actions._require_qt_widgets", return_value=qt_widgets),
            mock.patch("eegprep.functions.popfunc.pop_runscript.pop_runscript", side_effect=fake_runscript),
        ):
            dispatcher.dispatch("pop_runscript")

        self.assertEqual(session.CURRENTSET, [2])
        self.assertEqual(session.STUDY["name"], "script study")
        self.assertEqual(session.CURRENTSTUDY, 1)
        self.assertEqual(
            session.ALLCOM,
            ["EEG = script_command(EEG);", "LASTCOM = pop_runscript('/tmp/script.py');"],
        )
        self.assertEqual(session.LASTCOM, "LASTCOM = pop_runscript('/tmp/script.py');")

    def test_file_menu_runscript_clear_currentset_resets_eeg(self):
        session = EEGPrepSession()
        session.store_current(_demo_eeg(), new=True)
        dispatcher = MenuActionDispatcher(session)
        qt_widgets = _fake_qt_widgets(open_file="/tmp/script.py")

        def fake_runscript(_filename, namespace):
            namespace["CURRENTSET"] = 0
            return "LASTCOM = pop_runscript('/tmp/script.py');"

        with (
            mock.patch("eegprep.functions.guifunc.menu_actions._require_qt_widgets", return_value=qt_widgets),
            mock.patch("eegprep.functions.popfunc.pop_runscript.pop_runscript", side_effect=fake_runscript),
        ):
            dispatcher.dispatch("pop_runscript")

        self.assertEqual(session.CURRENTSET, [])
        self.assertIsInstance(session.EEG, dict)
        self.assertEqual(session.EEG.get("setname"), "")
        self.assertEqual(session.EEG["data"].size, 0)


class QtMainWindowTests(unittest.TestCase):
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
