import os
import unittest

import numpy as np
import pytest

from eegprep.functions.guifunc.eeglab_menu import eeglab_menus, menu_actions
from eegprep.functions.guifunc.menu_actions import action_kind
from eegprep.functions.guifunc.menu_placeholders import is_placeholder_action
from eegprep.functions.guifunc.menu_spec import menu_enabled
from eegprep.functions.guifunc.session import EEGPrepSession


def _labels(items):
    return [item.label for item in items]


def _child(menu, label):
    items = menu.children if hasattr(menu, "children") else menu
    for item in items:
        if item.label == label:
            return item
    raise AssertionError(f"missing menu item {label!r}")


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
        import_functions = _child(import_menu.children, "Using EEGLAB functions and plugins")
        export_menu = _child(file_menu.children, "Export")

        self.assertIn("From BIDS folder structure", _labels(import_functions.children))
        self.assertIn("Import Magstim/EGI .mff file", _labels(import_functions.children))
        self.assertIn("To BIDS folder structure", _labels(export_menu.children))
        self.assertEqual(_labels(file_menu.children)[4], "BIDS tools")

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

    def test_all_menu_actions_are_classified(self):
        actions = menu_actions(eeglab_menus(all_menus=True))

        self.assertIn("pop_reref", actions)
        self.assertEqual(action_kind("pop_reref"), "implemented")
        self.assertEqual(action_kind("pop_clean_rawdata"), "placeholder")
        self.assertEqual(action_kind("pop_exportbids"), "placeholder")
        self.assertEqual(action_kind("select_multiple_datasets"), "placeholder")
        self.assertTrue(all(action_kind(action) in {"implemented", "placeholder"} for action in actions))
        self.assertTrue(
            all(action_kind(action) == "implemented" or is_placeholder_action(action) for action in actions)
        )


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


class QtMainWindowTests(unittest.TestCase):
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
