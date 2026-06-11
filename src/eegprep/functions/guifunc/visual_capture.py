"""Screenshot capture entrypoint for EEGPrep visual parity cases."""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.guifunc.coregister import prepare_coregister_display
from eegprep.functions.guifunc.qt import QtDialogRenderer
from eegprep.functions.guifunc.listdlg2 import build_listdlg2_dialog
from eegprep.functions.guifunc.main_window import build_main_window
from eegprep.functions.guifunc.pophelp import pophelp
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.popfunc.pop_adjustevents import pop_adjustevents_dialog_spec
from eegprep.functions.popfunc.pop_autorej import pop_autorej_dialog_spec
from eegprep.functions.popfunc.pop_chanedit import pop_chanedit_dialog_spec
from eegprep.functions.popfunc.pop_chansel import pop_chansel_display_values
from eegprep.functions.popfunc.pop_comperp import pop_comperp_dialog_spec
from eegprep.functions.popfunc.pop_comments import pop_comments_dialog_spec
from eegprep.functions.popfunc.pop_copyset import pop_copyset_dialog_spec
from eegprep.functions.popfunc.pop_eegthresh import pop_eegthresh_dialog_spec
from eegprep.functions.popfunc.pop_editset import pop_editset_dialog_spec
from eegprep.functions.popfunc.pop_eegfilt import pop_eegfilt_dialog_spec
from eegprep.functions.popfunc.pop_epoch import pop_epoch_dialog_spec
from eegprep.functions.popfunc.pop_editeventfield import pop_editeventfield_dialog_spec
from eegprep.functions.popfunc.pop_editeventvals import pop_editeventvals_dialog_spec
from eegprep.functions.popfunc.pop_envtopo import pop_envtopo_dialog_spec
from eegprep.functions.popfunc.pop_erpimage import pop_erpimage_dialog_spec
from eegprep.functions.popfunc.pop_headplot import pop_headplot_dialog_spec
from eegprep.functions.popfunc.pop_interp import pop_interp_dialog_spec
from eegprep.functions.popfunc.pop_jointprob import pop_jointprob_dialog_spec
from eegprep.functions.popfunc.pop_mergeset import pop_mergeset_dialog_spec
from eegprep.functions.popfunc.pop_newset import pop_newset_dialog_spec
from eegprep.functions.popfunc.pop_eventstat import pop_eventstat_dialog_spec
from eegprep.functions.popfunc.pop_plotdata import pop_plotdata_dialog_spec
from eegprep.functions.popfunc.pop_plottopo import pop_plottopo_dialog_spec
from eegprep.functions.popfunc.pop_prop import pop_prop_dialog_spec
from eegprep.functions.popfunc.pop_newcrossf import pop_newcrossf_dialog_spec
from eegprep.functions.popfunc.pop_newtimef import pop_newtimef_dialog_spec
from eegprep.functions.popfunc.pop_reref import pop_reref_dialog_spec
from eegprep.functions.popfunc.pop_rejchan import pop_rejchan_dialog_spec
from eegprep.functions.popfunc.pop_rejcont import pop_rejcont_dialog_spec
from eegprep.functions.popfunc.pop_rejkurt import pop_rejkurt_dialog_spec
from eegprep.functions.popfunc.pop_rejmenu import pop_rejmenu_dialog_spec
from eegprep.functions.popfunc.pop_rejspec import pop_rejspec_dialog_spec
from eegprep.functions.popfunc.pop_rejtrend import pop_rejtrend_dialog_spec
from eegprep.functions.popfunc.pop_resample import pop_resample_dialog_spec
from eegprep.functions.popfunc.pop_rmdat import pop_rmdat_dialog_spec
from eegprep.functions.popfunc.pop_rmbase import pop_rmbase_dialog_spec
from eegprep.functions.popfunc.pop_runica import pop_runica_dialog_spec
from eegprep.functions.popfunc.pop_select import pop_select_dialog_spec
from eegprep.functions.popfunc.pop_selectevent import pop_selectevent_dialog_spec
from eegprep.functions.popfunc.pop_signalstat import pop_signalstat_dialog_spec
from eegprep.functions.popfunc.pop_selectcomps import pop_selectcomps_dialog_spec
from eegprep.functions.popfunc.pop_spectopo import pop_spectopo_dialog_spec
from eegprep.functions.popfunc.pop_subcomp import pop_subcomp_dialog_spec
from eegprep.functions.popfunc.pop_timtopo import pop_timtopo_dialog_spec
from eegprep.functions.popfunc.pop_topoplot import pop_topoplot_dialog_spec
from eegprep.functions.sigprocfunc.eegplot import build_eegplot_model
from eegprep.functions.studyfunc.pop_clust import pop_clust_dialog_spec
from eegprep.functions.studyfunc.pop_clustedit import pop_clustedit_dialog_spec
from eegprep.functions.studyfunc.pop_chanplot import pop_chanplot_dialog_spec
from eegprep.functions.studyfunc.pop_preclust import pop_preclust_dialog_spec
from eegprep.functions.studyfunc.pop_precomp import pop_precomp_dialog_spec
from eegprep.functions.studyfunc.pop_study import pop_study_dialog_spec
from eegprep.functions.studyfunc.pop_studydesign import pop_studydesign_dialog_spec
from eegprep.functions.studyfunc.std_checkset import std_checkset
from eegprep.plugins.ICLabel.pop_icflag import pop_icflag_dialog_spec
from eegprep.plugins.ICLabel.pop_iclabel import pop_iclabel_dialog_spec
from eegprep.plugins.ICLabel.pop_prop_extended import pop_prop_extended
from eegprep.plugins.ICLabel.pop_viewprops import pop_viewprops_dialog_spec
from eegprep.plugins.clean_rawdata.pop_clean_rawdata import pop_clean_rawdata_dialog_spec
from eegprep.plugins.dipfit.pop_dipfit_gridsearch import pop_dipfit_gridsearch_dialog_spec
from eegprep.plugins.dipfit.pop_dipfit_headmodel import pop_dipfit_headmodel_dialog_spec
from eegprep.plugins.dipfit.pop_dipfit_loreta import pop_dipfit_loreta_dialog_spec
from eegprep.plugins.dipfit.pop_dipfit_nonlinear import pop_dipfit_nonlinear_dialog_spec
from eegprep.plugins.dipfit.pop_dipfit_settings import pop_dipfit_settings_dialog_spec
from eegprep.plugins.dipfit.pop_dipplot import pop_dipplot_dialog_spec
from eegprep.plugins.dipfit.pop_leadfield import pop_leadfield_dialog_spec
from eegprep.plugins.dipfit.pop_multifit import pop_multifit_dialog_spec
from eegprep.plugins.firfilt.pop_eegfiltnew import pop_eegfiltnew_dialog_spec
from eegprep.plugins.firfilt.pop_firma import pop_firma_dialog_spec
from eegprep.plugins.firfilt.pop_firpm import pop_firpm_dialog_spec
from eegprep.plugins.firfilt.pop_firpmord import pop_firpmord_dialog_spec
from eegprep.plugins.firfilt.pop_firws import pop_firws_dialog_spec
from eegprep.plugins.firfilt.pop_firwsord import pop_firwsord_dialog_spec
from eegprep.plugins.firfilt.pop_kaiserbeta import pop_kaiserbeta_dialog_spec
from eegprep.plugins.firfilt.pop_xfirws import pop_xfirws_dialog_spec


def _demo_eeg() -> dict:
    return {
        "data": np.zeros((1, 1000), dtype=np.float32),
        "nbchan": 1,
        "pnts": 1000,
        "trials": 1,
        "srate": 250.0,
        "xmin": 0.0,
        "xmax": 3.996,
        "event": [
            {"type": "stim", "latency": 100.0},
            {"type": "resp", "latency": 350.0},
            {"type": "boundary", "latency": 500.5, "duration": 20.0},
        ],
    }


def _demo_reref_eeg() -> dict:
    chanlocs = [
        {
            "labels": "Fp1",
            "ref": "common",
            "theta": -18.0,
            "radius": 0.42,
            "X": -0.25,
            "Y": 0.75,
            "Z": 0.55,
            "type": "EEG",
        },
        {
            "labels": "Fp2",
            "ref": "common",
            "theta": 18.0,
            "radius": 0.42,
            "X": 0.25,
            "Y": 0.75,
            "Z": 0.55,
            "type": "EEG",
        },
        {
            "labels": "Cz",
            "ref": "common",
            "theta": 0.0,
            "radius": 0.0,
            "X": 0.0,
            "Y": 0.0,
            "Z": 1.0,
            "type": "EEG",
        },
        {
            "labels": "Oz",
            "ref": "common",
            "theta": 180.0,
            "radius": 0.42,
            "X": 0.0,
            "Y": -0.8,
            "Z": 0.55,
            "type": "EEG",
        },
    ]
    return {
        "data": np.zeros((4, 1000), dtype=np.float32),
        "nbchan": 4,
        "pnts": 1000,
        "trials": 1,
        "srate": 250.0,
        "xmin": 0.0,
        "xmax": 3.996,
        "chanlocs": chanlocs,
        "chaninfo": {
            "nodatchans": [{"labels": "M1", "theta": -90.0, "radius": 0.5, "type": "REF"}],
            "removedchans": [
                {
                    "labels": "Pz",
                    "theta": 180.0,
                    "radius": 0.25,
                    "X": 0.0,
                    "Y": -0.4,
                    "Z": 0.8,
                    "type": "EEG",
                }
            ],
        },
        "ref": "common",
    }


def _demo_interp_eeg(*, epoched: bool = False, removed: bool = False) -> dict:
    chanlocs = [
        {"labels": "Fp1", "theta": -18.0, "radius": 0.42, "X": -0.25, "Y": 0.75, "Z": 0.55},
        {"labels": "Fp2", "theta": 18.0, "radius": 0.42, "X": 0.25, "Y": 0.75, "Z": 0.55},
        {"labels": "Cz", "theta": 0.0, "radius": 0.0, "X": 0.0, "Y": 0.0, "Z": 1.0},
        {"labels": "Oz", "theta": 180.0, "radius": 0.42, "X": 0.0, "Y": -0.8, "Z": 0.55},
    ]
    eeg = {
        "data": np.zeros((4, 500, 2), dtype=np.float32) if epoched else np.zeros((4, 1000), dtype=np.float32),
        "nbchan": 4,
        "pnts": 500 if epoched else 1000,
        "trials": 2 if epoched else 1,
        "srate": 250.0,
        "xmin": 0.0,
        "xmax": 1.996 if epoched else 3.996,
        "chanlocs": chanlocs,
        "epoch": [{"event": [1]}, {"event": [2]}] if epoched else [],
        "chaninfo": {},
    }
    if removed:
        eeg["chaninfo"]["removedchans"] = [
            {"labels": "M1", "theta": -90.0, "radius": 0.5, "X": -0.8, "Y": 0.0, "Z": 0.4},
            {"labels": "M2", "theta": 90.0, "radius": 0.5, "X": 0.8, "Y": 0.0, "Z": 0.4},
        ]
    return eeg


def _demo_main_eeg(*, epoched: bool = False, setname: str = "menu demo") -> dict:
    eeg = _demo_reref_eeg()
    if epoched:
        eeg.update(
            {
                "data": np.zeros((4, 250, 2), dtype=np.float32),
                "pnts": 250,
                "trials": 2,
                "xmin": -0.2,
                "xmax": 0.796,
                "times": np.linspace(-200, 796, 250),
                "epoch": [{"event": [1]}, {"event": [2]}],
            }
        )
    eeg.update(
        {
            "setname": setname,
            "filename": f"{setname.replace(' ', '_')}.set",
            "filepath": "/tmp",
            "event": [
                {"type": "stim", "latency": 100.0, "duration": 0.0},
                {"type": "resp", "latency": 350.0, "duration": 0.0},
            ],
            "urevent": [],
            "history": "",
            "icaweights": np.eye(4, dtype=np.float32),
            "icasphere": np.eye(4, dtype=np.float32),
            "icawinv": np.eye(4, dtype=np.float32),
            "icachansind": np.arange(4),
            "icaact": np.zeros((4, 250, 2), dtype=np.float32) if epoched else np.zeros((4, 1000), dtype=np.float32),
        }
    )
    return eeg


def _demo_iclabel_dashboard_eeg() -> dict:
    eeg = _demo_main_eeg(setname="viewprops demo")
    samples = np.linspace(0.0, 1.0, int(eeg["pnts"]), dtype=float)
    eeg["data"] = np.vstack([np.sin(2.0 * np.pi * (index + 1) * samples) for index in range(4)]).astype(np.float32)
    eeg["icaact"] = np.vstack([np.cos(2.0 * np.pi * (index + 1) * samples) for index in range(4)]).astype(np.float32)
    eeg["times"] = np.linspace(0.0, float(eeg["xmax"]) * 1000.0, int(eeg["pnts"]))
    eeg["xmin"] = 0.0
    eeg["reject"] = {"gcompreject": np.zeros(4, dtype=int)}
    eeg["etc"] = {
        "ic_classification": {
            "ICLabel": {
                "classifications": np.array(
                    [
                        [0.70, 0.10, 0.10, 0.03, 0.02, 0.03, 0.02],
                        [0.02, 0.94, 0.02, 0.01, 0.00, 0.00, 0.01],
                        [0.05, 0.02, 0.91, 0.01, 0.00, 0.00, 0.01],
                        [0.80, 0.05, 0.05, 0.02, 0.02, 0.03, 0.03],
                    ],
                    dtype=float,
                ),
                "classes": ["Brain", "Muscle", "Eye", "Heart", "Line Noise", "Channel Noise", "Other"],
            }
        }
    }
    eeg["dipfit"] = {
        "coordformat": "MNI",
        "model": [
            {"posxyz": [0, -20, 40], "momxyz": [1, 0, 0], "rv": 0.12, "component": 1},
            {"posxyz": [25, 10, 35], "momxyz": [0, 1, 0], "rv": 0.2, "component": 2},
            {"posxyz": [], "momxyz": [], "rv": 1.0, "component": 3},
            {"posxyz": [], "momxyz": [], "rv": 1.0, "component": 4},
        ],
    }
    return eeg


def _demo_eegbrowser_eeg(*, epoched: bool = False, events: bool = True) -> dict:
    channel_count = 8
    points = 250 if epoched else 1000
    trials = 3 if epoched else 1
    times = np.arange(points, dtype=float) / 250.0
    data = np.empty((channel_count, points, trials), dtype=np.float32)
    for channel in range(channel_count):
        for trial in range(trials):
            trial_component = 0.2 * np.cos(2 * np.pi * (trial + 1) * times) if epoched else 0.0
            data[channel, :, trial] = np.sin(2 * np.pi * (channel + 1) * times) + trial_component + (channel + 1) * 0.05
    if not epoched:
        data = data[:, :, 0]
    eeg_events = []
    if events:
        eeg_events = [
            {"type": "stim", "latency": 80.0, "duration": 0.0},
            {"type": "resp", "latency": 350.0, "duration": 0.0},
            {"type": "boundary", "latency": 610.0, "duration": 25.0},
        ]
    return {
        "data": data,
        "nbchan": channel_count,
        "pnts": points,
        "trials": trials,
        "srate": 250.0,
        "xmin": -0.2 if epoched else 0.0,
        "xmax": 0.796 if epoched else 3.996,
        "chanlocs": [{"labels": label} for label in ("Fp1", "Fp2", "Cz", "Pz", "O1", "O2", "T7", "T8")],
        "event": eeg_events,
        "setname": "eegbrowser demo",
    }


def _demo_eegbrowser_ica_eeg() -> dict:
    eeg = _demo_eegbrowser_eeg(events=False)
    channel_count = int(eeg["nbchan"])
    times = np.arange(int(eeg["pnts"]), dtype=float) / float(eeg["srate"])
    eeg["icaweights"] = np.eye(channel_count)
    eeg["icasphere"] = np.eye(channel_count)
    eeg["icawinv"] = np.eye(channel_count)
    eeg["icachansind"] = np.arange(channel_count)
    icaact = np.vstack(
        [np.cos(2 * np.pi * (component + 1) * times) + 0.08 * component for component in range(channel_count)]
    )
    eeg["data"] = icaact.copy()
    eeg["icaact"] = icaact
    return eeg


def _demo_eegbrowser_winrej(variant: str) -> np.ndarray | None:
    if variant in {"continuous_marked", "pop_eegplot_reject_data"}:
        return np.array(
            [
                [125, 260, 0.7, 1.0, 0.9, 1, 1, 1, 1, 1, 1, 1, 1],
                [340, 460, 1.0, 0.9, 0.9, 0, 1, 0, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )
    if variant in {"epoched_marked", "rejection_epochs"}:
        return np.array(
            [
                [250, 499, 0.7, 1.0, 0.9, 1, 1, 1, 1, 1, 1, 1, 1],
                [500, 749, 1.0, 0.9, 0.9, 0, 0, 1, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )
    if variant == "component_activity":
        return np.array(
            [
                [180, 310, 1.0, 0.9, 0.9, 1, 0, 0, 0, 0, 0, 0, 0],
                [520, 640, 0.7, 1.0, 0.9, 0, 1, 1, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )
    if variant == "rejcont_continuous":
        return np.array(
            [
                [160, 260, 0.0, 0.9, 0.0, 1, 1, 1, 1],
                [620, 760, 0.0, 0.9, 0.0, 1, 0, 1, 0],
            ],
            dtype=float,
        )
    return None


def _configure_main_window_session(session: EEGPrepSession, state: str) -> None:
    if state == "startup":
        return
    if state == "continuous":
        session.store_current(_demo_main_eeg(), new=True)
        return
    if state == "epoched":
        session.store_current(_demo_main_eeg(epoched=True, setname="menu epoched"), new=True)
        return
    if state == "multiple":
        session.store_current(_demo_main_eeg(setname="menu one"), new=True)
        session.store_current(_demo_main_eeg(setname="menu two"), new=True)
        session.retrieve([1, 2])
        return
    if state == "study":
        session.store_current(_demo_main_eeg(setname="study demo"), new=True)
        session.STUDY = {
            "name": "menu study",
            "filename": "menu_study.study",
            "filepath": "/tmp",
            "task": "demo task",
            "subject": ["S01"],
            "condition": ["C1"],
            "session": [],
            "group": [],
            "cluster": [{"name": "ParentCluster"}],
        }
        session.CURRENTSTUDY = 1
        return
    raise ValueError(f"unsupported main-window state: {state}")


def _main_window_menu_state(menu_label: str | None, state: str) -> str:
    if menu_label is None or state != "startup":
        return state
    return "study" if menu_label == "Study" else "continuous"


def _grab_dialog(dialog, output: pathlib.Path, app) -> None:
    dialog.show()
    dialog.raise_()
    app.processEvents()
    pixmap = _matlab_scaled_pixmap(dialog.grab(), app)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not pixmap.save(str(output), "PNG"):
        raise RuntimeError(f"failed to save screenshot: {output}")
    dialog.close()
    app.processEvents()


def capture_main_window(output: pathlib.Path, *, state: str = "startup", menu_label: str | None = None) -> None:
    """Render and capture the EEGPrep main window, optionally with a menu open."""
    session = EEGPrepSession()
    state = _main_window_menu_state(menu_label, state)
    _configure_main_window_session(session, state)
    window = build_main_window(session, native_menu_bar=False if menu_label else None)
    if menu_label:
        window.window.setWindowFlag(window._qt_core.Qt.WindowStaysOnTopHint, True)
    window.show()
    window.window.activateWindow()
    window.window.raise_()
    window.app.processEvents()
    menu = None
    if menu_label:
        menu = window.open_menu(menu_label)
    if menu is not None:
        pixmap = _grab_main_window_with_menu(window, menu_label, menu)
    else:
        pixmap = window.window.grab()
    pixmap = _matlab_scaled_pixmap(pixmap, window.app)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not pixmap.save(str(output), "PNG"):
        raise RuntimeError(f"failed to save screenshot: {output}")
    window.window.close()
    window.app.processEvents()


def capture_eegbrowser(output: pathlib.Path, *, variant: str = "continuous") -> None:
    """Render and capture the EEGBrowser shell."""
    from PySide6 import QtWidgets
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    if variant == "component_activity":
        model = build_eegplot_model(
            _demo_eegbrowser_ica_eeg(),
            component=True,
            winlength=2.0,
            dispchans=6,
            spacing=1.2,
            xgrid="on",
            ygrid="on",
            winrej=_demo_eegbrowser_winrej(variant),
            command="TMPREJ=[];",
            butlabel="Update Marks",
            title="Scroll component activities -- eegplot() -- eegbrowser demo",
        )
    elif variant == "data2_overlay":
        eeg = _demo_eegbrowser_eeg(events=True)
        data = np.asarray(eeg["data"], dtype=float)
        overlay = data + 0.35 * np.cos(np.arange(data.shape[1], dtype=float)[None, :] / 12.0)
        model = build_eegplot_model(
            eeg,
            data2=overlay,
            winlength=2.0,
            dispchans=6,
            spacing=1.2,
            xgrid="on",
            ygrid="on",
            title="Scroll channel activities -- eegplot() -- data2 overlay demo",
        )
    elif variant == "spectral_overlay":
        freqs = np.linspace(1.0, 45.0, 90)
        data = np.vstack([np.log1p(freqs) + 0.12 * channel + np.sin(freqs / (channel + 2)) for channel in range(6)])
        overlay = data + 0.25 * np.cos(freqs / 4.0)
        model = build_eegplot_model(
            data,
            data2=overlay,
            freqs=freqs,
            freqlimits=[4, 32],
            winlength=12.0,
            dispchans=6,
            spacing=1.0,
            xgrid="on",
            ygrid="on",
            title="Scroll spectral activities -- eegplot() -- overlay demo",
        )
    else:
        epoched = variant in {"epoched", "epoched_marked", "rejection_epochs"}
        has_events = variant in {"events", "labels", "pop_eegplot_reject_data", "rejection_epochs"}
        winrej = _demo_eegbrowser_winrej(variant)
        eeg = _demo_eegbrowser_eeg(epoched=epoched, events=has_events)
        data = np.asarray(eeg["data"], dtype=float)
        if variant == "rejcont_continuous":
            selected_rows = [0, 1, 2, 3]
            data = np.asarray(eeg["data"], dtype=float)[selected_rows, :]
            eeg["chanlocs"] = [eeg["chanlocs"][index] for index in selected_rows]
        title = "Scroll channel activities -- eegplot() -- eegbrowser demo"
        if variant == "pop_eegplot_reject_data":
            title = "Scroll channel activities -- eegplot() -- pop_eegplot reject demo"
        elif variant == "rejection_epochs":
            title = "Trial rejection using data activity -- pop_eegthresh()"
        elif variant == "rejcont_continuous":
            title = "Scroll channel activities -- eegplot() -- pop_rejcont demo"
        model = build_eegplot_model(
            data,
            srate=float(eeg["srate"]),
            winlength=3.0 if variant in {"epoched_marked", "rejection_epochs"} else 1.0 if epoched else 2.0,
            dispchans=4 if variant == "rejcont_continuous" else 6,
            spacing=1.2,
            xgrid="off" if variant == "grid_off" or variant == "rejcont_continuous" else "on",
            ygrid="off" if variant == "grid_off" else "on",
            winrej=winrej,
            events=eeg["event"] if has_events else [],
            wincolor=(0.0, 0.9, 0.0) if variant == "rejcont_continuous" else None,
            command="TMPREJ=[];" if winrej is not None else None,
            butlabel="Reject",
            eloc_file=eeg.get("chanlocs", []) if variant == "rejcont_continuous" else None,
            title=title,
        )
    if variant != "labels":
        model.state.channel_label_mode = "numbers"
    window = EEGBrowserWindow(model)
    width = int(os.environ.get("EEGPREP_VISUAL_WINDOW_WIDTH", "960"))
    height = int(os.environ.get("EEGPREP_VISUAL_WINDOW_HEIGHT", "560"))
    window.resize(width, height)
    window.show()
    window.raise_()
    window.activateWindow()
    app.processEvents()
    pixmap = _matlab_scaled_pixmap(window.grab(), app)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not pixmap.save(str(output), "PNG"):
        raise RuntimeError(f"failed to save screenshot: {output}")
    window.close()
    app.processEvents()


def _grab_main_window_with_menu(window, menu_label: str, menu):
    window_pixmap = window.window.grab()
    menu_pixmap = menu.grab()
    ratio = float(window_pixmap.devicePixelRatio() or 1)
    window_pixmap.setDevicePixelRatio(1)
    menu_pixmap.setDevicePixelRatio(1)
    pos = _menu_popup_position(window, menu_label, ratio)
    width = max(window_pixmap.width(), pos[0] + menu_pixmap.width())
    height = max(window_pixmap.height(), pos[1] + menu_pixmap.height())
    canvas = window._qt_gui.QPixmap(width, height)
    canvas.fill(window._qt_gui.QColor("#a8c2ff"))
    painter = window._qt_gui.QPainter(canvas)
    painter.drawPixmap(0, 0, window_pixmap)
    painter.drawPixmap(pos[0], pos[1], menu_pixmap)
    painter.end()
    return canvas


def _menu_popup_position(window, menu_label: str, ratio: float) -> tuple[int, int]:
    menubar = window.window.menuBar()
    for action in menubar.actions():
        if action.text() != menu_label:
            continue
        geometry = menubar.actionGeometry(action)
        x = round((menubar.x() + geometry.x()) * ratio)
        y = round((menubar.y() + geometry.bottom()) * ratio)
        return max(0, x), max(0, y)
    return 0, round(menubar.height() * ratio)


def _matlab_scaled_pixmap(pixmap, app):
    if sys.platform != "darwin":
        return pixmap
    screen = app.primaryScreen()
    ratio = float(pixmap.devicePixelRatio() or 1)
    if screen is not None:
        ratio = max(ratio, float(screen.devicePixelRatio() or 1))
    matlab_ratio = 1.5
    scale = matlab_ratio / ratio
    if math.isclose(scale, 1.0):
        return pixmap
    return pixmap.scaled(max(1, math.floor(pixmap.width() * scale)), max(1, math.floor(pixmap.height() * scale)))


def capture_adjust_events_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_adjustevents dialog."""
    eeg = _demo_eeg()
    event_types = [event["type"] for event in eeg["event"]]
    spec = pop_adjustevents_dialog_spec(float(eeg["srate"]), event_types)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_comments_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_comments dialog."""
    spec = pop_comments_dialog_spec("About this dataset", "Existing sample dataset comments.")
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_editset_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_editset dialog."""
    eeg = _demo_main_eeg()
    eeg.update({"subject": "S01", "condition": "targets", "group": "control", "run": 1, "session": 1})
    spec = pop_editset_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_editeventfield_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_editeventfield dialog."""
    eeg = _demo_main_eeg()
    spec = pop_editeventfield_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_editeventvals_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_editeventvals dialog."""
    eeg = _demo_main_eeg()
    spec = pop_editeventvals_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_selectevent_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_selectevent dialog."""
    eeg = _demo_main_eeg(setname="event demo")
    spec = pop_selectevent_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_rmdat_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_rmdat dialog."""
    eeg = _demo_main_eeg()
    spec = pop_rmdat_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_chanedit_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_chanedit dialog."""
    eeg = _demo_main_eeg()
    spec = pop_chanedit_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_copyset_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_copyset dialog."""
    spec = pop_copyset_dialog_spec(1)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_mergeset_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_mergeset dialog."""
    eeg = _demo_main_eeg(setname="merge one")
    second = _demo_main_eeg(setname="merge two")
    spec = pop_mergeset_dialog_spec([eeg, second], default_indices=[1])
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def _demo_study() -> tuple[dict, list[dict]]:
    first = _demo_main_eeg(setname="study targets")
    first.update({"subject": "S01", "condition": "target", "group": "control", "session": 1, "run": 1})
    second = _demo_main_eeg(setname="study standards")
    second.update({"subject": "S02", "condition": "standard", "group": "control", "session": 1, "run": 1})
    study, alleeg = std_checkset({"name": "menu study", "task": "demo task"}, [first, second])
    return study, alleeg


def capture_pop_study_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_study dialog."""
    _study, alleeg = _demo_study()
    spec = pop_study_dialog_spec({}, alleeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_studydesign_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_studydesign dialog."""
    study, alleeg = _demo_study()
    spec = pop_studydesign_dialog_spec(study, alleeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def _demo_cluster_study() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    first = _demo_study()[1][0]
    first = dict(first)
    first["icaweights"] = np.eye(4)
    first["icasphere"] = np.eye(4)
    first["icawinv"] = np.eye(4)
    study, alleeg = std_checkset({"name": "menu study", "task": "demo task"}, [first])
    study["cluster"] = [
        {"name": "ParentCluster", "sets": [[1, 1, 1, 1]], "comps": [1, 2, 3, 4], "child": ["Cls 1", "Cls 2"]},
        {
            "name": "Cls 1",
            "sets": [[1, 1]],
            "comps": [1, 2],
            "parent": ["ParentCluster"],
            "child": [],
            "preclust": {"preclustdata": [[0.0], [0.1]], "preclustparams": []},
        },
        {
            "name": "Cls 2",
            "sets": [[1, 1]],
            "comps": [3, 4],
            "parent": ["ParentCluster"],
            "child": [],
            "preclust": {"preclustdata": [[1.0], [1.1]], "preclustparams": []},
        },
    ]
    study.setdefault("etc", {})["preclust"] = {
        "preclustdata": [[0.0, 0.1], [0.1, 0.1], [1.0, 1.1], [1.1, 1.0]],
        "preclustparams": [{"measure": "scalp", "npca": 2}],
        "clustlevel": 1,
    }
    return study, alleeg


def capture_pop_precomp_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_precomp dialog."""
    study, alleeg = _demo_study()
    spec = pop_precomp_dialog_spec(study, alleeg, "channels")
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_preclust_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_preclust dialog."""
    study, _alleeg = _demo_cluster_study()
    spec = pop_preclust_dialog_spec(study)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_clust_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_clust dialog."""
    study, _alleeg = _demo_cluster_study()
    spec = pop_clust_dialog_spec(study)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_chanplot_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_chanplot dialog."""
    study, alleeg = _demo_study()
    spec = pop_chanplot_dialog_spec(study, alleeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_clustedit_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_clustedit dialog."""
    study, _alleeg = _demo_cluster_study()
    spec = pop_clustedit_dialog_spec(study)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_reref_dialog(output: pathlib.Path, *, variant: str = "average") -> None:
    """Render and capture the pop_reref dialog."""
    eeg = _demo_reref_eeg()
    labels = [chan["labels"] for chan in eeg["chanlocs"]]
    refloc_labels = [chan["labels"] for chan in eeg["chaninfo"]["nodatchans"]]
    spec = pop_reref_dialog_spec("common", labels, refloc_labels)
    renderer = QtDialogRenderer()
    app, dialog, widgets = renderer.build_dialog(spec)
    if variant == "channels":
        widgets["rerefstr"].setChecked(True)
        renderer._set_reref_mode(widgets, "channels", True)
        widgets["reref"].setText("Fp1")
        widgets["keepref"].setChecked(True)
    elif variant == "huber":
        widgets["huberef"].setChecked(True)
        renderer._set_reref_mode(widgets, "huber", True)
    elif variant == "interp_removed":
        widgets["interp"].setChecked(True)
    _grab_dialog(dialog, output, app)


def capture_pop_interp_dialog(output: pathlib.Path, *, variant: str = "continuous") -> None:
    """Render and capture the pop_interp dialog."""
    eeg = _demo_interp_eeg(
        epoched=variant in {"epoched", "epoched_removed"},
        removed=variant in {"removed", "epoched_removed"},
    )
    spec = pop_interp_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_select_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_select dialog."""
    eeg = _demo_main_eeg()
    spec = pop_select_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_resample_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_resample dialog."""
    eeg = _demo_main_eeg()
    spec = pop_resample_dialog_spec(eeg["srate"])
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_newset_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_newset dialog."""
    eeg = _demo_main_eeg(setname="processed demo")
    spec = pop_newset_dialog_spec(eeg, 1)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_rmbase_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_rmbase dialog."""
    eeg = _demo_main_eeg(epoched=True, setname="baseline demo")
    spec = pop_rmbase_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_eegfilt_dialog(output: pathlib.Path) -> None:
    """Render and capture the legacy pop_eegfilt dialog."""
    eeg = _demo_main_eeg()
    spec = pop_eegfilt_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_eegfiltnew_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_eegfiltnew dialog."""
    eeg = _demo_main_eeg()
    spec = pop_eegfiltnew_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_firws_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_firws dialog."""
    eeg = _demo_main_eeg()
    spec = pop_firws_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_firpm_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_firpm dialog."""
    eeg = _demo_main_eeg()
    spec = pop_firpm_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_firma_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_firma dialog."""
    eeg = _demo_main_eeg()
    spec = pop_firma_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_kaiserbeta_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_kaiserbeta dialog."""
    spec = pop_kaiserbeta_dialog_spec()
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_firwsord_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_firwsord dialog."""
    spec = pop_firwsord_dialog_spec()
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_firpmord_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_firpmord dialog."""
    spec = pop_firpmord_dialog_spec()
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_xfirws_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_xfirws dialog."""
    spec = pop_xfirws_dialog_spec()
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_epoch_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_epoch dialog."""
    eeg = _demo_main_eeg(setname="pop demo")
    spec = pop_epoch_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_topoplot_dialog(output: pathlib.Path, *, variant: str = "erp") -> None:
    """Render and capture the pop_topoplot dialog."""
    eeg = _demo_main_eeg(setname="pop demo")
    spec = pop_topoplot_dialog_spec(eeg, typeplot=0 if variant == "components" else 1)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_spectopo_dialog(output: pathlib.Path, *, variant: str = "channels") -> None:
    """Render and capture the pop_spectopo dialog."""
    eeg = _demo_main_eeg(epoched=variant == "channels_epoched", setname="pop demo")
    spec = pop_spectopo_dialog_spec(eeg, dataflag=0 if variant == "components" else 1)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_prop_dialog(output: pathlib.Path, *, variant: str = "channels") -> None:
    """Render and capture the pop_prop dialog."""
    eeg = _demo_main_eeg(epoched=True, setname="pop demo")
    spec = pop_prop_dialog_spec(eeg, typecomp=0 if variant == "components" else 1)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_timtopo_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_timtopo dialog."""
    eeg = _demo_main_eeg(epoched=True, setname="pop demo")
    spec = pop_timtopo_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_plottopo_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_plottopo dialog."""
    eeg = _demo_main_eeg(epoched=True, setname="pop demo")
    spec = pop_plottopo_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_headplot_dialog(output: pathlib.Path, *, variant: str = "erp") -> None:
    """Render and capture the pop_headplot dialog."""
    eeg = _demo_main_eeg(epoched=True, setname="pop demo")
    spec = pop_headplot_dialog_spec(eeg, typeplot=0 if variant == "components" else 1)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_coregister_dialog(output: pathlib.Path) -> None:
    """Render and capture the manual headplot coregistration dialog."""
    eeg = _demo_main_eeg(epoched=True, setname="pop demo")
    app, dialog = prepare_coregister_display(
        eeg["chanlocs"],
        "mheadnew.xyz",
        chaninfo=eeg.get("chaninfo", {}),
        meshfile="mheadnew.mat",
        transform=[0, -10, 0, -0.1, 0, -1.6, 1100, 1100, 1100],
        parent=None,
        title="Co-registration plot for headplot mesh",
    )
    _grab_dialog(dialog, output, app)


def capture_pop_plotdata_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_plotdata dialog."""
    eeg = _demo_main_eeg(epoched=True, setname="pop demo")
    spec = pop_plotdata_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_erpimage_dialog(output: pathlib.Path, *, variant: str = "channels") -> None:
    """Render and capture the pop_erpimage dialog."""
    eeg = _demo_main_eeg(epoched=True, setname="pop demo")
    spec = pop_erpimage_dialog_spec(eeg, typeplot=0 if variant == "components" else 1)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_envtopo_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_envtopo dialog."""
    eeg = _demo_main_eeg(epoched=True, setname="pop demo")
    spec = pop_envtopo_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_comperp_dialog(output: pathlib.Path, *, variant: str = "channels") -> None:
    """Render and capture the pop_comperp dialog."""
    datasets = [
        _demo_main_eeg(epoched=True, setname="pop demo"),
        _demo_main_eeg(epoched=True, setname="pop demo two"),
    ]
    spec = pop_comperp_dialog_spec(datasets, flag=0 if variant == "components" else 1)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_newtimef_dialog(output: pathlib.Path, *, variant: str = "channels") -> None:
    """Render and capture the pop_newtimef dialog."""
    eeg = _demo_main_eeg(epoched=True, setname="pop demo")
    spec = pop_newtimef_dialog_spec(eeg, typeproc=0 if variant == "components" else 1)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_newcrossf_dialog(output: pathlib.Path, *, variant: str = "channels") -> None:
    """Render and capture the pop_newcrossf dialog."""
    eeg = _demo_main_eeg(epoched=True, setname="pop demo")
    spec = pop_newcrossf_dialog_spec(eeg, typeproc=0 if variant == "components" else 1)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_signalstat_dialog(output: pathlib.Path, *, variant: str = "channels") -> None:
    """Render and capture the pop_signalstat dialog."""
    eeg = _demo_main_eeg(epoched=True, setname="pop demo")
    spec = pop_signalstat_dialog_spec(eeg, typeproc=0 if variant == "components" else 1)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_eventstat_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_eventstat dialog."""
    eeg = _demo_main_eeg(epoched=True, setname="pop demo")
    spec = pop_eventstat_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_runica_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_runica dialog."""
    eeg = _demo_main_eeg()
    spec = pop_runica_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_runica_multiple_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_runica dialog for multiple datasets."""
    eeg = _demo_main_eeg()
    second = dict(eeg)
    second["setname"] = "second demo"
    spec = pop_runica_dialog_spec([eeg, second])
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_iclabel_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_iclabel dialog."""
    spec = pop_iclabel_dialog_spec()
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_icflag_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_icflag dialog."""
    spec = pop_icflag_dialog_spec()
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_prop_extended_dashboard(output: pathlib.Path) -> None:
    """Render and capture the ICLabel extended property dashboard."""
    figure = pop_prop_extended(
        _demo_iclabel_dashboard_eeg(),
        0,
        [1, 2],
        spec_opt="'freqrange', [2 40]",
        scroll_event=1,
    )
    figure.savefig(output, dpi=200)
    plt.close(figure)


def capture_pop_subcomp_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_subcomp dialog."""
    eeg = _demo_main_eeg()
    eeg["reject"] = {"gcompreject": np.zeros(4, dtype=int)}
    spec = pop_subcomp_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def _rejection_spec(case_id: str):
    eeg = _demo_main_eeg(epoched=True, setname="reject demo")
    eeg["reject"] = {
        "gcompreject": np.array([0, 1, 0, 0]),
        "rejthresh": np.array([0, 1], dtype=bool),
        "rejthreshE": np.zeros((4, 2), dtype=bool),
    }
    continuous = _demo_main_eeg()
    if case_id == "pop_eegthresh_dialog":
        return pop_eegthresh_dialog_spec(eeg, 1)
    if case_id == "pop_jointprob_dialog":
        return pop_jointprob_dialog_spec(eeg, 1)
    if case_id == "pop_rejkurt_dialog":
        return pop_rejkurt_dialog_spec(eeg, 1)
    if case_id == "pop_rejtrend_dialog":
        return pop_rejtrend_dialog_spec(eeg, 1)
    if case_id == "pop_rejspec_dialog":
        return pop_rejspec_dialog_spec(eeg, 1)
    if case_id == "pop_rejmenu_dialog":
        return pop_rejmenu_dialog_spec(eeg, 1)
    if case_id == "pop_autorej_dialog":
        return pop_autorej_dialog_spec(eeg)
    if case_id == "pop_selectcomps_dialog":
        return pop_selectcomps_dialog_spec(eeg)
    if case_id == "pop_viewprops_dialog":
        eeg["etc"] = {
            "ic_classification": {
                "ICLabel": {
                    "classifications": np.array([[0.7, 0.1, 0.1, 0.03, 0.02, 0.03, 0.02]] * 4),
                    "classes": ["Brain", "Muscle", "Eye", "Heart", "Line Noise", "Channel Noise", "Other"],
                }
            }
        }
        return pop_viewprops_dialog_spec(eeg, 0)
    if case_id == "pop_rejchan_dialog":
        return pop_rejchan_dialog_spec(continuous)
    if case_id == "pop_rejcont_dialog":
        return pop_rejcont_dialog_spec(continuous)
    raise ValueError(f"unsupported rejection visual case: {case_id}")


def capture_rejection_dialog(output: pathlib.Path, *, case_id: str) -> None:
    """Render and capture a rejection/component dialog."""
    spec = _rejection_spec(case_id)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def _dipfit_demo_eeg() -> dict:
    eeg = _demo_main_eeg(epoched=True, setname="dipfit demo")
    eeg["icaweights"] = np.eye(4)
    eeg["icasphere"] = np.eye(4)
    eeg["icawinv"] = np.eye(4)
    eeg["icachansind"] = np.arange(4)
    eeg["dipfit"] = {
        "hdmfile": "standard_BEM/standard_vol.mat",
        "mrifile": "standard_BEM/standard_mri.mat",
        "chanfile": "standard_BEM/elec/standard_1005.elc",
        "coordformat": "MNI",
        "coord_transform": [0, 0, 0, 0, 0, -1.5708, 1, 1, 1],
        "chansel": [1, 2, 3, 4],
        "model": [
            {"posxyz": [0, -20, 40], "momxyz": [1, 0, 0], "rv": 0.12, "component": 1},
            {"posxyz": [25, 10, 35], "momxyz": [0, 1, 0], "rv": 0.2, "component": 2},
        ],
    }
    return eeg


def _dipfit_spec(case_id: str):
    eeg = _dipfit_demo_eeg()
    if case_id == "pop_dipfit_settings_dialog":
        return pop_dipfit_settings_dialog_spec(eeg)
    if case_id == "pop_dipfit_headmodel_dialog":
        return pop_dipfit_headmodel_dialog_spec(eeg, "subject_T1.nii")
    if case_id == "pop_dipfit_gridsearch_dialog":
        return pop_dipfit_gridsearch_dialog_spec(eeg)
    if case_id == "pop_dipfit_nonlinear_dialog":
        return pop_dipfit_nonlinear_dialog_spec(eeg)
    if case_id == "pop_dipplot_dialog":
        return pop_dipplot_dialog_spec(eeg)
    if case_id == "pop_multifit_dialog":
        return pop_multifit_dialog_spec(eeg)
    if case_id == "pop_leadfield_dialog":
        return pop_leadfield_dialog_spec(eeg)
    if case_id == "pop_dipfit_loreta_dialog":
        eeg["dipfit"]["sourcemodel"] = {"pos": [[0, 0, 0]], "leadfield": []}
        return pop_dipfit_loreta_dialog_spec(eeg)
    raise ValueError(f"unsupported DIPFIT visual case: {case_id}")


def capture_dipfit_dialog(output: pathlib.Path, *, case_id: str) -> None:
    """Render and capture a DIPFIT dialog."""
    spec = _dipfit_spec(case_id)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_clean_rawdata_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_clean_rawdata dialog."""
    eeg = _demo_main_eeg()
    spec = pop_clean_rawdata_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_chansel_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_chansel/listdlg2 channel picker."""
    labels = ["Fp1", "Fp2", "Cz", "Oz"]
    display_values = pop_chansel_display_values(labels, withindex="on")
    app, dialog = build_listdlg2_dialog(
        promptstring="(use shift|Ctrl to\nselect several)",
        liststring=display_values,
        selectionmode="multiple",
    )
    _grab_dialog(dialog, output, app)


def capture_select_multiple_datasets_dialog(output: pathlib.Path) -> None:
    """Render and capture the EEGLAB-style multiple-dataset picker."""
    labels = ["Dataset 1:menu one", "Dataset 2:menu two", "Dataset 3:menu three"]
    display_values = pop_chansel_display_values(labels, withindex=[1, 2, 3])
    app, dialog = build_listdlg2_dialog(
        promptstring="(use shift|Ctrl to\nselect several)",
        liststring=display_values,
        selectionmode="multiple",
        initialvalue=[1, 2],
    )
    _grab_dialog(dialog, output, app)


def capture_dataset_index_dialog(output: pathlib.Path) -> None:
    """Render and capture the dataset index prompt used by pop_interp."""
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = QtWidgets.QDialog()
    dialog.setObjectName("inputdlg2")
    dialog.setWindowTitle("Choose dataset")
    dialog.resize(200, 127)
    dialog.setStyleSheet(
        """
        QDialog {
            background: #a8c2ff;
            color: #000066;
            font-size: 16px;
        }
        QLabel {
            color: #000066;
            background: transparent;
            font-size: 16px;
        }
        QLineEdit {
            background: white;
            border: 1px solid #7f7f7f;
            min-height: 18px;
            max-height: 18px;
            font-size: 16px;
        }
        QPushButton {
            background: #eeeeee;
            border: 1px solid #7f7f7f;
            min-width: 47px;
            max-width: 47px;
            min-height: 18px;
            max-height: 18px;
            padding: 0;
            color: #000066;
            font-size: 16px;
        }
        """
    )
    widgets = [
        (QtWidgets.QLabel("Dataset index", dialog), "", (9, 16, 180, 20)),
        (QtWidgets.QLineEdit(dialog), "input", (9, 44, 179, 19)),
        (QtWidgets.QPushButton("Help", dialog), "help", (9, 90, 49, 19)),
        (QtWidgets.QPushButton("Cancel", dialog), "cancel", (87, 90, 49, 19)),
        (QtWidgets.QPushButton("OK", dialog), "ok", (140, 90, 49, 19)),
    ]
    for widget, name, geometry in widgets:
        if name:
            widget.setObjectName(name)
        widget.setGeometry(*geometry)
    _grab_dialog(dialog, output, app)


def capture_pophelp_dialog(output: pathlib.Path, function_name: str) -> None:
    """Render and capture the pophelp browser for a pop function."""
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = pophelp(function_name)
    _grab_dialog(dialog, output, app)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)

    if args.case == "adjust_events_dialog":
        capture_adjust_events_dialog(args.output)
    elif args.case == "main_window":
        capture_main_window(args.output)
    elif args.case == "main_window_continuous":
        capture_main_window(args.output, state="continuous")
    elif args.case == "main_window_epoched":
        capture_main_window(args.output, state="epoched")
    elif args.case == "main_window_multiple":
        capture_main_window(args.output, state="multiple")
    elif args.case == "main_window_study":
        capture_main_window(args.output, state="study")
    elif args.case == "eegbrowser_continuous":
        capture_eegbrowser(args.output, variant="continuous")
    elif args.case == "eegbrowser_continuous_marked":
        capture_eegbrowser(args.output, variant="continuous_marked")
    elif args.case == "eegbrowser_epoched":
        capture_eegbrowser(args.output, variant="epoched")
    elif args.case == "eegbrowser_epoched_marked":
        capture_eegbrowser(args.output, variant="epoched_marked")
    elif args.case == "eegbrowser_events":
        capture_eegbrowser(args.output, variant="events")
    elif args.case == "eegbrowser_grid_off":
        capture_eegbrowser(args.output, variant="grid_off")
    elif args.case == "eegbrowser_labels":
        capture_eegbrowser(args.output, variant="labels")
    elif args.case == "eegbrowser_component_activity":
        capture_eegbrowser(args.output, variant="component_activity")
    elif args.case == "eegbrowser_data2_overlay":
        capture_eegbrowser(args.output, variant="data2_overlay")
    elif args.case == "eegbrowser_spectral_overlay":
        capture_eegbrowser(args.output, variant="spectral_overlay")
    elif args.case == "eegbrowser_pop_eegplot_reject_data":
        capture_eegbrowser(args.output, variant="pop_eegplot_reject_data")
    elif args.case == "eegbrowser_rejcont_continuous":
        capture_eegbrowser(args.output, variant="rejcont_continuous")
    elif args.case == "eegbrowser_rejection_epochs":
        capture_eegbrowser(args.output, variant="rejection_epochs")
    elif args.case == "file_menu":
        capture_main_window(args.output, menu_label="File")
    elif args.case == "edit_menu":
        capture_main_window(args.output, menu_label="Edit")
    elif args.case == "tools_menu":
        capture_main_window(args.output, menu_label="Tools")
    elif args.case == "plot_menu":
        capture_main_window(args.output, menu_label="Plot")
    elif args.case == "study_menu":
        capture_main_window(args.output, menu_label="Study")
    elif args.case == "datasets_menu":
        capture_main_window(args.output, menu_label="Datasets")
    elif args.case == "help_menu":
        capture_main_window(args.output, menu_label="Help")
    elif args.case == "pop_comments_dialog":
        capture_pop_comments_dialog(args.output)
    elif args.case == "pop_editset_dialog":
        capture_pop_editset_dialog(args.output)
    elif args.case == "pop_editeventfield_dialog":
        capture_pop_editeventfield_dialog(args.output)
    elif args.case == "pop_editeventvals_dialog":
        capture_pop_editeventvals_dialog(args.output)
    elif args.case == "pop_selectevent_dialog":
        capture_pop_selectevent_dialog(args.output)
    elif args.case == "pop_rmdat_dialog":
        capture_pop_rmdat_dialog(args.output)
    elif args.case == "pop_chanedit_dialog":
        capture_pop_chanedit_dialog(args.output)
    elif args.case == "pop_copyset_dialog":
        capture_pop_copyset_dialog(args.output)
    elif args.case == "pop_mergeset_dialog":
        capture_pop_mergeset_dialog(args.output)
    elif args.case == "pop_study_dialog":
        capture_pop_study_dialog(args.output)
    elif args.case == "pop_studydesign_dialog":
        capture_pop_studydesign_dialog(args.output)
    elif args.case == "pop_precomp_dialog":
        capture_pop_precomp_dialog(args.output)
    elif args.case == "pop_preclust_dialog":
        capture_pop_preclust_dialog(args.output)
    elif args.case == "pop_clust_dialog":
        capture_pop_clust_dialog(args.output)
    elif args.case == "pop_chanplot_dialog":
        capture_pop_chanplot_dialog(args.output)
    elif args.case == "pop_clustedit_dialog":
        capture_pop_clustedit_dialog(args.output)
    elif args.case == "reref_dialog":
        capture_reref_dialog(args.output)
    elif args.case == "reref_dialog_channel_ref":
        capture_reref_dialog(args.output, variant="channels")
    elif args.case == "reref_dialog_huber_ref":
        capture_reref_dialog(args.output, variant="huber")
    elif args.case == "reref_dialog_interp_removed":
        capture_reref_dialog(args.output, variant="interp_removed")
    elif args.case == "pop_interp_dialog":
        capture_pop_interp_dialog(args.output)
    elif args.case == "pop_interp_removed_dialog":
        capture_pop_interp_dialog(args.output, variant="removed")
    elif args.case == "pop_interp_epoched_dialog":
        capture_pop_interp_dialog(args.output, variant="epoched")
    elif args.case == "pop_select_dialog":
        capture_pop_select_dialog(args.output)
    elif args.case == "pop_resample_dialog":
        capture_pop_resample_dialog(args.output)
    elif args.case == "pop_newset_dialog":
        capture_pop_newset_dialog(args.output)
    elif args.case == "pop_rmbase_dialog":
        capture_pop_rmbase_dialog(args.output)
    elif args.case == "pop_eegfilt_dialog":
        capture_pop_eegfilt_dialog(args.output)
    elif args.case == "pop_eegfiltnew_dialog":
        capture_pop_eegfiltnew_dialog(args.output)
    elif args.case == "pop_firws_dialog":
        capture_pop_firws_dialog(args.output)
    elif args.case == "pop_firpm_dialog":
        capture_pop_firpm_dialog(args.output)
    elif args.case == "pop_firma_dialog":
        capture_pop_firma_dialog(args.output)
    elif args.case == "pop_kaiserbeta_dialog":
        capture_pop_kaiserbeta_dialog(args.output)
    elif args.case == "pop_firwsord_dialog":
        capture_pop_firwsord_dialog(args.output)
    elif args.case == "pop_firpmord_dialog":
        capture_pop_firpmord_dialog(args.output)
    elif args.case == "pop_xfirws_dialog":
        capture_pop_xfirws_dialog(args.output)
    elif args.case == "pop_epoch_dialog":
        capture_pop_epoch_dialog(args.output)
    elif args.case == "pop_topoplot_erp_dialog":
        capture_pop_topoplot_dialog(args.output, variant="erp")
    elif args.case == "pop_topoplot_components_dialog":
        capture_pop_topoplot_dialog(args.output, variant="components")
    elif args.case == "pop_spectopo_channels_dialog":
        capture_pop_spectopo_dialog(args.output, variant="channels")
    elif args.case == "pop_spectopo_components_dialog":
        capture_pop_spectopo_dialog(args.output, variant="components")
    elif args.case == "pop_prop_channels_dialog":
        capture_pop_prop_dialog(args.output, variant="channels")
    elif args.case == "pop_prop_components_dialog":
        capture_pop_prop_dialog(args.output, variant="components")
    elif args.case == "pop_timtopo_dialog":
        capture_pop_timtopo_dialog(args.output)
    elif args.case == "pop_plottopo_dialog":
        capture_pop_plottopo_dialog(args.output)
    elif args.case == "pop_headplot_erp_dialog":
        capture_pop_headplot_dialog(args.output, variant="erp")
    elif args.case == "pop_headplot_components_dialog":
        capture_pop_headplot_dialog(args.output, variant="components")
    elif args.case == "coregister_dialog":
        capture_coregister_dialog(args.output)
    elif args.case == "pop_plotdata_dialog":
        capture_pop_plotdata_dialog(args.output)
    elif args.case == "pop_erpimage_channels_dialog":
        capture_pop_erpimage_dialog(args.output, variant="channels")
    elif args.case == "pop_erpimage_components_dialog":
        capture_pop_erpimage_dialog(args.output, variant="components")
    elif args.case == "pop_envtopo_dialog":
        capture_pop_envtopo_dialog(args.output)
    elif args.case == "pop_comperp_channels_dialog":
        capture_pop_comperp_dialog(args.output, variant="channels")
    elif args.case == "pop_comperp_components_dialog":
        capture_pop_comperp_dialog(args.output, variant="components")
    elif args.case == "pop_newtimef_channels_dialog":
        capture_pop_newtimef_dialog(args.output, variant="channels")
    elif args.case == "pop_newtimef_components_dialog":
        capture_pop_newtimef_dialog(args.output, variant="components")
    elif args.case == "pop_newcrossf_channels_dialog":
        capture_pop_newcrossf_dialog(args.output, variant="channels")
    elif args.case == "pop_newcrossf_components_dialog":
        capture_pop_newcrossf_dialog(args.output, variant="components")
    elif args.case == "pop_signalstat_channels_dialog":
        capture_pop_signalstat_dialog(args.output, variant="channels")
    elif args.case == "pop_signalstat_components_dialog":
        capture_pop_signalstat_dialog(args.output, variant="components")
    elif args.case == "pop_eventstat_dialog":
        capture_pop_eventstat_dialog(args.output)
    elif args.case == "pop_runica_dialog":
        capture_pop_runica_dialog(args.output)
    elif args.case == "pop_runica_multiple_dialog":
        capture_pop_runica_multiple_dialog(args.output)
    elif args.case == "pop_iclabel_dialog":
        capture_pop_iclabel_dialog(args.output)
    elif args.case == "pop_icflag_dialog":
        capture_pop_icflag_dialog(args.output)
    elif args.case == "iclabel_pop_prop_extended_dashboard":
        capture_pop_prop_extended_dashboard(args.output)
    elif args.case == "pop_subcomp_dialog":
        capture_pop_subcomp_dialog(args.output)
    elif args.case in {
        "pop_autorej_dialog",
        "pop_eegthresh_dialog",
        "pop_jointprob_dialog",
        "pop_rejchan_dialog",
        "pop_rejcont_dialog",
        "pop_rejkurt_dialog",
        "pop_rejmenu_dialog",
        "pop_rejspec_dialog",
        "pop_rejtrend_dialog",
        "pop_selectcomps_dialog",
        "pop_viewprops_dialog",
    }:
        capture_rejection_dialog(args.output, case_id=args.case)
    elif args.case in {
        "pop_dipfit_settings_dialog",
        "pop_dipfit_headmodel_dialog",
        "pop_dipfit_gridsearch_dialog",
        "pop_dipfit_nonlinear_dialog",
        "pop_dipplot_dialog",
        "pop_multifit_dialog",
        "pop_leadfield_dialog",
        "pop_dipfit_loreta_dialog",
    }:
        capture_dipfit_dialog(args.output, case_id=args.case)
    elif args.case == "pop_clean_rawdata_dialog":
        capture_pop_clean_rawdata_dialog(args.output)
    elif args.case == "pop_chansel_dialog":
        capture_pop_chansel_dialog(args.output)
    elif args.case == "select_multiple_datasets_dialog":
        capture_select_multiple_datasets_dialog(args.output)
    elif args.case == "pop_interp_dataset_index_dialog":
        capture_dataset_index_dialog(args.output)
    elif args.case == "pop_reref_help_dialog":
        capture_pophelp_dialog(args.output, "pop_reref")
    elif args.case == "pop_interp_help_dialog":
        capture_pophelp_dialog(args.output, "pop_interp")
    else:
        parser.error(f"unsupported EEGPrep visual capture case: {args.case}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
