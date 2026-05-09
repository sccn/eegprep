"""Create an empty EEGLAB-like EEG dataset."""

from __future__ import annotations

import numpy as np


def eeg_emptyset() -> dict:
    """Return an empty EEG dictionary with EEGLAB core fields."""
    return {
        "setname": "",
        "filename": "",
        "filepath": "",
        "subject": "",
        "group": "",
        "condition": "",
        "session": [],
        "comments": "",
        "nbchan": 0,
        "trials": 0,
        "pnts": 0,
        "srate": 0,
        "xmin": 0,
        "xmax": 0,
        "times": np.array([]),
        "data": np.array([]),
        "chanlocs": [],
        "event": [],
        "urevent": [],
        "epoch": [],
        "history": "",
        "saved": "yes",
        "icaact": np.array([]),
        "icawinv": np.array([]),
        "icasphere": np.array([]),
        "icaweights": np.array([]),
        "icachansind": np.array([]),
        "chaninfo": {},
        "reject": {},
        "stats": {},
        "specdata": [],
        "specicaact": [],
        "splinefile": "",
        "icasplinefile": "",
        "dipfit": {},
        "etc": {},
    }
