"""A :class:`~eegprep_lean.Window` as an EEGLAB-compatible dictionary.

The key set is that of eegprep's ``eeg_emptyset()``, written out here literally because
this distribution must not import eegprep: it is the one that stays small, and eegprep
will depend on it. ``test_eeg.py`` checks the list against eegprep's own so that a key
added there is a failing test here.

Only the fields a reader can know are filled. Events, channel locations and the rest keep
their empty-set defaults, which is what ``eeg_checkset`` expects of a freshly read
recording.
"""

from __future__ import annotations

import numpy as np

from . import __version__
from .window import Window

#: Microvolts per one of each voltage unit a channel group can declare. Anything else is
#: not a voltage, or not one this table knows, and converting it would be a guess.
_MICROVOLTS_PER = {"V": 1e6, "mV": 1e3, "uV": 1.0}


def microvolts_per(unit: str | None) -> float:
    """The factor taking ``unit`` to microvolts, or a ``ValueError`` naming what is accepted."""
    if unit not in _MICROVOLTS_PER:
        raise ValueError(
            f"cannot treat unit {unit!r} as a voltage: only {', '.join(_MICROVOLTS_PER)} are accepted, "
            "and a window whose channels do not share one unit has none (unit is None)"
        )
    return _MICROVOLTS_PER[unit]


def require_physical(window: Window) -> tuple[str, ...]:
    """The window's labels, after checking it holds physical values that are named.

    Stored counts are in no unit, and unlabeled channels would have to be named by index,
    which makes a later mix-up between channels harder to see. Both are refused.
    """
    if not window.physical:
        raise ValueError("window holds stored counts (physical=False), which are not in any unit")
    if window.labels is None:
        raise ValueError("window has no channel labels for every row, and channels are not named by index")
    return window.labels


def _emptyset() -> dict:
    """The key set and defaults of eegprep's ``eeg_emptyset()``."""
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
        "urchanlocs": [],
        "ref": "common",
        "event": [],
        "urevent": [],
        "eventdescription": [],
        "epoch": [],
        "epochdescription": [],
        "history": "",
        "saved": "yes",
        "run": [],
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
        "datfile": "",
        "roi": {},
        "etc": {},
    }


def to_eeg(window: Window, *, dataset_id: str, store: str, index_digest: str) -> dict:
    """Turn a window into an EEGLAB-style EEG dictionary that ``eeg_checkset`` accepts.

    ``data`` is channels by samples, float64, in microvolts, converted from V, mV or uV;
    any other unit is refused with ``ValueError``. ``times`` is in milliseconds and
    ``xmin`` is 0 whatever the window's ``start_sample``, as EEGLAB reads a continuous
    recording. ``etc["nemar"]`` records where the data came from: dataset, store, the
    SHA-256 of the index it was read against (``index_digest``, from
    :func:`~eegprep_lean.index_sha256`) and this reader's version.
    """
    labels = require_physical(window)
    factor = microvolts_per(window.unit)
    data = np.array(window.data, dtype=np.float64) * factor
    nbchan, pnts = data.shape
    srate = float(window.rate)

    eeg = _emptyset()
    eeg.update(
        setname=f"{dataset_id}/{store}",
        nbchan=nbchan,
        trials=1,
        pnts=pnts,
        srate=srate,
        xmin=0.0,
        xmax=(pnts - 1) / srate,
        times=np.arange(pnts) / srate * 1000.0,
        data=data,
        chanlocs=[{"labels": label} for label in labels],
        etc={
            "nemar": {
                "dataset_id": dataset_id,
                "store": store,
                "index_sha256": index_digest,
                "reader": f"eegprep-lean@{__version__}",
            }
        },
    )
    return eeg
