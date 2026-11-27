"""EEGLAB channel locations from a store's electrode attributes.

The conversion eegprep's BIDS import makes from an ``electrodes.tsv``, so that a recording
read from a store and the same recording imported from its BIDS dataset get the same
``chanlocs``: millimeters (``coords_to_mm``), EEGLAB's ALS frame (``coords_RAS_to_ALS``),
then the angles (``coords_ALS_to_angular``). Those are in ``eegprep.plugins.EEG_BIDS.coords``
and this package must not import eegprep, which depends on it and not the other way round,
so the same few steps are written out here in NumPy, with the operations in the same order.
``test_eeg.py`` builds both routes from one set of coordinates and requires equal answers.

Both read a frame by the orientation the BIDS coordinate-systems appendix gives it, from the
same table (``BIDS_ALS_SYSTEMS`` and ``BIDS_RAS_SYSTEMS`` there), and give no positions for
a frame without one. **Where this differs from the import is what else it refuses, and that
is deliberate.** The import guesses a unit from the size of the coordinates when the
coordsystem names none, assumes a frame when there is no coordsystem file, matches a frame's
name without regard to case and reads any name with ``EEGLAB`` in it as ALS. Here a unit
that is not known, or a frame not spelled as in the table, gives no positions and a recorded
reason, because a wrong guess moves ``theta`` and ``radius`` and raises nothing.

The import also reads a coordsystem whose ``EEGCoordinateSystemDescription`` is ``EEGLAB`` as
ALS, whatever ``EEGCoordinateSystem`` says. A store carries no description, so a dataset that
states a RAS system with that description is ALS through the import and is turned by 90
degrees here. The converter writes no such description today; reading it here is a follow-up
if it ever does.

The origin is not moved. A template space has its origin at the anterior commissure, not at
the center of the head, and the import does not recenter either; ``radius`` is measured from
wherever the dataset put the origin.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .electrodes import ElectrodeAttributes

#: ``to_eeg`` sets ``chaninfo["nosedir"]`` to this when it fills locations: the nose is +X in
#: EEGLAB's ALS frame, which is the frame everything here ends in.
NOSE_DIRECTION = "+X"

#: Why no positions were filled, as recorded in ``etc["nemar"]["locations"]["reason"]``.
UNITS_UNKNOWN = "units_unknown"
COORDINATE_SYSTEM_UNKNOWN = "coordinate_system_unknown"
NO_MATCHING_LABELS = "no_matching_labels"

#: Millimeters per one of each unit. The spellings are those of eegprep's ``coords_to_mm``,
#: looked up in lower case as the BIDS import does. Any other text is not a unit this knows.
_MM_PER_UNIT = {"mm": 1.0, "millimeters": 1.0, "cm": 10.0, "centimeters": 10.0, "m": 1000.0, "meters": 1000.0}

#: The BIDS coordinate-systems appendix: the frames that are ALS already, and EEGLAB's own.
_ALS_SYSTEMS = frozenset({"EEGLAB", "EEGLAB-HJ", "CTF", "4DBti", "KitYokogawa"})

#: And the frames that are RAS, to be turned into ALS: the head-based ones, then every
#: standard template identifier, the deprecated spellings the appendix keeps included. The
#: appendix gives a template space RAS orientation and gives no orientation to the iEEG frames
#: ``Pixels``, ``ACPC`` and ``ScanRAS``, nor to ``Other``, ``individual``, ``study`` or
#: ``scanner``, which are therefore not here.
_RAS_SYSTEMS = frozenset(
    {
        "CapTrak",
        "NeuromagElektaMEGIN",
        "ElektaNeuromag",
        "ChietiItab",
        "ICBM452AirSpace",
        "ICBM452Warp5Space",
        "IXI549Space",
        "fsaverage",
        "fsaverageSym",
        "fsLR",
        "MNIColin27",
        "MNI152Lin",
        "MNI152NLin6Sym",
        # The appendix's table prints MNI152NLin6ASym; the schema's value is that up to BIDS
        # 1.10.1 and MNI152NLin6Asym from 1.11.0.
        "MNI152NLin6ASym",
        "MNI152NLin6Asym",
        "MNI305",
        "NIHPD",
        "OASIS30AntsOASISAnts",
        "OASIS30Atropos",
        "Talairach",
        "UNCInfant",
        "fsaverage3",
        "fsaverage4",
        "fsaverage5",
        "fsaverage6",
        "fsaveragesym",
        *(f"MNI152NLin2009{version}{symmetry}" for version in "abc" for symmetry in ("Sym", "Asym")),
        *(f"UNCInfant{age}V{volume}" for age in "012" for volume in ("21", "22", "23")),
    }
)


def chanlocs_from_electrodes(
    labels: tuple[str, ...], electrodes: ElectrodeAttributes
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """EEGLAB ``chanlocs`` for ``labels`` from a store's electrode attributes, and the record.

    Each channel is joined to the entry of exactly its label. When the unit and the frame
    are both known, a channel with an entry gets ``X``, ``Y``, ``Z`` (millimeters, ALS),
    ``sph_theta``, ``sph_phi``, ``sph_radius``, ``theta`` and ``radius``, and a channel
    without one gets an empty array for each of those, as the BIDS import leaves it. When a
    unit or a frame is not known, or no entry names a channel, nothing is filled and every
    channel keeps its label alone; the unit is checked first, and only the first reason is
    recorded.

    The record is the store's own words (``source`` and ``file`` when it gave them,
    ``coordinate_system`` and ``coordinate_units``), the count of channels that got a
    position as ``located``, and, when that is none, ``reason``.
    """
    record: dict[str, Any] = {
        "coordinate_system": electrodes.coordinate_system,
        "coordinate_units": electrodes.coordinate_units,
    }
    if electrodes.source is not None:
        record["source"] = electrodes.source
    if electrodes.file is not None:
        record["file"] = electrodes.file
    labels_only = [{"labels": label} for label in labels]

    factor = _MM_PER_UNIT.get(electrodes.coordinate_units.lower())
    if factor is None:
        return labels_only, {**record, "located": 0, "reason": UNITS_UNKNOWN}
    if electrodes.coordinate_system in _ALS_SYSTEMS:
        ras = False
    elif electrodes.coordinate_system in _RAS_SYSTEMS:
        ras = True
    else:
        return labels_only, {**record, "located": 0, "reason": COORDINATE_SYSTEM_UNKNOWN}
    rows = [row for row, label in enumerate(labels) if label in electrodes.positions]
    if not rows:
        return labels_only, {**record, "located": 0, "reason": NO_MATCHING_LABELS}

    coords = np.array([electrodes.positions[labels[row]] for row in rows], dtype=np.float64) * factor
    if ras:
        coords = coords[:, [1, 0, 2]]
        coords[:, 1] = -coords[:, 1]
    x, y, z = coords.T
    hypotxy = np.hypot(x, y)
    sph_theta = np.arctan2(y, x) / np.pi * 180
    sph_phi = np.arctan2(z, hypotxy) / np.pi * 180
    columns = {
        "X": x,
        "Y": y,
        "Z": z,
        "sph_theta": sph_theta,
        "sph_phi": sph_phi,
        "sph_radius": np.hypot(hypotxy, z),
        "theta": -sph_theta,
        "radius": 0.5 - sph_phi / 180,
    }

    place = {row: index for index, row in enumerate(rows)}
    chanlocs = []
    for row, label in enumerate(labels):
        chanloc: dict[str, Any] = {"labels": label}
        for field, column in columns.items():
            chanloc[field] = float(column[place[row]]) if row in place else np.array([])
        chanlocs.append(chanloc)
    return chanlocs, {**record, "located": len(rows)}
