"""Convert ICA maps and electrode coordinates for DIPFIT source localization."""

import logging

import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc.plot_utils import component_channel_indices, component_maps
from eegprep.plugins.dipfit._coordinates import apply_transform, traditionaldipfit
from eegprep.plugins.dipfit._utils import ensure_channel_locations, ensure_dipfit_settings, ensure_ica


def eeglab2fieldtrip(EEG: dict, fieldbox="componentanalysis", transform="dipfit") -> dict:
    """Return FieldTrip component topographies and electrodes as plain dictionaries."""
    if fieldbox not in ("componentanalysis", "comp"):
        raise ValueError("DIPFIT supports componentanalysis conversion only")
    ensure_ica(EEG)
    ensure_channel_locations(EEG)
    channels = chanlocs_as_list(EEG["chanlocs"])
    positions = np.zeros((len(channels), 3))
    for index, channel in enumerate(channels):
        xyz = [np.asarray(channel.get(key, [])).ravel() for key in ("X", "Y", "Z")]
        if all(value.size and np.isfinite(float(value[0])) for value in xyz):
            positions[index] = [value[0] for value in xyz]
        elif all(np.asarray(channel.get(key, [])).size for key in ("theta", "radius")):
            angle = np.deg2rad(float(channel["theta"]))
            radius = min(abs(float(channel["radius"])) * 2, 1)
            positions[index] = [radius * np.cos(angle), radius * np.sin(angle), np.sqrt(1 - radius**2)]
    if transform == "dipfit":
        cfg = ensure_dipfit_settings(EEG)
        matrix = np.asarray(cfg.get("coord_transform", []), dtype=float)
        if matrix.size:
            positions = apply_transform(traditionaldipfit(matrix), positions)
        elif str(cfg.get("coordformat", "")).lower() == "mni":
            logging.getLogger(__name__).warning("DIPFIT has an empty coord_transform")
    elif transform != "none":
        raise ValueError("transform must be 'dipfit' or 'none'")
    labels = [str(channel.get("labels") or index + 1) for index, channel in enumerate(channels)]
    maps = np.asarray(component_maps(EEG), dtype=float)
    rows = component_channel_indices(EEG, len(channels))
    return {
        "topo": maps,
        "topolabel": [labels[index] for index in rows],
        "label": [f"ica_{index + 1:03d}" for index in range(maps.shape[1])],
        "elec": {"elecpos": positions, "chanpos": positions.copy(), "label": labels, "unit": "mm"},
    }
