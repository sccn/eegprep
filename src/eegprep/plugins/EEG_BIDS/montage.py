"""Montage inference helpers for EEG-BIDS imports."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from importlib.resources import files
import os
from typing import Any

import numpy as np
from scipy.io.matlab import loadmat

from eegprep.plugins.EEG_BIDS.coords import (
    clear_chanloc,
    coords_ALS_to_angular,
    coords_RAS_to_ALS,
    coords_any_to_RAS,
    coords_to_mm,
)


def apply_montage_inference(
    EEG: MutableMapping[str, Any],
    infer_locations: bool | str,
    *,
    numeric_null: Any,
    report: MutableMapping[str, Any],
    warning: Callable[[str], None],
    error: Callable[[str], None],
) -> None:
    """Infer channel locations from packaged or user-specified montage files."""
    if not infer_locations:
        _assign_labelscheme_from_existing_labels(EEG)
        return

    EEG["chaninfo"]["nosedir"] = "+X"
    datalabels = _normalized_data_labels(EEG["chanlocs"])
    montage_path, filenames = _montage_candidates(infer_locations)
    opt_score, best_data, best_cap, fractions = _select_best_montage(montage_path, filenames, datalabels)
    best_fraction = opt_score[0]

    if best_data is None:
        if isinstance(infer_locations, str):
            raise RuntimeError(
                f"The channel labels in your data do not match the specified montage file ({infer_locations})."
            )
        raise RuntimeError("Channel labels do not match any known or specified montage.")

    skip_locations = _should_skip_or_warn(
        best_fraction,
        fractions,
        datalabels,
        warning=warning,
        error=error,
    )
    if skip_locations:
        return

    report["ChanlocsFrom"] = os.path.basename(best_cap)
    if "10-5" in best_cap:
        labeling = "10-20"
    else:
        labeling, _ext = os.path.splitext(os.path.basename(best_cap))
    EEG["etc"]["labelscheme"] = labeling
    _apply_montage_coordinates(EEG, best_data, datalabels, numeric_null)


def _strip_matching_quotes(name: str) -> str:
    while len(name) >= 2 and name[0] == name[-1] and name[0] in ("'", '"'):
        name = name[1:-1]
    return name


def _normalized_data_labels(chanlocs: list[dict[str, Any]]) -> list[str]:
    datalabels = [chanloc["labels"].lower() for chanloc in chanlocs]
    for prefix in ["brainvision rda_", "rda_", "eeg ", "eeg-", "eeg"]:
        datalabels = [label.replace(prefix, "") for label in datalabels]
    datalabels = [label.split("-")[0] for label in datalabels]
    return [_strip_matching_quotes(label) for label in datalabels]


def _montage_candidates(infer_locations: bool | str) -> tuple[str, list[str]]:
    montage_path = str(files("eegprep").joinpath("resources").joinpath("montages"))
    if not os.path.isdir(montage_path):
        raise RuntimeError(
            f"Could not find montages directory at {montage_path}. This may indicate a corrupted installation."
        )

    if isinstance(infer_locations, str):
        if os.path.isabs(infer_locations):
            return os.path.dirname(infer_locations), [os.path.basename(infer_locations)]
        return montage_path, [infer_locations]
    return montage_path, sorted(os.listdir(montage_path))


def _select_best_montage(
    montage_path: str, filenames: list[str], datalabels: list[str]
) -> tuple[tuple[float, int, float], Any | None, str, list[float]]:
    opt_score: tuple[float, int, float] = (0, 0, 0)
    best_data = None
    best_cap = "(not set)"
    fractions: list[float] = []
    for filename in filenames:
        if not filename.endswith(".locs"):
            continue
        try:
            data = loadmat(os.path.join(montage_path, filename), squeeze_me=True)
        except Exception as exc:
            raise ValueError(
                f"Failed to load montage file {filename}. Make sure it is a valid .locs file (MATLAB v7 .mat format)."
            ) from exc
        caplabels = [label.lower() for label in data["labels"]]
        fraction_in_data = np.mean([label in caplabels for label in datalabels])
        fraction_in_locfile = np.mean([label in datalabels for label in caplabels])
        bonus1020 = 1 if {"c3", "cz", "fcz", "c4"}.issubset(caplabels) else 0
        score = (fraction_in_data, bonus1020, fraction_in_locfile)
        if score > opt_score:
            opt_score = score
            best_data = data
            best_cap = filename
        fractions.append(fraction_in_data)
    return opt_score, best_data, best_cap, sorted(fractions, reverse=True)


def _should_skip_or_warn(
    best_fraction: float,
    fractions: list[float],
    datalabels: list[str],
    *,
    warning: Callable[[str], None],
    error: Callable[[str], None],
) -> bool:
    percent_found = int(100 * best_fraction)
    if best_fraction < 0.25:
        error(
            "The given data has a very poor match to all "
            "known montages (%s percent of channels found); "
            "not assigning locations (got: %s)" % (percent_found, datalabels)
        )
        return True
    if best_fraction < 0.5:
        if len(fractions) > 1 and best_fraction / 1.5 < fractions[1]:
            warning(
                "The given data has a poor match and multiple "
                "montages are partially matching potentially "
                "ambiguously (%s percent of channels found); "
                "please double-check assigned locations." % percent_found
            )
        else:
            warning(
                "The given data has a poor match to all known "
                "montages (%s percent of channels found); please "
                "double-check assigned locations." % percent_found
            )
    elif best_fraction < 0.75 and len(fractions) > 1 and best_fraction / 1.5 < fractions[1]:
        warning(
            "The given data has a reasonable match to known "
            "montages but multiple montages are potentially "
            "matching (%s percent of channels found); "
            "locations may be wrong." % percent_found
        )
    elif best_fraction < 1.0:
        warning(
            "Not all channel locations could be matched to a "
            "known montage; some channels may be non-EEG "
            "channels ({} percent of channels found).".format(percent_found)
        )
    return False


def _apply_montage_coordinates(
    EEG: MutableMapping[str, Any],
    best_data: MutableMapping[str, Any],
    datalabels: list[str],
    numeric_null: Any,
) -> None:
    unit = best_data["meta"]["unit"][()]
    x = best_data["meta"]["x"][()]
    y = best_data["meta"]["y"][()]
    z = best_data["meta"]["z"][()]
    coords = best_data["coordinates"]
    coords = coords_to_mm(coords, unit)
    coords = coords_any_to_RAS(coords, x, y, z)
    coords = coords_RAS_to_ALS(coords)
    sph_theta, sph_phi, sph_radius, polar_theta, polar_radius = coords_ALS_to_angular(coords)
    caplabels = [label.lower() for label in best_data["labels"]]
    for data_index, data_label in enumerate(datalabels):
        record = EEG["chanlocs"][data_index]
        for cap_index, cap_label in enumerate(caplabels):
            if data_label != cap_label:
                continue
            xyz = coords[cap_index, :]
            record["X"] = xyz[0]
            record["Y"] = xyz[1]
            record["Z"] = xyz[2]
            record["sph_radius"] = sph_radius[cap_index]
            record["sph_theta"] = sph_theta[cap_index]
            record["sph_phi"] = sph_phi[cap_index]
            record["theta"] = polar_theta[cap_index]
            record["radius"] = polar_radius[cap_index]
            break
        else:
            clear_chanloc(record, numeric_null)


def _assign_labelscheme_from_existing_labels(EEG: MutableMapping[str, Any]) -> None:
    candidate_locs = {"fp1", "fp2", "fz", "t3", "cz", "t4", "t5", "p3", "pz", "p4", "t6", "o1", "o2"}
    if any(chanloc["labels"].lower() in candidate_locs for chanloc in EEG["chanlocs"]):
        EEG["etc"]["labelscheme"] = "10-20"
    else:
        EEG["etc"]["labelscheme"] = "unknown"
