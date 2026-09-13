"""Extract channel or component data from EEG datasets."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from eegprep.functions.miscfunc.value_parsing import is_empty_value, parse_key_value_args, parse_numeric_sequence
from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc.eeg_interp import eeg_interp
from eegprep.functions.popfunc.pop_loadset import pop_loadset


def eeg_getdatact(EEG: Any, *args: Any, return_boundaries: bool = False, **kwargs: Any) -> Any:
    """Return selected channel data, component activity, or a backprojection.

    Channel, component, trial, and sample selectors are EEGLAB-facing 1-based
    indices. ``icachansind`` remains an internal 0-based EEGPrep field. By
    default the result has shape ``(signals, samples, trials)``; pass
    ``reshape="2d"`` to concatenate trials along the sample axis.

    Args:
        EEG: One EEG dictionary, a ``.set`` path, or a sequence of datasets.
        *args: Optional EEGLAB-style key/value pairs.
        return_boundaries: Also return continuous boundary offsets.
        **kwargs: ``channel``, ``component``, ``rmcomps``, ``projchan``,
            ``trialindices``, ``samples``, ``interp``, ``reshape``, and
            ``verbose``.

    Returns:
        The selected data, optionally paired with continuous boundary offsets.
    """
    options = parse_key_value_args(args, kwargs, lowercase_keys=True, lowercase_kwargs=True)
    allowed = {
        "channel",
        "component",
        "rmcomps",
        "projchan",
        "trialindices",
        "samples",
        "interp",
        "reshape",
        "verbose",
    }
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unsupported eeg_getdatact option: {sorted(unknown)[0]}")
    reshape = str(options.get("reshape", "3d")).lower()
    if reshape not in {"2d", "3d"}:
        raise ValueError("reshape must be '2d' or '3d'")
    verbose = str(options.get("verbose", "on")).lower()
    if verbose not in {"on", "off"}:
        raise ValueError("verbose must be 'on' or 'off'")
    if _indices(options.get("channel"), name="channel") and _indices(options.get("component"), name="component"):
        raise ValueError("channel and component cannot be used together")
    if options.get("component") is not None and options.get("rmcomps") is not None:
        if _indices(options["component"], name="component") and _indices(options["rmcomps"], name="rmcomps"):
            raise ValueError("component and rmcomps cannot be used together")

    if _is_dataset_sequence(EEG):
        data, boundaries = _concatenate_datasets(list(EEG), options)
    else:
        dataset = pop_loadset(EEG) if isinstance(EEG, (str, Path)) else EEG
        if not isinstance(dataset, dict):
            raise TypeError("EEG must be a dataset dictionary, .set path, or sequence of datasets")
        data, boundaries = _extract_one(dataset, options)
    return (data, boundaries) if return_boundaries else data


def _extract_one(EEG: dict[str, Any], options: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pnts, trials = _metadata_dimensions(EEG)
    channels = _indices(options.get("channel"), name="channel")
    components = _indices(options.get("component"), name="component")
    remove_components = _indices(options.get("rmcomps"), name="rmcomps")
    projection_channels = _projection_channel_indices(EEG, options.get("projchan"))
    raw_interpolation_locations = options.get("interp")
    interpolation_locations = None if is_empty_value(raw_interpolation_locations) else raw_interpolation_locations

    if components:
        if interpolation_locations is not None:
            raise ValueError("interp cannot be used with component extraction")
        selected = _component_data(EEG, None, components, pnts, trials)
    else:
        data3d = _as_3d(_dataset_data(EEG), pnts, trials)
        select_after_interpolation = interpolation_locations is not None
        channel_indices = (
            list(range(data3d.shape[0]))
            if select_after_interpolation
            else _zero_based_selection(channels, data3d.shape[0], "channel")
        )
        selected = np.asarray(data3d[channel_indices], dtype=float).copy()
        if remove_components:
            selected -= _removed_component_projection(EEG, data3d, channel_indices, remove_components)
        if select_after_interpolation:
            selected = _interpolate_channels(EEG, selected, interpolation_locations)
            old_all_channels = channels == list(range(1, data3d.shape[0] + 1))
            requested = [] if not channels or old_all_channels else channels
            selected = selected[_zero_based_selection(requested, selected.shape[0], "channel")]

    if projection_channels:
        if not components:
            raise ValueError("projchan requires a component selection")
        selected = _project_components(EEG, selected, components, projection_channels)

    trial_indices = _zero_based_selection(
        _indices(options.get("trialindices"), name="trialindices"), trials, "trialindices"
    )
    sample_indices = _zero_based_selection(_indices(options.get("samples"), name="samples"), pnts, "samples")
    selected = selected[:, sample_indices, :][:, :, trial_indices]

    reshape = str(options.get("reshape", "3d")).lower()
    if reshape == "2d":
        selected = selected.reshape(selected.shape[0], -1, order="F")
    return selected, _continuous_boundaries(EEG, options.get("samples"))


def _dataset_data(EEG: dict[str, Any]) -> np.ndarray:
    data = EEG.get("data")
    if not isinstance(data, str):
        array = np.asarray(data)
        if array.size == 0:
            raise ValueError("EEG.data is empty")
        return array

    filepath = Path(str(EEG.get("filepath", "")))
    if data.casefold() == "in set file":
        set_file = filepath / str(EEG.get("filename", ""))
        if not set_file.is_file():
            raise FileNotFoundError(f"EEG dataset file not found: {set_file}")
        loaded = pop_loadset(set_file)
        loaded_data = np.asarray(loaded.get("data"))
        if loaded_data.size == 0 or isinstance(loaded.get("data"), str):
            raise ValueError(f"EEG dataset does not contain readable sample data: {set_file}")
        return loaded_data
    filename = filepath / data
    if not filename.exists() and str(EEG.get("filename", "")):
        filename = filepath / Path(str(EEG["filename"])).with_suffix(".fdt")
    if not filename.exists():
        raise FileNotFoundError(f"EEG data file not found: {filename}")
    nbchan = int(EEG["nbchan"])
    pnts = int(EEG["pnts"])
    trials = int(EEG.get("trials", 1))
    raw = np.fromfile(filename, dtype="<f4")
    expected = nbchan * pnts * trials
    if raw.size != expected:
        raise ValueError(f"EEG data file contains {raw.size} values; expected {expected}")
    if filename.suffix.lower() == ".dat":
        frames = raw.reshape(pnts * trials, nbchan, order="F").T
        return frames.reshape(nbchan, pnts, trials, order="F")
    return raw.reshape(nbchan, pnts, trials, order="F")


def _metadata_dimensions(EEG: dict[str, Any]) -> tuple[int, int]:
    pnts = int(EEG.get("pnts", 0))
    trials = int(EEG.get("trials", 1))
    if pnts <= 0 or trials <= 0:
        raise ValueError("EEG pnts and trials must be positive")
    return pnts, trials


def _as_3d(data: np.ndarray, pnts: int, trials: int) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim == 3:
        if array.shape[1:] != (pnts, trials):
            raise ValueError("EEG.data shape does not match pnts and trials")
        return array
    if array.ndim != 2 or array.shape[1] != pnts * trials:
        raise ValueError("EEG.data shape does not match pnts and trials")
    return array.reshape(array.shape[0], pnts, trials, order="F")


def _component_data(
    EEG: dict[str, Any],
    data: np.ndarray | None,
    components: list[int],
    pnts: int | None = None,
    trials: int | None = None,
) -> np.ndarray:
    weights = np.asarray(EEG.get("icaweights", []), dtype=float)
    if weights.ndim != 2 or not weights.size:
        raise ValueError("No ICA weights in dataset")
    component_indices = _zero_based_selection(components, weights.shape[0], "component")
    cached = np.asarray(EEG.get("icaact", []))
    if pnts is None or trials is None:
        if data is None:
            pnts, trials = _metadata_dimensions(EEG)
        else:
            pnts, trials = data.shape[1:]
    if not cached.size:
        cached = _external_component_activity(EEG, weights.shape[0], pnts, trials)
    if cached.size:
        cached3d = _as_3d(cached, pnts, trials)
        return np.asarray(cached3d[component_indices], dtype=float).copy()
    if data is None:
        data = _as_3d(_dataset_data(EEG), pnts, trials)
    weights, sphere, _inverse, ica_channels = _ica_matrices(EEG, require_inverse=False)
    flattened = data[ica_channels].reshape(len(ica_channels), -1, order="F")
    activations = (weights[component_indices] @ sphere) @ flattened
    return activations.reshape(len(component_indices), pnts, trials, order="F")


def _external_component_activity(EEG: dict[str, Any], component_count: int, pnts: int, trials: int) -> np.ndarray:
    set_name = str(EEG.get("filename", ""))
    if not set_name:
        return np.array([])
    filename = Path(str(EEG.get("filepath", ""))) / f"{Path(set_name).stem}.icaact"
    if not filename.is_file():
        return np.array([])
    values = np.fromfile(filename, dtype="<f4")
    expected = component_count * pnts * trials
    if values.size != expected:
        raise ValueError(f"ICA activity file contains {values.size} values; expected {expected}")
    rows = values.reshape(component_count, pnts * trials)
    return np.stack([row.reshape(pnts, trials, order="F") for row in rows])


def _removed_component_projection(
    EEG: dict[str, Any],
    data: np.ndarray,
    selected_channels: list[int],
    components: list[int],
) -> np.ndarray:
    weights, _sphere, inverse, ica_channels = _ica_matrices(EEG)
    component_indices = _zero_based_selection(components, weights.shape[0], "rmcomps")
    activations = _component_data(EEG, data, components, data.shape[1], data.shape[2])
    output = np.zeros((len(selected_channels), data.shape[1], data.shape[2]), dtype=float)
    channel_lookup = {channel: index for index, channel in enumerate(ica_channels)}
    for output_index, channel in enumerate(selected_channels):
        if channel not in channel_lookup:
            continue
        mixing_row = inverse[channel_lookup[channel], component_indices]
        output[output_index] = np.tensordot(mixing_row, activations, axes=(0, 0))
    return output


def _project_components(
    EEG: dict[str, Any],
    activity: np.ndarray,
    components: list[int],
    projection_channels: list[int],
) -> np.ndarray:
    weights, _sphere, inverse, ica_channels = _ica_matrices(EEG)
    component_indices = _zero_based_selection(components, weights.shape[0], "component")
    requested_channels = _zero_based_selection(projection_channels, int(EEG["nbchan"]), "projchan")
    channel_lookup = {channel: index for index, channel in enumerate(ica_channels)}
    missing = [channel + 1 for channel in requested_channels if channel not in channel_lookup]
    if missing:
        raise ValueError(f"Cannot backproject components onto channel {missing[0]} because it was not used for ICA")
    rows = [channel_lookup[channel] for channel in requested_channels]
    flattened = activity.reshape(activity.shape[0], -1, order="F")
    projected = inverse[np.ix_(rows, component_indices)] @ flattened
    return projected.reshape(len(rows), activity.shape[1], activity.shape[2], order="F")


def _interpolate_channels(EEG: dict[str, Any], data: np.ndarray, locations: Any) -> np.ndarray:
    requested_locations = chanlocs_as_list(locations)
    if not requested_locations or not all(isinstance(location, dict) for location in requested_locations):
        raise TypeError("interp must be a channel-location dictionary or sequence of dictionaries")
    temporary = deepcopy(EEG)
    temporary["data"] = data[:, :, 0] if data.shape[2] == 1 else data
    temporary["event"] = []
    temporary["epoch"] = []
    interpolated = eeg_interp(temporary, requested_locations, "spherical", dtype="float64")
    return _as_3d(np.asarray(interpolated["data"]), int(EEG["pnts"]), int(EEG.get("trials", 1)))


def _projection_channel_indices(EEG: dict[str, Any], value: Any) -> list[int]:
    if value is None:
        return []
    raw = value.tolist() if isinstance(value, np.ndarray) else value
    values = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, Sequence) else [raw]
    if values and all(isinstance(item, str) for item in values):
        labels = [str(location.get("labels", "")) for location in chanlocs_as_list(EEG.get("chanlocs", []))]
        lookup: dict[str, list[int]] = {}
        for index, label in enumerate(labels, start=1):
            lookup.setdefault(label.casefold(), []).append(index)
        missing = [str(item) for item in values if str(item).casefold() not in lookup]
        if missing:
            raise ValueError(f"Unknown projection channel label: {missing[0]}")
        duplicates = [str(item) for item in values if len(lookup[str(item).casefold()]) > 1]
        if duplicates:
            raise ValueError(f"Projection channel label is not unique: {duplicates[0]}")
        return [lookup[str(item).casefold()][0] for item in values]
    return _indices(value, name="projchan")


def _ica_matrices(
    EEG: dict[str, Any], *, require_inverse: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    weights = np.asarray(EEG.get("icaweights", []), dtype=float)
    sphere = np.asarray(EEG.get("icasphere", []), dtype=float)
    inverse = np.asarray(EEG.get("icawinv", []), dtype=float)
    if not weights.size or not sphere.size or (require_inverse and not inverse.size):
        raise ValueError("No ICA weights in dataset")
    if weights.ndim != 2 or sphere.ndim != 2 or (require_inverse and inverse.ndim != 2):
        raise ValueError("ICA matrices must be two-dimensional")
    raw_channels = np.asarray(EEG.get("icachansind", []), dtype=int).reshape(-1)
    ica_channels = raw_channels.tolist() if raw_channels.size else list(range(sphere.shape[1]))
    if len(ica_channels) != sphere.shape[1] or any(index < 0 or index >= int(EEG["nbchan"]) for index in ica_channels):
        raise ValueError("icachansind does not match ICA matrix dimensions")
    if weights.shape[1] != sphere.shape[0]:
        raise ValueError("ICA matrix dimensions are inconsistent")
    if require_inverse and inverse.shape != (sphere.shape[1], weights.shape[0]):
        raise ValueError("ICA matrix dimensions are inconsistent")
    return weights, sphere, inverse, ica_channels


def _continuous_boundaries(EEG: dict[str, Any], samples: Any) -> np.ndarray:
    if int(EEG.get("trials", 1)) != 1:
        return np.array([], dtype=float)
    if _indices(samples, name="samples"):
        warnings.warn("Boundary offsets are not adjusted when samples are selected", RuntimeWarning, stacklevel=3)
    boundaries = []
    for event in _records(EEG.get("event", [])):
        if str(event.get("type", "")).casefold() == "boundary":
            boundaries.append(float(event["latency"]) - 0.5)
    return np.asarray(boundaries, dtype=float)


def _concatenate_datasets(datasets: list[dict[str, Any]], options: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if not datasets:
        raise ValueError("At least one EEG dataset is required")
    continuous_flags = [int(dataset.get("trials", 1)) == 1 for dataset in datasets]
    if any(continuous_flags) and not all(continuous_flags):
        raise ValueError("Continuous and epoched datasets cannot be concatenated")
    trial_options = _per_dataset_option(options.get("trialindices"), len(datasets))
    remove_options = _per_dataset_option(options.get("rmcomps"), len(datasets))
    parts: list[np.ndarray] = []
    boundaries: list[float] = []
    continuous = all(continuous_flags)
    offset = 0
    for index, dataset in enumerate(datasets):
        current_options = dict(options)
        current_options["trialindices"] = trial_options[index]
        current_options["rmcomps"] = remove_options[index]
        current_options["reshape"] = "3d"
        part, inner_boundaries = _extract_one(dataset, current_options)
        if parts and part.shape[0] != parts[0].shape[0]:
            raise ValueError("Datasets to be concatenated do not have the same number of signals")
        if continuous:
            if parts:
                boundaries.append(float(offset))
            boundaries.extend((inner_boundaries + offset).tolist())
            offset += part.shape[1]
        else:
            if parts and part.shape[1] != parts[0].shape[1]:
                raise ValueError("Epoched datasets must have the same number of samples")
        parts.append(part)
    axis = 1 if continuous else 2
    result = np.concatenate(parts, axis=axis)
    if str(options.get("reshape", "3d")).lower() == "2d":
        result = result.reshape(result.shape[0], -1, order="F")
    return result, np.asarray(boundaries, dtype=float)


def _indices(value: Any, *, name: str) -> list[int]:
    if value is None:
        return []
    values = parse_numeric_sequence(value, dtype=float)
    output = []
    for value in values:
        if not np.isfinite(value) or not float(value).is_integer() or value < 1:
            raise ValueError(f"{name} must contain positive 1-based integers")
        output.append(int(value))
    return output


def _zero_based_selection(indices: list[int], size: int, name: str) -> list[int]:
    if not indices:
        return list(range(size))
    output = [index - 1 for index in indices]
    if any(index >= size for index in output):
        raise IndexError(f"{name} contains an index outside 1..{size}")
    return output


def _is_dataset_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, Path, dict, np.ndarray))


def _per_dataset_option(value: Any, count: int) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, np.ndarray)):
        values = list(value)
        if len(values) == count and any(isinstance(item, (list, tuple, np.ndarray)) or item is None for item in values):
            return values
    return [value] * count


def _records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return [value]
    return [record for record in value if isinstance(record, dict)]
