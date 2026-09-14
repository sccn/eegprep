"""Match STUDY ICA components to a template scalp map by correlation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args
from eegprep.functions.popfunc.plot_utils import component_map_data
from eegprep.functions.studyfunc._cluster_utils import ensure_parent_cluster, rows_for_cluster
from eegprep.functions.studyfunc._study_utils import as_alleeg_list
from eegprep.functions.studyfunc.std_checkset import std_checkset
from eegprep.functions.studyfunc.std_createclust import std_createclust
from eegprep.functions.studyfunc.std_findsameica import std_findsameica


AUTO_THRESHOLDS = tuple(value / 100.0 for value in range(95, 54, -1))


def corrmap(
    STUDY: dict[str, Any] | None,
    ALLEEG: Any,
    n_tmp: int,
    index: int,
    *args: Any,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Find ICA maps correlated with one template component.

    The calculation follows CORRMAP's two-pass procedure: match every unique
    ICA decomposition to the requested component, build a polarity-aligned
    average map, then match again to that average. Automatic mode evaluates
    thresholds from 0.95 through 0.55 and chooses the most self-consistent
    pair of averages.

    Args:
        STUDY: EEGPrep STUDY dictionary.
        ALLEEG: Loaded EEG datasets.
        n_tmp: One-based template dataset index in ``ALLEEG``.
        index: One-based template ICA component.
        *args: EEGLAB-style option/value pairs.
        **kwargs: Options ``th``, ``ics``, ``clname``, ``badcomps``,
            ``resetclusters``, ``chanlocs``, ``title``, ``pl``, and ``plot``.

    Returns:
        ``(CORRMAP, STUDY, ALLEEG)`` with correlations, selected components,
        polarity-aligned averages, and any requested cluster updates.

    Notes:
        Component maps are RMS-normalized before averaging. Correlation is
        scale invariant, and the normalization prevents arbitrary ICA column
        scaling from weighting the second-pass template.
    """
    options = _options(args, kwargs)
    datasets = as_alleeg_list(ALLEEG)
    if not datasets:
        raise ValueError("corrmap requires loaded ALLEEG datasets")
    template_dataset = _one_based(n_tmp, len(datasets), "template dataset")
    if options["plot"]:
        raise NotImplementedError("CORRMAP summary plotting is not implemented; use plot='off'")
    if _has_value(options["chanlocs"]):
        raise NotImplementedError(
            "CORRMAP channel interpolation is not implemented; align montages with std_interp first"
        )

    study, datasets = std_checkset(STUDY, datasets)
    groups, _group_indices = std_findsameica(datasets)
    dataset_indices = [group[0] for group in groups]
    selected_datasets = [datasets[value - 1] for value in dataset_indices]
    template_group = next(
        (group_index for group_index, group in enumerate(groups) if template_dataset in group),
        None,
    )
    if template_group is None:
        raise ValueError("template dataset has no ICA decomposition")

    maps, chanlocs = _aligned_component_maps(selected_datasets)
    template_component = _one_based(index, maps[template_group].shape[1], "template component")
    template = maps[template_group][:, template_component - 1]
    _validate_template(template)
    match_count = options["ics"]
    if any(values.shape[1] < match_count for values in maps):
        raise ValueError("corrmap ics cannot exceed the component count of any selected dataset")

    if options["threshold"] == "auto":
        candidates = []
        for threshold in AUTO_THRESHOLDS:
            result = _two_pass(maps, template, template_group, template_component - 1, match_count, threshold)
            if result is not None:
                candidates.append((result["similarity"], threshold, result))
        if not candidates:
            raise ValueError("No ICA components exceed the lowest automatic CORRMAP threshold (0.55)")
        _similarity, threshold, selected = max(candidates, key=lambda item: (item[0], item[1]))
    else:
        threshold = float(options["threshold"])
        selected = _two_pass(maps, template, template_group, template_component - 1, match_count, threshold)
        if selected is None:
            raise ValueError(f"No ICA components exceed the CORRMAP threshold ({threshold:g})")

    first, second = selected["passes"]
    info = _result_structure(
        datasets,
        dataset_indices,
        template_dataset,
        template_component,
        chanlocs,
        options,
        threshold,
        first,
        second,
        selected["similarity"],
    )
    output_datasets = deepcopy(datasets)
    if options["badcomps"]:
        _store_bad_components(output_datasets, dataset_indices, second)
    if options["clname"]:
        study = _store_cluster(study, output_datasets, dataset_indices, second, options)
    study["saved"] = "no"
    return info, study, output_datasets


def _options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    supplied = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    allowed = {"chanlocs", "th", "ics", "pl", "resetclusters", "plot", "title", "clname", "badcomps"}
    unknown = set(supplied) - allowed
    if unknown:
        raise ValueError(f"Unknown corrmap option(s): {', '.join(sorted(unknown))}")
    threshold_value = supplied.get("th", "auto")
    if isinstance(threshold_value, str) and threshold_value.strip().lower() == "auto":
        threshold: str | float = "auto"
    else:
        threshold = float(threshold_value)
        if not 0.0 < threshold < 1.0:
            raise ValueError("corrmap th must be 'auto' or a value strictly between 0 and 1")
    ics = int(supplied.get("ics", 2))
    if ics not in {1, 2, 3}:
        raise ValueError("corrmap ics must be 1, 2, or 3")
    pl = str(supplied.get("pl", "2nd")).lower()
    if pl not in {"none", "2nd", "both"}:
        raise ValueError("corrmap pl must be 'none', '2nd', or 'both'")
    return {
        "chanlocs": supplied.get("chanlocs", []),
        "threshold": threshold,
        "ics": ics,
        "pl": pl,
        "resetclusters": is_on(supplied.get("resetclusters", False)),
        "plot": is_on(supplied.get("plot", False)),
        "title": str(supplied.get("title") or ""),
        "clname": str(supplied.get("clname") or ""),
        "badcomps": is_on(supplied.get("badcomps", False)),
    }


def _one_based(value: Any, maximum: int, label: str) -> int:
    number = int(value)
    if number != value or number < 1 or number > maximum:
        raise ValueError(f"corrmap {label} must be 1-based and within 1..{maximum}")
    return number


def _aligned_component_maps(datasets: list[dict[str, Any]]) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    first_maps, first_locs = component_map_data(datasets[0])
    reference_labels = _labels(first_locs)
    output = [np.asarray(first_maps, dtype=float)]
    for eeg in datasets[1:]:
        values, locs = component_map_data(eeg)
        values = np.asarray(values, dtype=float)
        labels = _labels(locs)
        if reference_labels and labels:
            if len(set(labels)) != len(labels) or len(set(reference_labels)) != len(reference_labels):
                raise ValueError("corrmap requires unique channel labels")
            lookup = {label: position for position, label in enumerate(labels)}
            if set(lookup) != set(reference_labels):
                raise ValueError("corrmap requires the same channel montage in every ICA decomposition")
            values = values[[lookup[label] for label in reference_labels], :]
        elif values.shape[0] != first_maps.shape[0]:
            raise ValueError("corrmap requires equal component-map channel counts")
        output.append(values)
    if any(not np.isfinite(values).all() for values in output):
        raise ValueError("corrmap component maps must contain only finite values")
    return output, chanlocs_as_list(first_locs)


def _labels(chanlocs: list[dict[str, Any]]) -> list[str]:
    labels = [str(loc.get("labels") or "").strip().casefold() for loc in chanlocs]
    return labels if labels and all(labels) else []


def _validate_template(template: np.ndarray) -> None:
    if template.size < 2 or not np.isfinite(template).all() or np.allclose(template, template[0]):
        raise ValueError("corrmap template component must be finite and spatially nonconstant")


def _two_pass(
    maps: list[np.ndarray],
    template: np.ndarray,
    template_dataset: int,
    template_component: int,
    ics: int,
    threshold: float,
) -> dict[str, Any] | None:
    first = _match_pass(maps, template, ics, threshold, exclude=(template_dataset, template_component))
    if first is None:
        return None
    second = _match_pass(maps, first["average"], ics, threshold)
    if second is None:
        return None
    similarity = abs(_correlation(first["average"], second["average"]))
    return {"passes": (first, second), "similarity": similarity}


def _match_pass(
    maps: list[np.ndarray],
    template: np.ndarray,
    ics: int,
    threshold: float,
    *,
    exclude: tuple[int, int] | None = None,
) -> dict[str, Any] | None:
    candidates = []
    for dataset_index, values in enumerate(maps):
        correlations = np.asarray(
            [_correlation(template, values[:, component]) for component in range(values.shape[1])]
        )
        if exclude is not None and exclude[0] == dataset_index:
            correlations[exclude[1]] = 0.0
        order = np.argsort(-np.abs(correlations), kind="stable")[:ics]
        for component in order:
            correlation = float(correlations[component])
            candidates.append((abs(correlation), dataset_index, int(component), correlation))
    candidates.sort(key=lambda item: -item[0])
    absolute = np.asarray([item[0] for item in candidates])
    candidate_sets = np.asarray([item[1] + 1 for item in candidates], dtype=int)
    candidate_components = np.asarray([item[2] + 1 for item in candidates], dtype=int)
    signed = np.asarray([item[3] for item in candidates])
    selected_count = int(np.count_nonzero(absolute > threshold))
    if selected_count == 0:
        return None
    selected_candidates = candidates[:selected_count]
    selected_absolute = absolute[:selected_count]
    sets = candidate_sets[:selected_count]
    components = candidate_components[:selected_count]
    selected_signed = signed[:selected_count]
    oriented = []
    for _absolute, dataset_index, component, correlation in selected_candidates:
        values = maps[dataset_index][:, component]
        rms = float(np.sqrt(np.mean(values**2)))
        if rms == 0.0:
            continue
        oriented.append(values / rms * (1.0 if correlation >= 0.0 else -1.0))
    if not oriented:
        return None
    counts = np.bincount(sets, minlength=len(maps) + 1)[1:]
    return {
        "abs_values": absolute,
        "signed_values": signed,
        "candidate_sets": candidate_sets,
        "candidate_ics": candidate_components,
        "sets": sets,
        "ics": components,
        "polarity": np.where(selected_signed < 0.0, -1, 1),
        "average": np.mean(np.vstack(oriented), axis=0),
        "mean_corr": float(np.tanh(np.mean(np.arctanh(np.clip(selected_absolute, 0.0, 1.0 - 1e-12))))),
        "dataset_counts": counts,
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = np.asarray(left, dtype=float) - float(np.mean(left))
    right_centered = np.asarray(right, dtype=float) - float(np.mean(right))
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0:
        return 0.0
    return float(np.clip(np.dot(left_centered, right_centered) / denominator, -1.0, 1.0))


def _result_structure(
    datasets: list[dict[str, Any]],
    dataset_indices: list[int],
    template_dataset: int,
    template_component: int,
    chanlocs: list[dict[str, Any]],
    options: dict[str, Any],
    threshold: float,
    first: dict[str, Any],
    second: dict[str, Any],
    similarity: float,
) -> dict[str, Any]:
    passes = (first, second)
    absent = [(np.flatnonzero(values["dataset_counts"] == 0) + 1).astype(int) for values in passes]
    return {
        "datasetindices": np.asarray(dataset_indices, dtype=int),
        "template": {
            "setname": str(datasets[template_dataset - 1].get("setname") or ""),
            "index": template_dataset,
            "ic": template_component,
        },
        "datasets": {
            "setnames": [str(datasets[value - 1].get("setname") or "") for value in dataset_indices],
            "index": np.arange(1, len(dataset_indices) + 1, dtype=int),
            "ics": np.asarray([np.asarray(datasets[value - 1]["icawinv"]).shape[1] for value in dataset_indices]),
        },
        "input": {
            "chanlocs": options["chanlocs"],
            "corr_th": options["threshold"],
            "ics_sel": options["ics"],
            "plots": options["pl"],
            "title": options["title"],
            "clname": options["clname"],
            "badcomps": "yes" if options["badcomps"] else "no",
        },
        "corr": {
            "abs_values": [values["abs_values"] for values in passes],
            "signed_values": [values["signed_values"] for values in passes],
            "sets": [values["candidate_sets"] for values in passes],
            "ics": [values["candidate_ics"] for values in passes],
        },
        "clust": {
            "best_th": threshold,
            "ics": np.asarray([values["ics"].size for values in passes], dtype=int),
            "sets": {
                "number": np.asarray([np.count_nonzero(values["dataset_counts"]) for values in passes], dtype=int),
                "more_oneIC": np.asarray(
                    [np.count_nonzero(values["dataset_counts"] > 1) for values in passes], dtype=int
                ),
                "absent": absent,
            },
            "mean_corr": np.asarray([values["mean_corr"] for values in passes]),
            "similarity": similarity,
        },
        "output": {
            "chanlocs": deepcopy(chanlocs),
            "average_plot": [values["average"] for values in passes],
            "sets": [values["sets"] for values in passes],
            "ics": [values["ics"] for values in passes],
            "polarity": [values["polarity"] for values in passes],
        },
    }


def _store_bad_components(datasets: list[dict[str, Any]], dataset_indices: list[int], selected: dict[str, Any]) -> None:
    for selected_set, component in zip(selected["sets"], selected["ics"], strict=True):
        dataset_index = dataset_indices[int(selected_set) - 1] - 1
        existing_value = datasets[dataset_index].get("badcomps")
        existing = (
            [int(value) for value in np.asarray(existing_value, dtype=int).ravel()]
            if _has_value(existing_value)
            else []
        )
        datasets[dataset_index]["badcomps"] = sorted(set([*existing, int(component)]))


def _has_value(value: Any) -> bool:
    if value is None or isinstance(value, str) and not value:
        return False
    if isinstance(value, np.ndarray):
        return value.size > 0
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True


def _store_cluster(
    study: dict[str, Any],
    datasets: list[dict[str, Any]],
    dataset_indices: list[int],
    selected: dict[str, Any],
    options: dict[str, Any],
) -> dict[str, Any]:
    study = deepcopy(study)
    if options["resetclusters"]:
        study["cluster"] = []
        for dataset_index, info in enumerate(study.get("datasetinfo") or []):
            component_count = np.asarray(datasets[dataset_index].get("icaweights", [])).shape[0]
            info["comps"] = list(range(1, component_count + 1))
    study = ensure_parent_cluster(study, datasets)
    parent_sets, parent_comps = rows_for_cluster(study, datasets, 1)
    matched = {
        (dataset_indices[int(selected_set) - 1], int(component))
        for selected_set, component in zip(selected["sets"], selected["ics"], strict=True)
    }
    labels = np.zeros(parent_comps.size, dtype=int)
    for row, component in enumerate(parent_comps):
        if (int(parent_sets[0, row]), int(component)) in matched:
            labels[row] = 1
    if not np.any(labels):
        raise ValueError("CORRMAP matches are absent from the STUDY parent cluster")
    return std_createclust(
        study,
        datasets,
        clusterind=labels,
        algorithm=["correlation (CORRMAP)", float(options["ics"])],
        name=options["clname"],
        ignore0="on",
    )


__all__ = ["AUTO_THRESHOLDS", "corrmap"]
