"""Query neighboring urevents around selected target events."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.popfunc._event_utils import events_as_list, is_boundary_event


def eeg_context(
    EEG: dict[str, Any],
    targets: Any | None = None,
    neighbors: Any | None = None,
    positions: Any | None = None,
    field: str | list[str] | None = None,
    alltargs: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, Any]:
    """Return target-event context from the original urevent sequence.

    Event identifiers in the returned matrices are 1-based. Neighbor searches
    stop at boundary events and delays are returned in milliseconds.
    """
    events = events_as_list(EEG.get("event", []))
    urevents = events_as_list(EEG.get("urevent", []))
    if not events or not urevents:
        raise ValueError("EEG.event and EEG.urevent are required")
    if any("urevent" not in event for event in events if not is_boundary_event(event)):
        raise ValueError("every EEG.event must point to an urevent")
    all_targets = alltargs.lower() if isinstance(alltargs, str) else alltargs
    if all_targets not in (None, "", "all"):
        raise ValueError("alltargs must be 'all' or empty")

    target_types = _type_list(targets)
    neighbor_types = _type_list(neighbors)
    requested_positions = np.sort(np.asarray([1] if _empty(positions) else positions, dtype=int).ravel())
    field_names = [] if field is None else ([field] if isinstance(field, str) else list(field))
    for name in field_names:
        if not any(name in event for event in urevents):
            raise ValueError(f"specified field {name!r} not found in urevent")
    continuous = int(EEG.get("trials", 1)) == 1 or not any("epoch" in event for event in events)
    return_all = continuous or all_targets == "all"

    target_rows: list[list[float]] = []
    neighbor_rows: list[list[float]] = []
    neighbor_type_rows: list[list[float]] = []
    delay_rows: list[list[float]] = []
    target_fields: list[list[Any]] = []
    neighbor_fields: list[list[list[Any]]] = []

    for event_index, event in enumerate(events):
        if is_boundary_event(event):
            continue
        urevent_index = int(event["urevent"])
        if urevent_index < 0 or urevent_index >= len(urevents):
            raise ValueError("event urevent pointer is outside EEG.urevent")
        target_type = _type_text(urevents[urevent_index].get("type"))
        target_type_index = _matching_type_index(target_type, target_types)
        if target_type_index is None:
            continue
        centered_epoch = _centered_epoch(EEG, event_index, event)
        if not return_all and not np.isfinite(centered_epoch):
            continue
        target_rows.append([event_index + 1, urevent_index + 1, centered_epoch, target_type_index])
        row_neighbors: list[float] = []
        row_neighbor_types: list[float] = []
        row_delays: list[float] = []
        row_fields: list[list[Any]] = []
        for position in requested_positions:
            match = _neighbor_at(urevents, urevent_index, int(position), neighbor_types)
            if match is None:
                row_neighbors.append(np.nan)
                row_neighbor_types.append(np.nan)
                row_delays.append(np.nan)
                row_fields.append([[] for _ in field_names])
                continue
            neighbor_index, neighbor_type_index = match
            row_neighbors.append(float(neighbor_index + 1))
            row_neighbor_types.append(float(neighbor_type_index))
            row_delays.append(
                1000
                / float(EEG["srate"])
                * (float(urevents[neighbor_index]["latency"]) - float(urevents[urevent_index]["latency"]))
            )
            row_fields.append([urevents[neighbor_index].get(name) for name in field_names])
        neighbor_rows.append(row_neighbors)
        neighbor_type_rows.append(row_neighbor_types)
        delay_rows.append(row_delays)
        target_fields.append([urevents[urevent_index].get(name) for name in field_names])
        neighbor_fields.append(row_fields)

    target_array = np.asarray(target_rows, dtype=float).reshape(-1, 4)
    output_shape = (len(target_rows), len(requested_positions))
    neighbor_array = np.asarray(neighbor_rows, dtype=float).reshape(output_shape)
    neighbor_type_array = np.asarray(neighbor_type_rows, dtype=float).reshape(output_shape)
    delay_array = np.asarray(delay_rows, dtype=float).reshape(output_shape)
    if not field_names:
        return target_array, neighbor_array, neighbor_type_array, delay_array, [], []
    if _fields_are_numeric(urevents, field_names):
        target_field_array = np.asarray(
            [[_numeric_or_nan(value) for value in values] for values in target_fields],
            dtype=float,
        ).reshape(len(target_fields), len(field_names))
        neighbor_field_array = np.asarray(
            [[[_numeric_or_nan(value) for value in values] for values in positions] for positions in neighbor_fields],
            dtype=float,
        ).reshape(len(neighbor_fields), len(requested_positions), len(field_names))
    else:
        target_field_array = np.empty((len(target_fields), len(field_names)), dtype=object)
        neighbor_field_array = np.empty(
            (len(neighbor_fields), len(requested_positions), len(field_names)),
            dtype=object,
        )
        for target_index, values in enumerate(target_fields):
            target_field_array[target_index, :] = values
            for position_index, neighbor_values in enumerate(neighbor_fields[target_index]):
                neighbor_field_array[target_index, position_index, :] = neighbor_values
    if len(field_names) == 1:
        target_field_array = target_field_array[:, 0]
        neighbor_field_array = neighbor_field_array[:, :, 0]
    return (
        target_array,
        neighbor_array,
        neighbor_type_array,
        delay_array,
        target_field_array,
        neighbor_field_array,
    )


def _type_list(value: Any | None) -> list[str]:
    if _empty(value):
        return ["_ALL"]
    values = [value] if isinstance(value, (str, int, float)) else list(value)
    return [_type_text(item) for item in values]


def _matching_type_index(value: str, types: list[str]) -> int | None:
    if "_ALL" in types:
        return 1
    for index, candidate in enumerate(types, start=1):
        if value.lower() == candidate.lower():
            return index
    return None


def _centered_epoch(EEG: dict[str, Any], event_index: int, event: dict[str, Any]) -> float:
    epochs = event.get("epoch", [])
    if not isinstance(epochs, (list, tuple, np.ndarray)):
        epochs = [epochs]
    epoch_records = EEG.get("epoch", [])
    for epoch_number in np.asarray(epochs).ravel():
        index = int(epoch_number) - 1
        if index < 0 or index >= len(epoch_records):
            continue
        record = epoch_records[index]
        event_numbers = np.asarray(record.get("event", [])).ravel()
        latencies = record.get("eventlatency", [])
        if not isinstance(latencies, (list, tuple, np.ndarray)):
            latencies = [latencies]
        for position, number in enumerate(event_numbers):
            if int(number) == event_index and position < len(latencies):
                latency = np.asarray(latencies[position]).ravel()
                if latency.size and float(latency[0]) == 0:
                    return float(epoch_number)
    return np.nan


def _neighbor_at(
    urevents: list[dict[str, Any]],
    target_index: int,
    position: int,
    types: list[str],
) -> tuple[int, int] | None:
    if position == 0:
        type_index = _matching_type_index(_type_text(urevents[target_index].get("type")), types)
        return (target_index, type_index) if type_index is not None else None
    direction = 1 if position > 0 else -1
    remaining = abs(position)
    index = target_index + direction
    while 0 <= index < len(urevents):
        event = urevents[index]
        if is_boundary_event(event):
            return None
        type_index = _matching_type_index(_type_text(event.get("type")), types)
        if type_index is not None:
            remaining -= 1
            if remaining == 0:
                return index, type_index
        index += direction
    return None


def _empty(value: Any | None) -> bool:
    if value is None or (isinstance(value, str) and value == ""):
        return True
    try:
        return np.asarray(value).size == 0
    except ValueError:
        return False


def _type_text(value: Any) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _fields_are_numeric(events: list[dict[str, Any]], field_names: list[str]) -> bool:
    for name in field_names:
        values = [event.get(name) for event in events if not _empty(event.get(name))]
        if not values or any(not _is_numeric_scalar(value) for value in values):
            return False
    return True


def _is_numeric_scalar(value: Any) -> bool:
    array = np.asarray(value)
    return array.size == 1 and np.issubdtype(array.dtype, np.number)


def _numeric_or_nan(value: Any) -> float:
    if _empty(value):
        return np.nan
    return float(np.asarray(value).reshape(-1)[0])


__all__ = ["eeg_context"]
