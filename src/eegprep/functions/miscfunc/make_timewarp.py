"""Select ordered epoch events for event-latency time warping."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np


Condition = str | Callable[[Mapping[str, Any]], bool] | None

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_COMPARISON_OPERATORS: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def make_timewarp(
    eeg: Mapping[str, Any],
    event_sequence: Sequence[Any],
    *,
    baseline_latency: float = 0.0,
    event_conditions: Sequence[Condition] | None = None,
    max_std_for_absolute: float = np.inf,
    max_std_for_relative: float = np.inf,
) -> dict[str, Any]:
    """Select epochs containing an ordered event sequence.

    Conditions may be callables receiving the current event-field mapping or
    simple arithmetic/comparison strings such as ``"latency < 200"``. Strings
    are parsed with a restricted expression evaluator; arbitrary code is never
    executed. Returned epoch indices are zero-based.
    """
    sequence = list(event_sequence)
    if not sequence:
        raise ValueError("event_sequence must contain at least one event type")
    if not np.isfinite(baseline_latency):
        raise ValueError("baseline_latency must be finite")
    absolute_limit = _standard_deviation_limit(max_std_for_absolute, "max_std_for_absolute")
    relative_limit = _standard_deviation_limit(max_std_for_relative, "max_std_for_relative")
    conditions = [] if event_conditions is None else list(event_conditions)
    if len(conditions) > len(sequence):
        raise ValueError("event_conditions cannot be longer than event_sequence")
    conditions.extend([None] * (len(sequence) - len(conditions)))

    selected_latencies: list[list[float]] = []
    selected_epochs: list[int] = []
    epochs = list(eeg.get("epoch", []))
    for epoch_index, epoch in enumerate(epochs):
        matches = _ordered_matches(epoch, sequence, conditions, float(baseline_latency))
        if matches is not None:
            selected_epochs.append(epoch_index)
            selected_latencies.append(matches)

    latencies = np.asarray(selected_latencies, dtype=float)
    if latencies.size == 0:
        latencies = np.empty((0, len(sequence)), dtype=float)
    rejected = _outlier_rows(latencies, absolute_limit, relative_limit)
    if rejected.size:
        keep = np.ones(latencies.shape[0], dtype=bool)
        keep[rejected] = False
        latencies = latencies[keep]
        selected_epochs = np.asarray(selected_epochs, dtype=int)[keep].tolist()
    return {
        "latencies": latencies,
        "epochs": np.asarray(selected_epochs, dtype=int),
        "event_sequence": sequence,
    }


def _ordered_matches(
    epoch: Mapping[str, Any],
    sequence: Sequence[Any],
    conditions: Sequence[Condition],
    baseline_latency: float,
) -> list[float] | None:
    event_types = _event_values(epoch.get("eventtype", []))
    event_latencies = _event_values(epoch.get("eventlatency", []))
    if len(event_types) != len(event_latencies):
        raise ValueError("each epoch must have one eventlatency per eventtype")
    records = [
        (float(_scalar_value(latency)), index) for index, latency in enumerate(event_latencies) if np.size(latency) == 1
    ]
    if len(records) != len(event_latencies) or not all(np.isfinite(latency) for latency, _index in records):
        raise ValueError("epoch event latencies must be finite scalars")
    records.sort(key=lambda item: (item[0], item[1]))

    minimum = baseline_latency
    result: list[float] = []
    for requested_type, condition in zip(sequence, conditions):
        match = None
        for latency, event_index in records:
            if latency < minimum:
                continue
            if not _event_type_matches(event_types[event_index], requested_type):
                continue
            fields = _condition_fields(epoch, event_index)
            if not _condition_matches(condition, fields):
                continue
            match = latency
            break
        if match is None:
            return None
        result.append(match)
        minimum = match
    return result


def _event_values(value: Any) -> list[Any]:
    if value is None:
        return []
    array = np.asarray(value, dtype=object)
    if array.ndim == 0:
        return [array.item()]
    return array.reshape(-1).tolist()


def _scalar_value(value: Any) -> Any:
    array = np.asarray(value)
    return array.reshape(-1)[0].item() if array.size == 1 else value


def _event_type_matches(value: Any, requested: Any) -> bool:
    choices = requested if isinstance(requested, (list, tuple, set, np.ndarray)) else [requested]
    actual = _scalar_value(value)
    for choice in choices:
        choice = _scalar_value(choice)
        if isinstance(choice, str) and isinstance(actual, (int, float, np.integer, np.floating)):
            if format(float(actual), "g") == choice:
                return True
        elif actual == choice:
            return True
    return False


def _condition_fields(epoch: Mapping[str, Any], event_index: int) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name, raw_value in epoch.items():
        if not name.startswith("event"):
            continue
        values = _event_values(raw_value)
        if not values:
            continue
        value = values[event_index] if event_index < len(values) else values[0] if len(values) == 1 else None
        if value is not None:
            fields[name.removeprefix("event")] = _scalar_value(value)
    return fields


def _condition_matches(condition: Condition, fields: Mapping[str, Any]) -> bool:
    if condition is None or condition == "" or condition == "true":
        return True
    if callable(condition):
        callback = cast(Callable[[Mapping[str, Any]], bool], condition)
        return bool(callback(fields))
    if not isinstance(condition, str):
        raise ValueError("event conditions must be strings, callables, or None")
    expression = condition.replace("&&", " and ").replace("||", " or ").replace("~=", "!=")
    try:
        parsed = ast.parse(expression, mode="eval")
        return bool(_evaluate_expression(parsed.body, fields))
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid event condition {condition!r}: {exc}") from exc


def _evaluate_expression(node: ast.AST, fields: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in fields:
            raise ValueError(f"unknown event field {node.id!r}")
        return fields[node.id]
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        values = [bool(_evaluate_expression(value, fields)) for value in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.UAdd, ast.USub)):
        value = _evaluate_expression(node.operand, fields)
        if isinstance(node.op, ast.Not):
            return not value
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        return _BINARY_OPERATORS[type(node.op)](
            _evaluate_expression(node.left, fields),
            _evaluate_expression(node.right, fields),
        )
    if isinstance(node, ast.Compare):
        left = _evaluate_expression(node.left, fields)
        for comparison, comparator in zip(node.ops, node.comparators):
            if type(comparison) not in _COMPARISON_OPERATORS:
                raise ValueError("unsupported comparison operator")
            right = _evaluate_expression(comparator, fields)
            if not _COMPARISON_OPERATORS[type(comparison)](left, right):
                return False
            left = right
        return True
    raise ValueError(f"unsupported expression element {type(node).__name__}")


def _standard_deviation_limit(value: float, name: str) -> float:
    numeric = float(value)
    if np.isnan(numeric) or numeric < 0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _outlier_rows(latencies: np.ndarray, absolute_limit: float, relative_limit: float) -> np.ndarray:
    if latencies.shape[0] < 2:
        return np.empty(0, dtype=int)
    rejected = np.zeros(latencies.shape[0], dtype=bool)
    if np.isfinite(absolute_limit):
        deviations = np.abs(latencies - np.mean(latencies, axis=0))
        threshold = absolute_limit * np.std(latencies, axis=0, ddof=1)
        rejected |= np.any(deviations > threshold, axis=1)
    if latencies.shape[1] > 1 and np.isfinite(relative_limit):
        relative = np.diff(latencies, axis=1)
        deviations = np.abs(relative - np.mean(relative, axis=0))
        threshold = relative_limit * np.std(relative, axis=0, ddof=1)
        rejected |= np.any(deviations > threshold, axis=1)
    return np.flatnonzero(rejected)


__all__ = ["make_timewarp"]
