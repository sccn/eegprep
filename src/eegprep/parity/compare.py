"""Comparison helpers for the EEGPrep parity harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
import re

import numpy as np


@dataclass
class ComparisonResult:
    """Structured comparison result with optional children."""

    label: str
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    children: list["ComparisonResult"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a JSON-friendly dictionary."""
        return {
            "label": self.label,
            "passed": self.passed,
            "metrics": dict(self.metrics),
            "failures": list(self.failures),
            "children": [child.to_dict() for child in self.children],
        }

    def raise_if_failed(self) -> None:
        """Raise an AssertionError when the comparison failed."""
        if not self.passed:
            lines = [f"{self.label} failed"]
            lines.extend(self.failures)
            raise AssertionError("\n".join(lines))


def _is_numeric_array(value: Any) -> bool:
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    return np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)


def _normalize_text(value: Any) -> str:
    text = str(value).replace("\\", "/")
    return " ".join(text.split())


def _normalize_trace_entry(value: Any) -> str:
    return re.sub(r"\s+", "", _normalize_text(value))


def _flatten_trace(trace: Sequence[Any] | str) -> list[str]:
    if isinstance(trace, str):
        items = trace.splitlines()
    else:
        items = [str(item) for item in trace]
    normalized = [_normalize_trace_entry(item) for item in items]
    return [item for item in normalized if item]


def compare_numeric(
    expected: Any,
    actual: Any,
    *,
    rtol: float = 0.0,
    atol: float = 1e-7,
    equal_nan: bool = True,
    label: str = "numeric",
) -> ComparisonResult:
    """Compare two numeric arrays or scalars."""
    exp = np.asarray(expected)
    act = np.asarray(actual)
    if exp.shape != act.shape:
        return ComparisonResult(
            label=label,
            passed=False,
            metrics={
                "expected_shape": exp.shape,
                "actual_shape": act.shape,
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "rms_diff": None,
                "mismatch_count": None,
                "total_count": None,
            },
            failures=[f"shape mismatch: expected {exp.shape}, actual {act.shape}"],
        )

    mismatched = ~np.isclose(exp, act, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if np.issubdtype(exp.dtype, np.complexfloating) or np.issubdtype(act.dtype, np.complexfloating):
        diff = exp.astype(np.complex128, copy=False) - act.astype(np.complex128, copy=False)
    else:
        diff = exp.astype(np.float64, copy=False) - act.astype(np.float64, copy=False)
    abs_diff = np.abs(diff)
    valid_metric_mask = np.isfinite(abs_diff)
    valid_abs_diff = abs_diff[valid_metric_mask]
    no_finite_diffs = not valid_abs_diff.size
    zero_metrics = exp.size == 0 or (not np.count_nonzero(mismatched) and no_finite_diffs)
    max_abs_diff = float(np.max(valid_abs_diff)) if valid_abs_diff.size else (0.0 if zero_metrics else None)
    mean_abs_diff = float(np.mean(valid_abs_diff)) if valid_abs_diff.size else (0.0 if zero_metrics else None)
    rms_diff = (
        float(np.sqrt(np.mean(np.square(valid_abs_diff))))
        if valid_abs_diff.size
        else (0.0 if zero_metrics else None)
    )
    metrics = {
        "expected_shape": exp.shape,
        "actual_shape": act.shape,
        "mismatch_count": int(np.count_nonzero(mismatched)),
        "total_count": int(exp.size),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "rms_diff": rms_diff,
        "nonfinite_diff_count": int(exp.size - np.count_nonzero(valid_metric_mask)),
        "rtol": float(rtol),
        "atol": float(atol),
    }
    failures = []
    if metrics["mismatch_count"]:
        failures.append(
            f"{metrics['mismatch_count']} / {metrics['total_count']} values exceeded "
            f"rtol={rtol} atol={atol}"
        )
    return ComparisonResult(label=label, passed=not failures, metrics=metrics, failures=failures)


def compare_workflow_trace(
    expected: Sequence[Any] | str,
    actual: Sequence[Any] | str,
    *,
    label: str = "workflow_trace",
) -> ComparisonResult:
    """Compare workflow traces after whitespace and path normalization."""
    exp = _flatten_trace(expected)
    act = _flatten_trace(actual)
    failures = []
    if exp != act:
        failures.append(f"trace mismatch: expected {len(exp)} entries, actual {len(act)}")
        mismatches = [
            f"{index}: expected={left!r} actual={right!r}"
            for index, (left, right) in enumerate(zip(exp, act))
            if left != right
        ]
        if mismatches:
            failures.append("first mismatches: " + "; ".join(mismatches[:5]))
        if len(exp) != len(act):
            failures.append(f"trace length differs after {min(len(exp), len(act))} shared positions")
    metrics = {
        "expected_steps": len(exp),
        "actual_steps": len(act),
        "matching_steps": sum(1 for left, right in zip(exp, act) if left == right),
    }
    return ComparisonResult(label=label, passed=not failures, metrics=metrics, failures=failures)


def _visual_array(image: Any) -> np.ndarray:
    if isinstance(image, (str, Path)):
        from matplotlib import image as mpimg

        arr = mpimg.imread(image)
    else:
        arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    return arr.astype(np.float64)


def compare_visual_output(
    expected: Any,
    actual: Any,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    label: str = "visual_output",
) -> ComparisonResult:
    """Compare figure rasters or image arrays."""
    return compare_numeric(_visual_array(expected), _visual_array(actual), rtol=rtol, atol=atol, label=label)


def _compare_mapping(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    label: str,
    rtol: float,
    atol: float,
    equal_nan: bool,
    ignore_fields: set[str],
) -> ComparisonResult:
    failures = []
    children: list[ComparisonResult] = []
    exp_keys = {key for key in expected.keys() if key not in ignore_fields}
    act_keys = {key for key in actual.keys() if key not in ignore_fields}
    missing = sorted(exp_keys - act_keys)
    extra = sorted(act_keys - exp_keys)
    if missing:
        failures.append(f"missing keys: {missing}")
    if extra:
        failures.append(f"unexpected keys: {extra}")
    for key in sorted(exp_keys & act_keys):
        child = _compare_any(
            expected[key],
            actual[key],
            label=f"{label}.{key}" if label else key,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            ignore_fields=ignore_fields,
        )
        children.append(child)
        if not child.passed:
            failures.append(f"child comparison failed: {child.label}")
    return ComparisonResult(
        label=label or "mapping",
        passed=not failures,
        metrics={"expected_keys": len(exp_keys), "actual_keys": len(act_keys)},
        failures=failures,
        children=children,
    )


def _compare_sequence(
    expected: Sequence[Any],
    actual: Sequence[Any],
    *,
    label: str,
    rtol: float,
    atol: float,
    equal_nan: bool,
    ignore_fields: set[str],
) -> ComparisonResult:
    failures = []
    children: list[ComparisonResult] = []
    if len(expected) != len(actual):
        failures.append(f"sequence length mismatch: expected {len(expected)}, actual {len(actual)}")
        return ComparisonResult(
            label=label or "sequence",
            passed=False,
            metrics={"expected_length": len(expected), "actual_length": len(actual)},
            failures=failures,
        )
    for index, (exp_item, act_item) in enumerate(zip(expected, actual)):
        child = _compare_any(
            exp_item,
            act_item,
            label=f"{label}[{index}]",
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            ignore_fields=ignore_fields,
        )
        children.append(child)
        if not child.passed:
            failures.append(f"child comparison failed: {child.label}")
    return ComparisonResult(
        label=label or "sequence",
        passed=not failures,
        metrics={"length": len(expected)},
        failures=failures,
        children=children,
    )


def _compare_any(
    expected: Any,
    actual: Any,
    *,
    label: str,
    rtol: float,
    atol: float,
    equal_nan: bool,
    ignore_fields: set[str],
) -> ComparisonResult:
    if isinstance(expected, dict) and isinstance(actual, dict):
        return _compare_mapping(
            expected,
            actual,
            label=label,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            ignore_fields=ignore_fields,
        )
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        return _compare_sequence(
            expected,
            actual,
            label=label,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            ignore_fields=ignore_fields,
        )
    if _is_numeric_array(expected) and _is_numeric_array(actual):
        return compare_numeric(expected, actual, rtol=rtol, atol=atol, equal_nan=equal_nan, label=label)
    passed = _normalize_text(expected) == _normalize_text(actual)
    failures = [] if passed else [f"value mismatch at {label}: expected={expected!r} actual={actual!r}"]
    return ComparisonResult(label=label, passed=passed, metrics={}, failures=failures)


def compare_eeg_struct(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    rtol: float = 0.0,
    atol: float = 1e-7,
    equal_nan: bool = True,
    ignore_fields: Iterable[str] = ("filepath", "filename", "saved", "history"),
    label: str = "eeg_struct",
) -> ComparisonResult:
    """Recursively compare two EEGLAB-style dictionaries."""
    return _compare_mapping(
        expected,
        actual,
        label=label,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        ignore_fields=set(ignore_fields),
    )


def compare_events_epochs(
    expected: Any,
    actual: Any,
    *,
    rtol: float = 0.0,
    atol: float = 1e-7,
    equal_nan: bool = True,
    label: str = "events_epochs",
) -> ComparisonResult:
    """Compare event and epoch structures with the same recursive rules as EEG dicts."""
    return _compare_any(
        expected,
        actual,
        label=label,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        ignore_fields=set(),
    )


def compare_stage_outputs(
    py_file: str | Path,
    mat_file: str | Path,
    *,
    data_rtol: float = 0.0,
    data_atol: float = 1e-7,
    struct_rtol: float = 0.0,
    struct_atol: float = 1e-7,
    equal_nan: bool = True,
) -> ComparisonResult:
    """Compare two saved EEG stage outputs."""
    from eegprep.eeg_checkset import strict_mode as eeg_checkset_strict_mode
    from eegprep.pop_loadset import pop_loadset

    with eeg_checkset_strict_mode(False):
        py_eeg = pop_loadset(str(py_file))
        mat_eeg = pop_loadset(str(mat_file))

    data_result = compare_numeric(
        py_eeg["data"],
        mat_eeg["data"],
        rtol=data_rtol,
        atol=data_atol,
        equal_nan=equal_nan,
        label="stage.data",
    )
    struct_result = compare_eeg_struct(
        py_eeg,
        mat_eeg,
        rtol=struct_rtol,
        atol=struct_atol,
        equal_nan=equal_nan,
        label="stage.struct",
    )
    event_result = compare_events_epochs(
        py_eeg.get("event", []),
        mat_eeg.get("event", []),
        rtol=struct_rtol,
        atol=struct_atol,
        equal_nan=equal_nan,
        label="stage.events",
    )
    epoch_result = compare_events_epochs(
        py_eeg.get("epoch", []),
        mat_eeg.get("epoch", []),
        rtol=struct_rtol,
        atol=struct_atol,
        equal_nan=equal_nan,
        label="stage.epochs",
    )
    children = [data_result, struct_result, event_result, epoch_result]
    failures = [f"child comparison failed: {child.label}" for child in children if not child.passed]
    metrics = {
        "py_file": str(py_file),
        "mat_file": str(mat_file),
        "data_max_abs_diff": data_result.metrics.get("max_abs_diff"),
        "data_mismatch_count": data_result.metrics.get("mismatch_count"),
    }
    return ComparisonResult(
        label="stage_outputs",
        passed=all(child.passed for child in children),
        metrics=metrics,
        failures=failures,
        children=children,
    )
