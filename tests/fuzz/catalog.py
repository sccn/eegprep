"""Target names for the nightly fuzzer.

The lists below are the registry. ``FUNCTION_NAMES``, the tolerance-file keys,
and the dashboard rows are derived from ``TARGETS`` so a new target is added
in one place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetMeta:
    """One nightly target, without its strategy or callables."""

    name: str
    tolerance_key: str | None
    comparators: tuple[str, ...]


# Existing five targets stay first so a seed still walks them in this order.
# pop_clean_rawdata reuses the clean_asr tolerance. pipeline borrows each
# step's tolerance and has no section of its own.
TARGETS: tuple[TargetMeta, ...] = (
    TargetMeta("pop_eegfiltnew", "pop_eegfiltnew", ("data",)),
    TargetMeta("pop_reref", "pop_reref", ("data",)),
    TargetMeta("pop_resample", "pop_resample", ("data",)),
    TargetMeta("clean_asr", "clean_asr", ("data",)),
    TargetMeta("pop_clean_rawdata", "clean_asr", ("data",)),
    TargetMeta("eeg_eegrej", "eeg_eegrej", ("data", "events", "meta")),
    TargetMeta("pop_select", "pop_select", ("data", "events", "meta")),
    TargetMeta("pop_epoch", "pop_epoch", ("data", "events", "meta", "extra")),
    TargetMeta("pop_rmbase", "pop_rmbase", ("data", "events", "meta")),
    TargetMeta("clean_windows", "clean_windows", ("data", "events", "meta", "mask", "extra")),
    TargetMeta("clean_channels", "clean_channels", ("data", "events", "meta", "mask")),
    TargetMeta("clean_flatlines", "clean_flatlines", ("data", "events", "meta", "mask")),
    TargetMeta("pipeline", None, ("data", "events", "meta", "mask", "extra")),
)

FUNCTION_NAMES: tuple[str, ...] = tuple(target.name for target in TARGETS)
FUNCTION_ROWS: tuple[str, ...] = FUNCTION_NAMES
TOLERANCE_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(target.tolerance_key for target in TARGETS if target.tolerance_key)
)
_BY_NAME = {target.name: target for target in TARGETS}

# Pipeline steps compare events and masks even when the single-function target
# (the original five) still compares data only.
PIPELINE_COMPARATORS: tuple[str, ...] = ("data", "events", "meta", "mask", "extra")

# 4 * 2^-53. Used when a pipeline step has no meta_rel of its own.
PIPELINE_META_REL = 4.0 * 2.0**-53


def target_meta(name: str) -> TargetMeta:
    return _BY_NAME[name]


def comparators_for(name: str, *, pipeline_step: bool = False) -> tuple[str, ...]:
    if pipeline_step:
        return PIPELINE_COMPARATORS
    return _BY_NAME[name].comparators


def tolerance_key_for(name: str) -> str:
    key = _BY_NAME[name].tolerance_key
    if key is None:
        raise KeyError(name)
    return key


def freeze_params(params: dict) -> tuple:
    return tuple(sorted((key, _freeze(value)) for key, value in params.items()))


def thaw_params(frozen: tuple) -> dict:
    return {key: _thaw(value) for key, value in frozen}


def _freeze(value):
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
