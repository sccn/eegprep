"""Generate EEG cases, run them through EEGPrep and an oracle, and compare.

EEGLAB output is the oracle. Absolute and relative tolerances are loaded from
``tolerances.toml`` and are not adjusted here when a case misses.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import tomllib
from typing import Callable

import numpy as np
from hypothesis import HealthCheck, Phase, Verbosity, given, seed, settings
from hypothesis import strategies as st

from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.miscfunc.misc import round_mat
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_resample import pop_resample
from eegprep.functions.popfunc.pop_reref import pop_reref
from eegprep.plugins.clean_rawdata.asr_calibrate import spectral_shaping_filter_available
from eegprep.plugins.clean_rawdata.clean_asr import clean_asr
from eegprep.plugins.clean_rawdata.pop_clean_rawdata import pop_clean_rawdata
from eegprep.plugins.firfilt.pop_eegfiltnew import pop_eegfiltnew
from tests.eeg_options_snapshot import eeg_options_snapshot, restore_eeg_options
from tests.fuzz.catalog import (
    FUNCTION_NAMES,
    FUNCTION_ROWS,
    PIPELINE_COMPARATORS,
    PIPELINE_META_REL,
    TOLERANCE_KEYS,
    comparators_for,
    freeze_params,
    is_legacy_spec,
    thaw_params,
    tolerance_key_for,
)
from tests.fuzz.issues import publish_findings
from tests.matlab_batch_engine import install

_ORIGINAL_FUNCTIONS = (
    "pop_eegfiltnew",
    "pop_reref",
    "pop_resample",
    "clean_asr",
    "pop_clean_rawdata",
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_SET = REPO_ROOT / "sample_data" / "eeglab_data.set"
TOLERANCES_PATH = Path(__file__).resolve().parent / "tolerances.toml"
_MIN_LENGTH = 256
_MAX_LENGTH = 1536
_SHORT_MAX = 96

Materialize = Callable[["FuzzCase"], dict]


class _Divergence(Exception):
    def __init__(self, outcome: "Outcome"):
        super().__init__(outcome.signature)
        self.outcome = outcome


@dataclass(frozen=True)
class Tolerance:
    abs: float
    rel: float
    event_latency_abs: float = 0.0
    meta_rel: float = 0.0


@dataclass(frozen=True)
class EEGSpec:
    channels: tuple[int, ...]
    start: int
    length: int
    decimate: int
    nan_channel: int
    nan_start: int
    nan_length: int
    source: str = "eeglab_data"
    events: str = "none"
    boundaries: tuple = ()
    trials: int = 1
    epoch_pnts: int = 0
    xmin: float = 0.0
    srate_hz: float = 0.0
    drop_xyz: tuple = ()
    artifact: str = ""
    artifact_channel: int = -1
    artifact_start: int = 0
    artifact_length: int = 0
    prior_mask: str = ""


@dataclass(frozen=True)
class FuzzCase:
    function: str
    spec: EEGSpec
    params: tuple
    sanity: bool = False
    pipeline: tuple = ()


@dataclass(frozen=True)
class SeedInfo:
    n_channels: int
    n_samples: int
    srate: float


@dataclass(frozen=True)
class SideResult:
    ok: bool
    data: np.ndarray | None
    srate: float | None
    error_type: str | None
    error_message: str | None
    events: tuple = ()
    labels: tuple = ()
    nbchan: int | None = None
    pnts: int | None = None
    trials: int | None = None
    xmin: float | None = None
    xmax: float | None = None
    clean_channel_mask: np.ndarray | None = None
    clean_sample_mask: np.ndarray | None = None
    extra: tuple = ()
    steps: tuple = ()
    halt: str = ""


@dataclass
class Outcome:
    case: FuzzCase
    diverged: bool
    both_ok: bool
    error_class: str
    max_abs: float
    max_rel: float
    detail: str
    input_sha256: str = ""
    repro_command: str = ""
    confirmed: bool = False

    @property
    def report(self) -> str:
        return report_name(self.case)

    @property
    def params(self) -> dict:
        return thaw_params(self.case.params)

    @property
    def signature(self) -> str:
        if not self.diverged:
            return ""
        if self.error_class == "numeric":
            magnitude = rounded_magnitude(self.max_abs if self.max_abs > 0 else self.max_rel)
        else:
            magnitude = "na"
        return f"{self.report}|{self.error_class}|{magnitude}"


@dataclass
class SessionResult:
    seed: int
    harness_ok: bool
    functions: dict[str, dict[str, int]]
    divergences: list[Outcome]
    output_dir: Path
    sanity: list[dict]
    created: list[int]
    commented: list[int]
    unfiled: list[str]


def nightly_seed(when: date | None = None) -> int:
    """Return the UTC date as YYYYMMDD, the deterministic nightly seed."""
    day = when or datetime.now(timezone.utc).date()
    return day.year * 10000 + day.month * 100 + day.day


def rounded_magnitude(value: float) -> str:
    """Round a positive error to one significant digit, such as ``1e-3``."""
    if not math.isfinite(value) or value <= 0:
        return "0"
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{int(mantissa)}e{int(exponent)}"


def report_name(case: FuzzCase) -> str:
    if case.function == "clean_asr" and thaw_params(case.params).get("entry") == "clean_rawdata":
        return "pop_clean_rawdata"
    return case.function


def load_tolerances(path: Path | None = None) -> dict[str, Tolerance]:
    """Load per-function tolerances. Missing functions are an error."""
    source = path or TOLERANCES_PATH
    payload = tomllib.loads(source.read_text())
    tolerances = {}
    for name in TOLERANCE_KEYS:
        row = payload[name]
        tolerances[name] = Tolerance(
            float(row["abs"]),
            float(row["rel"]),
            float(row.get("event_latency_abs", 0.0)),
            float(row.get("meta_rel", 0.0)),
        )
    return tolerances


def tolerance_for(name: str, tolerances: dict[str, Tolerance]) -> Tolerance:
    return tolerances[tolerance_key_for(name)]


def _event_targets():
    from tests.fuzz import targets as event_targets

    return event_targets


def load_seed() -> tuple[SeedInfo, dict]:
    """Load ``sample_data/eeglab_data.set`` once and reuse it for every case."""
    if not hasattr(load_seed, "_cache"):
        eeg = pop_loadset(str(SAMPLE_SET))
        info = SeedInfo(int(eeg["nbchan"]), int(eeg["pnts"]), float(eeg["srate"]))
        load_seed._cache = (info, eeg)  # type: ignore[attr-defined]
    return load_seed._cache  # type: ignore[attr-defined]


def materialize_eeg(case: FuzzCase) -> dict:
    """Crop a tutorial recording into the EEG described by ``case.spec``.

    Specs that only set the original crop fields keep today's continuous
    ``eeglab_data.set`` segment, including cleared events. Any other field
    keeps the cropped events from the named recording.
    """
    if is_legacy_spec(case.spec):
        return _materialize_legacy(case)
    return _event_targets().materialize_rich(case)


def _materialize_legacy(case: FuzzCase) -> dict:
    """Crop ``sample_data/eeglab_data.set`` into a continuous segment.

    Channel indices are into the sample file, in ascending order. ``decimate``
    keeps every Nth sample and divides the sampling rate by N. A non-negative
    ``nan_channel`` is an index into that crop, not into the original file.
    Events are cleared so both sides see the same continuous segment.
    """
    info, source = load_seed()
    spec = case.spec
    if spec.decimate < 1:
        raise ValueError(f"decimate must be positive, got {spec.decimate}")
    span = spec.length * spec.decimate
    if spec.start < 0 or spec.start + span > info.n_samples:
        raise ValueError(f"crop {spec.start}:{spec.start + span} exceeds {info.n_samples} samples")
    if any(index < 0 or index >= info.n_channels for index in spec.channels):
        raise ValueError(f"channels {spec.channels} outside 0..{info.n_channels - 1}")

    eeg = _copy_eeg(source)
    sample_index = np.arange(spec.start, spec.start + span, spec.decimate)
    data = np.asarray(source["data"])
    # The sample file is float32. EEGLAB's default option_single casts a
    # double recording to single before reref.m multiplies, about 1e-5 µV
    # from this float64 product. That single product also changes with the
    # BLAS kernel, so the survey compares float64 and connect_matlab turns
    # option_single off.
    cropped = np.array(data[np.ix_(list(spec.channels), sample_index)], dtype=np.float64, copy=True)
    if case.sanity and case.function == "pop_reref":
        # Sanity checks the bridge with values whose mean is exact in float32
        # and float64, rather than with the sample recording.
        cropped = _exact_reref_signal(cropped.shape)
    if spec.nan_channel >= 0 and spec.nan_length > 0:
        stop = spec.nan_start + spec.nan_length
        if spec.nan_channel >= cropped.shape[0] or spec.nan_start < 0 or stop > cropped.shape[1]:
            raise ValueError("NaN span is outside the cropped data")
        cropped[spec.nan_channel, spec.nan_start : stop] = np.nan

    srate = info.srate / spec.decimate
    pnts = int(cropped.shape[1])
    chosen = [_copy_channel(source["chanlocs"][index]) for index in spec.channels]
    eeg["data"] = cropped
    eeg["chanlocs"] = np.array(chosen, dtype=object)
    eeg["nbchan"] = int(cropped.shape[0])
    eeg["pnts"] = pnts
    eeg["trials"] = 1
    eeg["srate"] = float(srate)
    eeg["xmin"] = 0.0
    eeg["xmax"] = (pnts - 1) / srate if pnts else 0.0
    eeg["times"] = np.linspace(0.0, eeg["xmax"] * 1000.0, pnts) if pnts else np.array([])
    eeg["event"] = np.array([], dtype=object)
    eeg["urevent"] = np.array([], dtype=object)
    eeg["epoch"] = np.array([])
    eeg["saved"] = "no"
    return eeg


def data_sha256(data: np.ndarray) -> str:
    array = np.ascontiguousarray(data, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()


# EEGLAB clean_windows / fit_eeg_distribution defaults. The fuzzer does not
# override them. Short recordings reach m < 2 in the fit's width loop, and
# fit_eeg_distribution.m:126 then errors in bsxfun. clean_asr.m catches that
# and calibrates on the whole recording.
_FIT_MAX_WIDTH = 0.6 - 0.022
_FIT_MIN_WIDTH = 0.25 * _FIT_MAX_WIDTH
_FIT_WIDTH_STEP = 0.01
_CLEAN_WINDOW_OVERLAP = 0.66
_ASR_REF_WNDLEN = 1.0
# asr_calibrate's RMS window. clean_asr does not forward its processing window_len.
_ASR_CALIB_WINDOW = 0.5


def _matlab_colon(start: float, step: float, stop: float) -> np.ndarray:
    """MATLAB colon ``start:step:stop``, including the endpoint within a few ulps."""
    start = float(start)
    step = float(step)
    stop = float(stop)
    if step == 0.0 or (step > 0.0 and start > stop) or (step < 0.0 and start < stop):
        return np.array([], dtype=float)
    span = (stop - start) / step
    count = math.floor(span)
    tol = 3.0 * float(np.spacing(max(abs(start), abs(stop), abs(step)) * max(count, 1)))
    if span - count > 1.0 - tol / abs(step):
        count += 1
    values = start + step * np.arange(count + 1, dtype=float)
    if step > 0.0:
        return values[values <= stop + tol]
    return values[values >= stop - tol]


def _calibration_window_count(n_samples: int, srate: float) -> int:
    width = _ASR_REF_WNDLEN * srate
    step = width * (1.0 - _CLEAN_WINDOW_OVERLAP)
    stop = n_samples - width
    if width <= 0.0 or step <= 0.0 or stop < 1.0:
        return 0
    return int(_matlab_colon(1.0, step, stop).size)


def _clean_windows_count(n_samples: int, srate: float, window_len: float, window_overlap: float) -> int:
    """Windows in clean_windows.m's ``1:step:(S-N)`` colon."""
    width = float(window_len) * float(srate)
    if _matlab_colon(0.0, 1.0, width - 1.0).size == 0:
        return 0
    step = width * (1.0 - float(window_overlap))
    if step <= 0.0:
        step = 1.0
    stop = float(n_samples) - width
    if stop < 1.0:
        return 0
    return int(_matlab_colon(1.0, step, stop).size)


def _clean_windows_short_fit(case: FuzzCase, eeg: dict) -> bool:
    """True when fit_eeg_distribution.m:112 would index a rounded width of 1 or less."""
    if case.function != "clean_windows":
        return False
    data = np.asarray(eeg["data"])
    if data.ndim != 2:
        return False
    params = thaw_params(case.params)
    count = _clean_windows_count(
        int(data.shape[1]),
        float(eeg["srate"]),
        float(params["window_len"]),
        float(params["window_overlap"]),
    )
    return _fit_eeg_distribution_throws(count)


def _fit_eeg_distribution_throws(n_windows: int) -> bool:
    """True when EEGLAB's fit errors on this many per-channel RMS windows."""
    if n_windows <= 0:
        return False
    widths = _matlab_colon(_FIT_MAX_WIDTH, -_FIT_WIDTH_STEP, _FIT_MIN_WIDTH)
    return any(int(round_mat(n_windows * width)) < 2 for width in widths)


def _asr_rms_window_count(n_samples: int, srate: float) -> int:
    """Windows in asr_calibrate.m's ``1:N*(1-overlap):S-N`` RMS colon."""
    width = float(round_mat(_ASR_CALIB_WINDOW * srate))
    step = width * (1.0 - _CLEAN_WINDOW_OVERLAP)
    stop = n_samples - width
    if width <= 0.0 or step <= 0.0 or stop < 1.0:
        return 0
    return int(_matlab_colon(1.0, step, stop).size)


def _fit_width_reaches_zero(n_windows: int) -> bool:
    """True when asr_calibrate.m's nested fit indexes ``X(m,:)`` with m = 0.

    One window is rejected earlier with a different error. Two or three windows
    include a rounded width of 0; four or more stay at 1 or above.
    """
    if n_windows < 2:
        return False
    widths = _matlab_colon(_FIT_MAX_WIDTH, -_FIT_WIDTH_STEP, _FIT_MIN_WIDTH)
    return any(int(round_mat(n_windows * width)) == 0 for width in widths)


def compare_data(
    python_data: np.ndarray,
    oracle_data: np.ndarray,
    python_srate: float,
    oracle_srate: float,
    tolerance: Tolerance,
) -> tuple[bool, str, float, float, str]:
    """Compare one pair of outputs. The oracle array is the reference for ``rel``."""
    python = np.asarray(python_data, dtype=np.float64)
    oracle = np.asarray(oracle_data, dtype=np.float64)
    if python.shape != oracle.shape:
        return True, "shape", 0.0, 0.0, f"shape {python.shape} != {oracle.shape}"
    python_bad = ~np.isfinite(python)
    oracle_bad = ~np.isfinite(oracle)
    if np.any(python_bad != oracle_bad):
        count = int(np.count_nonzero(python_bad != oracle_bad))
        return True, "nan-mask", float(count), 0.0, f"non-finite mask differs at {count} samples"
    finite = ~oracle_bad
    if not np.any(finite):
        return _srate_or_match(python_srate, oracle_srate, 0.0, 0.0)
    diff = np.abs(python[finite] - oracle[finite])
    scale = np.abs(oracle[finite])
    limit = tolerance.abs + tolerance.rel * scale
    max_abs = float(np.max(diff))
    max_rel = float(np.max(diff / np.maximum(scale, 1e-300)))
    if np.any(diff > limit):
        flat = int(np.argmax(diff))
        return True, "numeric", max_abs, max_rel, f"max abs {max_abs:.3g} at finite index {flat}"
    return _srate_or_match(python_srate, oracle_srate, max_abs, max_rel)


def _same_failure(python: SideResult, oracle: SideResult) -> bool:
    if python.error_type == oracle.error_type:
        return True
    message = (python.error_message or "").strip()
    oracle_message = (oracle.error_message or "").strip()
    if message and message == oracle_message:
        return True
    # asr_calibrate.m uses one "No enough reference data" error both when the
    # calibration is shorter than one RMS window and when fewer than two
    # windows can be formed. EEGPrep names which of those happened.
    return "No enough reference data" in oracle_message and (
        "Fewer than 2 calibration windows" in message or "Not enough calibration data" in message
    )


def _oracle_filter_unavailable(oracle: SideResult) -> bool:
    # asr_calibrate.m reports this for every yulewalk failure, including a
    # missing Signal Processing Toolbox and a rejected breakpoint grid.
    if oracle.ok:
        return False
    return "yulewalk() function was not found" in (oracle.error_message or "")


def judge(
    python: SideResult,
    oracle: SideResult,
    tolerance: Tolerance,
    case: FuzzCase,
    digest: str,
    tolerances: dict[str, Tolerance] | None = None,
) -> Outcome:
    if case.function == "pipeline":
        return _judge_pipeline(python, oracle, tolerances or {}, case, digest)
    return _judge_pair(python, oracle, tolerance, case, digest, comparators_for(case.function))


def _judge_pair(
    python: SideResult,
    oracle: SideResult,
    tolerance: Tolerance,
    case: FuzzCase,
    digest: str,
    comparators: tuple[str, ...],
) -> Outcome:
    if _oracle_filter_unavailable(oracle):
        return _outcome(case, False, False, "oracle-infrastructure", 0.0, 0.0, oracle.error_message or "", digest)
    if not python.ok and not oracle.ok:
        detail = (
            f"python {python.error_type}: {python.error_message}; oracle {oracle.error_type}: {oracle.error_message}"
        )
        if _same_failure(python, oracle):
            return _outcome(case, False, False, "both-error", 0.0, 0.0, detail, digest)
        error_class = f"exception-mismatch:{python.error_type}/{oracle.error_type}"
        return _outcome(case, True, False, error_class, 0.0, 0.0, detail, digest)
    if not python.ok:
        detail = f"python {python.error_type}: {python.error_message}"
        return _outcome(case, True, False, f"python-only:{python.error_type}", 0.0, 0.0, detail, digest)
    if not oracle.ok:
        detail = f"oracle {oracle.error_type}: {oracle.error_message}"
        return _outcome(case, True, False, f"oracle-only:{oracle.error_type}", 0.0, 0.0, detail, digest)
    return _compare_success(python, oracle, tolerance, case, digest, comparators)


def _compare_success(
    python: SideResult,
    oracle: SideResult,
    tolerance: Tolerance,
    case: FuzzCase,
    digest: str,
    comparators: tuple[str, ...],
) -> Outcome:
    max_abs = 0.0
    max_rel = 0.0
    selected = set(comparators)
    if "data" in selected:
        diverged, error_class, max_abs, max_rel, detail = compare_data(
            python.data, oracle.data, float(python.srate), float(oracle.srate), tolerance
        )
        if diverged:
            return _outcome(case, True, True, error_class, max_abs, max_rel, detail, digest)
    if "events" in selected:
        diverged, error_class, detail = _compare_events(python, oracle, tolerance)
        if diverged:
            return _outcome(case, True, True, error_class, 0.0, 0.0, detail, digest)
    if "mask" in selected:
        diverged, error_class, detail = _compare_masks(python, oracle)
        if diverged:
            return _outcome(case, True, True, error_class, 0.0, 0.0, detail, digest)
    if "meta" in selected:
        diverged, error_class, detail = _compare_meta(python, oracle, tolerance)
        if diverged:
            return _outcome(case, True, True, error_class, 0.0, 0.0, detail, digest)
    if "extra" in selected:
        diverged, error_class, detail = _compare_extra(python, oracle)
        if diverged:
            return _outcome(case, True, True, error_class, 0.0, 0.0, detail, digest)
    return _outcome(case, False, True, "match", max_abs, max_rel, "match", digest)


def _judge_pipeline(
    python: SideResult, oracle: SideResult, tolerances: dict[str, Tolerance], case: FuzzCase, digest: str
) -> Outcome:
    if not python.steps and not oracle.steps:
        return _judge_pair(python, oracle, Tolerance(0.0, 0.0), case, digest, ("data",))
    count = min(len(python.steps), len(oracle.steps))
    last_abs = 0.0
    last_rel = 0.0
    for index in range(count):
        name, left = python.steps[index]
        _name, right = oracle.steps[index]
        if left.halt == "known-divergence" and right.halt == "known-divergence":
            return _outcome(
                case,
                False,
                True,
                "match",
                last_abs,
                last_rel,
                f"{name}: known divergence: continuous pop_rmbase with a boundary and pointrange starting after sample 1",
                digest,
            )
        params = case.pipeline[index][1] if index < len(case.pipeline) else ()
        step_case = FuzzCase(name, case.spec, params)
        outcome = _judge_pair(
            left, right, _pipeline_tolerance(name, tolerances), step_case, digest, PIPELINE_COMPARATORS
        )
        if outcome.error_class == "both-error":
            return _outcome(case, False, False, f"{name}:both-error", 0.0, 0.0, outcome.detail, digest)
        if outcome.error_class == "oracle-infrastructure":
            return _outcome(case, False, False, f"{name}:oracle-infrastructure", 0.0, 0.0, outcome.detail, digest)
        if outcome.diverged or outcome.error_class != "match":
            return _outcome(
                case,
                outcome.diverged,
                outcome.both_ok,
                f"{name}:{outcome.error_class}",
                outcome.max_abs,
                outcome.max_rel,
                outcome.detail,
                digest,
            )
        last_abs = outcome.max_abs
        last_rel = outcome.max_rel
    if len(python.steps) != len(oracle.steps):
        return _outcome(
            case,
            True,
            False,
            "step-count",
            0.0,
            0.0,
            f"python steps {len(python.steps)} != oracle steps {len(oracle.steps)}",
            digest,
        )
    return _outcome(case, False, True, "match", last_abs, last_rel, "match", digest)


def _pipeline_tolerance(name: str, tolerances: dict[str, Tolerance]) -> Tolerance:
    base = tolerance_for(name, tolerances)
    if base.meta_rel:
        return base
    return Tolerance(base.abs, base.rel, base.event_latency_abs, PIPELINE_META_REL)


class PythonBackend:
    """Run the EEGPrep implementation of a fuzz case."""

    def run(self, case: FuzzCase, eeg: dict) -> SideResult:
        try:
            return side_from_call(dispatch_python(case, eeg))
        except Exception as exc:
            return side_from_error(exc)


class MatlabBackend:
    """Run the matching EEGLAB function through the matlab-batch shim."""

    def __init__(self):
        self._engine = None
        self._options = None

    def run(self, case: FuzzCase, eeg: dict) -> SideResult:
        try:
            return side_from_call(dispatch_matlab(self.engine(), case, eeg))
        except Exception as exc:
            return side_from_error(exc)

    def engine(self):
        if self._engine is None:
            self._engine, self._options = connect_matlab()
        return self._engine

    def close(self):
        if self._engine is None or self._options is None:
            return
        snapshot = self._options
        self._options = None
        resume_eeg_options(self._engine.engine, snapshot)


class FakeOracle:
    """Stand-in oracle for tests and local runs without MATLAB.

    ``match`` repeats the Python result. ``survey-diverge`` does that for
    sanity cases and adds 1 to later successful outputs. ``sanity-fail``
    errors on every case.
    """

    def __init__(self, mode: str, python_backend: PythonBackend):
        self.mode = mode
        self.python_backend = python_backend

    def run(self, case: FuzzCase, eeg: dict) -> SideResult:
        if self.mode == "sanity-fail":
            return side_from_error(RuntimeError("fuzz sanity bridge failed"))
        result = self.python_backend.run(case, eeg)
        if self.mode == "match" or case.sanity or not result.ok or result.data is None:
            return result
        if self.mode == "survey-diverge":
            data = np.array(result.data + 1.0, copy=True)
            steps = result.steps
            if steps:
                name, last = steps[-1]
                if last.data is not None:
                    last = replace(last, data=np.array(last.data + 1.0, copy=True))
                    steps = steps[:-1] + ((name, last),)
            return replace(result, data=data, steps=steps)
        raise ValueError(f"unknown fake mode {self.mode}")


def connect_matlab():
    """Attach ``matlab.engine`` to the served matlab-batch directory."""
    batch = os.environ.get("EEGPREP_MATLAB_BATCH_DIR")
    if not batch:
        raise RuntimeError(
            "EEGPREP_MATLAB_BATCH_DIR is not set. Start matlab-batch with tests/matlab/eegprep_batch_engine_serve.m."
        )
    install(batch)
    os.environ.setdefault("EEGPREP_EEGLAB_ROOT", str(REPO_ROOT / "src" / "eegprep" / "eeglab"))
    engine = get_eeglab()
    # eeg_checkset applies option_single on pop_loadset, before the called
    # function runs. Leave the recording in double for the session, then
    # MatlabBackend.close writes the original options file back.
    return engine, suspend_option_single(engine.engine)


def suspend_option_single(engine):
    """Turn ``option_single`` off and return the options-file snapshot."""
    snapshot = eeg_options_snapshot(engine)
    try:
        engine.eval("pop_editoptions('option_single', 0);", nargout=0)
    except Exception:
        restore_eeg_options(*snapshot)
        raise
    return snapshot


def resume_eeg_options(engine, snapshot):
    """Write the snapshotted options file back and drop MATLAB's cached script."""
    restore_eeg_options(*snapshot)
    engine.eval("clear functions", nargout=0)


def default_strategies() -> dict:
    if not hasattr(default_strategies, "_cache"):
        info, _eeg = load_seed()
        default_strategies._cache = build_strategies(info)  # type: ignore[attr-defined]
    return default_strategies._cache  # type: ignore[attr-defined]


def _original_sanity() -> list[FuzzCase]:
    info, _eeg = load_seed()
    spec = EEGSpec(
        channels=tuple(range(info.n_channels)),
        start=0,
        length=min(1024, info.n_samples),
        decimate=1,
        nan_channel=-1,
        nan_start=0,
        nan_length=0,
    )
    return [
        FuzzCase("pop_eegfiltnew", spec, _filter_sanity(), sanity=True),
        FuzzCase("pop_reref", spec, _reref_sanity(), sanity=True),
        FuzzCase("pop_resample", spec, freeze_params({"freq": 96}), sanity=True),
        FuzzCase("clean_asr", spec, _asr_sanity(), sanity=True),
    ]


def default_sanity() -> list[FuzzCase]:
    return [*_original_sanity(), *_event_targets().extra_sanity(EEGSpec, FuzzCase)]


def draw_cases(function: str, *, seed_value: int, examples: int) -> list[FuzzCase]:
    """Draw the nightly case list for one function. The same seed redraws it."""
    strategies = default_strategies()
    if function not in strategies:
        strategies = _with_event_strategies(strategies)
    found: list[FuzzCase] = []

    def body(case: FuzzCase) -> None:
        found.append(case)

    run_given(strategies[function], seed_value, examples, (Phase.generate,), body)
    return found


def run_session(
    *,
    seed_value: int,
    output_dir: Path,
    examples: int = 6,
    shrink_budget: int = 12,
    max_shrinks: int = 4,
    issue_cap: int = 3,
    python_backend=None,
    oracle_backend=None,
    strategies: dict | None = None,
    sanity: list[FuzzCase] | None = None,
    materialize: Materialize | None = None,
    tolerances: dict[str, Tolerance] | None = None,
    tracker=None,
    run_url: str = "",
    summary_path: Path | None = None,
    functions: tuple[str, ...] | None = None,
) -> SessionResult:
    """Run sanity, then a seeded survey, then shrink confirmed divergences."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("fuzz seed %s", seed_value)
    (output_dir / "seed.txt").write_text(f"{seed_value}\n")

    python_backend = python_backend or PythonBackend()
    oracle_backend = oracle_backend or MatlabBackend()
    injected = strategies is not None
    caller_sanity = sanity
    strategies = dict(strategies or default_strategies())
    materialize = materialize or materialize_eeg
    tolerances = tolerances or load_tolerances()
    cache: dict[FuzzCase, Outcome] = {}

    def evaluate(case: FuzzCase, *, fresh: bool = False, budget: int | None = None, counter: dict | None = None):
        if not fresh and case in cache:
            return cache[case]
        if budget is not None and counter is not None and counter["n"] >= budget:
            return None
        if counter is not None:
            counter["n"] += 1
        outcome = evaluate_case(case, materialize, python_backend, oracle_backend, tolerances)
        cache[case] = outcome
        logger.info(
            "%s sanity=%s diverged=%s class=%s abs=%s rel=%s",
            outcome.report,
            case.sanity,
            outcome.diverged,
            outcome.error_class,
            outcome.max_abs,
            outcome.max_rel,
        )
        return outcome

    counts = {name: {"examples": 0, "divergences": 0} for name in FUNCTION_ROWS}

    def sanity_failure(cases: list[FuzzCase]) -> SessionResult | None:
        rows = []
        for case in cases:
            outcome = evaluate(case)
            if not outcome.both_ok or outcome.diverged:
                rows.append({"function": outcome.report, "error_class": outcome.error_class, "detail": outcome.detail})
        if not rows:
            return None
        for row in rows:
            logger.error("fuzz sanity %s %s: %s", row["function"], row["error_class"], row["detail"])
        logger.error("fuzz sanity failed; not surveying and not filing issues")
        result = SessionResult(seed_value, False, counts, [], output_dir, rows, [], [], [])
        _write_reports(result, summary_path, tracker, run_url, "sanity", issue_cap)
        return result

    def survey_names(names: list[str]) -> tuple[list[Outcome], list[Outcome]]:
        survey: list[Outcome] = []
        for name in names:
            if name not in strategies:
                continue
            found: list[Outcome] = []

            def body(case: FuzzCase, found=found) -> None:
                outcome = evaluate(case)
                if outcome is not None:
                    found.append(outcome)

            run_given(strategies[name], seed_value, examples, (Phase.generate,), body)
            survey.extend(found)
        by_signature: dict[str, Outcome] = {}
        for outcome in survey:
            if outcome.diverged and outcome.signature not in by_signature:
                by_signature[outcome.signature] = outcome
        ordered = sorted(by_signature.values(), key=lambda item: item.max_abs, reverse=True)[:max_shrinks]
        confirmed: list[Outcome] = []
        for original in ordered:
            strategy_name = original.report if original.report in strategies else original.case.function
            shrunk = shrink_signature(
                strategies[strategy_name],
                seed_value,
                examples,
                original.signature,
                evaluate,
                shrink_budget,
            )
            candidate = shrunk or original
            fresh = evaluate(candidate.case, fresh=True)
            if fresh.diverged and fresh.report == original.report and fresh.error_class == original.error_class:
                fresh.confirmed = True
                confirmed.append(fresh)
                logger.info("shrunk %s -> %s", original.signature, fresh.signature)
            else:
                logger.info("unconfirmed %s", original.signature)
        return survey, confirmed

    selected = list(functions) if functions else list(FUNCTION_NAMES)
    if injected:
        selected = [name for name in FUNCTION_NAMES if name in strategies]
        failed = sanity_failure(default_sanity() if caller_sanity is None else caller_sanity)
        if failed is not None:
            return failed
        survey, confirmed = survey_names(selected)
    else:
        original_names = [name for name in selected if name in _ORIGINAL_FUNCTIONS]
        extra_names = [name for name in selected if name not in _ORIGINAL_FUNCTIONS]
        if caller_sanity is None:
            original_cases = [case for case in _original_sanity() if case.function in original_names]
        else:
            original_cases = [case for case in caller_sanity if case.function in original_names]
        failed = sanity_failure(original_cases)
        if failed is not None:
            return failed
        survey, confirmed = survey_names(original_names)
        if extra_names:
            strategies = _with_event_strategies(strategies)
            if caller_sanity is None:
                extra_cases = [
                    case for case in _event_targets().extra_sanity(EEGSpec, FuzzCase) if case.function in extra_names
                ]
            else:
                extra_cases = [case for case in caller_sanity if case.function in extra_names]
            failed = sanity_failure(extra_cases)
            if failed is not None:
                return failed
            extra_survey, extra_confirmed = survey_names(extra_names)
            survey.extend(extra_survey)
            confirmed.extend(extra_confirmed)

    for outcome in survey:
        row = counts[outcome.report]
        row["examples"] += 1
        if outcome.diverged:
            row["divergences"] += 1
    for outcome in confirmed:
        eeg = materialize(outcome.case)
        write_repro(output_dir / "repros", outcome, seed_value, eeg)

    result = SessionResult(seed_value, True, counts, confirmed, output_dir, [], [], [], [])
    _write_reports(result, summary_path, tracker, run_url, "tracker" if tracker else "skipped", issue_cap)
    return result


def run_repro(
    path: Path | str, *, python_backend=None, oracle_backend=None, materialize: Materialize | None = None
) -> Outcome:
    """Re-run a saved repro against the oracle and check the saved input hash."""
    payload = json.loads(Path(path).read_text())
    case = case_from_payload(payload)
    outcome = evaluate_case(
        case,
        materialize or materialize_eeg,
        python_backend or PythonBackend(),
        oracle_backend or MatlabBackend(),
        load_tolerances(),
    )
    expected = payload.get("input_sha256", "")
    if expected and outcome.input_sha256 != expected:
        raise RuntimeError(f"input hash {outcome.input_sha256} does not match saved {expected}")
    return outcome


def case_from_payload(payload: dict) -> FuzzCase:
    spec = dict(payload["spec"])
    spec["channels"] = tuple(spec["channels"])
    if "drop_xyz" in spec:
        spec["drop_xyz"] = tuple(int(index) for index in spec["drop_xyz"])
    if "boundaries" in spec:
        spec["boundaries"] = tuple(tuple(float(item) for item in bound) for bound in spec["boundaries"])
    pipeline = tuple((name, freeze_params(params)) for name, params in payload.get("pipeline", ()))
    return FuzzCase(
        payload["function"],
        EEGSpec(**spec),
        freeze_params(payload["params"]),
        bool(payload.get("sanity", False)),
        pipeline,
    )


def write_repro(directory: Path, outcome: Outcome, seed_value: int, eeg: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stem = repro_stem(outcome.signature)
    json_path = directory / f"{stem}.json"
    outcome.repro_command = f"uv run python -m tests.fuzz reproduce repros/{json_path.name}"
    payload = {
        "signature": outcome.signature,
        "seed": seed_value,
        "function": outcome.case.function,
        "report": outcome.report,
        "spec": spec_to_json(outcome.case.spec),
        "params": outcome.params,
        "max_abs": outcome.max_abs,
        "max_rel": outcome.max_rel,
        "error_class": outcome.error_class,
        "detail": outcome.detail,
        "input_sha256": outcome.input_sha256,
        "command": outcome.repro_command,
    }
    if outcome.case.pipeline:
        payload["pipeline"] = [[name, thaw_params(params)] for name, params in outcome.case.pipeline]
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    np.save(directory / f"{stem}.npy", np.asarray(eeg["data"]))
    return json_path


def render_summary(result: SessionResult) -> str:
    lines = [
        f"# Fuzz seed {result.seed}",
        "",
        f"Harness: {'ok' if result.harness_ok else 'FAILED sanity'}",
        "",
        "| Function | Examples | Divergences |",
        "| --- | --- | --- |",
    ]
    for name in FUNCTION_ROWS:
        row = result.functions[name]
        lines.append(f"| {name} | {row['examples']} | {row['divergences']} |")
    if result.sanity:
        lines.extend(["", "## Sanity", ""])
        for row in result.sanity:
            lines.append(f"- {row['function']}: {row['error_class']}: {row['detail']}")
    if result.divergences:
        lines.extend(["", "## Confirmed", ""])
        for outcome in result.divergences:
            lines.append(
                f"- `{outcome.signature}` abs={outcome.max_abs} rel={outcome.max_rel} `{outcome.repro_command}`"
            )
    if result.unfiled:
        lines.extend(["", "Unfiled: " + ", ".join(result.unfiled)])
    lines.append("")
    return "\n".join(lines)


def build_strategies(info: SeedInfo) -> dict:
    """Hypothesis strategies for the nightly targets. Draws shrink toward smaller crops."""
    # Hypothesis unions literals from local modules already imported at draw time.
    # Loading the event targets first makes that set independent of import order.
    import tests.fuzz.targets  # noqa: F401

    @st.composite
    def eegfiltnew(draw):
        spec = _draw_spec(draw, info)
        return FuzzCase("pop_eegfiltnew", spec, freeze_params(_draw_filter(draw, info.srate / spec.decimate)))

    @st.composite
    def reref(draw):
        spec = _draw_spec(draw, info)
        return FuzzCase("pop_reref", spec, freeze_params(_draw_reref(draw, len(spec.channels))))

    @st.composite
    def resample(draw):
        spec = _draw_spec(draw, info)
        freq = draw(st.sampled_from([64, 90, 96, 100, 120, 160, 200, 256]))
        return FuzzCase("pop_resample", spec, freeze_params({"freq": freq}))

    @st.composite
    def asr(draw):
        spec = _draw_spec(draw, info)
        return FuzzCase("clean_asr", spec, freeze_params(_draw_asr(draw, "clean_asr")))

    @st.composite
    def rawdata(draw):
        spec = _draw_spec(draw, info)
        return FuzzCase("clean_asr", spec, freeze_params(_draw_asr(draw, "clean_rawdata")))

    strategies = {
        "pop_eegfiltnew": eegfiltnew(),
        "pop_reref": reref(),
        "pop_resample": resample(),
        "clean_asr": asr(),
        "pop_clean_rawdata": rawdata(),
    }
    return strategies


def _with_event_strategies(strategies: dict) -> dict:
    info, _eeg = load_seed()
    merged = dict(strategies)
    merged.update(_event_targets().extra_strategies(info, EEGSpec, FuzzCase, freeze_params))
    return merged


def run_given(strategy, seed_value: int, examples: int, phases, body: Callable) -> None:
    """Run a Hypothesis search. ``body`` may raise ``_Divergence`` to shrink that case."""

    def _inner(case):
        body(case)

    wrapped = given(case=strategy)(_inner)
    wrapped = settings(
        max_examples=examples,
        phases=phases,
        database=None,
        deadline=None,
        derandomize=False,
        print_blob=False,
        report_multiple_bugs=False,
        suppress_health_check=list(HealthCheck),
        verbosity=Verbosity.quiet,
    )(wrapped)
    wrapped = seed(seed_value)(wrapped)
    wrapped()


def shrink_signature(strategy, seed_value: int, examples: int, signature: str, evaluate, budget: int) -> Outcome | None:
    counter = {"n": 0}
    found = {"outcome": None}

    def body(case: FuzzCase) -> None:
        outcome = evaluate(case, budget=budget, counter=counter)
        if outcome is None:
            return
        if outcome.diverged and outcome.signature == signature:
            found["outcome"] = outcome
            raise _Divergence(outcome)

    try:
        run_given(strategy, seed_value, examples, (Phase.generate, Phase.shrink), body)
    except _Divergence as exc:
        return exc.outcome
    return found["outcome"]


def _explicit_reref_nan_is_oracle_defect(
    case: FuzzCase, eeg: dict, python: SideResult, oracle: SideResult, tolerance: Tolerance
) -> bool:
    """True when the only gap is EEGLAB's ``refmatrix * data`` spreading NaNs."""
    if case.function != "pop_reref" or not python.ok or not oracle.ok or python.data is None or oracle.data is None:
        return False
    params = thaw_params(case.params)
    ref = [int(index) for index in params["ref"]]
    if not ref:
        return False
    data = np.asarray(eeg["data"], dtype=np.float64)
    if data.ndim != 2:
        return False
    exclude = [int(index) for index in params.get("exclude") or ()]
    keepref = str(params.get("keepref") or "off")
    subtraction = _explicit_reref(data, ref, exclude, keepref, matrix=False)
    matrix = _explicit_reref(data, ref, exclude, keepref, matrix=True)
    python_gap, *_rest = compare_data(python.data, subtraction, float(python.srate), float(python.srate), tolerance)
    oracle_gap, *_rest = compare_data(oracle.data, matrix, float(oracle.srate), float(oracle.srate), tolerance)
    return not python_gap and not oracle_gap


def _stale_zero_trials_is_known(
    case: FuzzCase, eeg: dict, python: SideResult, oracle: SideResult, tolerance: Tolerance
) -> bool:
    """True when EEGLAB dropped events because ``trials`` was still 0.

    ``eeg_checkset.m`` lines 440-449 compare latency with ``pnts*trials+1``
    before lines 821-826 set ``trials`` from the data shape. EEGPrep sets
    ``trials`` first, so those events stay. A data mismatch is still filed.
    """
    if case.function != "eeg_eegrej" or not python.ok or not oracle.ok:
        return False
    if python.data is None or oracle.data is None or python.srate is None or oracle.srate is None:
        return False
    stored = eeg.get("trials")
    if isinstance(stored, bool) or not isinstance(stored, (int, float, np.integer, np.floating)):
        return False
    if int(stored) != 0:
        return False
    data_gap, *_rest = compare_data(python.data, oracle.data, float(python.srate), float(oracle.srate), tolerance)
    if data_gap:
        return False
    kept = tuple(event for event in python.events if event[1] <= 1)
    return kept == oracle.events and len(python.events) > len(oracle.events)


def _asr_short_calibration_is_oracle_defect(
    case: FuzzCase, eeg: dict, python: SideResult, oracle: SideResult, tolerance: Tolerance
) -> bool:
    """True when EEGLAB fell back to all-data ASR after fit_eeg_distribution threw."""
    if case.function != "clean_asr" or not python.ok or not oracle.ok or oracle.data is None:
        return False
    params = thaw_params(case.params)
    if isinstance(params["ref_maxbadchannels"], str):
        return False
    if not _fit_eeg_distribution_throws(_calibration_window_count(int(eeg["pnts"]), float(eeg["srate"]))):
        return False
    fallback_params = dict(params)
    fallback_params["ref_maxbadchannels"] = "off"
    fallback = dispatch_python(FuzzCase(case.function, case.spec, freeze_params(fallback_params)), copy.deepcopy(eeg))
    fallback_gap, *_rest = compare_data(
        fallback["data"], oracle.data, float(eeg["srate"]), float(oracle.srate), tolerance
    )
    return not fallback_gap


def _asr_zero_width_index_is_known(case: FuzzCase, eeg: dict, python: SideResult, oracle: SideResult) -> bool:
    """EEGLAB's calibration RMS fit indexes a zero window width and errors."""
    if case.function != "clean_asr" or not python.ok or oracle.ok:
        return False
    if "Index in position 1 is invalid" not in (oracle.error_message or ""):
        return False
    params = thaw_params(case.params)
    if params["ref_maxbadchannels"] != "off":
        return False
    return _fit_width_reaches_zero(_asr_rms_window_count(int(eeg["pnts"]), float(eeg["srate"])))


def _low_rate_spectral_filter_is_known(
    case: FuzzCase, eeg: dict, python: SideResult, oracle: SideResult, tolerance: Tolerance
) -> bool:
    """The default ASR breakpoints are not monotonic, so calibration is skipped.

    ``clean_artifacts`` returns the recording unchanged. A direct ``clean_asr``
    error stays with ``judge``, which already ignores EEGLAB's missing-yulewalk
    report. A Python error here is still a divergence.
    """
    if case.function != "clean_asr" or not python.ok or python.data is None:
        return False
    if spectral_shaping_filter_available(float(eeg["srate"])):
        return False
    original = np.asarray(eeg["data"], dtype=np.float64)
    if not np.array_equal(np.asarray(python.data, dtype=np.float64), original):
        return False
    if not oracle.ok:
        return "yulewalk() function was not found" in (oracle.error_message or "")
    if oracle.data is None:
        return False
    gap, *_rest = compare_data(original, oracle.data, float(eeg["srate"]), float(oracle.srate), tolerance)
    return not gap


def _explicit_reref(data: np.ndarray, ref: list[int], exclude: list[int], keepref: str, *, matrix: bool) -> np.ndarray:
    excluded = set(exclude)
    chansin = [index for index in range(data.shape[0]) if index not in excluded]
    position = {channel: index for index, channel in enumerate(chansin)}
    block = data[chansin]
    if matrix:
        refmatrix = np.eye(len(chansin))
        scale = 1.0 / len(ref)
        for index in ref:
            refmatrix[:, position[index]] -= scale
        with np.errstate(invalid="ignore"):
            updated = refmatrix @ block
    else:
        with np.errstate(invalid="ignore"):
            updated = block - data[ref].mean(axis=0)
    out = np.array(data, copy=True)
    out[chansin] = updated
    if keepref.lower() == "off":
        keep = [index for index in range(out.shape[0]) if index not in set(ref)]
        out = out[keep]
    return out


def evaluate_case(case, materialize, python_backend, oracle_backend, tolerances) -> Outcome:
    if case.sanity:
        try:
            eeg = materialize(case)
        except Exception as exc:
            return _outcome(case, True, False, "materialize", 0.0, 0.0, str(exc), "")
    else:
        eeg = materialize(case)
    digest = data_sha256(np.asarray(eeg["data"]))
    if case.function == "pipeline":
        return _evaluate_pipeline(case, eeg, oracle_backend, tolerances, digest)
    python = python_backend.run(case, copy.deepcopy(eeg))
    oracle = oracle_backend.run(case, copy.deepcopy(eeg))
    tolerance = tolerance_for(report_name(case), tolerances)
    if _explicit_reref_nan_is_oracle_defect(case, eeg, python, oracle, tolerance):
        # reref.m multiplies by a reference matrix that contains explicit zeros.
        # 0*NaN is NaN, so a NaN outside the reference spreads to every channel.
        # Subtraction, which EEGLAB's own alternate code comments as equivalent,
        # keeps that NaN on its own channel. EEGPrep follows the subtraction.
        return _outcome(case, False, True, "match", 0.0, 0.0, "match", digest)
    if _asr_short_calibration_is_oracle_defect(case, eeg, python, oracle, tolerance):
        # fit_eeg_distribution.m:112 indexes a rounded width of 1 or less.
        # clean_asr.m catches that and calibrates on every sample. EEGPrep raises
        # on the same grid and uses the same all-data calibration.
        return _outcome(case, False, True, "match", 0.0, 0.0, "match", digest)
    if _asr_zero_width_index_is_known(case, eeg, python, oracle):
        # asr_calibrate.m's own RMS fit, with reference selection off, indexes
        # X(m,:) at m = 0 for two or three windows. EEGPrep still calibrates.
        return _outcome(case, False, True, "match", 0.0, 0.0, "match", digest)
    if _low_rate_spectral_filter_is_known(case, eeg, python, oracle, tolerance):
        # Below the rate where the default breakpoints stay monotonic,
        # clean_artifacts.m catches the filter error and returns the recording.
        return _outcome(case, False, True, "match", 0.0, 0.0, "match", digest)
    if not python.ok and not oracle.ok and _clean_windows_short_fit(case, eeg):
        # fit_eeg_distribution.m:112 indexes a rounded width of 1 or less.
        # clean_windows has no all-data fallback, so both sides raise.
        return _outcome(
            case,
            False,
            True,
            "match",
            0.0,
            0.0,
            "both sides reject a recording whose clean_windows fit width rounds to 1 or less",
            digest,
        )
    outcome = judge(python, oracle, tolerance, case, digest, tolerances)
    if outcome.diverged and _stale_zero_trials_is_known(case, eeg, python, oracle, tolerance):
        # eeg_checkset.m:440-449 uses the stored trials value. A stored 0
        # drops events past sample 1, and lines 821-826 correct trials after
        # that. EEGPrep sets trials from the data shape first.
        return _outcome(
            case,
            False,
            True,
            "match",
            0.0,
            0.0,
            "known difference: eeg_eegrej keeps events when trials was 0",
            digest,
        )
    if outcome.diverged and case.function == "pop_rmbase" and _event_targets().rmbase_boundary_offset(case, eeg):
        # pop_rmbase.m shifts segment indices when the baseline starts after
        # sample 1. EEGPrep removes the mean of the listed samples.
        return _outcome(
            case,
            False,
            True,
            "match",
            0.0,
            0.0,
            "known divergence: continuous pop_rmbase with a boundary and pointrange starting after sample 1",
            digest,
        )
    if outcome.error_class == "mask" and case.function == "clean_channels":
        outcome.detail = f"{outcome.detail}; {_event_targets().clean_channels_margins(eeg, case.params)}"
    return outcome


def _evaluate_pipeline(case, eeg, oracle_backend, tolerances, digest) -> Outcome:
    """Compare each step on one shared input, then continue from the oracle EEG."""
    current = eeg
    last_abs = 0.0
    last_rel = 0.0
    last_detail = "match"
    for name, frozen in case.pipeline:
        step = FuzzCase(name, case.spec, frozen, sanity=case.sanity)
        python_side, _python_eeg = _invoke(dispatch_python, step, copy.deepcopy(current))
        oracle_side, oracle_eeg = _oracle_invoke(oracle_backend, step, copy.deepcopy(current))
        if not python_side.ok and not oracle_side.ok and _clean_windows_short_fit(step, current):
            return _outcome(
                case,
                False,
                True,
                "match",
                last_abs,
                last_rel,
                f"{name}: both sides reject a recording whose clean_windows fit width rounds to 1 or less",
                digest,
            )
        tolerance = _pipeline_tolerance(name, tolerances)
        outcome = _judge_pair(
            python_side,
            oracle_side,
            tolerance,
            step,
            digest,
            PIPELINE_COMPARATORS,
        )
        if outcome.error_class == "both-error":
            return _outcome(case, False, False, f"{name}:both-error", 0.0, 0.0, outcome.detail, digest)
        if outcome.error_class == "oracle-infrastructure":
            return _outcome(case, False, False, f"{name}:oracle-infrastructure", 0.0, 0.0, outcome.detail, digest)
        known_reref = outcome.diverged and _explicit_reref_nan_is_oracle_defect(
            step, current, python_side, oracle_side, tolerance
        )
        known_trials = outcome.diverged and _stale_zero_trials_is_known(
            step, current, python_side, oracle_side, tolerance
        )
        known_rmbase = (
            outcome.diverged and name == "pop_rmbase" and _event_targets().rmbase_boundary_offset(step, current)
        )
        if (
            not known_reref
            and not known_trials
            and not known_rmbase
            and (outcome.diverged or outcome.error_class != "match")
        ):
            return _outcome(
                case,
                outcome.diverged,
                outcome.both_ok,
                f"{name}:{outcome.error_class}",
                outcome.max_abs,
                outcome.max_rel,
                outcome.detail,
                digest,
            )
        # Continue from the Python EEG when MATLAB's output is the known
        # defect: a NaN spread by the reference matrix, or events dropped
        # because trials was still 0. Later steps then see a real signal.
        if (known_reref or known_trials) and _eeg_dict(_python_eeg) is not None:
            nxt = _python_eeg
        else:
            nxt = oracle_eeg if _eeg_dict(oracle_eeg) is not None else _python_eeg
        synced = _eeg_dict(nxt)
        if synced is None:
            return _outcome(case, True, False, f"{name}:resync", 0.0, 0.0, "step output has no EEG dict", digest)
        current = synced
        last_abs = outcome.max_abs
        last_rel = outcome.max_rel
        if known_reref:
            last_detail = f"{name}: known difference: a NaN stays on its channel under explicit re-reference"
        elif known_trials:
            last_detail = f"{name}: known difference: eeg_eegrej keeps events when trials was 0"
        elif known_rmbase:
            last_detail = (
                f"{name}: known divergence: continuous pop_rmbase with a boundary "
                "and pointrange starting after sample 1"
            )
    return _outcome(case, False, True, "match", last_abs, last_rel, last_detail, digest)


def _invoke(call, step: FuzzCase, eeg: dict):
    try:
        raw = call(step, eeg)
    except Exception as exc:
        return side_from_error(exc), None
    if isinstance(raw, SideResult):
        return raw, None
    return side_from_call(raw), raw


def _oracle_invoke(backend, step: FuzzCase, eeg: dict):
    if isinstance(backend, FakeOracle):
        if backend.mode == "sanity-fail":
            return side_from_error(RuntimeError("fuzz sanity bridge failed")), None
        side, produced = _invoke(dispatch_python, step, eeg)
        if backend.mode == "survey-diverge" and not step.sanity and side.ok and _eeg_dict(produced) is not None:
            produced = _perturb_eeg(produced)
            side = side_from_call(produced)
        return side, produced
    if isinstance(backend, MatlabBackend):
        return _invoke(lambda step_case, data: dispatch_matlab(backend.engine(), step_case, data), step, eeg)
    if hasattr(backend, "run_raw"):
        return _invoke(backend.run_raw, step, eeg)
    return _invoke(backend.run, step, eeg)


def _eeg_dict(raw):
    if isinstance(raw, tuple) and raw and isinstance(raw[0], dict) and "data" in raw[0]:
        return raw[0]
    if isinstance(raw, dict) and "data" in raw:
        return raw
    return None


def _perturb_eeg(raw):
    eeg = _eeg_dict(raw)
    eeg["data"] = np.array(np.asarray(eeg["data"]) + 1.0, copy=True)
    if isinstance(raw, tuple):
        return (eeg, *raw[1:])
    return eeg


def dispatch_python(case: FuzzCase, eeg: dict):
    params = thaw_params(case.params)
    if case.function == "pop_eegfiltnew":
        return pop_eegfiltnew(eeg, gui=False, plotfreqz=False, **_filter_kwargs(params))
    if case.function == "pop_reref":
        kwargs = {"gui": False}
        if params["ref"] or params["keepref"] != "off":
            kwargs["keepref"] = params["keepref"]
        if params["exclude"]:
            kwargs["exclude"] = list(params["exclude"])
        return pop_reref(eeg, list(params["ref"]), **kwargs)
    if case.function == "pop_resample":
        return pop_resample(eeg, float(params["freq"]), gui=False)
    if case.function == "clean_asr":
        if params["entry"] == "clean_rawdata":
            return pop_clean_rawdata(eeg, gui=False, **_rawdata_options(params))
        return clean_asr(eeg, **_asr_kwargs(params))
    if case.function == "pipeline":
        return run_pipeline(case, eeg, dispatch_python)
    target = _event_targets().PYTHON.get(case.function)
    if target is not None:
        return target(case, eeg)
    raise ValueError(f"unknown fuzz function {case.function}")


def _chaninfo_nones_to_empty(eeg: dict) -> dict:
    """Replace None chaninfo values with [] on a copy.

    ds003194 EDF imports leave plotrad and shrink as None. savemat cannot
    write None, so the MATLAB call never started.
    """
    info = eeg.get("chaninfo")
    if not isinstance(info, dict) or not any(value is None for value in info.values()):
        return eeg
    copied = dict(eeg)
    copied["chaninfo"] = {key: np.array([]) if value is None else value for key, value in info.items()}
    return copied


def dispatch_matlab(engine, case: FuzzCase, eeg: dict):
    eeg = _chaninfo_nones_to_empty(eeg)
    params = thaw_params(case.params)
    if case.function == "pop_eegfiltnew":
        args = []
        for key, value in _filter_kwargs(params).items():
            args.extend([key, int(value) if isinstance(value, bool) else value])
        args.extend(["plotfreqz", 0])
        return engine.pop_eegfiltnew(eeg, *args)
    if case.function == "pop_reref":
        ref = np.array([index + 1 for index in params["ref"]], dtype=float) if params["ref"] else np.array([])
        args = [ref]
        if params["ref"] or params["keepref"] != "off":
            args.extend(["keepref", params["keepref"]])
        if params["exclude"]:
            args.extend(["exclude", np.array([index + 1 for index in params["exclude"]], dtype=float)])
        return engine.pop_reref(eeg, *args)
    if case.function == "pop_resample":
        return engine.pop_resample(eeg, float(params["freq"]))
    if case.function == "clean_asr":
        if params["entry"] == "clean_rawdata":
            args = []
            for key, value in _rawdata_options(params).items():
                args.extend([key, value])
            return engine.pop_clean_rawdata(eeg, *args)
        return engine.clean_asr(eeg, *_matlab_asr_args(params))
    if case.function == "pipeline":
        return run_pipeline(case, eeg, lambda step, data: dispatch_matlab(engine, step, data))
    target = _event_targets().MATLAB.get(case.function)
    if target is not None:
        return target(engine, case, eeg)
    raise ValueError(f"unknown fuzz function {case.function}")


def run_pipeline(case: FuzzCase, eeg: dict, call) -> SideResult:
    """Run each pipeline step and keep that step's outputs for comparison."""
    current = eeg
    steps: list[tuple[str, SideResult]] = []
    for name, frozen in case.pipeline:
        step_case = FuzzCase(name, case.spec, frozen, sanity=case.sanity)
        try:
            raw = call(step_case, current)
        except Exception as exc:
            side = side_from_error(exc)
            steps.append((name, side))
            return replace(side, steps=tuple(steps))
        side = side_from_call(raw)
        steps.append((name, side))
        if not side.ok:
            return replace(side, steps=tuple(steps))
        current = raw[0] if isinstance(raw, tuple) else raw
    if not steps:
        return side_from_call(current)
    return replace(side_from_call(current), steps=tuple(steps))


_SPEC_JSON_DEFAULTS = (
    ("source", "eeglab_data"),
    ("events", "none"),
    ("boundaries", ()),
    ("trials", 1),
    ("epoch_pnts", 0),
    ("xmin", 0.0),
    ("srate_hz", 0.0),
    ("drop_xyz", ()),
    ("artifact", ""),
    ("artifact_channel", -1),
    ("artifact_start", 0),
    ("artifact_length", 0),
    ("prior_mask", ""),
)


def spec_to_json(spec: EEGSpec) -> dict:
    payload = {
        "channels": list(spec.channels),
        "start": spec.start,
        "length": spec.length,
        "decimate": spec.decimate,
        "nan_channel": spec.nan_channel,
        "nan_start": spec.nan_start,
        "nan_length": spec.nan_length,
    }
    for key, default in _SPEC_JSON_DEFAULTS:
        value = getattr(spec, key)
        if value == default:
            continue
        if isinstance(value, tuple):
            payload[key] = [list(item) if isinstance(item, tuple) else item for item in value]
        else:
            payload[key] = value
    return payload


def repro_stem(signature: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", signature).strip("-")


def side_from_call(result) -> SideResult:
    if isinstance(result, SideResult):
        return result
    if isinstance(result, tuple):
        if not result:
            return side_from_error(TypeError("call did not return an EEG dict"))
        return side_from_eeg(result[0], result[1:])
    return side_from_eeg(result)


def side_from_eeg(eeg, extra=()) -> SideResult:
    if not isinstance(eeg, dict) or "data" not in eeg:
        return side_from_error(TypeError("call did not return an EEG dict"))
    data = np.array(np.asarray(eeg["data"]), dtype=np.float64, copy=True)
    srate = eeg.get("srate", float("nan"))
    return SideResult(
        True,
        data,
        float(srate),
        None,
        None,
        _event_view(eeg),
        _labels(eeg),
        _int_field(eeg, "nbchan"),
        _int_field(eeg, "pnts"),
        _int_field(eeg, "trials"),
        _float_field(eeg, "xmin"),
        _float_field(eeg, "xmax"),
        _mask(eeg, "clean_channel_mask"),
        _mask(eeg, "clean_sample_mask"),
        _extra_arrays(extra),
    )


def side_from_error(exc: Exception) -> SideResult:
    message = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    return SideResult(False, None, None, exc.__class__.__name__, message[:300])


def _compare_events(python: SideResult, oracle: SideResult, tolerance: Tolerance):
    if len(python.events) != len(oracle.events):
        return True, "event-count", f"event count {len(python.events)} != {len(oracle.events)}"
    for index, (left, right) in enumerate(zip(python.events, oracle.events, strict=True)):
        if any(not _same_event_field(left[field], right[field]) for field in (0, 2, 3, 4)):
            return True, "event-field", f"event {index} field {left} != {right}"
        if abs(float(left[1]) - float(right[1])) > tolerance.event_latency_abs:
            return True, "event-latency", f"event {index} latency {left[1]} != {right[1]}"
    return False, "match", "match"


def _compare_masks(python: SideResult, oracle: SideResult):
    for field in ("clean_channel_mask", "clean_sample_mask"):
        left = getattr(python, field)
        right = getattr(oracle, field)
        if left is None and right is None:
            continue
        if left is None or right is None or left.shape != right.shape or not np.array_equal(left, right):
            return True, "mask", f"{field} differs"
    return False, "match", "match"


def _compare_meta(python: SideResult, oracle: SideResult, tolerance: Tolerance):
    if python.labels != oracle.labels:
        return True, "meta", f"labels {python.labels} != {oracle.labels}"
    for field in ("nbchan", "pnts", "trials"):
        if getattr(python, field) != getattr(oracle, field):
            return True, "meta", f"{field} {getattr(python, field)} != {getattr(oracle, field)}"
    for field in ("xmin", "xmax"):
        left = getattr(python, field)
        right = getattr(oracle, field)
        if left is None or right is None:
            if left != right:
                return True, "meta", f"{field} {left} != {right}"
            continue
        if abs(float(left) - float(right)) > tolerance.meta_rel * abs(float(right)):
            return True, "meta", f"{field} {left} != {right}"
    return False, "match", "match"


def _compare_extra(python: SideResult, oracle: SideResult):
    if len(python.extra) != len(oracle.extra):
        return True, "extra", f"extra count {len(python.extra)} != {len(oracle.extra)}"
    for index, (left, right) in enumerate(zip(python.extra, oracle.extra, strict=True)):
        if left.shape != right.shape or not np.array_equal(left, right):
            return True, "extra", f"extra[{index}] differs"
    return False, "match", "match"


def _same_event_field(left, right) -> bool:
    """True when two event fields match, including NaN with NaN."""
    if left == right:
        return True
    return (
        isinstance(left, (float, np.floating))
        and isinstance(right, (float, np.floating))
        and math.isnan(float(left))
        and math.isnan(float(right))
    )


def _event_view(eeg: dict) -> tuple:
    return tuple(
        (
            str(event.get("type", "")),
            float(_scalar(event.get("latency", np.nan))),
            _optional_float(event, "duration"),
            _optional_int(event, "epoch"),
            _optional_int(event, "urevent"),
        )
        for event in _iter_events(eeg)
    )


def _iter_events(eeg: dict):
    raw = eeg.get("event")
    if raw is None:
        return
    if isinstance(raw, np.ndarray):
        raw = list(raw.ravel())
    for event in raw:
        item = event.item() if isinstance(event, np.ndarray) else event
        if isinstance(item, dict):
            yield item


def _labels(eeg: dict) -> tuple:
    raw = eeg.get("chanlocs")
    if raw is None:
        return ()
    if isinstance(raw, np.ndarray):
        raw = list(raw.ravel())
    labels = []
    for channel in raw:
        item = channel.item() if isinstance(channel, np.ndarray) else channel
        if isinstance(item, dict):
            labels.append(str(item.get("labels", "")))
    return tuple(labels)


def _mask(eeg: dict, key: str):
    etc = eeg.get("etc")
    if not isinstance(etc, dict) or key not in etc:
        return None
    value = etc[key]
    if _missing(value):
        return None
    return np.array(np.asarray(value), dtype=bool, copy=True).reshape(-1)


def _extra_arrays(values) -> tuple:
    arrays = []
    for value in values:
        array = np.asarray(value)
        if array.dtype == bool or array.dtype.kind in "fc":
            array = np.rint(np.real(array)).astype(np.int64)
        else:
            array = np.array(array, dtype=np.int64, copy=True)
        arrays.append(array)
    return tuple(arrays)


def _optional_float(event: dict, key: str):
    if key not in event or _missing(event[key]):
        return None
    return float(_scalar(event[key]))


def _optional_int(event: dict, key: str):
    if key not in event or _missing(event[key]):
        return None
    return int(np.rint(float(_scalar(event[key]))))


def _int_field(eeg: dict, key: str):
    value = eeg.get(key)
    if _missing(value):
        return None
    return int(_scalar(value))


def _float_field(eeg: dict, key: str):
    value = eeg.get(key)
    if _missing(value):
        return None
    return float(_scalar(value))


def _scalar(value):
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0]
    return value


def _missing(value) -> bool:
    if isinstance(value, np.ndarray):
        return value.size == 0
    return value is None or value == ""


def _outcome(case, diverged, both_ok, error_class, max_abs, max_rel, detail, digest) -> Outcome:
    return Outcome(case, diverged, both_ok, error_class, max_abs, max_rel, detail, digest)


def _srate_or_match(python_srate: float, oracle_srate: float, max_abs: float, max_rel: float):
    if abs(float(python_srate) - float(oracle_srate)) > 1e-4:
        delta = abs(float(python_srate) - float(oracle_srate))
        return True, "srate", delta, 0.0, f"srate {python_srate} != {oracle_srate}"
    return False, "match", max_abs, max_rel, "match"


def _exact_reref_signal(shape: tuple[int, int]) -> np.ndarray:
    channels, samples = shape
    column = np.arange(samples, dtype=np.float64) % 7.0
    scale = np.arange(1, channels + 1, dtype=np.float64)[:, None]
    return scale * column


def _copy_eeg(source: dict) -> dict:
    return copy.deepcopy(source)


def _copy_channel(channel):
    return copy.deepcopy(channel.item() if isinstance(channel, np.ndarray) else channel)


def _draw_spec(draw, info: SeedInfo) -> EEGSpec:
    short = draw(st.sampled_from([False, False, False, False, True]))
    if short:
        length = draw(st.integers(32, min(_SHORT_MAX, info.n_samples)))
    else:
        length = draw(st.integers(_MIN_LENGTH, min(_MAX_LENGTH, info.n_samples)))
    decimate = draw(st.sampled_from([1, 1, 1, 2, 4]))
    while length * decimate > info.n_samples and decimate > 1:
        decimate //= 2
    span = length * decimate
    max_start = info.n_samples - span
    start = draw(st.integers(0, max_start)) if max_start else 0
    count = draw(st.integers(1, info.n_channels))
    offset_max = info.n_channels - count
    offset = draw(st.integers(0, offset_max)) if offset_max else 0
    inject = draw(st.sampled_from([False, False, False, True]))
    if inject:
        nan_channel = draw(st.integers(0, count - 1))
        nan_length = draw(st.integers(1, min(4, length)))
        nan_start = draw(st.integers(0, length - nan_length))
    else:
        nan_channel, nan_start, nan_length = -1, 0, 0
    return EEGSpec(tuple(range(offset, offset + count)), start, length, decimate, nan_channel, nan_start, nan_length)


def _draw_filter(draw, srate: float) -> dict:
    nyquist = srate / 2.0
    kind = draw(st.sampled_from(["lowpass", "highpass", "bandpass", "notch"]))
    order = draw(st.sampled_from([40, 60, 80, 100]))
    minphase = draw(st.sampled_from([False, False, True]))
    usefftfilt = draw(st.sampled_from([False, False, True]))
    common = {"filtorder": order, "minphase": minphase, "usefftfilt": usefftfilt}
    if kind == "lowpass":
        high = draw(st.integers(5, max(5, int(nyquist * 0.6))))
        return {**common, "locutoff": None, "hicutoff": high, "revfilt": False}
    if kind == "highpass":
        low = draw(st.integers(1, max(1, min(8, int(nyquist * 0.25)))))
        return {**common, "locutoff": low, "hicutoff": None, "revfilt": False}
    low = draw(st.integers(1, max(1, min(10, int(nyquist * 0.3)))))
    high_min = low + 4
    high_max = max(high_min, min(int(nyquist) - 1, int(nyquist * 0.7)))
    high = high_min if high_max <= high_min else draw(st.integers(high_min, high_max))
    return {**common, "locutoff": low, "hicutoff": high, "revfilt": kind == "notch"}


def _draw_reref(draw, n_channels: int) -> dict:
    mode = draw(st.sampled_from(["average", "average", "channel", "pair"]))
    if mode == "channel":
        ref = (draw(st.integers(0, n_channels - 1)),)
    elif mode == "pair" and n_channels >= 2:
        first = draw(st.integers(0, n_channels - 2))
        second = draw(st.integers(first + 1, n_channels - 1))
        ref = (first, second)
    else:
        ref = ()
    excluded = [index for index in range(n_channels) if index not in ref]
    use_exclude = bool(excluded) and draw(st.sampled_from([False, False, True]))
    exclude = (draw(st.sampled_from(excluded)),) if use_exclude else ()
    keepref = "off" if not ref else draw(st.sampled_from(["off", "on"]))
    return {"ref": ref, "keepref": keepref, "exclude": exclude}


def _draw_asr(draw, entry: str) -> dict:
    return {
        "entry": entry,
        "cutoff": draw(st.sampled_from([5.0, 5.0, 3.0, 10.0, 20.0])),
        "ref_maxbadchannels": draw(st.sampled_from(["off", "off", "off", 0.075, 0.2])),
        "window_len": draw(st.sampled_from([None, None, 0.5, 1.0])),
    }


def _filter_sanity() -> tuple:
    return freeze_params(
        {
            "locutoff": None,
            "hicutoff": 40,
            "filtorder": 100,
            "revfilt": False,
            "minphase": False,
            "usefftfilt": False,
        }
    )


def _reref_sanity() -> tuple:
    return freeze_params({"ref": (), "keepref": "off", "exclude": ()})


def _asr_sanity() -> tuple:
    return freeze_params({"entry": "clean_asr", "cutoff": 5.0, "ref_maxbadchannels": "off", "window_len": None})


def _filter_kwargs(params: dict) -> dict:
    kwargs = {
        "revfilt": bool(params["revfilt"]),
        "minphase": bool(params["minphase"]),
        "usefftfilt": bool(params["usefftfilt"]),
    }
    if params.get("filtorder") is not None:
        kwargs["filtorder"] = int(params["filtorder"])
    if params["locutoff"] is not None:
        kwargs["locutoff"] = params["locutoff"]
    if params["hicutoff"] is not None:
        kwargs["hicutoff"] = params["hicutoff"]
    return kwargs


def _asr_kwargs(params: dict) -> dict:
    kwargs = {"cutoff": float(params["cutoff"]), "ref_maxbadchannels": params["ref_maxbadchannels"]}
    if params["window_len"] is not None:
        kwargs["window_len"] = float(params["window_len"])
    if not isinstance(params["ref_maxbadchannels"], str):
        kwargs["ref_tolerances"] = (-3.5, 5.5)
        kwargs["ref_wndlen"] = 1.0
    return kwargs


def _matlab_asr_args(params: dict) -> list:
    ref = params["ref_maxbadchannels"]
    numeric_ref = not isinstance(ref, str)
    return [
        float(params["cutoff"]),
        np.array([]) if params["window_len"] is None else float(params["window_len"]),
        np.array([]),
        np.array([]),
        ref if isinstance(ref, str) else float(ref),
        np.array([-3.5, 5.5]) if numeric_ref else np.array([]),
        1.0 if numeric_ref else np.array([]),
    ]


def _rawdata_options(params: dict) -> dict:
    return {
        "FlatlineCriterion": "off",
        "Highpass": "off",
        "ChannelCriterion": "off",
        "LineNoiseCriterion": "off",
        "WindowCriterion": "off",
        "BurstCriterion": float(params["cutoff"]),
        "BurstCriterionRefMaxBadChns": params["ref_maxbadchannels"],
        "BurstRejection": "off",
    }


def _write_reports(
    result: SessionResult,
    summary_path: Path | None,
    tracker,
    run_url: str,
    filing: str,
    issue_cap: int,
) -> None:
    if tracker is not None:
        publish = publish_findings(
            seed=result.seed,
            functions=result.functions,
            findings=result.divergences if result.harness_ok else [],
            tracker=tracker,
            cap=issue_cap,
            run_url=run_url,
            harness_ok=result.harness_ok,
        )
        result.created = publish.created
        result.commented = publish.commented
        result.unfiled = publish.unfiled
    summary = render_summary(result)
    (result.output_dir / "summary.md").write_text(summary)
    if summary_path is not None:
        summary_path.write_text(summary)
    github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path is None and github_summary:
        with open(github_summary, "a", encoding="utf-8") as handle:
            handle.write(summary)
    metrics = {
        "seed": result.seed,
        "date": f"{result.seed // 10000:04d}-{result.seed % 10000 // 100:02d}-{result.seed % 100:02d}",
        "harness_ok": result.harness_ok,
        "filing": filing,
        "functions": result.functions,
        "sanity": result.sanity,
        "divergences": [
            {
                "signature": outcome.signature,
                "function": outcome.report,
                "max_abs": outcome.max_abs,
                "max_rel": outcome.max_rel,
                "error_class": outcome.error_class,
                "params": outcome.params,
                "repro": outcome.repro_command,
                "confirmed": outcome.confirmed,
            }
            for outcome in result.divergences
        ],
        "created_issues": result.created,
        "commented_issues": result.commented,
        "unfiled": result.unfiled,
    }
    (result.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
