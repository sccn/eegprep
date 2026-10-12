"""Generate EEG cases, run them through EEGPrep and an oracle, and compare.

EEGLAB output is the oracle. Absolute and relative tolerances are loaded from
``tolerances.toml`` and are not adjusted here when a case misses.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
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
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_resample import pop_resample
from eegprep.functions.popfunc.pop_reref import pop_reref
from eegprep.plugins.clean_rawdata.clean_asr import clean_asr
from eegprep.plugins.clean_rawdata.pop_clean_rawdata import pop_clean_rawdata
from eegprep.plugins.firfilt.pop_eegfiltnew import pop_eegfiltnew
from tests.fuzz.issues import FUNCTION_ROWS, publish_findings
from tests.matlab_batch_engine import install

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_SET = REPO_ROOT / "sample_data" / "eeglab_data.set"
TOLERANCES_PATH = Path(__file__).resolve().parent / "tolerances.toml"
FUNCTION_NAMES = ("pop_eegfiltnew", "pop_reref", "pop_resample", "clean_asr")
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


@dataclass(frozen=True)
class EEGSpec:
    channels: tuple[int, ...]
    start: int
    length: int
    decimate: int
    nan_channel: int
    nan_start: int
    nan_length: int


@dataclass(frozen=True)
class FuzzCase:
    function: str
    spec: EEGSpec
    params: tuple
    sanity: bool = False


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


def load_tolerances(path: Path | None = None) -> dict[str, Tolerance]:
    """Load per-function tolerances. Missing functions are an error."""
    source = path or TOLERANCES_PATH
    payload = tomllib.loads(source.read_text())
    tolerances = {}
    for name in ("pop_eegfiltnew", "pop_reref", "pop_resample", "clean_asr"):
        row = payload[name]
        tolerances[name] = Tolerance(float(row["abs"]), float(row["rel"]))
    return tolerances


def tolerance_for(name: str, tolerances: dict[str, Tolerance]) -> Tolerance:
    if name == "pop_clean_rawdata":
        return tolerances["clean_asr"]
    return tolerances[name]


def load_seed() -> tuple[SeedInfo, dict]:
    """Load ``sample_data/eeglab_data.set`` once and reuse it for every case."""
    if not hasattr(load_seed, "_cache"):
        eeg = pop_loadset(str(SAMPLE_SET))
        info = SeedInfo(int(eeg["nbchan"]), int(eeg["pnts"]), float(eeg["srate"]))
        load_seed._cache = (info, eeg)  # type: ignore[attr-defined]
    return load_seed._cache  # type: ignore[attr-defined]


def materialize_eeg(case: FuzzCase) -> dict:
    """Crop the sample recording into the EEG described by ``case.spec``.

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
    cropped = np.array(data[np.ix_(list(spec.channels), sample_index)], dtype=np.float64)
    if case.sanity and case.function == "pop_reref":
        # eeglab_data.set is float32. Average reference on those values differs
        # by about 1e-4 (see test_parity_data_reref_with_matlab), which is
        # outside the 1e-6 bar but is not a broken bridge. Sanity uses an
        # integer signal whose mean is exact in float32 and float64.
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


def judge(python: SideResult, oracle: SideResult, tolerance: Tolerance, case: FuzzCase, digest: str) -> Outcome:
    if not python.ok and not oracle.ok:
        same = python.error_type == oracle.error_type
        detail = (
            f"python {python.error_type}: {python.error_message}; oracle {oracle.error_type}: {oracle.error_message}"
        )
        if same:
            return _outcome(case, False, False, "both-error", 0.0, 0.0, detail, digest)
        error_class = f"exception-mismatch:{python.error_type}/{oracle.error_type}"
        return _outcome(case, True, False, error_class, 0.0, 0.0, detail, digest)
    if not python.ok:
        detail = f"python {python.error_type}: {python.error_message}"
        return _outcome(case, True, False, f"python-only:{python.error_type}", 0.0, 0.0, detail, digest)
    if not oracle.ok:
        detail = f"oracle {oracle.error_type}: {oracle.error_message}"
        return _outcome(case, True, False, f"oracle-only:{oracle.error_type}", 0.0, 0.0, detail, digest)
    diverged, error_class, max_abs, max_rel, detail = compare_data(
        python.data, oracle.data, float(python.srate), float(oracle.srate), tolerance
    )
    return _outcome(case, diverged, True, error_class, max_abs, max_rel, detail, digest)


class PythonBackend:
    """Run the EEGPrep implementation of a fuzz case."""

    def run(self, case: FuzzCase, eeg: dict) -> SideResult:
        try:
            return side_from_eeg(dispatch_python(case, eeg))
        except Exception as exc:
            return side_from_error(exc)


class MatlabBackend:
    """Run the matching EEGLAB function through the matlab-batch shim."""

    def __init__(self):
        self._engine = None

    def run(self, case: FuzzCase, eeg: dict) -> SideResult:
        try:
            return side_from_eeg(dispatch_matlab(self.engine(), case, eeg))
        except Exception as exc:
            return side_from_error(exc)

    def engine(self):
        if self._engine is None:
            self._engine = connect_matlab()
        return self._engine


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
            return SideResult(True, np.array(result.data + 1.0, copy=True), result.srate, None, None)
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
    return get_eeglab()


def default_strategies() -> dict:
    if not hasattr(default_strategies, "_cache"):
        info, _eeg = load_seed()
        default_strategies._cache = build_strategies(info)  # type: ignore[attr-defined]
    return default_strategies._cache  # type: ignore[attr-defined]


def default_sanity() -> list[FuzzCase]:
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


def draw_cases(function: str, *, seed_value: int, examples: int) -> list[FuzzCase]:
    """Draw the nightly case list for one function. The same seed redraws it."""
    found: list[FuzzCase] = []

    def body(case: FuzzCase) -> None:
        found.append(case)

    run_given(default_strategies()[function], seed_value, examples, (Phase.generate,), body)
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
) -> SessionResult:
    """Run sanity, then a seeded survey, then shrink confirmed divergences."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("fuzz seed %s", seed_value)
    (output_dir / "seed.txt").write_text(f"{seed_value}\n")

    python_backend = python_backend or PythonBackend()
    oracle_backend = oracle_backend or MatlabBackend()
    strategies = strategies or default_strategies()
    sanity = default_sanity() if sanity is None else sanity
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

    functions = {name: {"examples": 0, "divergences": 0} for name in FUNCTION_ROWS}
    sanity_rows = []
    harness_ok = True
    for case in sanity:
        outcome = evaluate(case)
        if not outcome.both_ok or outcome.diverged:
            harness_ok = False
            sanity_rows.append(
                {"function": outcome.report, "error_class": outcome.error_class, "detail": outcome.detail}
            )
    if not harness_ok:
        for row in sanity_rows:
            logger.error("fuzz sanity %s %s: %s", row["function"], row["error_class"], row["detail"])
        logger.error("fuzz sanity failed; not surveying and not filing issues")
        result = SessionResult(seed_value, False, functions, [], output_dir, sanity_rows, [], [], [])
        _write_reports(result, summary_path, tracker, run_url, "sanity", issue_cap)
        return result

    survey: list[Outcome] = []
    for name in FUNCTION_NAMES:
        if name not in strategies:
            continue
        found: list[Outcome] = []

        def body(case: FuzzCase, found=found) -> None:
            outcome = evaluate(case)
            if outcome is not None:
                found.append(outcome)

        run_given(strategies[name], seed_value, examples, (Phase.generate,), body)
        survey.extend(found)

    for outcome in survey:
        row = functions[outcome.report]
        row["examples"] += 1
        if outcome.diverged:
            row["divergences"] += 1

    by_signature: dict[str, Outcome] = {}
    for outcome in survey:
        if outcome.diverged and outcome.signature not in by_signature:
            by_signature[outcome.signature] = outcome
    ordered = sorted(by_signature.values(), key=lambda item: item.max_abs, reverse=True)[:max_shrinks]
    confirmed: list[Outcome] = []
    for original in ordered:
        shrunk = shrink_signature(
            strategies[original.case.function],
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
            eeg = materialize(fresh.case)
            write_repro(output_dir / "repros", fresh, seed_value, eeg)
            confirmed.append(fresh)
            logger.info("shrunk %s -> %s", original.signature, fresh.signature)
        else:
            logger.info("unconfirmed %s", original.signature)

    result = SessionResult(seed_value, True, functions, confirmed, output_dir, [], [], [], [])
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
    return FuzzCase(
        payload["function"], EEGSpec(**spec), freeze_params(payload["params"]), bool(payload.get("sanity", False))
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
    """Hypothesis strategies for the four nightly targets. Draws shrink toward smaller crops."""

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
        return FuzzCase("clean_asr", spec, freeze_params(_draw_asr(draw)))

    return {
        "pop_eegfiltnew": eegfiltnew(),
        "pop_reref": reref(),
        "pop_resample": resample(),
        "clean_asr": asr(),
    }


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
    exclude = [int(index) for index in params["exclude"]]
    keepref = str(params["keepref"])
    subtraction = _explicit_reref(data, ref, exclude, keepref, matrix=False)
    matrix = _explicit_reref(data, ref, exclude, keepref, matrix=True)
    python_gap, *_rest = compare_data(python.data, subtraction, float(python.srate), float(python.srate), tolerance)
    oracle_gap, *_rest = compare_data(oracle.data, matrix, float(oracle.srate), float(oracle.srate), tolerance)
    return not python_gap and not oracle_gap


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
    python = python_backend.run(case, copy.deepcopy(eeg))
    oracle = oracle_backend.run(case, copy.deepcopy(eeg))
    tolerance = tolerance_for(report_name(case), tolerances)
    if _explicit_reref_nan_is_oracle_defect(case, eeg, python, oracle, tolerance):
        # reref.m multiplies by a reference matrix that contains explicit zeros.
        # 0*NaN is NaN, so a NaN outside the reference spreads to every channel.
        # Subtraction, which EEGLAB's own alternate code comments as equivalent,
        # keeps that NaN on its own channel. EEGPrep follows the subtraction.
        return _outcome(case, False, True, "match", 0.0, 0.0, "match", digest)
    return judge(python, oracle, tolerance, case, digest)


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
    raise ValueError(f"unknown fuzz function {case.function}")


def dispatch_matlab(engine, case: FuzzCase, eeg: dict):
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
    raise ValueError(f"unknown fuzz function {case.function}")


def spec_to_json(spec: EEGSpec) -> dict:
    payload = asdict(spec)
    payload["channels"] = list(spec.channels)
    return payload


def repro_stem(signature: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", signature).strip("-")


def side_from_eeg(eeg) -> SideResult:
    if not isinstance(eeg, dict) or "data" not in eeg:
        return side_from_error(TypeError("call did not return an EEG dict"))
    data = np.array(np.asarray(eeg["data"]), dtype=np.float64, copy=True)
    srate = eeg.get("srate", float("nan"))
    return SideResult(True, data, float(srate), None, None)


def side_from_error(exc: Exception) -> SideResult:
    message = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    return SideResult(False, None, None, exc.__class__.__name__, message[:300])


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


def _draw_asr(draw) -> dict:
    return {
        "entry": draw(st.sampled_from(["clean_asr", "clean_asr", "clean_asr", "clean_rawdata"])),
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
        "filtorder": int(params["filtorder"]),
        "revfilt": bool(params["revfilt"]),
        "minphase": bool(params["minphase"]),
        "usefftfilt": bool(params["usefftfilt"]),
    }
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
