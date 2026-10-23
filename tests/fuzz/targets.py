"""Realistic-first generators and calls for the event-aware fuzz targets.

About 70% of draws use parameter values from EEGLAB defaults, the EEGPrep
dialogs, ``bids_preproc``, and the tutorial / ds003061 stage settings. The
other 30% walk the edge cases in the tier-1 plan (fractional rates, region
edges, empty selections, nested boundaries).

Recordings are the committed tutorial sets plus, when present, the same
ds003061 and ds003194 files the parity tests load. A missing suite or
OpenNeuro root leaves those recordings out. A root that is set and does not
contain the file raises ``FileNotFoundError``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from hypothesis import strategies as st

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.popfunc.eeg_eegrej import eeg_eegrej
from eegprep.functions.popfunc.pop_epoch import pop_epoch
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_rmbase import pop_rmbase
from eegprep.functions.popfunc.pop_select import pop_select
from eegprep.plugins.clean_rawdata.clean_channels import clean_channels
from eegprep.plugins.clean_rawdata.clean_flatlines import clean_flatlines
from eegprep.plugins.clean_rawdata.clean_windows import clean_windows
from eegprep.plugins.clean_rawdata.pop_clean_rawdata import pop_clean_rawdata
from tests.fuzz.catalog import freeze_params, thaw_params

_ICA_FIELDS = ("icaweights", "icasphere", "icawinv", "icaact", "icachansind")
_DS003194_EDF = "sub-NEPO01/ses-Nepopre/eeg/sub-NEPO01_ses-Nepopre_task-taskprenepo_eeg.edf"
_PREFIX_SECONDS = 30.0
_PREFIX_CAP = 32768

REPO_ROOT = Path(__file__).resolve().parents[2]


def _suite_root() -> Path | None:
    env = os.environ.get("EEGPREP_EEGLAB_SUITE_ROOT")
    if env:
        return Path(env)
    eeglab = os.environ.get("EEGPREP_EEGLAB_ROOT")
    if not eeglab:
        return None
    parent = Path(eeglab).resolve().parent
    if (parent / "ds003061").is_dir():
        return parent
    return None


def _openneuro_root() -> Path | None:
    override = os.environ.get("EEGPREP_OPENNEURO_ROOT")
    if override:
        root = Path(override)
        if (root / "dataset_description.json").is_file():
            return root
        nested = root / "ds003194"
        if (nested / "dataset_description.json").is_file():
            return nested
        if os.environ.get("EEGPREP_OPENNEURO_ROOT"):
            raise FileNotFoundError(
                f"ds003194 is not under {root}. The parity tests use EEGPREP_OPENNEURO_ROOT "
                "or tests/openneuro/ds003194, and fetch s3://openneuro.org/ds003194/."
            )
        return None
    candidate = REPO_ROOT / "tests" / "openneuro" / "ds003194"
    if (candidate / "dataset_description.json").is_file():
        return candidate
    return None


def _external_sources() -> list[tuple[str, Path, bool]]:
    """ds003061 and ds003194 paths used by the parity tests, when the files exist."""
    found = []
    suite = _suite_root()
    if suite is not None:
        configured = bool(os.environ.get("EEGPREP_EEGLAB_SUITE_ROOT"))
        for subject in ("sub-001", "sub-012"):
            directory = suite / "ds003061" / subject / "eeg"
            matches = sorted(directory.glob("*_run-1_eeg.set"))
            if len(matches) != 1:
                if configured or directory.is_dir():
                    raise FileNotFoundError(
                        f"expected one run-1 .set under {directory}, found {[str(path) for path in matches]}"
                    )
                continue
            found.append((f"ds003061_{subject}", matches[0], True))
    root = _openneuro_root()
    if root is not None:
        edf = root / _DS003194_EDF
        if not edf.is_file():
            raise FileNotFoundError(f"ds003194 recording is not at {edf}")
        found.append(("ds003194_NEPO01", edf, True))
    return found


def _load_recording(path: Path, prefix: bool) -> dict:
    if path.suffix.lower() == ".edf":
        from eegprep.functions.popfunc.pop_load_frombids import pop_load_frombids

        eeg = pop_load_frombids(
            str(path),
            eventtype="value",
            bidsevent=True,
            bidschanloc=True,
            verbose=False,
        )
    else:
        eeg = pop_loadset(str(path))
    if prefix:
        return _prefix_recording(eeg)
    return eeg


def _prefix_recording(eeg: dict) -> dict:
    trials = int(eeg.get("trials") or 1)
    if trials > 1:
        return eeg
    srate = float(eeg["srate"])
    keep = min(int(eeg["pnts"]), _PREFIX_CAP, max(1, int(round(_PREFIX_SECONDS * srate))))
    if keep >= int(eeg["pnts"]):
        return eeg
    eeg["data"] = np.array(np.asarray(eeg["data"])[:, :keep], dtype=np.float64, copy=True)
    eeg["pnts"] = keep
    xmin = float(eeg.get("xmin") or 0.0)
    eeg["xmax"] = xmin + (keep - 1) / srate
    eeg["times"] = np.linspace(xmin * 1000.0, eeg["xmax"] * 1000.0, keep)
    kept = []
    for event in _event_dicts(eeg):
        latency = event.get("latency")
        if latency is None or float(latency) <= keep + 0.5:
            kept.append(dict(event))
    eeg["event"] = kept
    return eeg


def _clear_ica(eeg, spec, rec) -> None:
    if tuple(spec.channels) == tuple(range(rec.nbchan)):
        return
    for key in _ICA_FIELDS:
        if key in eeg:
            eeg[key] = np.array([], dtype=int if key == "icachansind" else float)


_USER = (True, True, True, True, True, True, True, False, False, False)
_USER_LENGTHS = (512, 768, 1024, 1024, 1536, 2048)
_EDGE_LENGTHS = (256, 384, 1024, 4096, 30504)
_EDGE_RATES = (0.0, 64.0, 100.0, 100.5, 128.0, 250.0)
# pop_epoch.m default timelim, bids_preproc EpochLimits, the oddball tutorial window,
# and the ds003061 stage-parity window.
_EPOCH_WINDOWS = ((-1.0, 2.0), (-0.2, 0.8), (-0.5, 1.0))
_BASELINES_MS = ((-200.0, 0.0), (-500.0, 0.0))


@dataclass
class _Recording:
    name: str
    eeg: dict
    nbchan: int
    pnts: int
    trials: int
    srate: float
    xmin: float
    labels: tuple[str, ...]
    event_latencies: tuple[float, ...]
    event_types: tuple[str, ...]
    global_latency: bool


def _corpus() -> dict[str, _Recording]:
    if not hasattr(_corpus, "_cache"):
        loaded = {}
        sources = [
            ("eeglab_data", REPO_ROOT / "sample_data/eeglab_data.set", False),
            ("eeglab_data_with_ica_tmp", REPO_ROOT / "sample_data/eeglab_data_with_ica_tmp.set", False),
            ("eeglab_data_epochs_ica", REPO_ROOT / "sample_data/eeglab_data_epochs_ica.set", False),
            *_external_sources(),
        ]
        for name, path, prefix in sources:
            eeg = _load_recording(path, prefix)
            events = _event_dicts(eeg)
            latencies = tuple(float(event["latency"]) for event in events if "latency" in event)
            types = tuple(dict.fromkeys(str(event.get("type")) for event in events))
            pnts = int(eeg["pnts"])
            loaded[name] = _Recording(
                name,
                eeg,
                int(eeg["nbchan"]),
                pnts,
                int(eeg.get("trials") or 1),
                float(eeg["srate"]),
                float(eeg.get("xmin") or 0.0),
                _labels_of(eeg),
                latencies,
                types,
                any(latency > pnts + 0.5 for latency in latencies),
            )
        _corpus._cache = loaded  # type: ignore[attr-defined]
    return _corpus._cache  # type: ignore[attr-defined]


def recording(name: str) -> _Recording:
    return _corpus()[name]


def materialize_rich(case) -> dict:
    """Build an event-aware crop. The legacy crop stays in the harness."""
    spec = case.spec
    rec = recording(spec.source)
    eeg = copy.deepcopy(rec.eeg)
    if rec.trials > 1:
        cropped, kept_events = _crop_trials(rec, eeg, spec)
    else:
        cropped, kept_events = _crop_continuous(rec, eeg, spec)
    rate = float(spec.srate_hz) if spec.srate_hz else float(rec.srate) / float(spec.decimate)
    _apply_artifact(cropped, spec, rate)
    _apply_nan(cropped, spec)
    eeg["data"] = cropped
    eeg["event"] = kept_events
    if spec.boundaries:
        kept_events.extend(_boundary_events(spec.boundaries, len(kept_events)))
        kept_events.sort(key=lambda event: float(event.get("latency") or 0.0))
    _apply_rate_and_times(eeg, spec, rec)
    _drop_xyz(eeg, spec.drop_xyz)
    _apply_prior_mask(eeg, spec.prior_mask)
    _clear_ica(eeg, spec, rec)
    eeg["saved"] = "no"
    return eeg_checkset(eeg)


def python_eegrej(case, eeg):
    params = thaw_params(case.params)
    regions = _regions_array(params)
    return eeg_eegrej(eeg, regions)


def matlab_eegrej(engine, case, eeg):
    params = thaw_params(case.params)
    return engine.eeg_eegrej(eeg, _regions_array(params))


def python_select(case, eeg):
    return pop_select(eeg, gui=False, **_select_kwargs(thaw_params(case.params)))


def matlab_select(engine, case, eeg):
    args = []
    for key, value in _select_kwargs(thaw_params(case.params)).items():
        args.extend([key, _matlab_value(value)])
    return engine.pop_select(eeg, *args)


def python_epoch(case, eeg):
    params = thaw_params(case.params)
    return pop_epoch(eeg, _epoch_types(params), list(params["lim"]), gui=False, **_epoch_kwargs(params))


def matlab_epoch(engine, case, eeg):
    params = thaw_params(case.params)
    args = [_matlab_types(params), np.asarray(params["lim"], dtype=float)]
    for key, value in _epoch_kwargs(params).items():
        args.extend([key, _matlab_value(value)])
    eeg_out, indices = engine.pop_epoch(eeg, *args, nargout=2)
    # pop_epoch's Python indices are 0-based. MATLAB find() is 1-based.
    indices = np.rint(np.asarray(indices, dtype=float).ravel()) - 1.0
    return eeg_out, indices


def python_rmbase(case, eeg):
    timerange, pointrange, chanlist = _rmbase_args(thaw_params(case.params))
    return pop_rmbase(eeg, timerange, pointrange, chanlist, gui=False)


def matlab_rmbase(engine, case, eeg):
    timerange, pointrange, chanlist = _rmbase_args(thaw_params(case.params))
    return engine.pop_rmbase(
        eeg,
        _matlab_value(timerange),
        _matlab_value(pointrange),
        _matlab_value(chanlist),
    )


def python_windows(case, eeg):
    params = thaw_params(case.params)
    return clean_windows(eeg, **_windows_kwargs(params))


def matlab_windows(engine, case, eeg):
    params = _windows_kwargs(thaw_params(case.params))
    return engine.clean_windows(
        eeg,
        _matlab_value(params["max_bad_channels"]),
        _matlab_value(params["zthresholds"]),
        float(params["window_len"]),
        float(params["window_overlap"]),
        nargout=2,
    )


def python_channels(case, eeg):
    return clean_channels(eeg, **_channels_kwargs(thaw_params(case.params)))


def matlab_channels(engine, case, eeg):
    params = _channels_kwargs(thaw_params(case.params))
    return engine.clean_channels(
        eeg,
        float(params["corr_threshold"]),
        float(params["noise_threshold"]),
        float(params["window_len"]),
        float(params["max_broken_time"]),
        float(params["num_samples"]),
        float(params["subset_size"]),
    )


def python_flatlines(case, eeg):
    params = thaw_params(case.params)
    return clean_flatlines(
        eeg,
        max_flatline_duration=float(params["max_flatline_duration"]),
        max_allowed_jitter=float(params["max_allowed_jitter"]),
    )


def matlab_flatlines(engine, case, eeg):
    params = thaw_params(case.params)
    return engine.clean_flatlines(
        eeg,
        float(params["max_flatline_duration"]),
        float(params["max_allowed_jitter"]),
    )


def python_clean_rawdata(case, eeg):
    return pop_clean_rawdata(eeg, gui=False, **_rawdata_kwargs(thaw_params(case.params)))


def matlab_clean_rawdata(engine, case, eeg):
    args = []
    for key, value in _rawdata_kwargs(thaw_params(case.params)).items():
        args.extend([key, _matlab_value(value)])
    return engine.pop_clean_rawdata(eeg, *args)


PYTHON = {
    "eeg_eegrej": python_eegrej,
    "pop_select": python_select,
    "pop_epoch": python_epoch,
    "pop_rmbase": python_rmbase,
    "clean_windows": python_windows,
    "clean_channels": python_channels,
    "clean_flatlines": python_flatlines,
    "pop_clean_rawdata": python_clean_rawdata,
}

MATLAB = {
    "eeg_eegrej": matlab_eegrej,
    "pop_select": matlab_select,
    "pop_epoch": matlab_epoch,
    "pop_rmbase": matlab_rmbase,
    "clean_windows": matlab_windows,
    "clean_channels": matlab_channels,
    "clean_flatlines": matlab_flatlines,
    "pop_clean_rawdata": matlab_clean_rawdata,
}


def rmbase_boundary_offset(case, eeg) -> bool:
    """Continuous data, a boundary, and a baseline that starts after sample 1.

    EEGLAB shifts the segment indices (pop_rmbase.m). EEGPrep does not.
    """
    if case.function != "pop_rmbase":
        return False
    points = thaw_params(case.params).get("pointrange") or []
    if not points or int(points[0]) <= 1:
        return False
    if int(eeg.get("trials") or 1) != 1:
        return False
    return any(str(event.get("type")) == "boundary" for event in _event_dicts(eeg))


def clean_channels_margins(eeg, params) -> str:
    """Report the channel count ``clean_channels`` itself keeps.

    Diagnostic only. It does not forgive a mask mismatch.
    """
    thawed = thaw_params(params) if isinstance(params, tuple) else params
    try:
        cleaned = clean_channels(copy.deepcopy(eeg), **_channels_kwargs(thawed))
    except Exception as exc:
        return f"python clean_channels diagnostic failed: {exc.__class__.__name__}: {exc}"
    return f"python clean_channels kept {int(cleaned['nbchan'])} of {int(eeg['nbchan'])} channels"


def extra_strategies(info, spec_cls, case_cls, freeze):
    """Hypothesis strategies for the seven new targets and the pipeline."""
    del info, freeze
    corpus = _corpus()
    continuous = (corpus["eeglab_data"], corpus["eeglab_data_with_ica_tmp"])
    epoched = corpus["eeglab_data_epochs_ica"]

    @st.composite
    def eegrej(draw):
        user = _user_draw(draw)
        rec = _pick_continuous(draw, continuous, user)
        spec = _draw_continuous(draw, spec_cls, rec, user=user, events="keep+boundary")
        params = {"regions": _draw_regions(draw, spec, rec, user), "eegplot": (not user) and draw(st.booleans())}
        return case_cls("eeg_eegrej", spec, freeze_params(params))

    @st.composite
    def select(draw):
        user = _user_draw(draw)
        if user and draw(st.booleans()):
            spec = _draw_trials(draw, spec_cls, epoched, user=True)
            count = spec.length
            first = draw(st.integers(1, max(1, count // 2)))
            last = draw(st.integers(first, count))
            params = {"trial": list(range(first, last + 1)), "sorttrial": "on"}
        elif user:
            rec = _pick_continuous(draw, continuous, True)
            spec = _draw_continuous(draw, spec_cls, rec, user=True, events="keep")
            params = _user_select(draw, spec, rec)
        else:
            spec, params = _edge_select(draw, spec_cls, continuous, epoched)
        return case_cls("pop_select", spec, freeze_params(params))

    @st.composite
    def epoch(draw):
        user = _user_draw(draw)
        rec = _pick_continuous(draw, continuous, user)
        spec = _draw_continuous(
            draw,
            spec_cls,
            rec,
            user=user,
            events="keep+boundary" if not user else "keep",
            rates=_EDGE_RATES if not user else (0.0,),
        )
        types = _user_types(draw, rec) if user else _edge_types(draw, rec)
        window = draw(st.sampled_from(_EPOCH_WINDOWS)) if user else _edge_window(draw, spec.srate_hz or rec.srate)
        params = {"types": types, "lim": list(window)}
        if not user and draw(st.booleans()):
            params["valuelim"] = draw(st.sampled_from(([-200.0, 200.0], [50.0], [-20.0, 20.0])))
        if not user and draw(st.booleans()):
            params["eventindices"] = [1, 2]
        return case_cls("pop_epoch", spec, freeze_params(params))

    @st.composite
    def rmbase(draw):
        user = _user_draw(draw)
        if user:
            spec = _draw_trials(draw, spec_cls, epoched, user=True)
            params = {"timerange": list(draw(st.sampled_from(_BASELINES_MS))), "pointrange": [], "chanlist": []}
            if draw(st.booleans()):
                params = {"timerange": [], "pointrange": [], "chanlist": []}
        else:
            spec, params = _edge_rmbase(draw, spec_cls, continuous, epoched)
        return case_cls("pop_rmbase", spec, freeze_params(params))

    @st.composite
    def windows(draw):
        user = _user_draw(draw)
        rec = _pick_continuous(draw, continuous, user)
        seconds = 8.0 if user else draw(st.sampled_from((4.0, 8.0, 16.0)))
        spec = _draw_continuous(
            draw, spec_cls, rec, user=user, events="keep", rates=_EDGE_RATES, seconds=seconds, max_length=16384
        )
        if user:
            params = {
                "max_bad_channels": draw(st.sampled_from((0.2, 0.25))),
                "zthresholds": draw(st.sampled_from(([-3.5, 5.0], ["-inf", 7.0]))),
                "window_len": draw(st.sampled_from((0.5, 1.0))),
                "window_overlap": 0.66,
            }
        else:
            params = {
                "max_bad_channels": draw(st.sampled_from((0.05, 0.1, 0.25, 0.5, 0, 1, 3))),
                "zthresholds": draw(st.sampled_from(([-3.5, 5.0], ["-inf", 7.0], [-3.0, "inf"]))),
                "window_len": draw(st.sampled_from((0.5, 1.0, 1.5, 2.0))),
                "window_overlap": draw(st.sampled_from((0.0, 0.5, 0.66, 0.9))),
            }
            if draw(st.booleans()):
                spec = _with_prior(spec, "sample" if draw(st.booleans()) else "sample-short")
        return case_cls("clean_windows", spec, freeze_params(params))

    @st.composite
    def channels(draw):
        user = _user_draw(draw)
        rec = _pick_continuous(draw, continuous, user)
        seconds = 20.0 if user else draw(st.sampled_from((15.0, 20.0, 30.0)))
        spec = _draw_continuous(
            draw,
            spec_cls,
            rec,
            user=user,
            events="keep",
            rates=_EDGE_RATES,
            seconds=seconds,
            max_length=32768,
        )
        if user and draw(st.booleans()):
            spec = _with_artifact(spec, "copy", 1, 0, spec.length)
        if user:
            params = {
                "corr_threshold": 0.8,
                "noise_threshold": draw(st.sampled_from((4.0, 4.0, 5.0))),
                "window_len": 5.0,
                "max_broken_time": draw(st.sampled_from((0.4, 0.5))),
                "num_samples": 50,
                "subset_size": 0.25,
            }
        else:
            params = {
                "corr_threshold": draw(st.sampled_from((0.7, 0.8, 0.85))),
                "noise_threshold": draw(st.sampled_from((3.0, 4.0, 6.0))),
                "window_len": draw(st.sampled_from((2.0, 5.0))),
                "max_broken_time": draw(st.sampled_from((0.2, 0.4, 0.5, 2.0))),
                "num_samples": draw(st.sampled_from((20, 50, 51))),
                "subset_size": 0.25,
            }
            spec = _with_artifact(spec, draw(st.sampled_from(("copy", "hf", "flat"))), 1, 0, spec.length)
        return case_cls("clean_channels", spec, freeze_params(params))

    @st.composite
    def flatlines(draw):
        user = _user_draw(draw)
        rec = _pick_continuous(draw, continuous, user)
        duration = 5.0 if user else draw(st.sampled_from((5.0, 1.0, 0.25)))
        spec = _draw_continuous(draw, spec_cls, rec, user=user, events="keep", rates=_EDGE_RATES, max_length=2048)
        rate = spec.srate_hz or rec.srate / spec.decimate
        run = int(duration * rate)
        if user:
            jitter = 20.0
            if draw(st.booleans()):
                spec = _with_artifact(spec, "flat", 0, 0, run + 2)
        else:
            jitter = draw(st.sampled_from((20.0, 10.0, 40.0)))
            kind = draw(st.sampled_from(("flat", "jitter10", "jitter20", "jitter40", "allflat")))
            extra = 2 if draw(st.booleans()) else 1
            spec = _with_artifact(spec, kind, 0, 0, max(2, run + extra))
            if draw(st.booleans()):
                spec = _with_prior(spec, "channel")
        params = {"max_flatline_duration": duration, "max_allowed_jitter": jitter}
        return case_cls("clean_flatlines", spec, freeze_params(params))

    @st.composite
    def pipeline(draw):
        user = _user_draw(draw)
        rec = continuous[0]
        spec = _draw_continuous(
            draw,
            spec_cls,
            rec,
            user=user,
            events="keep",
            min_length=1024,
            max_length=1536,
        )
        if not user:
            spec = _with_boundaries(draw, spec, nested=draw(st.booleans()))
        steps = _tutorial_chain(draw) if user else _edge_chain(draw, rec)
        return case_cls("pipeline", spec, freeze_params({"profile": steps[0]}), pipeline=tuple(steps[1]))

    return {
        "eeg_eegrej": eegrej(),
        "pop_select": select(),
        "pop_epoch": epoch(),
        "pop_rmbase": rmbase(),
        "clean_windows": windows(),
        "clean_channels": channels(),
        "clean_flatlines": flatlines(),
        "pipeline": pipeline(),
    }


def extra_sanity(spec_cls, case_cls):
    """One known-good case per new target, on the tutorial recording."""
    rec = recording("eeglab_data")
    base = spec_cls(
        tuple(range(rec.nbchan)),
        0,
        1024,
        1,
        -1,
        0,
        0,
        source=rec.name,
        events="keep",
    )
    epochs = recording("eeglab_data_epochs_ica")
    epoch_spec = spec_cls(
        tuple(range(epochs.nbchan)),
        0,
        8,
        1,
        -1,
        0,
        0,
        source=epochs.name,
        events="keep",
    )
    chain = (
        ("pop_eegfiltnew", _filter_params(1.0)),
        ("pop_reref", freeze_params({"ref": [], "keepref": "off", "exclude": []})),
        ("pop_epoch", freeze_params({"types": ["square"], "lim": [-0.2, 0.8]})),
        ("pop_rmbase", freeze_params({"timerange": [-200.0, 0.0], "pointrange": [], "chanlist": []})),
    )
    return [
        case_cls("eeg_eegrej", base, freeze_params({"regions": [[101, 200]], "eegplot": False}), sanity=True),
        case_cls("pop_select", base, freeze_params({"time": [[0.0, 1.0]]}), sanity=True),
        case_cls(
            "pop_epoch",
            base,
            freeze_params({"types": ["square"], "lim": [-0.2, 0.8]}),
            sanity=True,
        ),
        case_cls(
            "pop_rmbase",
            epoch_spec,
            freeze_params({"timerange": [-200.0, 0.0], "pointrange": [], "chanlist": []}),
            sanity=True,
        ),
        case_cls("clean_windows", base, freeze_params(_window_defaults()), sanity=True),
        case_cls("clean_channels", base, freeze_params(_channel_defaults()), sanity=True),
        case_cls("clean_flatlines", base, freeze_params(_flatline_defaults()), sanity=True),
        case_cls("pipeline", base, freeze_params({"profile": "tutorial-epoch"}), pipeline=chain, sanity=True),
    ]


def _user_draw(draw) -> bool:
    return draw(st.sampled_from(_USER))


def _pick_continuous(draw, recordings, user: bool) -> _Recording:
    del user
    return draw(st.sampled_from(recordings))


def _duration_samples(rec, seconds: float, cap: int) -> int:
    return min(rec.pnts, cap, max(256, int(round(seconds * rec.srate))))


def _draw_continuous(draw, spec_cls, rec, *, user, events, rates=(0.0,), min_length=256, max_length=None, seconds=None):
    if seconds is not None:
        length = _duration_samples(rec, seconds, 32768 if max_length is None else max_length)
    else:
        choices = _USER_LENGTHS if user else _EDGE_LENGTHS
        if max_length is not None:
            choices = tuple(length for length in choices if length <= max_length)
        choices = tuple(length for length in choices if min_length <= length <= rec.pnts) or (
            min(rec.pnts, max(min_length, 256)),
        )
        length = min(rec.pnts, draw(st.sampled_from(choices)))
    decimate = 1 if user else draw(st.sampled_from((1, 1, 2, 4)))
    while length * decimate > rec.pnts and decimate > 1:
        decimate //= 2
    span = length * decimate
    start_max = rec.pnts - span
    if user:
        start = draw(st.integers(0, min(start_max, max(rec.pnts // 5, 1)))) if start_max else 0
        channels = tuple(range(rec.nbchan))
        nan = (-1, 0, 0)
        srate_hz = 0.0
        drop = ()
    else:
        start = draw(st.integers(0, start_max)) if start_max else 0
        count = draw(st.integers(max(4, rec.nbchan // 4), rec.nbchan))
        offset = draw(st.integers(0, rec.nbchan - count))
        channels = tuple(range(offset, offset + count))
        if draw(st.sampled_from((False, False, True))):
            nan_length = draw(st.integers(1, min(4, length)))
            nan = (draw(st.integers(0, count - 1)), draw(st.integers(0, length - nan_length)), nan_length)
        else:
            nan = (-1, 0, 0)
        srate_hz = draw(st.sampled_from(rates))
        drop_count = draw(st.integers(0, min(4, count)))
        drop = tuple(sorted({draw(st.integers(0, count - 1)) for _ in range(drop_count)}))
    return spec_cls(
        channels,
        start,
        length,
        decimate,
        nan[0],
        nan[1],
        nan[2],
        source=rec.name,
        events=events,
        srate_hz=float(srate_hz),
        drop_xyz=drop,
    )


def _draw_trials(draw, spec_cls, rec, *, user):
    choices = (8, 16, 20, 40) if user else (1, 2, 5, rec.trials)
    count = min(rec.trials, draw(st.sampled_from(choices)))
    start = 0 if user else draw(st.integers(0, rec.trials - count))
    return spec_cls(
        tuple(range(rec.nbchan if user else max(1, rec.nbchan // 2))),
        start,
        count,
        1,
        -1,
        0,
        0,
        source=rec.name,
        events="keep",
    )


def _draw_regions(draw, spec, rec, user: bool):
    length = spec.length
    if user:
        count = draw(st.sampled_from((1, 1, 2)))
        regions = []
        cursor = 32
        for _ in range(count):
            if cursor >= length - 16:
                break
            width = draw(st.sampled_from((16, 32, 64, 128)))
            stop = min(length, cursor + width)
            regions.append([cursor, stop])
            cursor = stop + draw(st.integers(8, 64))
        return regions or [[32, min(length, 96)]]
    count = draw(st.integers(1, 6))
    regions = []
    for _ in range(count):
        left = _quarter(draw, -2, length + 4)
        right = _quarter(draw, -2, length + 4)
        regions.append([left, right])
    if draw(st.booleans()) and regions:
        regions[0] = [regions[0][1], regions[0][0]]
    latencies = _crop_latencies(rec, spec)
    if latencies and regions:
        regions[0][1] = round(latencies[0])
    if draw(st.booleans()) and length > 4:
        regions.append([1, length])
    return regions


def _user_select(draw, spec, rec):
    mode = draw(st.sampled_from(("time", "time", "channel", "nochannel")))
    if mode == "time":
        stop = min(2.0, max(0.5, (spec.length - 1) / rec.srate))
        return {"time": [[0.0, round(stop, 3)]]}
    label = rec.labels[0]
    if mode == "channel":
        return {"channel": list(rec.labels[: max(1, len(rec.labels) // 2)])}
    return {"nochannel": [label]}


def _edge_select(draw, spec_cls, continuous, epoched):
    kind = draw(st.sampled_from(("point", "nopoint", "notime", "trial", "dupes")))
    if kind == "trial":
        spec = _draw_trials(draw, spec_cls, epoched, user=False)
        trials = list(range(1, spec.length + 1))
        draw_order = trials[::-1]
        if draw(st.booleans()):
            draw_order = draw_order + draw_order[:1]
        return spec, {"trial": draw_order, "sorttrial": draw(st.sampled_from(("on", "off")))}
    rec = _pick_continuous(draw, continuous, False)
    spec = _draw_continuous(draw, spec_cls, rec, user=False, events="keep+boundary")
    if kind == "point":
        return spec, {"point": [1, spec.length / 2, spec.length / 2 + 0.5, spec.length]}
    if kind == "nopoint":
        return spec, {"nopoint": [[1, min(8, spec.length)]]}
    if kind == "dupes":
        return spec, {"channel": [1, 1, rec.nbchan, 2]}
    spec = _draw_trials(draw, spec_cls, epoched, user=False)
    # pop_select.m rejects a notime span that does not touch either epoch edge.
    return spec, {"notime": [[-0.2, 0.2]]}


def _user_types(draw, rec):
    present = [kind for kind in rec.event_types if kind not in {"boundary", "None"}]
    if not present:
        return []
    mode = draw(st.sampled_from(("one", "one", "both", "all")))
    if mode == "all":
        return []
    if mode == "both" and len(present) > 1:
        return present[:2]
    return [draw(st.sampled_from(present))]


def _edge_types(draw, rec):
    return draw(st.sampled_from((["sq.*"], ["^r"], [], _user_types(draw, rec))))


def _edge_window(draw, srate):
    rate = srate or 128.0
    half = 0.5 / rate
    return draw(st.sampled_from(((-1.0, 2.0), (-0.2, 0.8), (0.0, 0.5), (half, half + 0.25), (-0.5, -0.1), (0.1, 0.4))))


def _edge_rmbase(draw, spec_cls, continuous, epoched):
    kind = draw(st.sampled_from(("points", "offset", "timerange", "nan", "error")))
    if kind in {"timerange", "error"}:
        spec = _draw_trials(draw, spec_cls, epoched, user=False)
        if kind == "error":
            return spec, {"timerange": [-1.0e6, 1.0e6], "pointrange": [], "chanlist": []}
        return spec, {"timerange": list(draw(st.sampled_from(_BASELINES_MS))), "pointrange": [], "chanlist": [1, 2]}
    rec = _pick_continuous(draw, continuous, False)
    spec = _draw_continuous(draw, spec_cls, rec, user=False, events="keep+boundary")
    if kind == "offset":
        return spec, {"timerange": [], "pointrange": [5, 6, 7, 8, 40], "chanlist": []}
    if kind == "nan":
        spec = _with_nan(spec)
        return spec, {"timerange": [], "pointrange": list(range(1, min(spec.length, 30) + 1)), "chanlist": []}
    return spec, {
        "timerange": [],
        "pointrange": [1, 2, 4, min(spec.length, 10)],
        "chanlist": [],
    }


def _tutorial_chain(draw):
    profile = draw(st.sampled_from(("tutorial-epoch", "bids-clean", "gui-clean", "ds003061-params")))
    if profile == "tutorial-epoch":
        steps = (
            ("pop_eegfiltnew", _filter_params(1.0)),
            ("pop_reref", freeze_params({"ref": [], "keepref": "off", "exclude": []})),
            ("pop_epoch", freeze_params({"types": ["square"], "lim": [-1.0, 2.0]})),
            ("pop_rmbase", freeze_params({"timerange": [-200.0, 0.0], "pointrange": [], "chanlist": []})),
        )
    elif profile == "bids-clean":
        steps = (
            ("pop_clean_rawdata", _clean_params(burst=5.0, rejection="off", highpass=(0.25, 0.75))),
            ("pop_reref", freeze_params({"ref": [], "keepref": "off", "exclude": []})),
            ("pop_epoch", freeze_params({"types": ["square"], "lim": [-1.0, 2.0]})),
            ("pop_select", freeze_params({"trial": [1, 2, 3], "sorttrial": "on"})),
        )
    elif profile == "gui-clean":
        steps = (
            ("pop_clean_rawdata", _clean_params(burst=20.0, rejection="on", highpass="off", window_tol=("-inf", 7.0))),
            ("pop_reref", freeze_params({"ref": [], "keepref": "off", "exclude": []})),
        )
    else:
        steps = (
            ("pop_clean_rawdata", _clean_params(burst=20.0, rejection="on", highpass=(0.25, 0.75))),
            ("pop_reref", freeze_params({"ref": [], "keepref": "off", "exclude": []})),
            ("pop_epoch", freeze_params({"types": ["square", "rt"], "lim": [-0.5, 1.0]})),
            ("pop_rmbase", freeze_params({"timerange": [-500.0, 0.0], "pointrange": [], "chanlist": []})),
        )
    return profile, steps


def _edge_chain(draw, rec):
    del rec
    steps = (
        ("pop_eegfiltnew", _filter_params(0.5)),
        ("pop_reref", freeze_params({"ref": [0], "keepref": "on", "exclude": []})),
        ("pop_epoch", freeze_params({"types": ["sq.*"], "lim": [-0.2, 0.8]})),
        ("pop_rmbase", freeze_params({"timerange": [], "pointrange": [1, 2, 3, 4], "chanlist": []})),
        ("pop_select", freeze_params({"notrial": [1], "sorttrial": "off"})),
    )
    return "edge", steps


def _filter_params(low: float) -> tuple:
    return freeze_params(
        {
            "locutoff": low,
            "hicutoff": None,
            "revfilt": False,
            "minphase": False,
            "usefftfilt": False,
        }
    )


def _clean_params(*, burst, rejection, highpass, window_tol=None) -> tuple:
    return freeze_params(
        {
            "FlatlineCriterion": 5.0,
            "Highpass": highpass,
            "ChannelCriterion": 0.8,
            "LineNoiseCriterion": 4.0,
            "BurstCriterion": burst,
            "WindowCriterion": 0.25,
            "BurstRejection": rejection,
            "WindowCriterionTolerances": window_tol if window_tol is not None else ["-inf", 7.0],
            "ChannelCriterionMaxBadTime": 0.5,
            "Distance": "euclidian",
        }
    )


def _window_defaults():
    return {"max_bad_channels": 0.2, "zthresholds": [-3.5, 5.0], "window_len": 1.0, "window_overlap": 0.66}


def _channel_defaults():
    return {
        "corr_threshold": 0.8,
        "noise_threshold": 4.0,
        "window_len": 5.0,
        "max_broken_time": 0.4,
        "num_samples": 50,
        "subset_size": 0.25,
    }


def _flatline_defaults():
    return {"max_flatline_duration": 5.0, "max_allowed_jitter": 20.0}


def _with_prior(spec, kind: str):
    return _replace_spec(spec, prior_mask=kind)


def _with_artifact(spec, kind: str, channel: int, start: int, length: int):
    return _replace_spec(
        spec,
        artifact=kind,
        artifact_channel=channel,
        artifact_start=start,
        artifact_length=length,
    )


def _with_nan(spec):
    return _replace_spec(spec, nan_channel=0, nan_start=0, nan_length=min(4, spec.length))


def _with_boundaries(draw, spec, *, nested: bool):
    length = spec.length
    latency = _quarter(draw, 8, max(8, length / 2))
    duration = float(draw(st.sampled_from((1.0, 4.0, 8.0))))
    bounds = [(latency, duration)]
    if nested:
        bounds.append((latency + 0.25, 0.5))
    return _replace_spec(spec, boundaries=tuple(bounds), events="keep+boundary")


def _replace_spec(spec, **changes):
    values = {
        "channels": spec.channels,
        "start": spec.start,
        "length": spec.length,
        "decimate": spec.decimate,
        "nan_channel": spec.nan_channel,
        "nan_start": spec.nan_start,
        "nan_length": spec.nan_length,
        "source": spec.source,
        "events": spec.events,
        "boundaries": spec.boundaries,
        "trials": spec.trials,
        "epoch_pnts": spec.epoch_pnts,
        "xmin": spec.xmin,
        "srate_hz": spec.srate_hz,
        "drop_xyz": spec.drop_xyz,
        "artifact": spec.artifact,
        "artifact_channel": spec.artifact_channel,
        "artifact_start": spec.artifact_start,
        "artifact_length": spec.artifact_length,
        "prior_mask": spec.prior_mask,
    }
    values.update(changes)
    return spec.__class__(**values)


def _quarter(draw, low: float, high: float) -> float:
    start_q = int(np.ceil(float(low) * 4.0))
    stop_q = int(np.floor(float(high) * 4.0))
    if stop_q < start_q:
        return float(low)
    return draw(st.integers(start_q, stop_q)) / 4.0


def _crop_latencies(rec, spec) -> list[float]:
    span = spec.length * spec.decimate
    shifted = []
    for latency in rec.event_latencies:
        if spec.start < latency <= spec.start + span:
            shifted.append((latency - spec.start - 1.0) / spec.decimate + 1.0)
    return shifted


def _crop_continuous(rec, eeg, spec):
    span = spec.length * spec.decimate
    sample_index = np.arange(spec.start, spec.start + span, spec.decimate)
    data = np.asarray(rec.eeg["data"])
    cropped = np.array(data[np.ix_(list(spec.channels), sample_index)], dtype=np.float64, copy=True)
    chosen = [_copy_channel(rec.eeg["chanlocs"][index]) for index in spec.channels]
    eeg["chanlocs"] = np.array(chosen, dtype=object)
    eeg["nbchan"] = int(cropped.shape[0])
    eeg["pnts"] = int(cropped.shape[1])
    eeg["trials"] = 1
    kept = []
    if spec.events != "none":
        for event in _event_dicts(rec.eeg):
            latency = event.get("latency")
            if latency is None:
                continue
            latency = float(latency)
            if spec.start < latency <= spec.start + span:
                shifted = copy.deepcopy(event)
                shifted["latency"] = (latency - spec.start - 1.0) / spec.decimate + 1.0
                kept.append(shifted)
    return cropped, kept


def _crop_trials(rec, eeg, spec):
    start = int(spec.start)
    count = int(spec.length)
    data = np.asarray(rec.eeg["data"])
    cropped = np.array(
        data[np.ix_(list(spec.channels), np.arange(rec.pnts), np.arange(start, start + count))],
        dtype=np.float64,
        copy=True,
    )
    chosen = [_copy_channel(rec.eeg["chanlocs"][index]) for index in spec.channels]
    eeg["chanlocs"] = np.array(chosen, dtype=object)
    eeg["nbchan"] = int(cropped.shape[0])
    eeg["pnts"] = int(rec.pnts)
    eeg["trials"] = count
    eeg["xmin"] = rec.xmin
    kept = []
    if spec.events != "none":
        for event in _event_dicts(rec.eeg):
            epoch = int(event.get("epoch", 1) or 1)
            if start < epoch <= start + count:
                shifted = copy.deepcopy(event)
                shifted["epoch"] = epoch - start
                if rec.global_latency and "latency" in shifted:
                    shifted["latency"] = float(shifted["latency"]) - start * rec.pnts
                kept.append(shifted)
    return cropped, kept


def _apply_artifact(data, spec, srate: float):
    if not spec.artifact:
        return
    if spec.artifact == "allflat":
        if data.ndim == 2:
            data[:] = data[:, :1]
        elif data.ndim == 3:
            data[:] = data[:, :1, :]
        return
    if spec.artifact_channel < 0:
        return
    channel = min(spec.artifact_channel, data.shape[0] - 1)
    start = max(0, spec.artifact_start)
    width = data.shape[1]
    stop = min(width, start + max(spec.artifact_length, 1))
    if spec.artifact == "copy" and data.ndim == 2 and data.shape[0] > 1:
        source = 0 if channel else 1
        count = max(stop - start, 1)
        stop = start + count
        noise = np.random.RandomState(0).normal(scale=1e-3, size=count)
        data[channel, start:stop] = data[source, start:stop] + noise
        return
    if data.ndim == 3:
        flat = data[channel].reshape(-1)
        _write_artifact(flat, spec.artifact, start, stop if stop > start else min(flat.size, start + 1), srate)
        return
    _write_artifact(data[channel], spec.artifact, start, stop, srate)


def _write_artifact(samples, kind: str, start: int, stop: int, srate: float):
    stop = min(samples.size, max(start + 1, stop))
    segment = samples[start:stop]
    if kind == "hf":
        time = np.arange(segment.size, dtype=np.float64) / float(srate)
        segment += 50.0 * np.sin(2.0 * np.pi * 40.0 * time)
        return
    if kind.startswith("jitter"):
        scale = float(kind.removeprefix("jitter"))
        base = float(samples[start])
        ulp = float(np.spacing(abs(base)))
        if ulp == 0.0:
            ulp = float(np.finfo(np.float64).eps)
        signs = np.random.RandomState(0).choice(np.array([-1.0, 1.0]), size=segment.size)
        segment[:] = base + signs * scale * ulp
        return
    segment[:] = float(samples[start])


def _apply_nan(data, spec):
    if spec.nan_channel < 0 or spec.nan_length <= 0 or data.ndim != 2:
        return
    stop = min(data.shape[1], spec.nan_start + spec.nan_length)
    data[spec.nan_channel, spec.nan_start : stop] = np.nan


def _boundary_events(bounds, count: int) -> list[dict]:
    events = []
    for index, (latency, duration) in enumerate(bounds, start=1):
        events.append(
            {
                "type": "boundary",
                "latency": float(latency),
                "duration": float(duration),
                "urevent": count + index - 1,
            }
        )
    return events


def _apply_rate_and_times(eeg, spec, rec):
    srate = float(spec.srate_hz) if spec.srate_hz else float(rec.srate) / float(spec.decimate)
    pnts = int(eeg["pnts"])
    xmin = float(eeg.get("xmin") or 0.0)
    eeg["srate"] = srate
    eeg["xmin"] = xmin
    eeg["xmax"] = xmin + (pnts - 1) / srate if pnts else xmin
    eeg["times"] = np.linspace(xmin * 1000.0, eeg["xmax"] * 1000.0, pnts) if pnts else np.array([])


def _drop_xyz(eeg, channels):
    if not channels:
        return
    for index in channels:
        if index < 0 or index >= len(eeg["chanlocs"]):
            continue
        channel = eeg["chanlocs"][index]
        if isinstance(channel, np.ndarray):
            channel = channel.item()
            eeg["chanlocs"][index] = channel
        for key in ("X", "Y", "Z"):
            channel[key] = np.array([])


def _apply_prior_mask(eeg, kind: str):
    if not kind:
        return
    etc = eeg.setdefault("etc", {})
    if kind == "channel":
        etc["clean_channel_mask"] = np.ones(int(eeg["nbchan"]), dtype=bool)
        return
    pnts = int(eeg["pnts"])
    mask = np.ones(pnts, dtype=bool)
    if kind == "sample":
        mask[: min(8, pnts)] = False
        etc["clean_sample_mask"] = mask
    elif kind == "sample-short":
        etc["clean_sample_mask"] = mask[: max(1, pnts // 2)]


def _event_dicts(eeg) -> list[dict]:
    raw = eeg.get("event")
    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        raw = list(raw.ravel())
    events = []
    for event in raw:
        item = event.item() if isinstance(event, np.ndarray) else event
        if isinstance(item, dict):
            events.append(item)
    return events


def _labels_of(eeg) -> tuple[str, ...]:
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


def _copy_channel(channel):
    return copy.deepcopy(channel.item() if isinstance(channel, np.ndarray) else channel)


def _regions_array(params) -> np.ndarray:
    regions = np.asarray(params["regions"], dtype=float)
    if regions.size == 0:
        return np.zeros((0, 2))
    if regions.ndim == 1:
        regions = regions.reshape(1, -1)
    if params.get("eegplot"):
        pad = np.zeros((regions.shape[0], 2))
        regions = np.concatenate([pad, regions], axis=1)
    return regions


def _select_kwargs(params) -> dict:
    skip = {"kind"}
    return {key: value for key, value in params.items() if key not in skip and value not in ([], None)}


def _epoch_types(params):
    types = params.get("types") or []
    if isinstance(types, str):
        return types
    return list(types)


def _matlab_types(params):
    types = _epoch_types(params)
    if isinstance(types, str):
        return types
    if not types:
        return np.array([], dtype=float)
    return np.array(types, dtype=object).reshape(1, -1)


def _epoch_kwargs(params) -> dict:
    kwargs = {}
    if params.get("valuelim"):
        kwargs["valuelim"] = list(params["valuelim"])
    if params.get("eventindices"):
        kwargs["eventindices"] = [int(index) for index in params["eventindices"]]
    return kwargs


def _rmbase_args(params):
    timerange = list(params.get("timerange") or [])
    pointrange = list(params.get("pointrange") or [])
    chanlist = list(params.get("chanlist") or [])
    return timerange or None, pointrange or None, chanlist or None


def _windows_kwargs(params) -> dict:
    return {
        "max_bad_channels": _number(params["max_bad_channels"]),
        "zthresholds": (_number(params["zthresholds"][0]), _number(params["zthresholds"][1])),
        "window_len": float(params["window_len"]),
        "window_overlap": float(params["window_overlap"]),
    }


def _channels_kwargs(params) -> dict:
    return {
        "corr_threshold": float(params["corr_threshold"]),
        "noise_threshold": float(params["noise_threshold"]),
        "window_len": float(params["window_len"]),
        "max_broken_time": float(params["max_broken_time"]),
        "num_samples": int(params["num_samples"]),
        "subset_size": float(params["subset_size"]),
    }


def _rawdata_kwargs(params) -> dict:
    cleaned = {}
    for key, value in params.items():
        if key == "WindowCriterionTolerances":
            cleaned[key] = (_number(value[0]), _number(value[1]))
        elif key == "Highpass" and not isinstance(value, str):
            cleaned[key] = (float(value[0]), float(value[1]))
        else:
            cleaned[key] = value
    return cleaned


def _number(value):
    if value in {"inf", "+inf"}:
        return float("inf")
    if value == "-inf":
        return float("-inf")
    return float(value)


def _matlab_value(value):
    if value is None:
        return np.array([])
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            return np.array([])
        if all(isinstance(item, str) for item in value):
            return np.array(value, dtype=object).reshape(1, -1)
        array = np.asarray(value, dtype=float)
        return array
    if isinstance(value, (int, float)):
        return float(value)
    return value
