"""NumPy-only numerical engine for EEGLAB-style ERP images (``erpimage.m``).

This module computes everything EEGLAB's ``erpimage`` returns, stage by stage
in EEGLAB's order: data preparation and baselines, sortvar renormalization and
alignment, trial sorting (sortvar, value, phase or amplitude), moving-average
smoothing with decimation, the ERP and its permutation threshold, wavelet
amplitude and inter-trial coherence with bootstrap significance, and the axis
limits EEGLAB draws with. Plotting lives in
:mod:`eegprep.functions.sigprocfunc.erpimage`.

The module is deliberately self-contained: it imports only NumPy and the
standard library (``ast``, ``dataclasses``, ``typing``) and nothing from
``eegprep``, so the browser distribution can include this single file.

Conventions differ from MATLAB only at the Python boundary: absent outputs are
``None``, row vectors are 1-D arrays, and ``sortidx`` holds 0-based indices
into the original input trials. ``outtrials`` keeps EEGLAB's 1-based trial-axis
coordinates because they are plot positions, not indices.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

DEFAULT_CYCLES = 3.0
DEFAULT_SDEV = 1.0 / 7.0
DEFAULT_BASELINE_END_MS = 0.0
YEXPAND = 1.3
MIN_AMP = 1e-6
COHSIG_REPS = 500
ERP_PERMUTATIONS = 500
MIN_ERPALPHA = 0.001
MAX_ERPALPHA = 0.1
AMPSORT_WINDOW_STEP_MS = 10.0
WAVELET_TRIAL_BLOCK = 256  # trials per FFT block, bounding memory for large datasets
PLOTAMPS_CAXIS_FRACTION = 0.8
NEAR_ZERO = 1e-22
EEGLAB_SIGN_BIAS_ISSUE = "https://github.com/sccn/eeglab/issues/976"

# Flag keys take an optional 'on'/'off' value; value keys always consume the next token.
_FLAG_KEYS = {
    "nosort": "nosort",
    "showwin": "showwin",
    "noplot": "noplot",
    "cbar": "cbar",
    "colorbar": "cbar",
    "allamps": "plotamps",
    "plotamps": "plotamps",
    "noxlabel": "noxlabel",
    "noxlabels": "noxlabel",
    "nox": "noxlabel",
}
_FLAG_KEYS_CI = {"erp": "erp", "rmerp": "rmerp", "erpstd": "erpstd"}
_VALUE_KEYS = {
    "caxis": "caxis",
    "timestretch": "timewarp",
    "timewarp": "timewarp",
    "coher": "coher",
    "topo": "topo",
    "topoplot": "topo",
    "spec": "spec",
    "spectrum": "spec",
    "Specaxis": "specaxis",
    "specaxis": "specaxis",
    "SpecAxis": "specaxis",
    "renorm": "renorm",
    "NoShow": "noshow",
    "align": "align",
    "sortvarpercent": "sortvarpercent",
    "limits": "limits",
    "auxvar": "auxvar",
    "vert": "vert",
    "verttimes": "vert",
    "horz": "horz",
    "horiz": "horz",
    "horizontal": "horz",
    "signif": "signif",
    "signifs": "signif",
    "sig": "signif",
    "sigs": "signif",
    "allcohers": "allcohers",
    "phase": "phasesort",
    "phasesort": "phasesort",
    "sortwin": "sortwin",
    "ampsort": "ampsort",
    "valsort": "valsort",
    "plotmode": "plotmode",
    "title": "title",
    "cycles": "cycles",
}
_VALUE_KEYS_CI = {
    "yerplabel": "yerplabel",
    "srate": "srate",
    "erpalpha": "erpalpha",
    "avg_type": "avg_type",
    "replace_ties": "replace_ties",
    "sortvar_limits": "sortvar_limits",
    "baseline": "baseline",
    "baselinedb": "baselinedb",
    "filt": "filt",
    "erp_vltg_ticks": "erp_vltg_ticks",
    "img_trialax_label": "img_trialax_label",
    "img_trialax_ticks": "img_trialax_ticks",
    "cbar_title": "cbar_title",
}
_BARE_KEYS_CI = {"erp_grid": "erp_grid"}
_KWARG_NAMES = (
    set(_FLAG_KEYS.values())
    | set(_FLAG_KEYS_CI.values())
    | set(_VALUE_KEYS.values())
    | set(_VALUE_KEYS_CI.values())
    | set(_BARE_KEYS_CI.values())
)


@dataclass(frozen=True, eq=False)
class ErpimageOptions:
    """Normalized EEGLAB ``erpimage`` options (engine and plotting)."""

    title: str | None = None
    align: float | None = None
    timewarp: tuple[np.ndarray, np.ndarray | None, tuple[str, ...]] | None = None
    renorm: str = "no"
    noplot: bool = False
    noshow: bool = False
    nosort: bool = False
    replace_ties: bool = False
    valsort: np.ndarray | None = None
    phasesort: np.ndarray | None = None
    ampsort: np.ndarray | None = None
    sortwin: np.ndarray | None = None
    showwin: bool = False
    plotamps: bool = False
    limits: np.ndarray | None = None
    sortvar_limits: np.ndarray | None = None
    signif: np.ndarray | None = None
    caxis: np.ndarray | None = None
    erp: int = 0
    erpalpha: float | None = None
    erpstd: bool = False
    erp_grid: bool = False
    rmerp: bool = False
    coher: np.ndarray | None = None
    srate: float | None = None
    cycles: float = DEFAULT_CYCLES
    cbar: bool = False
    cbar_title: str | None = None
    topo: tuple[Any, ...] | None = None
    spec: np.ndarray | None = None
    specaxis: str = "log"
    horz: np.ndarray | None = None
    vert: np.ndarray | None = None
    auxvar: np.ndarray | None = None
    auxcolors: tuple[str, ...] | None = None
    sortvarpercent: np.ndarray | None = None
    noxlabel: bool = False
    yerplabel: str = "ERP"
    avg_type: str = "boxcar"
    img_trialax_label: str | None = "Trials"
    img_trialax_ticks: np.ndarray | None = None
    baseline: np.ndarray | None = None
    baselinedb: np.ndarray | None = None
    filt: np.ndarray | None = None
    erp_vltg_ticks: np.ndarray | None = None
    plotmode: str = "normal"


@dataclass(frozen=True, eq=False)
class ErpimageComputation:
    """EEGLAB ``erpimage`` outputs plus the values its plots are drawn from.

    The first fields mirror EEGLAB's outputs in order (``axhndls`` excepted):
    ``outdata`` is (frames, n_out), ``outvar`` is (n_out,) or
    ``(outsort, percentiles)`` with ``sortvarpercent``, ``outtrials`` (n_out,),
    ``limits`` (10,), ``erp`` (ptiles, frames), ``amps``/``cohers`` (frames,),
    ``cohsig`` scalar, ``ampsig`` (2,), ``outamps`` (frames, trials),
    ``phsangls``/``phsamp`` (trials,), ``sortidx`` (trials,) 0-based indices
    into the input trials, ``erpsig`` (frames,) or (2, frames) when plotted.
    """

    outdata: np.ndarray
    outvar: Any
    outtrials: np.ndarray
    limits: np.ndarray
    erp: np.ndarray | None
    amps: np.ndarray | None
    cohers: np.ndarray | None
    cohsig: float | None
    ampsig: np.ndarray | None
    outamps: np.ndarray | None
    phsangls: np.ndarray | None
    phsamp: np.ndarray | None
    sortidx: np.ndarray
    erpsig: np.ndarray | None
    times: np.ndarray
    srate: float
    timelimits: np.ndarray
    clim: tuple[float, float]
    sort_mode: str
    times_passed: bool
    show_sortvar: bool
    aligntime: float
    urdata: np.ndarray
    erp_data: np.ndarray
    erp_std: np.ndarray | None
    erp_limits: tuple[float, float]
    amp_limits: tuple[float, float]
    coh_limits: tuple[float, float]
    baseamp: float
    coherfreq: float
    cohsig_flag: bool
    verttimes: np.ndarray | None
    vert_colors: tuple[str, ...] | None
    vert_smoothed: tuple[np.ndarray, ...]
    auxvar: np.ndarray | None
    auxcolors: tuple[str, ...] | None
    percentiles: tuple[np.ndarray, ...]
    winlocs: np.ndarray | None
    trial_limits: tuple[float, float] | None


def parse_erpimage_options(tokens: tuple[Any, ...] | list[Any] = (), **kwargs: Any) -> ErpimageOptions:
    """Parse EEGLAB ``'key', value`` tokens and Python keyword options.

    Tokens follow EEGLAB's grammar: flag keys (``'erp'``, ``'cbar'``,
    ``'nosort'``, ...) take an optional ``'on'``/``'off'`` value, value keys
    always consume the next token. Keyword arguments use the same names in
    lower case (``noshow`` for ``'NoShow'``) and accept booleans for flags.
    Keyword options are applied after the tokens.
    """
    values: dict[str, Any] = {}
    index = 0
    tokens = list(tokens)
    while index < len(tokens):
        key = tokens[index]
        if not isinstance(key, str):
            shape = np.shape(key)
            raise ValueError(f"erpimage(): unknown arg {index + 1}, size{shape}")
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        has_following = index + 1 < len(tokens)
        lower = key.lower()
        if key in _FLAG_KEYS or lower in _FLAG_KEYS_CI:
            name = _FLAG_KEYS.get(key) or _FLAG_KEYS_CI[lower]
            consumed, value = _flag_value(name, following) if has_following else (False, True)
            _set_option(values, name, value)
            index += 2 if consumed else 1
            continue
        if lower in _BARE_KEYS_CI:
            values[_BARE_KEYS_CI[lower]] = True
            index += 1
            continue
        name = _VALUE_KEYS.get(key) or _VALUE_KEYS_CI.get(lower)
        if name is None:
            raise ValueError(f"erpimage(): unknown arg {key}")
        if not has_following:
            raise ValueError(f"erpimage(): option '{key}' needs a value")
        _set_option(values, name, following)
        index += 2
    for key, value in kwargs.items():
        name = key.lower()
        if name not in _KWARG_NAMES:
            raise ValueError(f"erpimage(): unknown option '{key}'")
        if name in {"nosort", "showwin", "noplot", "cbar", "plotamps", "noxlabel", "rmerp", "erpstd", "erp_grid"}:
            value = _on_off(value, key)
        _set_option(values, name, value)
    return ErpimageOptions(**values)


def _flag_value(name: str, following: Any) -> tuple[bool, Any]:
    if isinstance(following, str):
        if following.lower() == "on":
            return True, True
        if following.lower() == "off":
            return True, False
        if name == "erp" and following in {"1", "2", "3", "4"}:
            return True, int(following)
        return False, True
    if (
        name == "erp"
        and isinstance(following, (int, float, np.ndarray, list, tuple))
        and not isinstance(following, bool)
    ):
        first = np.asarray(following, dtype=float).ravel()
        if first.size and first[0] in (1.0, 2.0, 3.0, 4.0):
            return True, int(first[0])
    return False, True


def _on_off(value: Any, key: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"on", "off"}:
        return value.lower() == "on"
    raise ValueError(f"erpimage(): '{key}' must be True/False or 'on'/'off'")


def _vector(value: Any, name: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError) as error:
        raise ValueError(f"erpimage(): '{name}' must be numeric") from error


def _set_option(values: dict[str, Any], name: str, value: Any) -> None:  # noqa: C901
    if name == "erp":
        if isinstance(value, (bool, np.bool_)):
            values["erp"] = 1 if value else 0
        elif isinstance(value, str):
            ptiles = {"on": 1, "off": 0, "1": 1, "2": 2, "3": 3, "4": 4}.get(value.lower())
            if ptiles is None:
                raise ValueError("erpimage(): 'erp' must be 'on', 'off' or 1 to 4")
            values["erp"] = ptiles
        else:
            ptiles = int(_vector(value, "erp")[0])
            if ptiles not in (0, 1, 2, 3, 4):
                raise ValueError("erpimage(): 'erp' must be 'on', 'off' or 1 to 4")
            values["erp"] = ptiles
        return
    if name in {"nosort", "showwin", "noplot", "cbar", "plotamps", "noxlabel", "rmerp", "erpstd", "erp_grid"}:
        values[name] = bool(value)
        return
    if name == "caxis":
        caxis = np.asarray(value, dtype=float)
        if caxis.ndim > 1 and caxis.shape[0] != 1 or caxis.size not in (1, 2):
            raise ValueError("erpimage(): caxis arg must be a scalar or (1,2) vector.")
        values["caxis"] = caxis.ravel()
        return
    if name == "coher":
        coher = _vector(value, "coher")
        if coher.size < 1 or coher.size > 3:
            raise ValueError("erpimage(): coher arg must be size <= 3.")
        if coher.size == 3 and (coher[2] < 0 or coher[2] > 0.1):
            raise ValueError(f"erpimage(): alpha value {coher[2]:g} out of bounds.")
        values["coher"] = coher
        values["erp"] = max(values.get("erp", 0), 1)
        return
    if name == "timewarp":
        parts = list(value) if isinstance(value, (list, tuple)) else [value]
        events = np.atleast_2d(np.asarray(parts[0], dtype=float))
        warpms = _vector(parts[1], "timewarp") if len(parts) > 1 and np.size(parts[1]) else None
        colors = tuple(str(color) for color in parts[2]) if len(parts) > 2 and parts[2] is not None else ()
        values["timewarp"] = (events, warpms, colors)
        return
    if name == "topo":
        parts = tuple(value) if isinstance(value, (list, tuple)) else tuple(np.asarray(value, dtype=object).ravel())
        if len(parts) < 2:
            raise ValueError("erpimage(): topo arg must be a list of length 2 or 3.")
        values["topo"] = parts
        return
    if name == "spec":
        spec = _vector(value, "spec")
        if spec.size != 2:
            raise ValueError("erpimage(): 'spec' flag argument must be a numeric array of length 2.")
        values["spec"] = spec
        return
    if name == "specaxis":
        axis = str(value).lower()
        if axis not in {"lin", "log"}:
            raise ValueError("erpimage(): spectrum axis type must be 'lin' or 'log'.")
        values["specaxis"] = "linear" if axis == "lin" else "log"
        return
    if name == "noshow":
        values["noshow"] = (
            bool(value) if isinstance(value, (bool, np.bool_)) else str(value).lower() not in {"off", "no"}
        )
        return
    if name == "align":
        values["align"] = float(_vector(value, "align")[0])
        return
    if name == "limits":
        limits = np.asarray(value, dtype=float)
        if limits.ndim > 1 and limits.shape[0] != 1 or not 2 <= limits.size <= 9:
            raise ValueError("erpimage(): limits arg must be a vector sized (1,2<->9).")
        limits = limits.ravel()
        if not np.isnan(limits[0]) and limits[1] <= limits[0]:
            raise ValueError("erpimage(): time limits out of order or out of range.")
        padded = np.full(9, np.nan)
        padded[: limits.size] = limits
        if not np.isnan(padded[3]) and padded[3] <= padded[2]:
            raise ValueError("erpimage(): erp limits args out of order.")
        if padded[5] <= padded[4]:
            raise ValueError("erpimage(): amp limits args out of order.")
        if padded[7] <= padded[6]:
            raise ValueError("erpimage(): coh limits args out of order.")
        values["limits"] = limits
        return
    if name in {"srate", "cycles"}:
        values[name] = float(_vector(value, name)[0])
        return
    if name == "auxvar":
        if isinstance(value, (list, tuple)) and len(value) == 2 and not np.isscalar(value[0]):
            values["auxvar"] = np.atleast_2d(np.asarray(value[0], dtype=float))
            values["auxcolors"] = tuple(str(color) for color in value[1])
        elif isinstance(value, (list, tuple)) and value and not np.isscalar(value[0]) and len(value) != 2:
            raise ValueError("erpimage(): auxvars argument must be a matrix or length-2 cell array.")
        else:
            values["auxvar"] = np.atleast_2d(np.asarray(value, dtype=float))
        return
    if name in {"vert", "horz"}:
        array = np.asarray(value, dtype=float)
        values[name] = np.atleast_2d(array) if array.ndim == 2 else array.ravel()
        return
    if name == "signif":
        signif = _vector(value, "signif")
        if signif.size != 3:
            raise ValueError("erpimage(): signif arg must have 3 values")
        values["signif"] = signif
        return
    if name == "allcohers":
        raise ValueError("erpimage(): 'allcohers' is deprecated and not functional in EEGLAB")
    if name == "phasesort":
        phargs = _vector(value, "phasesort")
        if phargs.size > 5:
            raise ValueError("erpimage(): Too many arguments for keyword 'phasesort'")
        if phargs.size < 3:
            raise ValueError("erpimage(): 'phasesort' needs at least [ms_center prct freq]")
        if phargs[2] < 0:
            raise ValueError("erpimage(): Invalid negative frequency argument for keyword 'phasesort'")
        if phargs.size >= 4 and phargs[3] < 0:
            raise ValueError("erpimage(): Invalid negative argument for keyword 'phasesort'")
        if phargs[1] >= 100 or phargs[1] < -100:
            raise ValueError("erpimage(): %-argument for keyword 'phasesort' must be (-100;100)")
        if phargs.size >= 4 and phargs[2] > phargs[3]:
            raise ValueError("erpimage(): Phase sorting frequency range must be increasing.")
        values["phasesort"] = phargs
        return
    if name == "ampsort":
        ampargs = _vector(value, "ampsort")
        if ampargs.size > 4:
            raise ValueError("erpimage(): Too many arguments for keyword 'ampsort'")
        if ampargs.size < 3:
            raise ValueError("erpimage(): 'ampsort' needs at least [center_ms prcnt freq]")
        if ampargs.size >= 4 and ampargs[3] < 0:
            raise ValueError("erpimage(): Invalid negative argument for keyword 'ampsort'")
        if ampargs[1] >= 100 or ampargs[1] < -100:
            raise ValueError("erpimage(): percentile argument for keyword 'ampsort' must be (-100;100)")
        if ampargs.size == 4 and abs(ampargs[2]) > abs(ampargs[3]):
            raise ValueError("erpimage(): Amplitude sorting frequency range must be increasing.")
        values["ampsort"] = ampargs
        return
    if name == "sortwin":
        sortwin = _vector(value, "sortwin")
        if sortwin.size > 2:
            raise ValueError("erpimage(): Too many arguments for keyword 'sortwin'")
        if sortwin.size > 1 and sortwin[0] > sortwin[1]:
            raise ValueError("erpimage(): Value sorting time range must be increasing.")
        values["sortwin"] = sortwin
        return
    if name == "valsort":
        valargs = _vector(value, "valsort")
        if valargs.size > 3 or valargs.size == 0:
            raise ValueError("erpimage(): Too many arguments for keyword 'valsort'")
        if valargs.size > 1 and valargs[0] > valargs[1]:
            raise ValueError("erpimage(): Value sorting time range must be increasing.")
        if valargs.size == 3 and valargs[2] == 0:
            raise ValueError("erpimage(): Value sorting direction must be +1 or -1.")
        values["valsort"] = valargs
        return
    if name == "erpalpha":
        alpha = float(_vector(value, "erpalpha")[0])
        if alpha < MIN_ERPALPHA or alpha > MAX_ERPALPHA:
            raise ValueError(f"erpimage(): erpalpha value is out of bounds [{MIN_ERPALPHA:g}, {MAX_ERPALPHA:g}]")
        values["erpalpha"] = alpha
        return
    if name == "avg_type":
        avg_type = str(value).lower()
        if avg_type not in {"gaussian", "boxcar"}:
            raise ValueError("erpimage(): Invalid value for optional argument 'avg_type'.")
        values["avg_type"] = avg_type
        return
    if name == "replace_ties":
        if isinstance(value, (bool, np.bool_)):
            values["replace_ties"] = bool(value)
        elif isinstance(value, str) and value.lower() in {"on", "off"}:
            values["replace_ties"] = value.lower() == "on"
        else:
            raise ValueError("erpimage(): Argument 'replace_ties' needs to be followed by the string 'on' or 'off'.")
        return
    if name == "sortvar_limits":
        if isinstance(value, str) or np.size(value) != 2:
            raise ValueError("erpimage(): Argument 'sortvar_limits' needs to be followed by a two element vector.")
        values["sortvar_limits"] = _vector(value, name)
        return
    if name == "baselinedb":
        baselinedb = _vector(value, name)
        if baselinedb.size > 2:
            raise ValueError("'baselinedb' need to be a 2 argument vector")
        values["baselinedb"] = baselinedb
        return
    if name in {"baseline", "filt", "erp_vltg_ticks", "img_trialax_ticks", "sortvarpercent"}:
        values[name] = _vector(value, name)
        return
    if name in {"renorm", "yerplabel", "plotmode", "title", "img_trialax_label", "cbar_title"}:
        values[name] = "" if value is None else str(value)
        return
    raise ValueError(f"erpimage(): unknown option '{name}'")  # pragma: no cover - table and setter agree


@dataclass(frozen=True, eq=False)
class _Trials:
    """Trials in their current order, with every per-trial array that follows them."""

    values: np.ndarray  # (frames, n)
    sortvar: np.ndarray  # (n,)
    index: np.ndarray  # input-trial index of each column
    auxvar: np.ndarray | None  # (k, n)
    vert: np.ndarray | None  # (n, k) per-trial vertical-line latencies
    marks: np.ndarray | None  # (n, events + 2) timewarp event frames

    @property
    def count(self) -> int:
        return self.sortvar.size

    def take(self, order: np.ndarray) -> _Trials:
        return _Trials(
            self.values[:, order],
            self.sortvar[order],
            self.index[order],
            None if self.auxvar is None else self.auxvar[:, order],
            None if self.vert is None else self.vert[order],
            None if self.marks is None else self.marks[order],
        )


@dataclass(frozen=True, eq=False)
class _Sorted:
    """Sorted trials and what the sorting step returns besides them."""

    trials: _Trials
    mode: str
    freq: np.ndarray  # phase/amplitude analysis frequencies, [0] for other modes
    rejection: float | None  # EEGLAB's converted rejection fraction, reused for decimation
    phsangls: np.ndarray | None = None
    phsamp: np.ndarray | None = None
    outamps: np.ndarray | None = None
    winlocs: np.ndarray | None = None


@dataclass(frozen=True, eq=False)
class _Image:
    outdata: np.ndarray
    outtrials: np.ndarray
    outsort: np.ndarray
    auxvar: np.ndarray | None
    percentiles: tuple[np.ndarray, ...]
    clim: tuple[float, float]


@dataclass(frozen=True, eq=False)
class _Spectral:
    amps: np.ndarray | None = None
    cohers: np.ndarray | None = None
    cohsig: float | None = None
    ampsig: np.ndarray | None = None
    baseamp: float = np.nan


def compute_erpimage(
    data: Any,
    sortvar: Any = None,
    times: Any = None,
    avewidth: Any = 1,
    decimate: Any = 0,
    options: ErpimageOptions | None = None,
    *,
    rng: Any = None,
    plotted: bool = True,
    trial_filter: Callable[[np.ndarray, np.ndarray, float], np.ndarray] | None = None,
) -> ErpimageComputation:
    """Compute EEGLAB ``erpimage`` outputs without plotting.

    Args:
        data: (frames, trials) array, or a (1, frames*trials) concatenation
            in column-major order, as EEGLAB accepts.
        sortvar: one sorting value per trial; ``None`` or empty means trial
            order with no sortvar overplot. Trials with NaN values are removed.
        times: latencies in ms, or ``[start_ms, frames, srate]``; ``None``
            means 0..frames-1 and an empty array means 1..frames, as in EEGLAB.
        avewidth: boxcar width in trials (an integer), or the Gaussian
            standard deviation with ``avg_type='gaussian'``.
        decimate: window advance in trials; values above sqrt(trials) give
            that many output rows; negative values subtract the window width.
        options: options from :func:`parse_erpimage_options`.
        rng: ``numpy.random.Generator`` or ``RandomState`` used for the
            coherence bootstrap and the ``erpalpha`` permutations.
            ``RandomState(seed)`` reproduces MATLAB ``rng(seed,'twister')``
            uniform draws.
        plotted: mirror EEGLAB's plotted call (axis limits in ``limits`` and a
            two-row ``erpsig``) rather than its no-plot call.
        trial_filter: function ``(data, filt, srate) -> data`` that applies the
            ``'filt'`` Butterworth filter; required only with ``filt``.
    """
    opts = options if options is not None else ErpimageOptions()
    rng = np.random.default_rng() if rng is None else rng
    values, times_passed = _data_matrix(data, times)
    sortvar_values, show_sortvar = _sort_values(sortvar, values.shape[1], opts.noplot)
    ntrials = sortvar_values.size
    frames = values.size // ntrials
    if frames * ntrials != values.size:
        raise ValueError("erpimage(): length of sortvar doesn't divide number of data elements")
    time_values, srate = _time_axis(times, frames)
    srate = opts.srate if opts.srate is not None else srate
    decfactor = _first_value(decimate, 0.0)
    decfactor = 1.0 if decfactor == 0 else min(decfactor, float(ntrials))

    _check_option_times(opts, time_values)
    if opts.plotamps and (opts.coher is None or opts.coher.size != 3):
        raise ValueError("erpimage(): allamps and allcohers flags require coher freq, srate, and cohsig.")
    if not srate > 0:
        raise ValueError("erpimage(): Data srate must be specified and > 0.")
    auxvar = None if opts.auxvar is None else _trial_matrix(opts.auxvar, ntrials, "auxvar")
    verttimes = None if opts.vert is None else _vert_times(opts.vert, ntrials)
    marks, ref, vert_colors = _timewarp_setup(opts, auxvar is None, time_values, srate)
    if vert_colors is not None:
        verttimes = ((ref[[bool(color) for color in vert_colors]] - 1) / srate + time_values[0] / 1000) * 1000
        vert_colors = tuple(color for color in vert_colors if color)
    auxcolors = vert_colors if vert_colors is not None else opts.auxcolors
    phargs, ampargs = _check_sort_frequencies(opts, srate, frames)
    coherfreq = _check_coherence(opts.coher, opts.cycles, srate, frames)
    limits, timelimits, aligntime = _limit_settings(opts, time_values)

    values[np.isnan(values)] = 0
    if values.shape[1] != ntrials:
        values = values.reshape(-1, order="F").reshape(frames, ntrials, order="F")
    per_trial_vert = verttimes if verttimes is not None and verttimes.ndim == 2 else None
    trials = _Trials(values, sortvar_values, np.arange(ntrials), auxvar, per_trial_vert, marks)
    if np.any(np.isnan(sortvar_values)):
        trials = trials.take(np.flatnonzero(~np.isnan(sortvar_values)))
        if trials.count <= 1:
            raise ValueError("erpimage(): Too few trials")
    avewidth, weights = _smoothing_window(_first_value(avewidth, 1.0), opts.avg_type, trials.count)
    trials, aligntime = _preprocess(trials, opts, time_values, srate, aligntime, trial_filter)
    ordered = _sort_trials(trials, opts, phargs, ampargs, time_values, srate)
    trials = ordered.trials
    decfactor = _adjust_decimation(decfactor, trials.count, avewidth, ordered.rejection)
    smoothing = (avewidth, decfactor, weights, ordered.mode == "phase", opts.sortvarpercent)

    urdata = trials.values.copy()
    coherfreq = _analysis_frequency(coherfreq, ordered.freq, urdata, frames, srate)
    alpha = float(opts.coher[2]) if opts.coher is not None and opts.coher.size == 3 else 0.0
    spectral = _Spectral(baseamp=limits[8])
    if opts.plotamps:
        image, spectral = _amplitude_image(
            trials, opts, coherfreq, alpha, time_values, srate, rng, limits[8], smoothing
        )
        erp_data = urdata
    else:
        erp_data = urdata if trials.marks is None else _apply_timewarp(urdata, trials.marks, ref)
        image = _potential_image(trials, erp_data, opts.caxis, smoothing)
    trial_limits = None
    if opts.sortvar_limits is not None and ordered.mode in {"sortvar", "nosort"}:
        trial_limits, inside = _sortvar_trial_limits(
            opts.sortvar_limits, image.outsort, image.outtrials, trials.sortvar
        )
        urdata, erp_data = urdata[:, inside], erp_data[:, inside]

    erp, erpsig, erp_std, minerp, maxerp = _erp_stage(erp_data, urdata, opts, rng, plotted, limits[2], limits[3])
    minamp, maxamp, mincoh, maxcoh = limits[4:8]
    if not np.isnan(coherfreq):
        if ordered.freq[0] > 0:
            coherfreq = float(np.mean(ordered.freq))
        if not opts.plotamps:
            spectral = _coherence(urdata, opts, coherfreq, alpha, time_values, srate, rng, limits[8])
        if plotted:
            maxamp, minamp, maxcoh, mincoh = _amp_coh_limits(
                spectral.amps, spectral.cohers, spectral.ampsig, _alpha_given(opts), maxamp, minamp, maxcoh, mincoh
            )
    else:
        spectral = replace(spectral, amps=None, cohers=None)

    outvar: Any = image.outsort
    if opts.sortvarpercent is not None and image.percentiles:
        outvar = (image.outsort, image.percentiles)
    return ErpimageComputation(
        outdata=image.outdata,
        outvar=outvar,
        outtrials=np.asarray(image.outtrials, dtype=float),
        limits=np.array(
            [
                *(np.min(time_values), np.max(time_values)),
                *(minerp, maxerp, minamp, maxamp, mincoh, maxcoh),
                *(spectral.baseamp, coherfreq),
            ],
            dtype=float,
        ),
        erp=erp,
        amps=spectral.amps,
        cohers=spectral.cohers,
        cohsig=spectral.cohsig,
        ampsig=spectral.ampsig,
        outamps=image.outdata if opts.plotamps else ordered.outamps,
        phsangls=ordered.phsangls,
        phsamp=ordered.phsamp,
        sortidx=trials.index,
        erpsig=erpsig,
        times=time_values,
        srate=float(srate),
        timelimits=timelimits,
        clim=image.clim,
        sort_mode=ordered.mode,
        times_passed=times_passed,
        show_sortvar=show_sortvar,
        aligntime=float(aligntime),
        urdata=urdata,
        erp_data=erp_data,
        erp_std=erp_std,
        erp_limits=(float(minerp), float(maxerp)),
        amp_limits=(float(minamp), float(maxamp)),
        coh_limits=(float(mincoh), float(maxcoh)),
        baseamp=float(spectral.baseamp),
        coherfreq=float(coherfreq),
        cohsig_flag=_alpha_given(opts),
        verttimes=trials.vert if per_trial_vert is not None else verttimes,
        vert_colors=vert_colors,
        vert_smoothed=_smooth_vert(trials.vert, avewidth, decfactor, weights),
        auxvar=image.auxvar,
        auxcolors=auxcolors,
        percentiles=image.percentiles,
        winlocs=ordered.winlocs,
        trial_limits=trial_limits,
    )


def _alpha_given(options: ErpimageOptions) -> bool:
    """EEGLAB's ``Cohsigflag``: 'coher' was given a significance level."""
    return options.coher is not None and options.coher.size == 3


def _first_value(value: Any, default: float) -> float:
    return default if value is None or np.size(value) == 0 else float(np.ravel(value)[0])


def _data_matrix(data: Any, times: Any) -> tuple[np.ndarray, bool]:
    values = np.asarray(data, dtype=float)
    while values.ndim > 2 and 1 in values.shape:
        values = np.squeeze(values, axis=values.shape.index(1))
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2:
        raise ValueError("erpimage(): data must be (frames, trials) or (1, frames*trials)")
    times_passed = not (times is None or np.size(times) == 0)
    if not times_passed and 1 in values.shape:
        raise ValueError("erpimage(): either input a times vector or make data size = (frames,trials).")
    return values.copy(), times_passed


def _sort_values(sortvar: Any, columns: int, noplot: bool) -> tuple[np.ndarray, bool]:
    values = None if sortvar is None else np.asarray(sortvar, dtype=float).ravel()
    if values is None or values.size == 0:
        values, show = np.arange(1.0, columns + 1.0), False
    else:
        show = not noplot
    if values.size < 2:
        raise ValueError("erpimage(): too few trials.")
    return values, show


def _check_option_times(opts: ErpimageOptions, times: np.ndarray) -> None:
    """Latency checks EEGLAB makes while parsing 'phasesort', 'ampsort', 'sortwin' and 'valsort'."""
    if opts.phasesort is not None and not times[0] <= opts.phasesort[0] <= times[-1]:
        raise ValueError("erpimage(): time for phase sorting filter out of bound.")
    if opts.ampsort is not None and not np.isinf(opts.ampsort[0]) and not times[0] <= opts.ampsort[0] <= times[-1]:
        raise ValueError("erpimage(): time for amplitude sorting filter out of bounds.")
    for name, values in (("sortwin", opts.sortwin), ("valsort", opts.valsort)):
        if values is None:
            continue
        for value, which in zip(values[:2], ("start", "end"), strict=False):
            if not times[0] <= value <= times[-1]:
                raise ValueError(f"erpimage(): {which} time for value sorting out of bounds ('{name}').")


def _check_sort_frequencies(
    opts: ErpimageOptions, srate: float, frames: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    phargs = None if opts.phasesort is None else opts.phasesort.copy()
    ampargs = None if opts.ampsort is None else opts.ampsort.copy()
    cycles = opts.cycles
    if phargs is not None:
        if frames < cycles * srate / phargs[2]:
            raise ValueError(
                f"erpimage(): phase-sorting freq. ({phargs[2]:g}) too low: epoch length < {cycles:g} cycles."
            )
        if phargs.size == 4 and phargs[3] > srate / 2:
            phargs[3] = srate / 2
        if phargs.size == 5 and not -180 <= phargs[4] <= 180:
            raise ValueError(f"erpimage(): coher topphase ({phargs[4]:g}) out of range.")
    if ampargs is not None:
        if frames < cycles * srate / abs(ampargs[2]):
            raise ValueError(
                f"erpimage(): amplitude-sorting freq. ({abs(ampargs[2]):g}) too low: epoch length < {cycles:g} cycles."
            )
        if ampargs.size == 4 and abs(ampargs[3]) > srate / 2:
            ampargs[3] = srate / 2
    return phargs, ampargs


def _check_coherence(coher: np.ndarray | None, cycles: float, srate: float, frames: int) -> Any:
    if coher is None:
        return np.nan
    coherfreq = coher[:1] if coher.size == 1 else coher[:2]
    if coherfreq[0] <= 0 or coherfreq[-1] > srate / 2:
        raise ValueError(f"erpimage(): coher frequency ({coherfreq[0]:g}) out of range.")
    if frames < cycles * srate / coherfreq[0]:
        raise ValueError(f"erpimage(): coher freq. ({coherfreq[0]:g}) too low:  epoch length < {cycles:g} cycles.")
    return coherfreq


def _limit_settings(opts: ErpimageOptions, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The nine 'limits' values (clipped to the data), the plotted time range and the align time."""
    limits = np.full(9, np.nan)
    timelimits = np.array([np.min(times), np.max(times)])
    align_window = timelimits
    if opts.limits is not None:
        limits[: opts.limits.size] = opts.limits
        limits[0] = max(limits[0], np.min(times)) if not np.isnan(limits[0]) else limits[0]
        limits[1] = min(limits[1], np.max(times))
        align_window = limits[:2]
        if not np.isnan(limits[0]):
            timelimits = limits[:2].copy()
    aligntime = np.nan if opts.align is None else opts.align
    if (
        not np.isnan(aligntime)
        and not np.isinf(aligntime)
        and (aligntime < align_window[0] or aligntime > align_window[1])
    ):
        raise ValueError("erpimage(): requested align time outside of time limits.")
    return limits, timelimits, aligntime


def _preprocess(
    trials: _Trials,
    opts: ErpimageOptions,
    times: np.ndarray,
    srate: float,
    aligntime: float,
    trial_filter: Callable[[np.ndarray, np.ndarray, float], np.ndarray] | None,
) -> tuple[_Trials, float]:
    """'filt', 'baseline', 'renorm', 'align' and 'rmerp', in EEGLAB's order."""
    values = trials.values
    if opts.filt is not None:
        _check_filter(opts.filt, srate)
        if trial_filter is None:
            raise ValueError("erpimage(): 'filt' needs a trial filter (use eegprep.erpimage)")
        values = np.asarray(trial_filter(values, opts.filt, srate), dtype=float)
    if opts.baseline is not None:
        values = _remove_baseline(values, opts.baseline, times)
    sortvar = _renorm(trials.sortvar, opts.renorm, times)
    auxvar = trials.auxvar
    if not np.isnan(aligntime):
        if np.isinf(aligntime):
            aligntime = float(np.median(sortvar))
        shifts = _mround((aligntime - sortvar) * srate / 1000).astype(int)
        values = _shift_trials(values, shifts)
        if auxvar is not None:
            auxvar = auxvar + shifts  # EEGLAB adds frame shifts to the (ms) auxvar values
    if opts.rmerp:
        values = values - _nan_mean(values, axis=1)[:, np.newaxis]
    return replace(trials, values=values, sortvar=sortvar, auxvar=auxvar), aligntime


def _sort_trials(
    trials: _Trials,
    opts: ErpimageOptions,
    phargs: np.ndarray | None,
    ampargs: np.ndarray | None,
    times: np.ndarray,
    srate: float,
) -> _Sorted:
    if phargs is not None:
        return _phase_sort(trials, phargs, times, srate, opts.cycles)
    if ampargs is not None:
        return _amplitude_sort(trials, ampargs, opts.sortwin, times, srate, opts.cycles)
    if opts.nosort:
        return _Sorted(trials, "nosort", np.zeros(1), None)
    if opts.valsort is not None:
        return _value_sort(trials, opts.valsort, times)
    trials = trials.take(np.argsort(trials.sortvar, kind="stable"))
    return _Sorted(_replace_ties(trials) if opts.replace_ties else trials, "sortvar", np.zeros(1), None)


def _phase_sort(trials: _Trials, phargs: np.ndarray, times: np.ndarray, srate: float, cycles: float) -> _Sorted:
    frames = times.size
    if phargs.size >= 4 and phargs[2] != phargs[3]:
        freq = _peak_frequency(trials.values, frames, srate, phargs[2], phargs[3])
    else:
        freq = float(phargs[2])
    _amps, _cohers, _cohsig, _ampsig, allamps, allphs = _phasecoher(trials.values, srate, freq, cycles)
    center = int(np.argmin(np.abs(times - phargs[0])))
    rejection = phargs[1] / 100
    keep = _reject_by_amplitude(allamps[center], rejection)
    phases = -allphs[center, keep]
    topphase = phargs[4] if phargs.size == 5 else 180.0
    phases[phases > topphase / 360 * 2 * np.pi] -= 2 * np.pi
    order = np.argsort(phases, kind="stable")
    return _Sorted(
        trials.take(keep).take(order),
        "phase",
        np.array([freq]),
        rejection,
        phsangls=-phases[order],
        phsamp=allamps[center, keep][order],
        outamps=allamps,
        winlocs=_sort_window(center, cycles, srate, freq, frames)[np.newaxis, :],
    )


def _amplitude_sort(
    trials: _Trials, ampargs: np.ndarray, sortwin: np.ndarray | None, times: np.ndarray, srate: float, cycles: float
) -> _Sorted:
    frames = times.size
    freqs = _ampsort_frequencies(trials.values, frames, srate, ampargs)
    if np.isinf(ampargs[0]):
        if sortwin is None or sortwin.size < 2:
            raise ValueError("erpimage(): 'ampsort' with an Inf center needs 'sortwin' [start_ms end_ms]")
        centers = _colon(sortwin[0], AMPSORT_WINDOW_STEP_MS, sortwin[1])
    else:
        centers = ampargs[:1]
    amplitude = np.zeros(trials.count)
    windows: list[np.ndarray] = []
    for findex, freq in enumerate(freqs):
        _amps, _cohers, _cohsig, _ampsig, allamps, allphs = _phasecoher(trials.values, srate, float(freq), cycles)
        for center_ms in centers:
            center = int(np.argmin(np.abs(times - center_ms)))
            window = _sort_window(center, cycles, srate, float(freq), frames)
            if findex == 0:
                windows.append(window)
            phases = allphs[center].copy()
            amplitude = amplitude + allamps[center]
    if len({window.size for window in windows}) > 1:
        raise ValueError("erpimage(): 'sortwin' windows reach the epoch edges with different lengths")
    if window.size != int(np.floor(cycles * srate / freqs[-1])) + 1:
        # EEGLAB's phase correction for a window clipped at the epoch edge (1-based frames, as in EEGLAB).
        center_ms = np.median(window + 1) / srate * 1000 + times[0]
        phases = phases + 2 * np.pi * (center_ms - ampargs[0]) * freqs[-1]
    rejection = ampargs[1] / 100
    keep = np.argsort(amplitude, kind="stable")
    if rejection >= 0:
        keep = keep[int(np.ceil(rejection * keep.size)) :]
    else:
        rejection = 1 + rejection
        keep = keep[: int(np.floor(rejection * keep.size))]
    order = np.argsort(amplitude[keep], kind="stable")
    return _Sorted(
        trials.take(keep).take(order),
        "amp",
        freqs,
        rejection,
        phsangls=phases[keep][order],
        outamps=allamps,
        winlocs=np.vstack(windows),
    )


def _value_sort(trials: _Trials, valargs: np.ndarray, times: np.ndarray) -> _Sorted:
    start = int(np.argmin(np.abs(times - valargs[0])))
    stop = int(np.argmin(np.abs(times - valargs[1]))) if valargs.size > 1 else start
    window = trials.values[start : stop + 1]
    sortval = _seqsum(window, axis=0) / window.shape[0] if stop > start else trials.values[start]
    order = np.argsort(sortval, kind="stable")
    if valargs.size > 2 and valargs[2] < 0:
        order = order[::-1]
    return _Sorted(trials.take(order), "value", np.zeros(1), None, winlocs=np.array([[start, stop]]))


def _analysis_frequency(coherfreq: Any, freq: np.ndarray, urdata: np.ndarray, frames: int, srate: float) -> float:
    if isinstance(coherfreq, np.ndarray) and coherfreq.size == 2 and coherfreq[0] != coherfreq[1] and freq[0] <= 0:
        return _peak_frequency(urdata, frames, srate, coherfreq[0], coherfreq[1])
    return float(np.atleast_1d(coherfreq)[0])


def _smoothed(
    values: np.ndarray, trials: _Trials, smoothing: tuple[Any, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, tuple[np.ndarray, ...]]:
    avewidth, decfactor, weights, circular, percent = smoothing
    if avewidth > 1 or decfactor > 1:
        return _smooth_trials(values, trials.sortvar, trials.auxvar, avewidth, decfactor, weights, circular, percent)
    return values, np.arange(1.0, trials.count + 1.0), trials.sortvar.copy(), trials.auxvar, ()


def _potential_image(
    trials: _Trials, values: np.ndarray, caxis: np.ndarray | None, smoothing: tuple[Any, ...]
) -> _Image:
    outdata, outtrials, outsort, auxvar, percentiles = _smoothed(values, trials, smoothing)
    if caxis is not None and caxis.size == 2:
        clim = (float(caxis[0]), float(caxis[1]))
    else:
        maxdat = float(max(abs(np.nanmin(outdata)), abs(np.nanmax(outdata))))
        clim = _fraction_clim(-maxdat, maxdat, None if caxis is None else float(caxis[0]))
    rows, columns = np.nonzero(np.isnan(outdata))
    if rows.size:
        # EEGLAB zeroes the full row-by-column block of NaN positions (data(i,j) = 0).
        outdata = outdata.copy()
        outdata[np.ix_(np.unique(rows), np.unique(columns))] = 0
    return _Image(outdata, outtrials, outsort, auxvar, percentiles, clim)


def _amplitude_image(
    trials: _Trials,
    opts: ErpimageOptions,
    coherfreq: float,
    alpha: float,
    times: np.ndarray,
    srate: float,
    rng: Any,
    baseamp: float,
    smoothing: tuple[Any, ...],
) -> tuple[_Image, _Spectral]:
    """'plotamps': image smoothed wavelet amplitudes in dB instead of potentials."""
    signifs = _given_signif(opts.signif)
    if signifs is not None:
        amps, cohers, _cohsig, _ampsig, allamps, _phases = _phasecoher(trials.values, srate, coherfreq, opts.cycles)
        ampsig, cohsig = signifs[:2].copy(), float(signifs[2])
    else:
        amps, cohers, cohsig, ampsig, allamps, _phases = _phasecoher(
            trials.values, srate, coherfreq, opts.cycles, alpha, rng
        )
    base = np.flatnonzero(times <= DEFAULT_BASELINE_END_MS)
    if base.size < 2:
        base = np.arange(times.size // 4)
    outdata, outtrials, outsort, auxvar, percentiles = _smoothed(allamps, trials, smoothing)
    outdata = 20 * np.log10(outdata)
    amps = 20 * np.log10(amps)
    # EEGLAB converts given 'signif' amplitudes to dB again here; emath keeps MATLAB's complex results.
    ampsig = None if ampsig is None else 20 * np.emath.log10(ampsig)
    if np.isnan(baseamp):
        baseamp = float(_nan_mean(amps[base], axis=0))
        ampsig = None if ampsig is None else ampsig - baseamp
    elif signifs is None and ampsig is not None:
        ampsig = ampsig - baseamp
    amps, outdata = amps - baseamp, outdata - baseamp
    if opts.caxis is not None and opts.caxis.size == 2:
        clim = (float(opts.caxis[0]), float(opts.caxis[1]))
    else:
        maxdat = PLOTAMPS_CAXIS_FRACTION * float(np.max(np.abs(outdata)))
        clim = _fraction_clim(-maxdat, maxdat, None if opts.caxis is None else float(opts.caxis[0]))
    image = _Image(outdata, outtrials, outsort, auxvar, percentiles, clim)
    return image, _Spectral(amps, cohers, cohsig, ampsig, baseamp)


def _given_signif(signif: np.ndarray | None) -> np.ndarray | None:
    return None if signif is None or np.any(np.isnan(signif)) else signif


def _coherence(
    urdata: np.ndarray,
    opts: ErpimageOptions,
    coherfreq: float,
    alpha: float,
    times: np.ndarray,
    srate: float,
    rng: Any,
    baseamp: float,
) -> _Spectral:
    """'coher': wavelet amplitude (dB, baseline removed) and inter-trial coherence."""
    signifs = _given_signif(opts.signif)
    cohsig = ampsig = None
    if signifs is not None or not _alpha_given(opts):
        amps, cohers, _cohsig, _ampsig, _allamps, _phases = _phasecoher(urdata, srate, coherfreq, opts.cycles)
        if signifs is not None:
            ampsig, cohsig = signifs[:2].copy(), float(signifs[2])
    else:
        amps, cohers, cohsig, ampsig, _allamps, _phases = _phasecoher(urdata, srate, coherfreq, opts.cycles, alpha, rng)
        ampsig = None if ampsig is None else 20 * np.emath.log10(ampsig)
    amps = 20 * np.log10(amps)
    if not np.isnan(baseamp):
        amps = amps - baseamp
    elif opts.baselinedb is None or not opts.baselinedb.size or not np.isnan(opts.baselinedb[0]):
        window = opts.baselinedb if opts.baselinedb is not None and opts.baselinedb.size else (times[0], 0.0)
        base = np.flatnonzero((times >= window[0]) & (times <= window[-1]))
        if base.size < 2:
            base = np.arange(times.size // 4)
        baseamp = float(_nan_mean(amps[base], axis=0))
        amps = amps - baseamp
    if alpha > 0 and ampsig is not None:
        ampsig = ampsig - baseamp
    return _Spectral(amps, cohers, cohsig, ampsig, baseamp)


def _erp_stage(
    erp_data: np.ndarray,
    urdata: np.ndarray,
    opts: ErpimageOptions,
    rng: Any,
    plotted: bool,
    minerp: float,
    maxerp: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, float, float]:
    """ERP (or percentile ERPs), its permutation threshold and, when plotted, the axis limits."""
    if not opts.erp:
        return None, None, None, minerp, maxerp
    erpsig = erp_std = None
    if opts.erpalpha is not None:
        signs = _permutation_signs(rng, _permutation_count(opts.erpalpha), erp_data.shape[1])
        mean, erpsig = _permutation_threshold(erp_data, opts.erpalpha, signs)
        erp = mean[np.newaxis, :]
    else:
        erp = _percentile_erps(erp_data, opts.erp)
    if not plotted:
        return erp, erpsig, None, minerp, maxerp
    if opts.erpstd:
        erp_std = _nan_std(urdata)
    if erpsig is not None:
        erpsig = np.vstack([erpsig, -erpsig])
    if np.isnan(maxerp):
        maxerp = _max_erp_limit(erp, erp_std, erpsig)
    if np.isnan(minerp):
        minerp = _min_erp_limit(erp, erp_std, erpsig)
    return erp, erpsig, erp_std, minerp, maxerp


# ----------------------------------------------------------------------------- MATLAB arithmetic


def _mround(values: Any) -> Any:
    """MATLAB ``round``: halves away from zero."""
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def _colon(start: float, step: float, stop: float) -> np.ndarray:
    """MATLAB ``start:step:stop`` (symmetric halves, endpoint snapping)."""
    if step == 0 or (step > 0 and start > stop) or (step < 0 and start < stop) or np.isnan(step):
        return np.zeros(0)
    if np.isinf(step):
        return np.array([float(start)])
    tol = 2.0 * np.finfo(float).eps * max(abs(start), abs(stop))
    sign = np.sign(step)
    if start == np.floor(start) and step == 1:
        count = int(np.floor(stop) - start)
    elif start == np.floor(start) and step == np.floor(step):
        count = int(np.floor((stop - start) / step))
    else:
        count = int(np.round((stop - start) / step))
        if sign * (start + count * step - stop) > tol:
            count -= 1
    end = start + count * step
    if sign * (end - stop) > -tol:
        end = stop
    result = np.empty(count + 1)
    half = np.arange(count // 2 + 1)
    result[half] = start + half * step
    result[count - half] = end - half * step
    if count % 2 == 0:
        result[count // 2] = (start + end) / 2
    return result


def _seqsum(values: np.ndarray, axis: int) -> np.ndarray:
    """Left-to-right sum along ``axis``, the order MATLAB's ``sum`` uses for these sizes.

    NumPy's pairwise summation can differ in the last bit, which matters where EEGLAB
    feeds a sum into a discontinuous rule (axis-limit rounding, baseline means of
    identical padded values).
    """
    if values.shape[axis] == 0:
        return np.zeros(np.delete(values.shape, axis), dtype=values.dtype)
    return np.take(np.cumsum(values, axis=axis), -1, axis=axis)


def _nan_mean(values: np.ndarray, axis: int) -> np.ndarray:
    """EEGLAB ``nan_mean``: mean ignoring NaN; all-NaN slices give NaN."""
    finite = ~np.isnan(values)
    counts = finite.sum(axis=axis)
    totals = _seqsum(np.where(finite, values, 0.0), axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)


def _nan_std(values: np.ndarray) -> np.ndarray:
    """EEGLAB ``nan_std`` over trials (columns) of a (frames, trials) array."""
    finite = ~np.isnan(values)
    clean = np.where(finite, values, 0.0)
    counts = finite.sum(axis=1).astype(float)
    empty = counts == 0
    counts[empty] = 1
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.sqrt((_seqsum(clean**2, axis=1) - _seqsum(clean, axis=1) ** 2 / counts) / (counts - 1))
    result[empty] = np.nan
    return result


# ----------------------------------------------------------------------------- inputs


def _time_axis(times: Any, frames: int) -> tuple[np.ndarray, float]:
    if times is None:
        values = np.zeros(1)
    elif np.size(times) == 0:
        values = np.arange(1.0, frames + 1.0)
    else:
        values = np.asarray(times, dtype=float).ravel()
    if values.size == 1:
        values = np.arange(0.0, frames, dtype=float)
        srate = 1000 * (values.size - 1) / (values[-1] - values[0]) if frames > 1 else np.nan
    elif values.size == 3 and frames != 3:
        # EEGLAB's compact [start_ms frames srate] form; a real 3-sample time vector keeps its meaning.
        mintime, count, srate = float(values[0]), values[1], float(values[2])
        if count != frames:
            raise ValueError(f"erpimage(): times [start frames srate] gives {count:g} frames, data has {frames}.")
        values = _colon(mintime, 1000 / srate, mintime + (frames - 1) * 1000 / srate)
    else:
        srate = 1000 * (values.size - 1) / (values[-1] - values[0])
    if values.size != frames:
        raise ValueError(f"erpimage(): length(times) ({values.size}) differs from frames per epoch ({frames}).")
    return values, float(srate)


def _trial_matrix(values: np.ndarray, ntrials: int, name: str) -> np.ndarray:
    matrix = np.atleast_2d(np.asarray(values, dtype=float))
    rows, cols = matrix.shape
    if rows != ntrials and cols != ntrials:
        raise ValueError(f"erpimage(): {name} size should be (N,ntrials), e.g., (N,{ntrials})")
    if rows == ntrials and cols != ntrials:
        matrix = matrix.T
    return matrix.copy()


def _vert_times(vert: np.ndarray, ntrials: int) -> np.ndarray:
    """Row of latencies, or a (trials, k) matrix of per-trial latencies."""
    vert = np.asarray(vert, dtype=float)
    if vert.ndim < 2 or vert.shape[0] == 1 or (vert.shape[1] == 1 and vert.shape[0] != ntrials):
        return vert.ravel()
    if vert.shape[0] != ntrials:
        raise ValueError(f"erpimage(): vert arg matrix must have 1 or {ntrials} rows")
    return vert.copy()


def _timewarp_marks(
    spec: tuple[np.ndarray, np.ndarray | None, tuple[str, ...]], times: np.ndarray, srate: float, frames: int
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    events, warpms, colors = spec
    marks = _mround(1 + (np.asarray(events, dtype=float) - times[0]) * srate / 1000)
    rows, _cols = np.nonzero(np.diff(marks, axis=1) < 0)
    if rows.size:
        raise ValueError(f"erpimage(): Timewarp event latencies not in ascending order in trial {rows[0] + 1}.")
    marks = np.column_stack([np.ones(marks.shape[0]), marks, np.full(marks.shape[0], float(frames))])
    if warpms is None:
        ref = np.median(marks, axis=0)
    else:
        ref = np.concatenate([[1.0], _mround(1 + (warpms - times[0]) * srate / 1000), [float(frames)]])
    if colors:
        filled = list(colors)
        for k in range(len(colors) + 1, len(ref) - 1):
            filled.append(colors[(k - 1) % len(colors)])
        draw = ("", *filled, "")
    else:
        draw = ("k--",) * len(ref)
    return marks, ref, draw


def _timewarp_setup(
    opts: ErpimageOptions, no_auxvar: bool, times: np.ndarray, srate: float
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[str, ...] | None]:
    """Warp marks and reference frames; the line styles when the warped events replace 'vert'."""
    if opts.timewarp is None:
        return None, None, None
    if opts.phasesort is not None or opts.ampsort is not None or opts.plotamps:
        raise ValueError(
            "erpimage(): 'timewarp' with phase sorting, amplitude sorting or 'plotamps' is not supported "
            "(EEGLAB applies the warp marks inconsistently there)"
        )
    marks, ref, styles = _timewarp_marks(opts.timewarp, times, srate, times.size)
    if not no_auxvar:
        return marks, ref, None
    if opts.align is not None:
        raise ValueError('erpimage(): options "align" and "timewarp" are not compatible.')
    return marks, ref, styles


def _apply_timewarp(values: np.ndarray, marks: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """EEGLAB ``timewarp`` per trial: piecewise-linear resampling from each trial's events to ``ref``."""
    warped = np.empty_like(values)
    for trial in range(values.shape[1]):
        left, weight_left, weight_right = _timewarp_weights(marks[trial], ref)
        right = np.minimum(left + 1, values.shape[0] - 1)
        warped[:, trial] = weight_left * values[left, trial] + weight_right * values[right, trial]
    return warped


def _timewarp_weights(old: np.ndarray, new: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nonzero entries of EEGLAB's ``timewarp(old, new)`` matrix: two taps per output frame."""
    n_old = int(np.max(old))
    positions = np.arange(1.0, n_old + 1.0)
    warped = np.empty(n_old)
    for k in range(old.size - 1):
        span = np.arange(int(old[k]), int(old[k + 1])) - 1
        warped[span] = (positions[span] - old[k]) * (new[k + 1] - new[k]) / (old[k + 1] - old[k]) + new[k]
    warped[n_old - 1] = np.max(new)
    shifted = warped - np.min(new) + 1
    target = np.arange(1.0, int(np.max(new) - np.min(new) + 1) + 1.0)
    upper = np.searchsorted(shifted, target, side="left")  # first warped frame at or after each target
    first = upper == 0
    upper[first] = 1
    gap = shifted[upper] - shifted[upper - 1]
    weight_left = 1 - (target - shifted[upper - 1]) / gap
    weight_right = 1 - (shifted[upper] - target) / gap
    weight_left[first], weight_right[first] = 1.0, 0.0
    return upper - 1, weight_left, weight_right


def _smoothing_window(avewidth: float, avg_type: str, ntrials: int) -> tuple[float, np.ndarray]:
    if avg_type == "gaussian":
        if avewidth == 0:
            avewidth = DEFAULT_SDEV
        elif avewidth < 1:
            raise ValueError("erpimage(): Variable avewidth cannot be < 1.")
        weights = _gaussian_weights(avewidth)
        if weights.size > ntrials:
            sdev = float(np.floor((ntrials - 1) / 6))
            weights = _gaussian_weights(DEFAULT_SDEV if sdev == 0 else sdev)
        return float(weights.size), weights
    if avewidth == 0:
        avewidth = 1.0
    elif avewidth < 1:
        raise ValueError("erpimage(): Variable avewidth cannot be < 1.")
    elif avewidth > ntrials:
        avewidth = float(ntrials)
    if avewidth != np.floor(avewidth):
        # EEGLAB documents non-integer widths, but ones(1,avewidth) rejects them on MATLAB R2025b.
        raise ValueError("erpimage(): boxcar avewidth must be an integer number of trials.")
    return float(avewidth), np.full(int(avewidth), 1.0 / avewidth)


def _gaussian_weights(sdev: float) -> np.ndarray:
    weights = np.exp(-0.5 * (_colon(-3 * sdev, 1.0, 3 * sdev) / sdev) ** 2)
    return weights / _seqsum(weights, axis=0)


def _check_filter(filt: np.ndarray, srate: float) -> None:
    if filt.size != 2:
        raise ValueError("erpimage(): 'filt' parameter argument should be a two element vector.")
    if np.max(filt) > srate / 2:
        raise ValueError(
            f"erpimage(): 'filt' parameters need to be less than or equal to sampling rate/2 (i.e., {srate / 2:f})."
        )
    if filt[1] == srate / 2 and filt[0] == 0:
        raise ValueError(
            "erpimage(): If second element of 'filt' parameter is srate/2, then the first element must be greater than 0."
        )
    if abs(filt[1]) <= abs(filt[0]):
        raise ValueError(
            "erpimage(): Second element of 'filt' parameters must be greater than first in absolute value."
        )
    if filt[0] < 0 or filt[1] < 0:
        if filt[0] >= 0 or filt[1] >= 0:
            raise ValueError(
                "erpimage(): BOTH parameters of 'filt' need to be greater than or equal to zero OR need to be negative."
            )
        if np.min(filt) <= -srate / 2:
            raise ValueError(
                f"erpimage(): 'filt' parameters need to be greater than sampling rate/2 (i.e., -{srate / 2:f}) "
                "when creating a stop band."
            )


def find_corresponding_point(target: float, values: np.ndarray, positions: np.ndarray) -> float | None:
    """EEGLAB ``find_crspnd_pt``: position of ``target`` along monotonic ``values``.

    Exact matches return the median matching position, other targets are
    interpolated between the neighbors, and targets outside ``values`` give None.
    """
    above = np.flatnonzero(values >= target)
    below = np.flatnonzero(values <= target)
    if above.size == 0 or below.size == 0:
        return None
    above_index, below_index = above[0], below[-1]
    if values[above_index] == values[below_index]:
        return float(np.median(positions[values == target]))
    slope = (positions[above_index] - positions[below_index]) / (values[above_index] - values[below_index])
    return float(positions[below_index] + (target - values[below_index]) * slope)


def _remove_baseline(values: np.ndarray, baseline: np.ndarray, times: np.ndarray) -> np.ndarray:
    if baseline[1] < baseline[0]:
        raise ValueError(
            "erpimage(): First element of 'baseline' argument needs to be less than or equal to second argument."
        )
    if baseline[1] < times[0]:
        raise ValueError(
            "erpimage(): Second element of 'baseline' argument needs to be greater than or equal to epoch start time "
            f"{times[0]:.1f}."
        )
    if baseline[0] > times[-1]:
        raise ValueError(
            "erpimage(): First element of 'baseline' argument needs to be less than or equal to epoch end time "
            f"{times[-1]:.1f}."
        )
    positions = np.arange(1.0, times.size + 1.0)
    start = 1 if baseline[0] < times[0] else int(np.ceil(find_corresponding_point(baseline[0], times, positions)))
    stop = (
        times.size
        if baseline[-1] > times[-1]
        else int(np.floor(find_corresponding_point(baseline[1], times, positions)))
    )
    window = values[start - 1 : stop]
    return values - _seqsum(window, axis=0) / window.shape[0]


def _renorm(sortvar: np.ndarray, renorm: str, times: np.ndarray) -> np.ndarray:
    key = renorm.lower()
    if key == "yes":
        span = np.max(times) - np.min(times)
        with np.errstate(invalid="ignore", divide="ignore"):
            scaled = (sortvar - np.min(sortvar)) / (np.max(sortvar) - np.min(sortvar))
        return scaled * 0.5 * span + np.min(times) + 0.4 * span
    if key == "no" or not renorm:
        return sortvar
    if key.count("x") != 1:
        raise ValueError("erpimage: unrecognized renormalizing formula")
    position = key.index("x")
    expression = renorm[:position] + "x" + renorm[position + 1 :]
    for matlab, python in ((".*", "*"), ("./", "/"), (".^", "**"), ("^", "**")):
        expression = expression.replace(matlab, python)
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as error:
        raise ValueError(f"erpimage: unrecognized renormalizing formula '{renorm}'") from error
    return np.asarray(_evaluate_formula(tree.body, sortvar, renorm), dtype=float)


def _evaluate_formula(node: ast.AST, x: np.ndarray, text: str) -> Any:
    operators: dict[type, Callable[[Any, Any], Any]] = {
        ast.Add: np.add,
        ast.Sub: np.subtract,
        ast.Mult: np.multiply,
        ast.Div: np.divide,
        ast.Pow: np.power,
    }
    if isinstance(node, ast.BinOp) and type(node.op) in operators:
        return operators[type(node.op)](_evaluate_formula(node.left, x, text), _evaluate_formula(node.right, x, text))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _evaluate_formula(node.operand, x, text)
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id == "x":
        return x
    raise ValueError(f"erpimage: unrecognized renormalizing formula '{text}'")


def _shift_trials(values: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    frames, ntrials = values.shape
    aligned = np.zeros_like(values)
    for trial, shift in enumerate(shifts):
        if shift > 0 and frames - shift > 0:
            aligned[shift:, trial] = values[: frames - shift, trial]
        elif shift < 0 and frames + shift > 0:
            aligned[: frames + shift, trial] = values[-shift:, trial]
        elif shift == 0:
            aligned[:, trial] = values[:, trial]
    return aligned


# ----------------------------------------------------------------------------- sorting


def _reject_by_amplitude(amplitudes: np.ndarray, fraction: float) -> np.ndarray:
    order = np.argsort(amplitudes, kind="stable")
    if fraction >= 0:
        return order[int(np.ceil(fraction * order.size)) :]
    return order[: int(np.floor((1 + fraction) * order.size))]


def _sort_window(minx: int, cycles: float, srate: float, freq: float, frames: int) -> np.ndarray:
    """0-based frames of the EEGLAB sorting window (for 'showwin')."""
    winlen = int(np.floor(cycles * srate / freq))
    window = (minx + 1) - np.linspace(np.floor(winlen / 2), np.floor(-winlen / 2), winlen + 1)
    return (window[(window > 0) & (window <= frames)] - 1).astype(int)


def _replace_ties(trials: _Trials) -> _Trials:
    """'replace_ties': trials sharing a sortvar value are replaced by their mean."""
    values = trials.values.copy()
    auxvar = None if trials.auxvar is None else trials.auxvar.copy()
    for value in np.unique(trials.sortvar):
        ids = np.flatnonzero(trials.sortvar == value)
        if ids.size > 1:
            values[:, ids] = (_seqsum(values[:, ids], axis=1) / ids.size)[:, np.newaxis]
            if auxvar is not None:
                auxvar[:, ids] = (_seqsum(auxvar[:, ids], axis=1) / ids.size)[:, np.newaxis]
    return replace(trials, values=values, auxvar=auxvar)


def _peak_frequency(values: np.ndarray, frames: int, srate: float, low: float, high: float) -> float:
    psd, freqs = welch_psd(values.reshape(-1, order="F"), frames, srate)
    band = np.flatnonzero((freqs >= low) & (freqs <= high))
    if band.size == 0:
        raise ValueError(f"erpimage(): no spectral estimate between {low:g} and {high:g} Hz")
    with np.errstate(divide="ignore"):
        return float(freqs[band[np.argmax(10 * np.log10(psd[band]))]])


def _ampsort_frequencies(values: np.ndarray, frames: int, srate: float, ampargs: np.ndarray) -> np.ndarray:
    if ampargs.size != 4:
        return np.array([abs(ampargs[2])])
    psd, freqs = welch_psd(values.reshape(-1, order="F"), frames, srate)
    with np.errstate(divide="ignore"):
        decibels = 10 * np.log10(psd)
    if ampargs[2] == ampargs[3]:
        band = np.array([int(np.argmin(np.abs(freqs - ampargs[2])))])
    else:
        band = np.flatnonzero((freqs >= abs(ampargs[2])) & (freqs <= abs(ampargs[3])))
    if band.size == 0:
        raise ValueError(f"erpimage(): no spectral estimate between {abs(ampargs[2]):g} and {abs(ampargs[3]):g} Hz")
    if ampargs[2] >= 0:
        return np.array([freqs[band[np.argmax(decibels[band])]]])
    return freqs[band]


def _adjust_decimation(decfactor: float, ntrials: int, avewidth: float, rejection: float | None) -> float:
    """Decimation above sqrt(trials) means "output this many trials" (negative: minus the window)."""
    inverse = decfactor < 0
    decfactor = abs(decfactor)
    if decfactor <= np.sqrt(ntrials):
        return float(decfactor)
    count = ntrials
    # EEGLAB applies the (already applied) amplitude-rejection fraction a second time here.
    if rejection is not None and rejection > 0:
        count = max(ntrials - int(np.ceil(rejection * ntrials)), 0)
    elif rejection is not None and rejection < 0:
        count = max(int(np.floor(rejection * ntrials)), 0)
    return float((count - avewidth) / decfactor if inverse else count / decfactor)


# ----------------------------------------------------------------------------- moving average


def _moving_average(
    values: np.ndarray, width: float, advance: float, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """EEGLAB ``movav(values, 1:n, width, advance, [], [], weights)`` on (rows, n) data."""
    rows, ntrials = values.shape
    if ntrials < 4:
        raise ValueError("movav(): data are too short")
    advance = 1.0 if advance == 0 else float(advance)
    width = float(width)
    count = max(int(np.floor(((ntrials - 1.0 + advance + 1) - width) / advance)), 1)
    steps = np.full(count, advance)
    steps[0] = 1.0
    low = np.cumsum(steps)  # repeated addition, bitwise equal to EEGLAB's loop
    steps[0] = 1.0 + width / 2
    centers = np.cumsum(steps)
    high = low + width
    first = np.maximum(np.ceil(low), 1).astype(int)  # 1-based first trial in each window
    last = np.minimum(np.ceil(high) - 1, ntrials).astype(int)
    lengths = last - first + 1
    output = np.zeros((rows, count))
    scalar_weight = weights.size == 1
    taps = 1 if scalar_weight else weights.size
    offsets = np.arange(taps)
    regular = lengths == taps
    if not scalar_weight and np.any(regular):
        spacing = (high[regular] - low[regular]) / taps
        index = np.floor((first[regular, np.newaxis] + offsets - low[regular, np.newaxis]) / spacing[:, np.newaxis])
        regular[np.flatnonzero(regular)[np.any(index != offsets, axis=1)]] = False
    if np.any(regular):
        columns = first[regular] - 1
        output[:, regular] = _weighted_windows(values, columns, weights)
    for frame in np.flatnonzero(~regular):
        if lengths[frame] <= 0:
            output[:, frame] = output[:, frame - 1] if frame else 0.0
            continue
        trials = np.arange(first[frame], last[frame] + 1)
        window = values[:, trials - 1]
        if scalar_weight:
            output[:, frame] = _nan_mean(window, axis=1)
            continue
        spacing = (high[frame] - low[frame]) / taps
        index = np.floor((trials - low[frame]) / spacing).astype(int)
        if np.any(index >= taps):
            raise ValueError("movav(): window weights do not cover the smoothing window")
        selected = weights[index]
        total = _nan_sum(window * selected, axis=1)
        weight_sum = _seqsum(selected, axis=0)
        output[:, frame] = total / weight_sum if abs(weight_sum) > NEAR_ZERO else total
    return output, centers


def _weighted_windows(values: np.ndarray, columns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted sums over ``weights.size`` consecutive trials starting at ``columns``."""
    if weights.size == 1:
        return values[:, columns]
    finite = ~np.isnan(values)
    if not finite.all():
        clean = np.where(finite, values, 0.0)
        total = np.zeros((values.shape[0], columns.size))
        present = np.zeros((values.shape[0], columns.size), dtype=bool)
        for tap, weight in enumerate(weights):
            total += clean[:, columns + tap] * weight
            present |= finite[:, columns + tap]
        total[~present] = np.nan
    elif np.all(weights == weights[0]):
        # Boxcar: O(trials) sliding sums instead of one pass per tap. The same terms are
        # added in a different order, so results agree within (2*width)*eps*max|x|.
        total = _sliding_sums(values, columns, weights.size) * weights[0]
    else:
        total = np.zeros((values.shape[0], columns.size))
        for tap, weight in enumerate(weights):
            total += values[:, columns + tap] * weight
    weight_sum = _seqsum(weights, axis=0)
    return total / weight_sum if abs(weight_sum) > NEAR_ZERO else total


def _sliding_sums(values: np.ndarray, starts: np.ndarray, width: int) -> np.ndarray:
    """Sums of ``width`` consecutive columns at each start (van Herk/Gil-Werman).

    Columns are split into blocks of ``width``; a window is the suffix of one block
    plus the prefix of the next, so each sum uses only the window's own terms.
    """
    rows, ntrials = values.shape
    blocks = -(-ntrials // width) + 1
    padded = np.zeros((rows, blocks * width))
    padded[:, :ntrials] = values
    shaped = padded.reshape(rows, blocks, width)
    prefix = np.cumsum(shaped, axis=2).reshape(rows, -1)
    suffix = np.cumsum(shaped[:, :, ::-1], axis=2)[:, :, ::-1].reshape(rows, -1)
    sums = prefix[:, starts + width - 1]
    partial = starts % width != 0
    sums[:, partial] += suffix[:, starts[partial]]
    return sums


def _nan_sum(values: np.ndarray, axis: int) -> np.ndarray:
    finite = ~np.isnan(values)
    total = _seqsum(np.where(finite, values, 0.0), axis=axis)
    return np.where(finite.any(axis=axis), total, np.nan)


def _smooth_trials(
    values: np.ndarray,
    sortvar: np.ndarray,
    auxvar: np.ndarray | None,
    avewidth: float,
    decfactor: float,
    weights: np.ndarray,
    circular: bool,
    percent: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, tuple[np.ndarray, ...]]:
    if circular:
        values_ext, sortvar_ext, aux_ext = _wrap_trials(values, sortvar, auxvar, avewidth)
    else:
        values_ext, sortvar_ext, aux_ext = values, sortvar, auxvar
    smoothed, outtrials = _moving_average(values_ext, avewidth, decfactor, weights)
    outsort, outtrials = _moving_average(sortvar_ext[np.newaxis, :], avewidth, decfactor, weights)
    if circular:
        outtrials = outtrials - outtrials[0] + 1
    smoothed_aux = None if aux_ext is None else _moving_average(aux_ext, avewidth, decfactor, weights)[0]
    percentiles = tuple(
        _percentile_trace(sortvar, float(p), outtrials, avewidth) for p in (() if percent is None else percent)
    )
    return smoothed, outtrials, outsort.ravel(), smoothed_aux, percentiles


def _wrap_trials(
    values: np.ndarray, sortvar: np.ndarray, auxvar: np.ndarray | None, avewidth: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Circular padding EEGLAB uses before smoothing phase-sorted trials."""
    if avewidth > 2:
        back, front = int(np.floor(avewidth / 2)), int(np.floor((avewidth - 1) / 2))
        index = np.concatenate(
            [np.arange(sortvar.size - back, sortvar.size), np.arange(sortvar.size), np.arange(front)]
        )
    else:
        index = np.concatenate([[sortvar.size - 1], np.arange(sortvar.size)])
    return values[:, index], sortvar[index], None if auxvar is None else auxvar[:, index]


def _percentile_trace(sortvar: np.ndarray, percent: float, outtrials: np.ndarray, winsize: float) -> np.ndarray:
    """EEGLAB ``compute_percentile`` over a circular sortvar window."""
    ntrials = sortvar.size
    rounded = _mround(outtrials).astype(int)
    tripled = np.concatenate([sortvar, sortvar, sortvar])
    offsets = np.arange(int(_mround(-winsize / 2)), int(_mround(winsize / 2)) + 1)
    pick = int(_mround((offsets.size - 1) * percent))
    trace = np.empty(rounded.size)
    for index, trial in enumerate(rounded):
        window = np.sort(tripled[trial + ntrials + offsets - 1])
        trace[index] = window[pick]
    return trace


def _smooth_vert(
    verttimes: np.ndarray | None, avewidth: float, decfactor: float, weights: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Smoothed per-trial vertical-line traces (one per column of a trials x k matrix)."""
    if verttimes is None or verttimes.shape[0] < 4:
        return ()
    return tuple(
        _moving_average(column[np.newaxis, :], avewidth, decfactor, weights)[0].ravel() for column in verttimes.T
    )


def _fraction_clim(mindat: float, maxdat: float, caxfraction: float | None) -> tuple[float, float]:
    if caxfraction is not None:
        adjust = (1 - caxfraction) / 2 * (maxdat - mindat)
        return mindat + adjust, maxdat - adjust
    return mindat, maxdat


def _sortvar_trial_limits(
    limits: np.ndarray, outsort: np.ndarray, outtrials: np.ndarray, sortvar: np.ndarray
) -> tuple[tuple[float, float], np.ndarray]:
    lower, upper = float(limits[0]), float(limits[1])
    low = find_corresponding_point(lower, outsort, outtrials)
    if low is None:
        low, lower = 1.0, float(outsort[0])
    high = find_corresponding_point(upper, outsort, outtrials)
    if high is None:
        high, upper = float(outsort.size), float(outsort[-1])
    inside = np.flatnonzero((sortvar >= lower) & (sortvar <= upper))
    return (low, high), inside


# ----------------------------------------------------------------------------- ERP


def _percentile_erps(values: np.ndarray, ptiles: int) -> np.ndarray:
    ntrials = values.shape[1]
    step = int(_mround(ntrials / ptiles))
    erp = np.empty((ptiles, values.shape[0]))
    for ptile in range(ptiles):
        trials = np.arange(step) + ptile * step
        if trials.size and trials.max() >= ntrials:
            trials = np.arange(trials[0], ntrials)
        erp[ptile] = _nan_mean(values[:, trials], axis=1) if trials.size else np.nan
    return erp


def _permutation_count(alpha: float) -> int:
    return max(ERP_PERMUTATIONS, int(_mround(3 / alpha)))


def _permutation_signs(rng: Any, permutations: int, ntrials: int) -> np.ndarray:
    # EEGLAB draws sign(randn - 0.5), which is +1 with probability 0.31 and biases the
    # threshold; EEGPrep uses unbiased signs, so 'erpalpha' bands differ slightly from
    # MATLAB. See https://github.com/sccn/eeglab/issues/976.
    return np.sign(rng.random((permutations, ntrials)) - 0.5)


def _permutation_threshold(values: np.ndarray, alpha: float, signs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """EEGLAB ``nan_mean(in, alpha)`` on (frames, trials) data with given (perm, trials) signs."""
    trials_by_frames = values.T
    finite = ~np.isnan(trials_by_frames)
    clean = np.where(finite, trials_by_frames, 0.0)
    counts = finite.sum(axis=0).astype(float)
    empty = counts == 0
    counts[empty] = 1
    mean = _seqsum(clean, axis=0) / counts
    # macOS Accelerate raises spurious floating-point flags inside matmul; the inputs are finite.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sums = signs @ clean
    permuted = np.sort(np.abs(sums / counts), axis=0)
    bound = int(np.floor(2 * alpha * signs.shape[0]))
    threshold = permuted[signs.shape[0] - 1 - bound]
    mean[empty] = np.nan
    return mean, threshold


def _max_erp_limit(erp: np.ndarray, std: np.ndarray | None, erpsig: np.ndarray | None) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        peak = _matlab_max(erp)
        fac, limit = 10.0, np.zeros(1)
        while np.all(limit == 0):
            limit = _mround(fac * YEXPAND * peak) / fac
            fac *= 10
        if std is not None:
            fac /= 10
            limit = np.maximum(limit, _mround(fac * YEXPAND * _matlab_max(erp + std)) / fac)
        if erpsig is not None:
            limit = np.maximum(limit, _mround(fac * YEXPAND * _matlab_max(erpsig)) / fac)
    return float(np.max(limit))


def _min_erp_limit(erp: np.ndarray, std: np.ndarray | None, erpsig: np.ndarray | None) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        trough = _matlab_min(erp)
        fac, limit = 1.0, np.zeros(1)
        while np.all(limit == 0):
            limit = _mround(fac * YEXPAND * trough) / fac
            fac *= 10
        if std is not None:
            fac /= 10
            limit = np.minimum(limit, _mround(fac * YEXPAND * _matlab_min(erp - std)) / fac)
        if erpsig is not None:
            limit = np.minimum(limit, _mround(fac * YEXPAND * _matlab_min(erpsig)) / fac)
    return float(np.min(limit))


def _matlab_max(values: np.ndarray) -> np.ndarray:
    """MATLAB ``max``: scalar for a vector, column maxima for a matrix (NaN ignored)."""
    array = np.atleast_2d(values)
    with np.errstate(invalid="ignore"):
        return np.atleast_1d(np.nanmax(array) if array.shape[0] == 1 else np.nanmax(array, axis=0))


def _matlab_min(values: np.ndarray) -> np.ndarray:
    array = np.atleast_2d(values)
    with np.errstate(invalid="ignore"):
        return np.atleast_1d(np.nanmin(array) if array.shape[0] == 1 else np.nanmin(array, axis=0))


def _amp_coh_limits(
    amps: np.ndarray,
    cohers: np.ndarray,
    ampsig: np.ndarray | None,
    cohsig_flag: bool,
    maxamp: float,
    minamp: float,
    maxcoh: float,
    mincoh: float,
) -> tuple[float, float, float, float]:
    with np.errstate(over="ignore", invalid="ignore"):
        if np.isnan(maxamp):
            fac, maxamp = 1.0, 0.0
            while maxamp == 0:
                maxamp = np.floor(YEXPAND * fac * np.max(amps)) / fac
                fac *= 10
            maxamp = maxamp + 10 / fac
            if cohsig_flag and ampsig is not None and np.real(ampsig[1]) > maxamp:
                maxamp = 1.01 * np.real(ampsig[1]) if np.real(ampsig[1]) > 0 else 0.99 * np.real(ampsig[1])
        if np.isnan(maxamp):
            maxamp = 0.0
        if np.isnan(minamp):
            fac, minamp = 1.0, 0.0
            while minamp == 0:
                minamp = np.floor(YEXPAND * fac * np.max(-amps)) / fac
                fac *= 10
            minamp = -(minamp + 10 / fac)
            if cohsig_flag and ampsig is not None and np.real(ampsig[0]) < minamp:
                minamp = 1.01 * np.real(ampsig[0]) if np.real(ampsig[0]) < 0 else 0.99 * np.real(ampsig[0])
        if np.isnan(minamp):
            minamp = 0.0
        if np.isnan(maxcoh):
            fac, maxcoh = 1.0, 0.0
            while maxcoh == 0:
                maxcoh = np.floor(YEXPAND * fac * np.max(cohers)) / fac
                fac *= 10
            maxcoh = maxcoh + 10 / fac
            if maxcoh > 1:
                maxcoh = 1.0
        if np.isnan(mincoh):
            mincoh = 0.0
    return float(maxamp), float(minamp), float(maxcoh), float(mincoh)


# ----------------------------------------------------------------------------- spectra


def welch_psd(signal: np.ndarray, segment: int, srate: float) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB ``pwelch(signal, segment, 0, nfft, srate)`` with EEGLAB's nfft.

    Symmetric Hamming window of ``segment`` samples, no overlap, no detrending,
    one-sided density, ``nfft = max(1024, 2^ceil(log2(segment)))``.
    """
    segment = int(segment)
    nfft = max(1024, 2 ** int(np.ceil(np.log2(segment))))
    if segment > 1:
        window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(segment) / (segment - 1))
    else:
        window = np.ones(1)
    count = signal.size // segment
    segments = signal[: count * segment].reshape(count, segment) * window
    power = np.abs(np.fft.rfft(segments, nfft, axis=1)) ** 2
    psd = power.mean(axis=0) / (srate * np.sum(window**2))
    psd[1 : -1 if nfft % 2 == 0 else None] *= 2
    return psd, np.arange(psd.size) * srate / nfft


def _gauss(length: int, sds: float) -> np.ndarray:
    if length < 1:
        raise ValueError("phasecoher(): wavelet window is empty")
    if length == 1:
        return np.exp(-(np.array([-sds]) ** 2))
    return np.exp(-(_colon(-sds, 2 * sds / (length - 1), sds) ** 2))


def _wavelet(srate: float, freq: float, cycles: float) -> tuple[np.ndarray, np.ndarray]:
    window = _gauss(int(np.ceil(srate / freq * cycles)), 2.0)
    length = window.size
    angles = _colon(-cycles * np.pi, cycles * 2 * np.pi / length, cycles * np.pi)[:length]
    cosine, sine = np.cos(angles), np.sin(angles)
    coswin = window * cosine
    sinwin = window * sine
    return coswin / (coswin @ cosine), sinwin / (sinwin @ sine)


def _phasecoher(  # noqa: C901
    values: np.ndarray, srate: float, freq: float, cycles: float, alpha: float = np.nan, rng: Any = None
) -> tuple[np.ndarray, np.ndarray, float | None, np.ndarray | None, np.ndarray, np.ndarray]:
    """EEGLAB ``phasecoher``: wavelet amplitude, ITC, bootstrap thresholds, per-trial amps and phases."""
    frames, trials = values.shape
    coswin, sinwin = _wavelet(srate, freq, cycles)
    length = coswin.size
    half = length // 2
    if frames < length:
        raise ValueError(f"phasecoher(): Epoch length ({frames} frames) too short for analysis with {cycles:g} cycles!")
    head = np.arange(half, -1, -1)
    tail = np.arange(frames - 1, frames - 1 - (length - half), -1)
    extended = np.vstack([values[head], values, values[tail]])
    complex_values = _wavelet_transform(extended, frames, coswin, sinwin)
    allamps = np.sqrt(complex_values.real**2 + complex_values.imag**2)
    allphs = np.angle(complex_values)
    amps, cohers = _amp_coh(allamps, allphs)
    cohsig = ampsig = None
    if alpha == 0:
        alpha = np.nan
    if not np.isnan(alpha):
        if alpha < 0 or alpha > 0.1:
            raise ValueError("phasecoher(): alpha out of bounds.")
        reps = COHSIG_REPS if alpha * COHSIG_REPS >= 5 else int(np.ceil(5 / alpha))
        offsets = np.floor(rng.random((reps, trials)) * (frames - length)).astype(int)
        columns = np.arange(trials)
        boot_amps, boot_cohers = _amp_coh(allamps[offsets, columns], allphs[offsets, columns])
        boot_cohers, boot_amps = np.sort(boot_cohers), np.sort(boot_amps)
        cohsig = float(boot_cohers[int(_mround(reps * (1.0 - alpha))) - 1])
        ampsig = np.array(
            [boot_amps[int(_mround(reps * alpha)) - 1], boot_amps[int(_mround(reps * (1.0 - alpha))) - 1]]
        )
    amps[:half] = amps[half]
    cohers[:half] = cohers[half]
    amps[frames - half - 1 :] = amps[frames - half - 1]
    cohers[frames - half - 1 :] = cohers[frames - half - 1]
    return amps, cohers, cohsig, ampsig, allamps, allphs


def _wavelet_transform(extended: np.ndarray, frames: int, coswin: np.ndarray, sinwin: np.ndarray) -> np.ndarray:
    """EEGLAB's per-frame projection of each demeaned window onto the wavelet.

    ``coswin . (epoch - mean(epoch))`` equals a correlation of the data with the
    demeaned kernel, computed here with FFTs (agreeing within a few eps of
    ``max|result|``). The kernel sums to zero, so removing each trial's mean first
    changes nothing but keeps rounding relative to the signal, not its offset.
    """
    length = coswin.size
    real_kernel = coswin - _seqsum(coswin, axis=0) / length
    imag_kernel = sinwin - _seqsum(sinwin, axis=0) / length
    size = 1 << int(np.ceil(np.log2(extended.shape[0] + length - 1)))
    real_spectrum = np.fft.rfft(real_kernel[::-1], size)
    imag_spectrum = np.fft.rfft(imag_kernel[::-1], size)
    result = np.empty((frames, extended.shape[1]), dtype=complex)
    for start in range(0, extended.shape[1], WAVELET_TRIAL_BLOCK):
        block = extended[:, start : start + WAVELET_TRIAL_BLOCK]
        spectrum = np.fft.rfft(block - block.mean(axis=0), size, axis=0)
        window = slice(length - 1, length - 1 + frames)
        result[:, start : start + block.shape[1]].real = np.fft.irfft(spectrum * real_spectrum[:, None], size, axis=0)[
            window
        ]
        result[:, start : start + block.shape[1]].imag = np.fft.irfft(spectrum * imag_spectrum[:, None], size, axis=0)[
            window
        ]
    return result


def _amp_coh(allamps: np.ndarray, allphs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = allamps >= MIN_AMP
    counts = keep.sum(axis=1)
    amps = np.zeros(allamps.shape[0])
    cohers = np.zeros(allamps.shape[0])
    present = counts != 0
    amps[present] = _seqsum(allamps * keep, axis=1)[present] / counts[present]
    phasors = (np.cos(allphs) + 1j * np.sin(allphs)) * keep
    mean_phasor = _seqsum(phasors, axis=1)[present] / counts[present]
    cohers[present] = np.sqrt((mean_phasor * np.conj(mean_phasor)).real)
    return amps, cohers


def order_of_magnitude(value: float) -> float:
    """EEGLAB ``orderofmag``: power of ten of ``value`` for tick spacing.

    Zero gives 1 (EEGLAB's loop never ends for zero).
    """
    value = abs(value)
    if value == 0:
        return 1.0
    if value >= 1:
        order = 1.0
        value = np.floor(value / 10)
        while value >= 1:
            order *= 10
            value = np.floor(value / 10)
        return order
    order = 0.1
    value = value * 10
    while value < 1:
        order /= 10
        value *= 10
    return order


__all__ = [
    "ErpimageComputation",
    "ErpimageOptions",
    "compute_erpimage",
    "find_corresponding_point",
    "order_of_magnitude",
    "parse_erpimage_options",
    "welch_psd",
]
