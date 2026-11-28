"""EEG topographic plotting functions."""

import warnings

import numpy as np
from matplotlib import colors as mcolors
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from eegprep.functions.miscfunc.linspace import linspace
from eegprep.functions.popfunc.plot_utils import show_figures


_IMPLEMENTED_OPTIONS = frozenset(
    {
        "noplot",
        "plotchans",
        "gridscale",
        "electrodes",
        "intrad",
        "plotrad",
        "method",
        "style",
        "colormap",
        "maplimits",
        "numcontour",
        "colorbar",
        "markersize",
        "showlabels",
        "axes",
        "title",
        "emarker",
        "emarker2",
        "emarkersize1chan",
        "emarkersizemark",
        "emarkersize",
        "emsize",
        "electcolor",
        "ecolor",
    }
)
# Accepted without effect: EEGLAB ignores 'iclabel' (topoplot.m:613), and
# existing callers/tests pass 'verbose', 'chaninfo', and 'masksurf'.
_COMPAT_IGNORED_OPTIONS = frozenset({"iclabel", "verbose", "chaninfo", "masksurf"})
_ALLOWED_OPTIONS = _IMPLEMENTED_OPTIONS | _COMPAT_IGNORED_OPTIONS
# Valid EEGLAB topoplot.m (lines 389-616) options/aliases not yet implemented in EEGPrep.
_UNIMPLEMENTED_EEGLAB_OPTIONS = frozenset(
    {
        "ccolor",
        "chantype",
        "circgrid",
        "contourcolor",
        "contourvals",
        "conv",
        "dipcolor",
        "diplen",
        "dipnorm",
        "dipnormmax",
        "dipole",
        "diporient",
        "dipscale",
        "dipsphere",
        "drawaxis",
        "efontsize",
        "efsize",
        "emarkercolors",
        "forprint",
        "gridplot",
        "hcolor",
        "headcolor",
        "headlimits",
        "headrad",
        "interplimits",
        "intsquare",
        "nosedir",
        "plotdisk",
        "plotgrid",
        "pmask",
        "scatter",
        "shading",
        "shrink",
        "whiteback",
        "whitebk",
    }
)

_ELECTRODE_ALIASES = {
    "on": "on",
    "pts": "on",
    "off": "off",
    "labels": "labels",
    "numbers": "numbers",
    "nums": "numbers",
    "labelpoint": "labelpoint",
    "pointlabels": "labelpoint",
    "ptslabels": "labelpoint",
    "labelspts": "labelpoint",
    "ptlabels": "labelpoint",
    "labelpts": "labelpoint",
    "numpoint": "numpoint",
    "pointnumbers": "numpoint",
    "ptsnumbers": "numpoint",
    "numberspts": "numpoint",
    "ptnumbers": "numpoint",
    "numberpts": "numpoint",
    "ptsnums": "numpoint",
    "numspts": "numpoint",
}

_MATLAB_MARKERS: dict[str, str | tuple[int, int, int]] = {
    ".": ".",
    "o": "o",
    "+": "+",
    "*": (8, 2, 0),  # MATLAB '*' is an 8-spoke line asterisk, not a 5-point star
    "x": "x",
    "s": "s",
    "square": "s",
    "d": "d",
    "diamond": "d",
    "^": "^",
    "v": "v",
    "<": "<",
    ">": ">",
    "p": "*",  # MATLAB pentagram -> Matplotlib 5-point fillable star
    "pentagram": "*",
    "h": (6, 1, 0),  # MATLAB hexagram -> Matplotlib 6-point fillable star
    "hexagram": (6, 1, 0),
}
_BLANK_COLOR_ARRAY = ((1.0, 0.0, 0.0, 1.0), (0.5, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))
_DEFAULT_EMARKERSIZE1CHAN = 20.0
_EMARKERSIZE_THRESHOLDS = ((100, 3.0), (80, 4.0), (64, 5.0), (48, 6.0), (32, 8.0))


def griddata_v4(x, y, v, xq, yq):
    """Python version of MATLAB's GDATAV4 interpolation based on David T. Sandwell's biharmonic spline interpolation.

    Duplicate sensors are averaged first. The warning names this function and
    uses MATLAB's ``DuplicateDataPoints`` wording. One unique location has no
    biharmonic interpolant: MATLAB ``griddata`` returns a non-finite map and
    warns, and this function raises ``ValueError``.

    Parameters
    ----------
    x, y : 1D arrays
        Coordinates for known points.
    v : 1D array
        Values at known points.
    xq, yq : 2D arrays
        Query points coordinates.

    Returns
    -------
    vq : 2D array
        Interpolated values at query points.
    """
    n_in = np.asarray(x, dtype=float).size
    x, y, v, _merged_count = _merge_coincident_points(x, y, v)
    if x.size < n_in:
        warnings.warn(
            "topoplot griddata_v4: Duplicate x-y data points detected: using average values for duplicate points.",
            UserWarning,
            stacklevel=2,
        )
    if x.size < 2:
        raise ValueError(
            "topoplot griddata_v4: at least two distinct sensor locations are needed. "
            "Pass values at two or more electrodes that stay apart after duplicate averaging."
        )

    xy = x + 1j * y
    d = np.abs(xy[:, None] - xy[None, :])

    with np.errstate(divide='ignore', invalid='ignore'):
        g = (d**2) * (np.log(d) - 1)  # Green's function
    np.fill_diagonal(g, 0)  # Fix value along diagonal
    weights = np.linalg.solve(g, v)

    # Evaluate at requested points (vectorized)
    # q is (M, N), xy is (L,) -> d_q is (M, N, L)
    q = xq + 1j * yq
    d_q = np.abs(q[:, :, np.newaxis] - xy[np.newaxis, np.newaxis, :])

    with np.errstate(divide='ignore', invalid='ignore'):
        g_q = (d_q**2) * (np.log(d_q) - 1)  # Green's function
    g_q[d_q == 0] = 0  # Handle Green's function at zero

    # Weights is (L,), g_q is (M, N, L) -> vq is (M, N)
    # NumPy 2 may expose spurious BLAS floating-point flags for this finite
    # product; preserve the computed values without leaking those flags.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        vq = g_q @ weights

    return vq


def _merge_coincident_points(x, y, v):
    """Average locations MATLAB ``griddata`` merges before ``gdatav4``.

    ``mergepoints2D`` calls ``mergesimpts([y, x, v], [tol_y, tol_x, Inf],
    'average')`` with ``tol = spacing(0.5 * (max - min)) ** (1/3)``. Rows
    are sorted by ``(y, x, v)``. Each unclaimed row is an anchor, and every
    unclaimed row strictly inside that anchor's tolerance on both axes joins
    it. ``mergesimpts`` uses ``<``, so a separation equal to the tolerance
    stays separate: ``mergesimpts([0; 0.5; 1], 0.5, 'average')`` keeps all
    three rows. The group is replaced by the mean of its coordinates and
    values. A chain does not merge by neighbor transitivity: ``C`` stays
    separate when it is outside tolerance of anchor ``A``, even if it is
    within tolerance of ``B``. A row that sorts between ``A`` and ``D`` does
    not keep ``D`` out when ``D`` is within tolerance of ``A``.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)
    tol_x = float(np.spacing(0.5 * (np.max(x) - np.min(x))) ** (1.0 / 3.0))
    tol_y = float(np.spacing(0.5 * (np.max(y) - np.min(y))) ** (1.0 / 3.0))
    return _average_similar_points(x, y, v, tol_x, tol_y)


def _average_similar_points(x, y, v, tol_x, tol_y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)
    order = np.lexsort((v, x, y))
    y_sorted = y[order]
    x_sorted = x[order]
    v_sorted = v[order]
    claimed = np.zeros(x.size, dtype=bool)
    merged_x = []
    merged_y = []
    merged_v = []
    merged_count = 0
    for index in range(x.size):
        if claimed[index]:
            continue
        close = (np.abs(y_sorted - y_sorted[index]) < tol_y) & (np.abs(x_sorted - x_sorted[index]) < tol_x)
        close &= ~claimed
        close[index] = True
        claimed[close] = True
        group_size = int(np.count_nonzero(close))
        if group_size > 1:
            merged_count += group_size
        merged_x.append(float(np.mean(x_sorted[close])))
        merged_y.append(float(np.mean(y_sorted[close])))
        merged_v.append(float(np.mean(v_sorted[close])))
    return np.asarray(merged_x), np.asarray(merged_y), np.asarray(merged_v), merged_count


def _plot_channel_indices(plotchans, located, n_channels):
    """Return sorted 0-based channels to draw.

    Empty ``plotchans`` means every located channel. Otherwise the values
    are 1-based integers in ``1 .. n_channels``. They are converted to
    0-based indices and intersected with the located channels, matching
    ``topoplot.m`` lines 703-716.
    """
    located = np.asarray(located, dtype=int)
    if plotchans is None or np.asarray(plotchans).size == 0:
        return located
    requested = np.asarray(plotchans, dtype=float).reshape(-1)
    if not np.all(np.isfinite(requested)) or np.any(requested != np.trunc(requested)):
        raise ValueError(
            "topoplot: plotchans must be integer 1-based channel indices. "
            f"Got {np.asarray(plotchans).reshape(-1).tolist()}. "
            "Pass whole indices such as [1, 3, 6]."
        )
    requested = requested.astype(int)
    if np.any(requested < 1) or np.any(requested > n_channels):
        raise ValueError(
            f"topoplot: plotchans must be 1-based indices in 1..{n_channels}. "
            f"Got {requested.tolist()}. Pass the channel indices to include, for example [1, 3]."
        )
    return np.intersect1d(requested - 1, located)


def _scalar_finite_coord(loc: dict, field: str) -> float:
    if field not in loc:
        return np.nan
    value = loc[field]
    if value is None or isinstance(value, (str, bytes, bool, np.bool_)):
        return np.nan
    try:
        arr = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.nan
    if arr.size != 1 or not np.isfinite(arr[0]):
        return np.nan
    return float(arr[0])


def _require_channel_locations(chan_locs) -> None:
    if chan_locs is None or len(chan_locs) == 0:
        raise ValueError(
            "topoplot: Channel locations are required. "
            "Load locations with pop_chanedit or pop_readlocs before plotting."
        )


def _is_empty_placeholder(value) -> bool:
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    return isinstance(value, (list, tuple)) and len(value) == 0


def _normalize_electrodes(value) -> str:
    if isinstance(value, str):
        normalized = _ELECTRODE_ALIASES.get(value.lower())
        if normalized is not None:
            return normalized
    raise ValueError(f"topoplot: Unknown value for keyword 'electrodes': {value!r}")


def _parse_marker_symbol(value, option_name: str) -> str | tuple[int, int, int]:
    if not isinstance(value, str):
        raise ValueError(f"topoplot: invalid {option_name} marker {value!r}")
    if value.lower() == "none":
        return "none"
    mapped = _MATLAB_MARKERS.get(value)
    if mapped is None:
        mapped = _MATLAB_MARKERS.get(value.lower())
    if mapped is None:
        raise ValueError(f"topoplot: invalid {option_name} marker {value!r}")
    return mapped


def _parse_color_spec(value, option_name: str):
    if isinstance(value, str):
        if value.lower() == "none":
            return "none"
        try:
            return mcolors.to_rgba(value)
        except ValueError:
            raise ValueError(f"topoplot: invalid {option_name} color {value!r}") from None
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            raise ValueError(f"topoplot: invalid {option_name} color {value!r}") from None
        if arr.size in (3, 4) and np.all(np.isfinite(arr)) and np.all((arr >= 0.0) & (arr <= 1.0)):
            return mcolors.to_rgba(tuple(arr.tolist()))
    raise ValueError(f"topoplot: invalid {option_name} color {value!r}")


def _parse_positive_number(value, option_name: str, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"topoplot: {option_name} {field_name} must be a positive number, got {value!r}")
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise ValueError(f"topoplot: {option_name} {field_name} must be a positive number, got {value!r}") from None
    if arr.size != 1 or not np.isfinite(arr[0]) or arr[0] <= 0.0:
        raise ValueError(f"topoplot: {option_name} {field_name} must be a positive number, got {value!r}")
    return float(arr[0])


def _parse_emarker2_channels(value) -> np.ndarray:
    if _is_empty_placeholder(value):
        return np.array([], dtype=int)
    if isinstance(value, (str, bytes, bool, np.bool_)):
        raise ValueError(f"topoplot: emarker2 channel indices must be whole numbers, got {value!r}")
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise ValueError(f"topoplot: emarker2 channel indices must be whole numbers, got {value!r}") from None
    if not np.all(np.isfinite(arr)) or np.any(arr != np.trunc(arr)):
        raise ValueError(f"topoplot: emarker2 channel indices must be whole numbers, got {value!r}")
    # topoplot.m:523 takes abs(Value{1}) for negative channel indices.
    return np.abs(arr).astype(int)


def _default_emarkersize(n_channels: int) -> float:
    """EEGLAB channel-count marker size table (topoplot.m:1466-1483)."""
    for threshold, size in _EMARKERSIZE_THRESHOLDS:
        if n_channels >= threshold:
            return size
    return 10.0


def _apply_marker_spec(target: dict, items, option_name: str, *, allow_auto_size: bool) -> None:
    if items and not _is_empty_placeholder(items[0]):
        target["marker"] = _parse_marker_symbol(items[0], option_name)
    if len(items) > 1 and not _is_empty_placeholder(items[1]):
        target["color"] = _parse_color_spec(items[1], option_name)
    if len(items) > 2:
        if _is_empty_placeholder(items[2]):
            if allow_auto_size:
                target["size"] = None
        else:
            target["size"] = _parse_positive_number(items[2], option_name, "size")
    if len(items) > 3 and not _is_empty_placeholder(items[3]):
        target["linewidth"] = _parse_positive_number(items[3], option_name, "linewidth")


def _parse_topoplot_options(kwargs: dict) -> dict:
    opts: dict = {}
    marker_style: dict = {"marker": ".", "color": mcolors.to_rgba("k"), "size": None, "linewidth": 1.0}
    marker2_style: dict = {
        "chans": np.array([], dtype=int),
        "marker": "o",
        "color": mcolors.to_rgba("r"),
        "size": 10.0,
        "linewidth": 1.0,
    }
    emarkersize1chan = _DEFAULT_EMARKERSIZE1CHAN
    legacy_markersize = None
    electrodes_arg = None

    for raw_key, value in kwargs.items():
        key = str(raw_key).lower()
        if key not in _ALLOWED_OPTIONS:
            if key in _UNIMPLEMENTED_EEGLAB_OPTIONS:
                raise ValueError(
                    f"topoplot: '{raw_key}' is an EEGLAB topoplot option that EEGPrep does not implement yet. "
                    "Remove it from the call."
                )
            raise ValueError(f"topoplot: Unknown input parameter '{raw_key}'")
        opts[key] = value
        if key == "electrodes":
            electrodes_arg = _normalize_electrodes(value)
        elif key == "emarker":
            if isinstance(value, str):
                marker_style["marker"] = _parse_marker_symbol(value, "emarker")
            elif isinstance(value, (list, tuple)) and 1 <= len(value) <= 4:
                _apply_marker_spec(marker_style, value, "emarker", allow_auto_size=True)
            else:
                raise ValueError(
                    "topoplot: 'emarker' argument must be a marker string or a 1-to-4 item sequence "
                    "{marker, color, size, linewidth}"
                )
        elif key == "emarker2":
            if not isinstance(value, (list, tuple)) or not (1 <= len(value) <= 5):
                raise ValueError(
                    "topoplot: 'emarker2' argument must be a 1-to-5 item sequence "
                    "{chans, marker, color, size, linewidth}"
                )
            marker2_style["chans"] = _parse_emarker2_channels(value[0])
            _apply_marker_spec(marker2_style, value[1:], "emarker2", allow_auto_size=False)
        elif key in {"emarkersize", "emsize"}:
            marker_style["size"] = None if _is_empty_placeholder(value) else _parse_positive_number(value, key, "size")
        elif key in {"electcolor", "ecolor"}:
            if not _is_empty_placeholder(value):
                marker_style["color"] = _parse_color_spec(value, key)
        elif key in {"emarkersize1chan", "emarkersizemark"}:
            if not _is_empty_placeholder(value):
                emarkersize1chan = _parse_positive_number(value, key, "size")
        elif key == "markersize":
            legacy_markersize = value

    return {
        "opts": opts,
        "marker_style": marker_style,
        "marker2_style": marker2_style,
        "emarkersize1chan": emarkersize1chan,
        "legacy_markersize": legacy_markersize,
        "electrodes": electrodes_arg,
    }


def _matlab_marker_size_sq(marker: str | tuple[int, int, int], pt: float) -> float:
    """Convert MATLAB ``MarkerSize`` (pt) to Matplotlib ``scatter(..., s=...)`` area.

    MATLAB renders the point marker ``'.'`` at ``MarkerSize / 3`` ignoring
    ``LineWidth``. Matplotlib renders ``'.'`` with diameter ``sqrt(s) / 2``
    when ``linewidths=0``, so ``s = (2 * pt / 3) ** 2`` matches MATLAB's dot
    diameter (~3.24 pt at 10 pt, ~6.60 pt at 20 pt).
    """
    if marker == ".":
        return (2.0 * float(pt) / 3.0) ** 2
    return float(pt) ** 2


def _resolve_emarker_size_sq(marker_style: dict, n_channels: int, legacy_markersize) -> float:
    if marker_style["size"] is not None:
        return _matlab_marker_size_sq(marker_style["marker"], marker_style["size"])
    if legacy_markersize is not None:
        return float(legacy_markersize)
    return _matlab_marker_size_sq(marker_style["marker"], _default_emarkersize(n_channels))


def _draw_markers(
    ax,
    x,
    y,
    *,
    marker: str | tuple[int, int, int],
    color,
    size_sq: float,
    linewidth: float,
    filled: bool,
    zorder: int,
):
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if x_arr.size == 0 or marker == "none" or color == "none":
        return None
    if marker == ".":
        return ax.scatter(
            x_arr,
            y_arr,
            marker=".",
            s=size_sq,
            facecolors=color,
            edgecolors="none",
            linewidths=0.0,
            zorder=zorder,
        )
    if not MarkerStyle(marker).is_filled():
        return ax.scatter(x_arr, y_arr, marker=marker, s=size_sq, color=color, linewidths=linewidth, zorder=zorder)
    facecolors = color if filled else "none"
    return ax.scatter(
        x_arr,
        y_arr,
        marker=marker,
        s=size_sq,
        facecolors=facecolors,
        edgecolors=color,
        linewidths=linewidth,
        zorder=zorder,
    )


def topoplot(datavector, chan_locs, **kwargs):
    """Plot a 2D topographic map of EEG data.

    Parameters
    ----------
    datavector : array-like
        Values to plot at each channel location, or 1-based channel indices to
        mark when ``style='blank'``.
    chan_locs : list of dict
        Channel location structures with 'labels', 'theta', and 'radius' fields.
        ``None`` raises ``ValueError``. An empty list raises ``ValueError``
        for an interpolated map. A blank map with an empty list draws the head.
    kwargs : dict
        Case-insensitive options from an explicit allow-list. Valid EEGLAB
        ``topoplot.m`` options that EEGPrep does not implement yet raise
        ``ValueError`` naming the option and explaining that it is not
        implemented yet; truly unknown keys raise ``ValueError`` naming
        ``topoplot`` and the key. Implemented options:
        ``noplot``, ``plotchans`` (1-based indices), ``electrodes`` (``on``/``pts``,
        ``off``, ``labels``, ``numbers``/``nums``, ``labelpoint``/``ptslabels``,
        ``numpoint``/``ptsnumbers``), ``emarker`` (``marker`` or
        ``{marker, color, size, linewidth}``), ``emarker2``
        (``{chans, marker, color, size, linewidth}``), ``emarkersize``/``emsize``,
        ``electcolor``/``ecolor``, ``emarkersize1chan``/``emarkersizemark``
        (default 20), ``gridscale`` (67), ``intrad``, ``plotrad``, ``method``
        (``'v4'``), ``style`` (``'map'``), ``colormap`` (``'turbo'``),
        ``maplimits``, ``numcontour`` (6), ``colorbar``, ``markersize``,
        ``showlabels``, ``axes``, and ``title``. Compatibility keys accepted
        without effect: ``iclabel``, ``verbose``, ``chaninfo``, ``masksurf``.

    Returns
    -------
    handle : matplotlib.figure.Figure or None
        Figure handle if plotted, None otherwise.
    """
    parsed = _parse_topoplot_options(kwargs)
    opts = parsed["opts"]
    marker_style = parsed["marker_style"]
    marker2_style = parsed["marker2_style"]
    emarkersize1chan = parsed["emarkersize1chan"]
    legacy_markersize = parsed["legacy_markersize"]
    electrodes_arg = parsed["electrodes"]

    if chan_locs is None:
        _require_channel_locations(chan_locs)
    # Set default values
    noplot = opts.get('noplot', 'off')
    plotchans = opts.get('plotchans', [])
    gridscale = opts.get('gridscale', 67)  # Default to 67 (EEGLAB default)
    handle = None
    Zi = None
    rmax = 0.5  # actual head radius
    INTSQUARE = 'on'
    default_intrad = 1
    MAXDEFAULTSHOWLOCS = 64
    intrad = opts.get('intrad', np.nan)
    plotrad = opts.get('plotrad', np.nan)
    squeezefac = 1.0
    ContourVals = datavector
    # MATLAB uses 'v4' (biharmonic spline) by default, which extrapolates
    # more aggressively into regions beyond electrodes (75.9% of 67x67 grid).
    # Python's griddata_v4 now properly filters NaN coordinates before interpolation.
    # This matches MATLAB's coverage and properly interpolates frontal regions.
    # Users can override with method='griddata' for scipy's cubic interpolation (56.2% coverage).
    method = opts.get('method', 'v4')
    style = str(opts.get('style', 'map')).lower()

    if 'noplot' in opts:
        noplot = opts['noplot']
        if not isinstance(noplot, str):
            if len(noplot) != 2:
                raise ValueError("'noplot' location should be [radius, angle]")
            else:
                noplot = 'on'

    # Set colormap
    cmap = plt.get_cmap(opts.get('colormap') or 'turbo')
    GRID_SCALE = gridscale

    datavector = np.array([] if datavector is None else datavector).flatten()
    is_blank = datavector.size == 0 or style == 'blank'
    if is_blank and marker2_style["chans"].size > 0:
        raise ValueError(
            "topoplot: emarker2 not defined for style 'blank' - use marking channel numbers in place of data"
        )

    if electrodes_arg is None:
        ELECTRODES = 'off' if (not is_blank and len(datavector) > MAXDEFAULTSHOWLOCS) else 'on'
    else:
        ELECTRODES = electrodes_arg

    if is_blank:
        return _blank_topoplot(
            chan_locs,
            datavector=datavector,
            noplot=noplot,
            electrodes=ELECTRODES,
            gridscale=gridscale,
            marker_style=marker_style,
            emarkersize1chan=emarkersize1chan,
            legacy_markersize=legacy_markersize,
            axes=opts.get('axes', None),
            title=opts.get('title', 'Channel locations'),
            showlabels=opts.get('showlabels', False),
        )

    _require_channel_locations(chan_locs)

    ContourVals = np.array(ContourVals).flatten()

    tmpeloc = chan_locs
    labels = [loc['labels'] for loc in tmpeloc]
    Th = np.array([_scalar_finite_coord(loc, 'theta') for loc in tmpeloc], dtype=float)
    Rd = np.array([_scalar_finite_coord(loc, 'radius') for loc in tmpeloc], dtype=float)
    # Match readlocs() in topoplot.m:694-716: only channels with finite coordinates are located.
    indices = np.flatnonzero(np.isfinite(Th) & np.isfinite(Rd)).tolist()
    Th = np.deg2rad(Th)
    allchansind = list(range(len(Th)))
    plotchans = _plot_channel_indices(plotchans, indices, len(tmpeloc))
    if len(tmpeloc) == len(datavector) + 1 and plotchans.size and int(plotchans[-1]) == len(tmpeloc) - 1:
        plotchans = plotchans[:-1]
    if len(datavector) > 1 or len(datavector) == len(Th):
        nonfinite = np.flatnonzero(~np.isfinite(datavector))
        plotchans = np.setdiff1d(plotchans, nonfinite)

    x, y = np.cos(Th) * Rd, np.sin(Th) * Rd
    if plotchans.size == 0 or not np.any(np.isfinite(Th[plotchans]) & np.isfinite(Rd[plotchans])):
        raise ValueError(
            "topoplot: no channel has a finite theta and radius. "
            "Import channel locations with Edit > Channel locations before drawing a scalp map."
        )
    plotchans = np.abs(plotchans)
    allchansind = np.array(allchansind)[plotchans]
    Th, Rd, x, y = Th[plotchans], Rd[plotchans], x[plotchans], y[plotchans]
    labels = np.array(labels)[plotchans]
    if len(datavector) > 1 or len(datavector) == len(Th):
        datavector = datavector[plotchans]
        ContourVals = ContourVals[plotchans]

    if np.isnan(plotrad):
        plotrad = min(1.0, max(Rd) * 1.02)
        plotrad = max(plotrad, 0.5)

    if np.isnan(intrad):
        default_intrad = 1
        intrad = min(1.0, max(Rd) * 1.02)
    else:
        default_intrad = 0
        if plotrad > intrad:
            plotrad = intrad

    pltchans = np.where(Rd <= plotrad)[0]
    intchans = np.where((x <= intrad) & (y <= intrad) if INTSQUARE == 'on' else (Rd <= intrad))[0]

    # topoplot.m:913-914 splits pltchans (1-based positions after plotchans filtering).
    mark2_mask = np.zeros(pltchans.size, dtype=bool)
    if marker2_style["chans"].size > 0:
        mark2_set = set(marker2_style["chans"].tolist())
        mark2_mask = np.array([int(idx + 1) in mark2_set for idx in pltchans], dtype=bool)
    mark1_mask = ~mark2_mask

    allchansind = allchansind[pltchans]
    labels = labels[pltchans]

    # topoplot.m:918-923 keeps a one-channel value vector aligned with Th.
    if datavector is not None and len(datavector) == len(Th):
        intdatavector = datavector[intchans]
        datavector = datavector[pltchans]
        ContourVals = ContourVals[pltchans]

    squeezefac = rmax / plotrad
    Rd = Rd * squeezefac
    x, y, intx, inty = (
        x[pltchans] * squeezefac,
        y[pltchans] * squeezefac,
        x[intchans] * squeezefac,
        y[intchans] * squeezefac,
    )

    if default_intrad:
        xmin, xmax = min(-rmax, min(intx)), max(rmax, max(intx))
        ymin, ymax = min(-rmax, min(inty)), max(rmax, max(inty))
    else:
        xmin, xmax, ymin, ymax = -intrad * squeezefac, intrad * squeezefac, -intrad * squeezefac, intrad * squeezefac

    xi, yi = linspace(xmin, xmax, GRID_SCALE), linspace(ymin, ymax, GRID_SCALE)
    yi, xi = np.meshgrid(yi, xi)

    if method == 'griddata':
        Zi = griddata((inty, intx), intdatavector, (yi, xi), method='cubic')
    else:
        coords = np.array([intx.ravel(), inty.ravel()])
        values = intdatavector.ravel()

        nanidx_values = np.where(np.isnan(values))[0]
        nanidx_coords = np.where(np.isnan(coords[0]) | np.isnan(coords[1]))[0]
        nanidx = np.union1d(nanidx_values, nanidx_coords)
        coords = np.delete(coords, nanidx, axis=1)
        values = np.delete(values, nanidx)

        Zi = griddata_v4(coords[0], coords[1], values, xi, yi)

    # Mask outside the head circle (same as MATLAB)
    mask = np.sqrt(xi**2 + yi**2) <= rmax
    Zi[~mask] = np.nan

    if noplot == 'off':
        ax = opts.get('axes', None)
        own_figure = ax is None
        if own_figure:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.figure

        # Screen coords match topo_screen_coords: screen X = sin (left-right), screen Y = cos
        # (front-back). x, y here are already cartesian (cos*Rd, sin*Rd), so this is the swap (y, x).
        # imshow: columns span y over [ymin, ymax]; rows span x over [xmin, xmax], origin='lower'.
        screen_x = y.copy()
        screen_y = x.copy()
        screen_extent = (ymin, ymax, xmin, xmax)

        im = ax.imshow(
            Zi, extent=screen_extent, origin='lower', cmap=cmap, **_maplimits_kwargs(opts.get('maplimits'), Zi)
        )
        # Contour lines
        if np.count_nonzero(np.isfinite(Zi)) > 1 and np.nanmin(Zi) < np.nanmax(Zi):
            grid_x, grid_y = np.meshgrid(
                linspace(screen_extent[0], screen_extent[1], Zi.shape[1]),
                linspace(screen_extent[2], screen_extent[3], Zi.shape[0]),
            )
            levels = _contour_levels(np.nanmin(Zi), np.nanmax(Zi), int(opts.get('numcontour', 6)))
            ax.contour(
                grid_x,
                grid_y,
                Zi,
                levels=levels,
                colors=[(0.2, 0.2, 0.2)],
                linewidths=0.5,
                linestyles='solid',
                zorder=2,
            )
        if opts.get('colorbar', own_figure):
            fig.colorbar(im, ax=ax, shrink=0.7)

        # topoplot.m:1489-1617 draws emarker+emarker2 for on/labelpoint/numpoint,
        # emarker2 only for off, and text only for labels/numbers.
        if ELECTRODES in {'on', 'labelpoint', 'numpoint'}:
            _draw_markers(
                ax,
                screen_x[mark1_mask],
                screen_y[mark1_mask],
                marker=marker_style["marker"],
                color=marker_style["color"],
                size_sq=_resolve_emarker_size_sq(marker_style, screen_x.size, legacy_markersize),
                linewidth=marker_style["linewidth"],
                filled=False,
                zorder=5,
            )
        if ELECTRODES in {'on', 'labelpoint', 'numpoint', 'off'} and np.any(mark2_mask):
            _draw_markers(
                ax,
                screen_x[mark2_mask],
                screen_y[mark2_mask],
                marker=marker2_style["marker"],
                color=marker2_style["color"],
                size_sq=_matlab_marker_size_sq(marker2_style["marker"], marker2_style["size"]),
                linewidth=marker2_style["linewidth"],
                filled=True,
                zorder=6,
            )

        theta_c = linspace(0, 2 * np.pi, 100)
        # white ring hides the jagged color edge at rmax
        ax.plot(
            np.cos(theta_c) * (rmax * 0.99), np.sin(theta_c) * (rmax * 0.99), color='white', linewidth=2.5, zorder=3
        )
        # head drawn at squeezefac*rmax, so it sits inside the color skirt (EEGLAB)
        headrad = squeezefac * rmax
        ax.plot(np.cos(theta_c) * headrad, np.sin(theta_c) * headrad, 'k', linewidth=_HEAD_LINEWIDTH, zorder=4)
        nose_w = 0.08 * squeezefac
        ax.plot(
            [nose_w, 0, -nose_w],
            [headrad, headrad + 0.06 * squeezefac, headrad],
            'k',
            linewidth=_HEAD_LINEWIDTH,
            zorder=4,
        )
        _draw_ears(ax, scale=squeezefac)

        _draw_electrode_labels(
            ax,
            screen_x,
            screen_y,
            labels,
            ELECTRODES,
            showlabels=opts.get('showlabels', False),
            location_indices=allchansind,
            color=marker_style["color"],
        )

        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.65)
        ax.set_aspect('equal')
        ax.axis('off')

        if own_figure:
            ax.set_title(opts.get('title', 'Topoplot'))
            # Display only when we own the figure, like a bare EEGLAB topoplot() call.
            show_figures(fig)

        handle = fig

    return handle, Zi, plotrad, xi, yi


def _blank_selected_markers(datavector: np.ndarray, n_chanlocs: int, location_indices, screen_x, screen_y):
    """Map style='blank' channel numbers or indicator values to (x, y, color) groups."""
    if datavector.size == 0 or location_indices.size == 0:
        return {}
    coords_by_index = {int(idx): (float(sx), float(sy)) for idx, sx, sy in zip(location_indices, screen_x, screen_y)}
    grouped: dict = {}
    if datavector.size < n_chanlocs:
        for raw_val in datavector:
            val = float(raw_val)
            if not np.isfinite(val) or val != np.trunc(val):
                continue
            idx = int(val) - 1
            if idx in coords_by_index:
                grouped.setdefault(_BLANK_COLOR_ARRAY[0], []).append(coords_by_index[idx])
    else:
        for idx, raw_val in enumerate(datavector[:n_chanlocs]):
            val = float(raw_val)
            if not np.isfinite(val) or abs(val) == 0.0 or idx not in coords_by_index:
                continue
            color_idx = max(0, min(int(round(abs(val))) - 1, len(_BLANK_COLOR_ARRAY) - 1))
            grouped.setdefault(_BLANK_COLOR_ARRAY[color_idx], []).append(coords_by_index[idx])
    return grouped


def _blank_topoplot(
    chan_locs,
    *,
    datavector=None,
    noplot='off',
    electrodes='on',
    gridscale=67,
    marker_style=None,
    emarkersize1chan=_DEFAULT_EMARKERSIZE1CHAN,
    legacy_markersize=None,
    axes=None,
    title='Channel locations',
    showlabels=False,
):
    """Draw channel locations without interpolated data."""
    if marker_style is None:
        marker_style = {"marker": ".", "color": mcolors.to_rgba("k"), "size": None, "linewidth": 1.0}
    labels, screen_x, screen_y, location_indices = _channel_location_points(chan_locs)
    xi_1d = np.linspace(-0.5, 0.5, int(gridscale))
    yi_1d = np.linspace(-0.5, 0.5, int(gridscale))
    xi, yi = np.meshgrid(xi_1d, yi_1d)
    # Blank topoplots are decorative channel-location views; Zi is not a
    # numerical interpolation result.
    Zi = np.full_like(xi, np.nan, dtype=float)
    if str(noplot).lower() != 'off':
        return None, Zi, 0.5, xi, yi

    ax = axes
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    _draw_head(ax)
    electrode_mode = _normalize_electrodes(electrodes)
    if electrode_mode in {'on', 'labelpoint', 'numpoint'}:
        size_sq = _resolve_emarker_size_sq(marker_style, screen_x.size, legacy_markersize)
        _draw_markers(
            ax,
            screen_x,
            screen_y,
            marker=marker_style["marker"],
            color=marker_style["color"],
            size_sq=size_sq,
            linewidth=marker_style["linewidth"],
            filled=False,
            zorder=5,
        )
    if datavector is not None and datavector.size > 0:
        selected = _blank_selected_markers(datavector, len(chan_locs), location_indices, screen_x, screen_y)
        selected_size_sq = _matlab_marker_size_sq(marker_style["marker"], emarkersize1chan)
        for color, points in selected.items():
            xs = [pt[0] for pt in points]
            ys = [pt[1] for pt in points]
            _draw_markers(
                ax,
                xs,
                ys,
                marker=marker_style["marker"],
                color=color,
                size_sq=selected_size_sq,
                linewidth=marker_style["linewidth"],
                filled=True,
                zorder=6,
            )
    _draw_electrode_labels(
        ax,
        screen_x,
        screen_y,
        labels,
        electrode_mode,
        showlabels=showlabels,
        location_indices=location_indices,
        color=marker_style["color"],
    )
    if electrode_mode in {'labelpoint', 'numpoint'}:
        ax.set_xlim(-0.85, 0.85)
        ax.set_ylim(-0.95, 0.78)
        shown = int(labels.size)
        total = len(chan_locs)
        ax.text(
            0,
            -0.70,
            f"{shown} of {total} electrode locations shown",
            ha='center',
            va='center',
            fontsize=8,
            clip_on=False,
        )
        ax.text(
            0,
            -0.82,
            "Click on electrodes to toggle name/number",
            ha='center',
            va='center',
            fontsize=8,
            clip_on=False,
        )
        if own_figure:
            fig.canvas.mpl_connect('pick_event', _toggle_electrode_label)
    else:
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.65)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)
    if own_figure:
        show_figures(fig)
    return fig, Zi, 0.5, xi, yi


def topo_screen_coords(theta_deg, radius):
    """Map EEGLAB polar channel coordinates to topoplot screen coordinates.

    EEGLAB draws scalp maps nose-up via ``plot(y, x)``: screen X is the sine
    (left-right) component and screen Y the cosine (front-back) component of a
    channel's ``(theta, radius)``. Routing every scalp renderer through this one
    mapping keeps electrode markers, labels, and interpolated data on the same
    side, with no left-right mirror.

    Args:
        theta_deg: Channel angle in degrees; scalar or array-like.
        radius: Channel radius; scalar or array-like.

    Returns:
        ``(screen_x, screen_y)`` in axes data coordinates.
    """
    theta = np.deg2rad(theta_deg)
    return np.sin(theta) * radius, np.cos(theta) * radius


def _contour_levels(zmin, zmax, numcontour):
    """Return ``numcontour`` interior contour levels between ``zmin`` and ``zmax``."""
    return np.linspace(zmin, zmax, int(numcontour) + 2)[1:-1]


def plot_channel_location(ax, chan_locs, channel_index, *, markersize=24, color="k"):
    """Draw a blank scalp map and mark a single channel's location.

    Mirrors EEGLAB's single-channel ``topoplot(..., 'style', 'blank',
    'emarkersize1chan', ...)`` used by the property and ERP-image plots.
    ``channel_index`` is 1-based; an out-of-range or coordinate-less channel draws
    just the blank head.
    """
    topoplot([], chan_locs, style="blank", electrodes="off", axes=ax, title="")
    if not 1 <= channel_index <= len(chan_locs):
        return
    loc = chan_locs[channel_index - 1]
    try:
        theta_deg = float(loc.get("theta"))
        radius_value = float(loc.get("radius"))
    except (TypeError, ValueError):
        return
    if not (np.isfinite(theta_deg) and np.isfinite(radius_value)):
        return
    screen_x, screen_y = topo_screen_coords(theta_deg, radius_value)
    ax.scatter(screen_x, screen_y, c=color, s=markersize, zorder=6)


def _channel_location_points(chan_locs):
    labels = []
    screen_xs = []
    screen_ys = []
    indices = []
    for index, loc in enumerate(chan_locs):
        if 'theta' not in loc or 'radius' not in loc:
            continue
        try:
            theta_deg = float(np.asarray(loc.get('theta'), dtype=float).reshape(-1)[0])
            radius_value = float(np.asarray(loc.get('radius'), dtype=float).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            continue
        if not np.isfinite(theta_deg) or not np.isfinite(radius_value):
            continue
        labels.append(str(loc.get('labels', index + 1)))
        screen_x, screen_y = topo_screen_coords(theta_deg, radius_value)
        screen_xs.append(screen_x)
        screen_ys.append(screen_y)
        indices.append(index)
    return np.asarray(labels), np.asarray(screen_xs), np.asarray(screen_ys), np.asarray(indices, dtype=int)


# EEGLAB topoplot ear outline (rmax = 0.5); the left ear mirrors these x-coords.
_EAR_X = np.array([0.492, 0.510, 0.518, 0.5299, 0.5419, 0.540, 0.547, 0.532, 0.510, 0.484])
_EAR_Y = np.array([0.0955, 0.1175, 0.1183, 0.1146, 0.0955, -0.0055, -0.0932, -0.1313, -0.1384, -0.1199])
_HEAD_LINEWIDTH = 2.5
_CLIM_MARGIN = 0.05  # EEGLAB expands the color axis by 5% beyond the data limits (topoplot caxis)
_LABEL_OFFSET = 0.01  # text offset beside marker in axes data coords (EEGLAB topoplot.m text(y+0.01, x))


def _draw_ears(ax, scale=1.0):
    """Draw both ear outlines; ``scale`` shrinks them with the head (EEGLAB ``sf``)."""
    ax.plot(_EAR_X * scale, _EAR_Y * scale, 'k', linewidth=_HEAD_LINEWIDTH, zorder=4)
    ax.plot(-_EAR_X * scale, _EAR_Y * scale, 'k', linewidth=_HEAD_LINEWIDTH, zorder=4)


def _draw_head(ax):
    theta_c = linspace(0, 2 * np.pi, 100)
    rmax = 0.5
    ax.plot(np.cos(theta_c) * rmax, np.sin(theta_c) * rmax, 'k', linewidth=_HEAD_LINEWIDTH, zorder=4)
    nose_w = 0.08
    ax.plot([nose_w, 0, -nose_w], [rmax, rmax + 0.06, rmax], 'k', linewidth=_HEAD_LINEWIDTH, zorder=4)
    _draw_ears(ax)


def _toggle_electrode_label(event):
    artist = event.artist
    name = getattr(artist, "_eegprep_electrode_name", None)
    number = getattr(artist, "_eegprep_electrode_number", None)
    if name is None or number is None:
        return
    artist.set_text(number if artist.get_text() == name else name)
    if artist.figure is not None:
        artist.figure.canvas.draw_idle()


def _draw_electrode_labels(ax, x, y, labels, electrodes, *, showlabels=False, location_indices=None, color='k'):
    electrode_mode = _normalize_electrodes(electrodes)
    if location_indices is None:
        location_indices = range(len(labels))
    if electrode_mode in {'numpoint', 'numbers'}:
        text_labels = [str(int(index) + 1) for index in location_indices]
    elif electrode_mode in {'labelpoint', 'labels'} or showlabels:
        text_labels = labels
    else:
        return
    # Offset the text beside the marker dot (EEGLAB text(y+0.01, x), left-aligned) whenever a dot
    # is drawn under it: labelpoint/numpoint, or 'on' with showlabels. 'labels' and 'numbers'
    # have no dot and stay centered.
    if electrode_mode in {'labelpoint', 'numpoint'} or (showlabels and electrode_mode == 'on'):
        x_offset = _LABEL_OFFSET
        halign = 'left'
    else:
        x_offset = 0.0
        halign = 'center'
    text_color = 'k' if color == 'none' else color
    names = [str(label) for label in labels]
    numbers = [str(int(index) + 1) for index in location_indices]
    for x_pos, y_pos, text, name, number in zip(x, y, text_labels, names, numbers):
        artist = ax.annotate(
            text,
            (x_pos + x_offset, y_pos),
            fontsize=7,
            color=text_color,
            ha=halign,
            va='center',
            clip_on=False,
            zorder=7,
        )
        if electrode_mode in {'labelpoint', 'numpoint'}:
            artist.set_picker(5)
            artist._eegprep_electrode_name = name
            artist._eegprep_electrode_number = number


def _maplimits_kwargs(maplimits, data):
    if maplimits is None:
        return {}
    if isinstance(maplimits, str):
        if maplimits.lower() == 'absmax':
            limit = np.nanmax(np.abs(data))
            return _clim_kwargs(-limit, limit) if np.isfinite(limit) and limit > 0 else {}
        if maplimits.lower() == 'maxmin':
            return {}
    values = np.asarray(maplimits, dtype=float).ravel()
    if values.size >= 2 and np.all(np.isfinite(values[:2])):
        return _clim_kwargs(float(values[0]), float(values[1]))
    return {}


def _clim_kwargs(amin, amax):
    # Scale each limit to 1.05*v away from zero, matching EEGLAB topoplot's caxis 5% margin.
    return {
        "vmin": amin + np.sign(amin) * _CLIM_MARGIN * abs(amin),
        "vmax": amax + np.sign(amax) * _CLIM_MARGIN * abs(amax),
    }
