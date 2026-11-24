"""EEG topographic plotting functions."""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from eegprep.functions.miscfunc.linspace import linspace
from eegprep.functions.popfunc.plot_utils import show_figures


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
    must be integers in ``0 .. n_channels - 1`` and are intersected with the
    located channels, matching ``topoplot.m`` lines 703-716.
    """
    located = np.asarray(located, dtype=int)
    if plotchans is None or np.asarray(plotchans).size == 0:
        return located
    requested = np.asarray(plotchans, dtype=float).reshape(-1)
    if not np.all(np.isfinite(requested)) or np.any(requested != np.trunc(requested)):
        raise ValueError(
            "topoplot: plotchans must be integer 0-based channel indices. "
            f"Got {np.asarray(plotchans).reshape(-1).tolist()}. "
            "Pass whole indices such as [0, 2, 5]."
        )
    requested = requested.astype(int)
    if np.any(requested < 0) or np.any(requested >= n_channels):
        raise ValueError(
            f"topoplot: plotchans must be 0-based indices from 0 through {n_channels - 1}. "
            f"Got {requested.tolist()}. Pass the channel indices to include, for example [0, 2]."
        )
    return np.intersect1d(requested, located)


def _require_channel_locations(chan_locs) -> None:
    if chan_locs is None or len(chan_locs) == 0:
        raise ValueError(
            "topoplot: Channel locations are required. "
            "Load locations with pop_chanedit or pop_readlocs before plotting."
        )


def topoplot(datavector, chan_locs, **kwargs):
    """Plot a 2D topographic map of EEG data.

    Parameters
    ----------
    datavector : array-like
        Values to plot at each channel location.
    chan_locs : list of dict
        Channel location structures with 'labels', 'theta', and 'radius' fields.
        ``None`` raises ``ValueError``. An empty list raises ``ValueError``
        for an interpolated map. A blank map with an empty list draws the head.
    kwargs : dict
        Additional keyword arguments for customization:

        - noplot : str or tuple, default 'off'
        - plotgrid : str, default 'off'
        - plotchans : sequence of int, default []
          0-based channel indices to include. Empty draws every channel that
          has ``theta``. A single index is applied the same way. EEGLAB's
          ``plotchans`` is 1-based. One remaining location raises
          ``ValueError`` from ``griddata_v4``. EEGLAB ``topoplot`` follows
          MATLAB ``griddata`` and returns a non-finite map instead.
        - ELECTRODES : str, default 'on'
        - intrad : float, default nan
        - plotrad : float, default nan
        - headrad : float, default 0.5
        - method : str, default 'rbf'

    Returns
    -------
    handle : matplotlib.figure.Figure or None
        Figure handle if plotted, None otherwise.
    """
    if chan_locs is None:
        _require_channel_locations(chan_locs)
    # Set default values
    noplot = kwargs.get('noplot', 'off')
    plotchans = kwargs.get('plotchans', [])
    gridscale = kwargs.get('gridscale', 67)  # Default to 67 (EEGLAB default)
    handle = None
    Zi = None
    rmax = 0.5  # actual head radius
    INTSQUARE = 'on'
    default_intrad = 1
    ELECTRODES = kwargs.get('electrodes', kwargs.get('ELECTRODES', 'on'))
    MAXDEFAULTSHOWLOCS = 64
    intrad = kwargs.get('intrad', np.nan)
    plotrad = kwargs.get('plotrad', np.nan)
    squeezefac = 1.0
    ContourVals = datavector
    # MATLAB uses 'v4' (biharmonic spline) by default, which extrapolates
    # more aggressively into regions beyond electrodes (75.9% of 67x67 grid).
    # Python's griddata_v4 now properly filters NaN coordinates before interpolation.
    # This matches MATLAB's coverage and properly interpolates frontal regions.
    # Users can override with method='griddata' for scipy's cubic interpolation (56.2% coverage).
    method = kwargs.get('method', 'v4')
    style = str(kwargs.get('style', 'map')).lower()

    if 'noplot' in kwargs:
        noplot = kwargs['noplot']
        if not isinstance(noplot, str):
            if len(noplot) != 2:
                raise ValueError("'noplot' location should be [radius, angle]")
            else:
                noplot = 'on'

    # Set colormap
    cmap = plt.get_cmap(kwargs.get('colormap') or 'turbo')
    GRID_SCALE = gridscale

    datavector = np.array([] if datavector is None else datavector).flatten()
    if datavector.size == 0 or style == 'blank':
        blank_kwargs = dict(kwargs)
        blank_kwargs.pop('noplot', None)
        blank_kwargs.pop('electrodes', None)
        blank_kwargs.pop('ELECTRODES', None)
        return _blank_topoplot(chan_locs, noplot=noplot, electrodes=ELECTRODES, gridscale=gridscale, **blank_kwargs)

    _require_channel_locations(chan_locs)
    if len(datavector) > MAXDEFAULTSHOWLOCS and str(ELECTRODES).lower() == 'on':
        ELECTRODES = 'off'
    elif str(ELECTRODES).lower() == 'on':
        ELECTRODES = 'on'

    ContourVals = np.array(ContourVals).flatten()

    tmpeloc = chan_locs
    labels = [loc['labels'] for loc in tmpeloc]
    indices = [i for i, loc in enumerate(tmpeloc) if 'theta' in loc]
    Th = np.array(
        [
            tmpeloc[i]['theta'] if i in indices and not isinstance(tmpeloc[i]['theta'], np.ndarray) else np.nan
            for i in range(len(tmpeloc))
        ]
    )
    Rd = np.array(
        [
            tmpeloc[i]['radius'] if i in indices and not isinstance(tmpeloc[i]['radius'], np.ndarray) else np.nan
            for i in range(len(tmpeloc))
        ]
    )
    Th = np.deg2rad(Th)
    allchansind = list(range(len(Th)))
    # topoplot.m:703-781, with plotchans 0-based.
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
        ax = kwargs.get('axes', None)
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
            Zi, extent=screen_extent, origin='lower', cmap=cmap, **_maplimits_kwargs(kwargs.get('maplimits'), Zi)
        )
        # Contour lines
        if np.count_nonzero(np.isfinite(Zi)) > 1 and np.nanmin(Zi) < np.nanmax(Zi):
            grid_x, grid_y = np.meshgrid(
                linspace(screen_extent[0], screen_extent[1], Zi.shape[1]),
                linspace(screen_extent[2], screen_extent[3], Zi.shape[0]),
            )
            levels = _contour_levels(np.nanmin(Zi), np.nanmax(Zi), int(kwargs.get('numcontour', 6)))
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
        if kwargs.get('colorbar', own_figure):
            fig.colorbar(im, ax=ax, shrink=0.7)

        markersize = kwargs.get('markersize', 6)
        if str(ELECTRODES).lower() == 'on':
            ax.scatter(screen_x, screen_y, c='k', s=markersize, zorder=5)
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

        _draw_electrode_labels(ax, screen_x, screen_y, labels, ELECTRODES, showlabels=kwargs.get('showlabels', False))

        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.65)
        ax.set_aspect('equal')
        ax.axis('off')

        if own_figure:
            ax.set_title('Topoplot')
            # Display only when we own the figure, like a bare EEGLAB topoplot() call.
            show_figures(fig)

        handle = fig

    return handle, Zi, plotrad, xi, yi


def _blank_topoplot(chan_locs, *, noplot='off', electrodes='on', gridscale=67, **kwargs):
    """Draw channel locations without interpolated data."""
    labels, screen_x, screen_y, location_indices = _channel_location_points(chan_locs)
    xi_1d = np.linspace(-0.5, 0.5, int(gridscale))
    yi_1d = np.linspace(-0.5, 0.5, int(gridscale))
    xi, yi = np.meshgrid(xi_1d, yi_1d)
    # Blank topoplots are decorative channel-location views; Zi is not a
    # numerical interpolation result.
    Zi = np.full_like(xi, np.nan, dtype=float)
    if str(noplot).lower() != 'off':
        return None, Zi, 0.5, xi, yi

    ax = kwargs.get('axes', None)
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    _draw_head(ax)
    markersize = kwargs.get('markersize', 6)
    electrode_mode = str(electrodes).lower()
    if electrode_mode in {'on', 'labelpoint', 'numpoint'}:
        ax.scatter(screen_x, screen_y, c='k', s=markersize, zorder=5)
    _draw_electrode_labels(ax, screen_x, screen_y, labels, electrodes, location_indices=location_indices)
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
    ax.set_title(kwargs.get('title', 'Channel locations'))
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


def _draw_electrode_labels(ax, x, y, labels, electrodes, *, showlabels=False, location_indices=None):
    electrode_mode = str(electrodes).lower()
    if location_indices is None:
        location_indices = range(len(labels))
    if electrode_mode == 'numpoint':
        text_labels = [str(int(index) + 1) for index in location_indices]
    elif electrode_mode in {'labelpoint', 'labels'} or showlabels:
        text_labels = labels
    else:
        return
    # Offset the text beside the marker dot (EEGLAB text(y+0.01, x), left-aligned) whenever a dot
    # is drawn under it: labelpoint/numpoint, or 'on' with showlabels. 'labels' has no dot and
    # stays centered.
    if electrode_mode in {'labelpoint', 'numpoint'} or (showlabels and electrode_mode == 'on'):
        x_offset = _LABEL_OFFSET
        halign = 'left'
    else:
        x_offset = 0.0
        halign = 'center'
    names = [str(label) for label in labels]
    numbers = [str(int(index) + 1) for index in location_indices]
    for x_pos, y_pos, text, name, number in zip(x, y, text_labels, names, numbers):
        artist = ax.annotate(text, (x_pos + x_offset, y_pos), fontsize=7, ha=halign, va='center', clip_on=False)
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
