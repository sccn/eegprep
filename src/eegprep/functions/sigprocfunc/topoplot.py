"""EEG topographic plotting functions."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def griddata_v4(x, y, v, xq, yq):
    """Python version of MATLAB's GDATAV4 interpolation based on David T. Sandwell's biharmonic spline interpolation.

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
    # Combine x and y into complex numbers for convenience
    xy = x + 1j * y

    # Determine distances between points
    d = np.abs(xy[:, None] - xy[None, :])

    # Determine weights for interpolation
    with np.errstate(divide='ignore', invalid='ignore'):
        g = (d**2) * (np.log(d) - 1)  # Green's function
    np.fill_diagonal(g, 0)  # Fix value along diagonal

    # Add Tikhonov regularization to handle ill-conditioned matrices
    # This prevents numerical issues when electrodes are very close together
    n = len(v)
    regularization = 1e-8 * np.trace(g) / n * np.eye(n)
    g_reg = g + regularization

    try:
        weights = np.linalg.solve(g_reg, v)
    except np.linalg.LinAlgError:
        # If still singular, use pseudoinverse as last resort
        weights = np.linalg.pinv(g_reg) @ v

    # Vectorized evaluation at query points:
    # 1. Flatten the coordinates to form a 1D complex array of query locations
    xy_query_flat = (xq + 1j * yq).ravel()
    xy_flat = xy.ravel()

    # 2. Compute pairwise distances using broadcasting
    d_q = np.abs(xy_query_flat[:, None] - xy_flat[None, :])

    # 3. Compute Green's function values
    with np.errstate(divide='ignore', invalid='ignore'):
        g_q = (d_q**2) * (np.log(d_q) - 1)
    g_q[d_q == 0] = 0

    # 4. Perform matrix multiplication and reshape back to original grid shape
    vq = (g_q @ weights).reshape(xq.shape)

    return vq


def topoplot(datavector, chan_locs, **kwargs):
    """Plot a 2D topographic map of EEG data.

    Parameters
    ----------
    datavector : array-like
        Values to plot at each channel location.
    chan_locs : list of dict
        Channel location structures with 'labels', 'theta', and 'radius' fields.
    kwargs : dict
        Additional keyword arguments for customization:

        - noplot : str or tuple, default 'off'
        - plotgrid : str, default 'off'
        - plotchans : list, default []
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

    # print method
    # print(f'method = {method}')

    # Handle additional arguments
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

    if len(datavector) > MAXDEFAULTSHOWLOCS and str(ELECTRODES).lower() == 'on':
        ELECTRODES = 'off'
    elif str(ELECTRODES).lower() == 'on':
        ELECTRODES = 'on'

    ContourVals = np.array(ContourVals).flatten()

    # Read the channel location information
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
    plotchans = indices

    if len(tmpeloc) == len(datavector) + 1:
        if plotchans and plotchans[-1] == len(tmpeloc):
            plotchans.pop()

    if len(datavector) > 1:
        inds = np.union1d(np.where(np.isnan(datavector))[0], np.where(np.isinf(datavector))[0])
        plotchans = list(set(plotchans) - set(inds))

    x, y = np.cos(Th) * Rd, np.sin(Th) * Rd
    plotchans = np.abs(plotchans)
    allchansind = np.array(allchansind)[plotchans]
    Th, Rd, x, y = Th[plotchans], Rd[plotchans], x[plotchans], y[plotchans]
    labels = np.array(labels)[plotchans]

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

    if datavector is not None and len(datavector) > 1:
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

    xi, yi = np.linspace(xmin, xmax, GRID_SCALE), np.linspace(ymin, ymax, GRID_SCALE)
    yi, xi = np.meshgrid(yi, xi)

    if method == 'griddata':
        Zi = griddata((inty, intx), intdatavector, (yi, xi), method='cubic')
    else:
        coords = np.array([intx.ravel(), inty.ravel()])
        values = intdatavector.ravel()

        # find nan in values OR coordinates and remove them
        nanidx_values = np.where(np.isnan(values))[0]
        nanidx_coords = np.where(np.isnan(coords[0]) | np.isnan(coords[1]))[0]
        nanidx = np.union1d(nanidx_values, nanidx_coords)
        coords = np.delete(coords, nanidx, axis=1)
        values = np.delete(values, nanidx)

        # use griddata_v4 to interpolate
        Zi = griddata_v4(coords[0], coords[1], values, xi, yi)

        # Create the RBF interpolator

        # rbf = Rbf(coords[0], coords[1], values, function='inverse') #, function='linear')
        # Zi1 = rbf(xi, yi)

        # rbf = Rbf(coords[0], coords[1], values, function='thin_plate') #, function='linear')
        # Zi2 = rbf(xi, yi)

        # # average the two results
        # Zi = (Zi1 + Zi2) / 2

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

        # Rotate electrode positions by -90 degrees: (x, y) -> (y, -x)
        x_rotated = -y.copy()
        y_rotated = x.copy()
        extent_rotated = (ymin, ymax, -xmax, -xmin)

        im = ax.imshow(
            Zi, extent=extent_rotated, origin='lower', cmap=cmap, **_maplimits_kwargs(kwargs.get('maplimits'), Zi)
        )
        # Contour lines
        if np.count_nonzero(np.isfinite(Zi)) > 1 and np.nanmin(Zi) < np.nanmax(Zi):
            grid_x, grid_y = np.meshgrid(
                np.linspace(extent_rotated[0], extent_rotated[1], Zi.shape[1]),
                np.linspace(extent_rotated[2], extent_rotated[3], Zi.shape[0]),
            )
            levels = np.linspace(np.nanmin(Zi), np.nanmax(Zi), 8)[1:-1]
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
            ax.scatter(x_rotated, y_rotated, c='k', s=markersize, zorder=5)
        theta_c = np.linspace(0, 2 * np.pi, 100)
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

        _draw_electrode_labels(ax, x_rotated, y_rotated, labels, ELECTRODES, showlabels=kwargs.get('showlabels', False))

        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.65)
        ax.set_aspect('equal')
        ax.axis('off')

        if own_figure:
            ax.set_title('Topoplot')
            _show_if_interactive()

        handle = fig

    return handle, Zi, plotrad, xi, yi


def _blank_topoplot(chan_locs, *, noplot='off', electrodes='on', gridscale=67, **kwargs):
    """Draw channel locations without interpolated data."""
    labels, x_rotated, y_rotated = _channel_location_points(chan_locs)
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
        ax.scatter(x_rotated, y_rotated, c='k', s=markersize, zorder=5)
    _draw_electrode_labels(ax, x_rotated, y_rotated, labels, electrodes)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.65)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(kwargs.get('title', 'Channel locations'))
    if own_figure:
        _show_if_interactive()
    return fig, Zi, 0.5, xi, yi


def _channel_location_points(chan_locs):
    labels = []
    xs = []
    ys = []
    for index, loc in enumerate(chan_locs):
        if 'theta' not in loc or 'radius' not in loc:
            continue
        try:
            theta_rad = np.deg2rad(float(loc.get('theta')))
            radius_value = float(loc.get('radius'))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(theta_rad) or not np.isfinite(radius_value):
            continue
        labels.append(str(loc.get('labels', index + 1)))
        x = np.cos(theta_rad) * radius_value
        y = np.sin(theta_rad) * radius_value
        xs.append(-y)
        ys.append(x)
    return np.asarray(labels), np.asarray(xs), np.asarray(ys)


# EEGLAB topoplot ear outline (rmax = 0.5); the left ear mirrors these x-coords.
_EAR_X = np.array([0.492, 0.510, 0.518, 0.5299, 0.5419, 0.540, 0.547, 0.532, 0.510, 0.484])
_EAR_Y = np.array([0.0955, 0.1175, 0.1183, 0.1146, 0.0955, -0.0055, -0.0932, -0.1313, -0.1384, -0.1199])
_HEAD_LINEWIDTH = 2.5


def _draw_ears(ax, scale=1.0):
    """Draw both ear outlines; ``scale`` shrinks them with the head (EEGLAB ``sf``)."""
    ax.plot(_EAR_X * scale, _EAR_Y * scale, 'k', linewidth=_HEAD_LINEWIDTH, zorder=4)
    ax.plot(-_EAR_X * scale, _EAR_Y * scale, 'k', linewidth=_HEAD_LINEWIDTH, zorder=4)


def _draw_head(ax):
    theta_c = np.linspace(0, 2 * np.pi, 100)
    rmax = 0.5
    ax.plot(np.cos(theta_c) * rmax, np.sin(theta_c) * rmax, 'k', linewidth=_HEAD_LINEWIDTH, zorder=4)
    nose_w = 0.08
    ax.plot([nose_w, 0, -nose_w], [rmax, rmax + 0.06, rmax], 'k', linewidth=_HEAD_LINEWIDTH, zorder=4)
    _draw_ears(ax)


def _draw_electrode_labels(ax, x, y, labels, electrodes, *, showlabels=False):
    electrode_mode = str(electrodes).lower()
    if electrode_mode == 'numpoint':
        text_labels = [str(index + 1) for index in range(len(labels))]
    elif electrode_mode in {'labelpoint', 'labels'} or showlabels:
        text_labels = labels
    else:
        return
    for x_pos, y_pos, text in zip(x, y, text_labels):
        ax.annotate(text, (x_pos, y_pos), fontsize=7, ha='center', va='center')


def _show_if_interactive():
    if 'agg' not in plt.get_backend().lower():
        plt.show()


def _maplimits_kwargs(maplimits, data):
    if maplimits is None:
        return {}
    if isinstance(maplimits, str):
        if maplimits.lower() == 'absmax':
            limit = np.nanmax(np.abs(data))
            return {"vmin": -limit, "vmax": limit} if np.isfinite(limit) and limit > 0 else {}
        if maplimits.lower() == 'maxmin':
            return {}
    values = np.asarray(maplimits, dtype=float).ravel()
    if values.size >= 2 and np.all(np.isfinite(values[:2])):
        return {"vmin": float(values[0]), "vmax": float(values[1])}
    return {}
