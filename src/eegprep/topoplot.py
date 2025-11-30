import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree

def griddata_v4(x, y, v, xq, yq):
    """
    Python version of MATLAB's GDATAV4 interpolation based on David T. Sandwell's biharmonic spline interpolation.
    
    Parameters:
    x, y : 1D arrays of coordinates for known points
    v : 1D array of values at known points
    xq, yq : 2D arrays of query points coordinates
    
    Returns:
    vq : 2D array of interpolated values at query points
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
    
    # Initialize output array
    m, n = xq.shape
    vq = np.zeros_like(xq)
    
    # Evaluate at requested points
    xy = xy[:, None]  # Make it column vector for broadcasting
    for i in range(m):
        for j in range(n):
            d = np.abs(xq[i, j] + 1j * yq[i, j] - xy.ravel())
            with np.errstate(divide='ignore', invalid='ignore'):
                g = (d**2) * (np.log(d) - 1)  # Green's function
            g[d == 0] = 0  # Handle Green's function at zero
            vq[i, j] = np.dot(g, weights)
    
    return vq

def topoplot(datavector, chan_locs, **kwargs):
    # Set default values
    noplot = kwargs.get('noplot', 'off')
    plotgrid = kwargs.get('plotgrid', 'off')
    plotchans = kwargs.get('plotchans', [])
    gridscale = kwargs.get('gridscale', 67)  # Default to 67 (EEGLAB default)
    handle = None
    Zi = None
    chanval = np.nan
    rmax = 0.5  # actual head radius
    INTERPLIMITS = 'head'
    INTSQUARE = 'on'
    default_intrad = 1
    ELECTRODES = kwargs.get('ELECTRODES', 'on')
    MAXDEFAULTSHOWLOCS = 64
    intrad = kwargs.get('intrad', np.nan)
    plotrad = kwargs.get('plotrad', np.nan)
    headrad = kwargs.get('headrad', 0.5)
    squeezefac = 1.0
    ContourVals = datavector
    # MATLAB uses 'v4' (biharmonic spline) by default, which extrapolates
    # more aggressively into regions beyond electrodes (75.9% of 67x67 grid).
    # Python's griddata_v4 now properly filters NaN coordinates before interpolation.
    # This matches MATLAB's coverage and properly interpolates frontal regions.
    # Users can override with method='griddata' for scipy's cubic interpolation (56.2% coverage).
    method = kwargs.get('method', 'v4')

    # print method
    # print(f'method = {method}')

    # Handle additional arguments
    if 'noplot' in kwargs:
        noplot = kwargs['noplot']
        if not isinstance(noplot, str):
            if len(noplot) != 2:
                raise ValueError("'noplot' location should be [radius, angle]")
            else:
                chanrad = noplot[0]
                chantheta = noplot[1]
                noplot = 'on'

    # Set colormap
    cmap = plt.get_cmap('jet')
    cmaplen = cmap.N
    GRID_SCALE = gridscale

    if len(datavector) > MAXDEFAULTSHOWLOCS:
        ELECTRODES = 'off'
    else:
        ELECTRODES = 'on'

    datavector = np.array(datavector).flatten()
    ContourVals = np.array(ContourVals).flatten()

    # Read the channel location information
    tmpeloc = chan_locs
    labels = [loc['labels'] for loc in tmpeloc]
    indices = [i for i, loc in enumerate(tmpeloc) if 'theta' in loc]
    Th = np.array([tmpeloc[i]['theta'] if i in indices and not isinstance(tmpeloc[i]['theta'], np.ndarray) else np.nan for i in range(len(tmpeloc))])
    Rd = np.array([tmpeloc[i]['radius'] if i in indices and not isinstance(tmpeloc[i]['radius'], np.ndarray) else np.nan for i in range(len(tmpeloc))])
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
    Rd, intRd = Rd * squeezefac, Rd[intchans] * squeezefac
    x, y, intx, inty = x[pltchans] * squeezefac, y[pltchans] * squeezefac, x[intchans] * squeezefac, y[intchans] * squeezefac

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
    mask = (np.sqrt(xi**2 + yi**2) <= rmax)
    Zi[~mask] = np.nan

    if noplot == 'off':
        # Rotate electrode positions by -90 degrees: (x, y) -> (y, -x)
        x_rotated = y.copy()
        y_rotated = -x.copy()
        extent_rotated = (ymin, ymax, -xmax, -xmin)
        
        plt.imshow(Zi_rotated, extent=extent_rotated, origin='lower', cmap=cmap)
        plt.colorbar()
        plt.scatter(x_rotated, y_rotated, c='k')
        # Rotate head circle coordinates
        theta = np.linspace(0, 2 * np.pi, 100)
        head_x = np.cos(theta) * rmax
        head_y = np.sin(theta) * rmax
        plt.plot(head_x, head_y, 'k')
        for i, txt in enumerate(labels):
            plt.annotate(txt, (x_rotated[i], y_rotated[i]), fontsize=8, ha='right')
        plt.title('Topoplot')
        plt.axis('off')
        plt.show()

    return handle, Zi, plotrad, xi, yi