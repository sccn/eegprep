"""EEGLAB-style EEG resampling pop function."""

import math
from math import ceil, floor, gcd

import numpy as np
import sympy as sp
from scipy import signal
from scipy.signal import resample, resample_poly
from scipy.signal.windows import kaiser

from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.miscfunc.misc import aslist
from eegprep.plugins.firfilt import firws, firwsord

# TO DO TO ADDRESS DIFFERENCES BETWEEN MATLAB AND PYTHON
# - Do a simple resample 500 to 250 Hz, there only the filter should matter (subsampling is just a decimation)
# - Check the filter result in MATLAB and Python
# - Check the options of the resample function in MATLAB and Python
# - Try the pyresample package
# - Check for boundary effects in MATLAB and Python (different padding)


def pop_resample(
    EEG,
    freq=None,
    engine=None,
    *,
    gui=None,
    renderer=None,
    return_com=False,
    fc=None,
    df=None,
):
    """Resample EEG data to a new sampling rate.

    Parameters
    ----------
    EEG : dict
        EEGLAB EEG structure.
    freq : float
        New sampling rate in Hz.
    engine : str or None
        Engine to use for implementation. Options are:
        - None: Use the default Python implementation
        - 'poly': Use scipy's resample_poly function
        - 'matlab': Use MATLAB engine
        - 'octave': Use Octave engine

    Returns
    -------
    EEG : dict
        EEGLAB EEG structure with resampled data.
    """
    if EEG is None:
        return (None, "") if return_com else None
    if gui is None:
        gui = freq is None
    if gui:
        result = _run_gui(EEG[0] if isinstance(EEG, list) else EEG, renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        freq = result["freq"]
    if freq is None:
        raise ValueError("freq argument is required when gui=False")
    freq = float(freq)
    if freq <= 0:
        raise ValueError("New sampling rate must be positive")
    fc = 0.9 if fc is None else fc
    df = 0.2 if df is None else df

    if isinstance(EEG, list):
        output = [
            pop_resample(item, freq, engine=engine, gui=False, fc=fc, df=df)
            for item in EEG
        ]
        command = _history_command(freq)
        return (output, command) if return_com else output

    # Check if using MATLAB or Octave implementation
    if engine in ['matlab', 'octave']:
        eeglab = get_eeglab(runtime='MAT' if engine == 'matlab' else 'OCT')
        EEG_new = eeglab.pop_resample(EEG, freq)
        command = _history_command(freq)
        return (EEG_new, command) if return_com else EEG_new

    # Default Python implementation
    else:
        if engine is None:
            # use the resample_eeg function
            EEG_new = resample_eeg(EEG, freq, method='poly', fc=fc, df=df)

        elif engine == 'poly':
            # use the resample_poly function
            EEG_new = resample_eeg(EEG, freq, method='poly', fc=fc, df=df)

        elif engine == 'scipy':
            # Calculate the new number of points
            # Resample the data

            # Create a copy of the EEG structure
            EEG_new = EEG.copy()
            old_srate = EEG['srate']
            old_pnts = EEG['pnts']
            new_pnts = int(old_pnts * freq / old_srate)

            if 'data' in EEG:
                EEG_new['data'] = resample(EEG['data'].astype(np.float64), new_pnts, axis=1).astype(np.float32)

        else:
            raise ValueError(f"Unsupported engine: {engine}. Should be None, 'matlab', or 'octave'")

        # Update EEG structure
        new_pnts = EEG_new['data'].shape[1]
        EEG_new['pnts'] = new_pnts
        EEG_new['srate'] = freq

        # Update xmin and xmax if present
        if 'xmin' in EEG and 'xmax' in EEG:
            duration = EEG['xmax'] - EEG['xmin']
            EEG_new['xmin'] = EEG['xmin']
            EEG_new['xmax'] = EEG['xmin'] + (EEG_new['pnts']-1)/EEG_new['srate'] # was: EEG['xmin'] + duration

        # Update times if present
        EEG_new['times'] = np.linspace(EEG_new['xmin']*1000, EEG_new['xmax']*1000, new_pnts)

        # Update event/urevent latencies if present
        orig_ratio = freq / EEG['srate']
        rational_approx = sp.nsimplify(orig_ratio, tolerance=1e-12)
        p, q = rational_approx.as_numer_denom()
        ratio = float(p/q)

        for event in aslist(EEG_new.get('event', [])) + aslist(EEG_new.get('urevent', [])):
            if isinstance(event, dict) and 'latency' in event:
                event['latency'] = np.clip((event['latency']-1) * ratio + 1, 1, new_pnts)

        command = _history_command(freq)
        return (EEG_new, command) if return_com else EEG_new


def pop_resample_dialog_spec(srate) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_resample``."""
    return DialogSpec(
        title="Resample current dataset -- pop_resample()",
        function_name="pop_resample",
        eeglab_source="functions/popfunc/pop_resample.m",
        geometry=((1,), (1,)),
        size=(300, 199),
        help_text="pophelp('pop_resample')",
        controls=(
            ControlSpec("text", "New sampling rate"),
            ControlSpec(
                "edit",
                tag="freq",
                value=f"{float(srate):g}",
                callback=CallbackSpec("validate_numeric_range", params={"columns": 1, "lower": 0, "upper": np.inf}),
            ),
        ),
    )


def _run_gui(EEG, renderer=None):
    spec = pop_resample_dialog_spec(EEG.get("srate", 1))
    result = inputgui(spec, renderer=renderer)
    if result is None:
        return None
    text = str(result.get("freq", "")).strip()
    if not text:
        return None
    return {"freq": float(text)}


def _history_command(freq):
    return f"EEG = pop_resample( EEG, {_format_number(freq)});"


def _format_number(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"

def resample_eeg(EEG, freq, method='poly', fc=0.9, df=0.2):
    """Port of EEGLAB's pop_resample behavior.

    This currently supports only filtering of continuous / gap-free data.

    Parameters
    ----------
    EEG : dict
        EEGLAB EEG structure.
    freq : float
        New sampling rate in Hz.
    method : str
        Resampling method. Options are 'poly' or 'octave'.
    fc : float
        Anti-aliasing filter cutoff frequency.
    df : float
        Transition width of the filter.

    Returns
    -------
    EEG : dict
        EEGLAB EEG structure with resampled data.
    """
    assert 0 <= fc <= 1, "Anti-aliasing filter cutoff frequency out of range"

    # Calculate the ratio
    ratio = freq / EEG['srate']
    rational_approx = sp.nsimplify(ratio, tolerance=1e-12)
    p, q = rational_approx.as_numer_denom()
    p = int(p)
    q = int(q)

    # Prepare new data
    EEG_new = EEG.copy()
    EEG_new['data'] = np.zeros((EEG['nbchan'], int(EEG['pnts'] * p / q)))

    if method == 'poly':
        # use scipy's resample_poly() function
        nyq = 1 / np.maximum(p, q)
        fc *= nyq
        df *= nyq

        # determine filter order
        m, _ = firwsord('kaiser', 2, df, 0.002)

        # design windowed-sinc filter
        wnd = kaiser(m + 1, beta=5)
        b, _ = firws(m, fc, w=wnd)

        nPad = int(np.ceil((m / 2) / q) * q)
        # constant-pad the data along axis=1
        tmpdata = np.pad(EEG['data'], ((0, 0), (nPad, nPad)), mode='edge').astype(np.float64)
        tmpdata = resample_poly(tmpdata, p, q, axis=1, window=b).astype(np.float32)
        nPadAfter = nPad * p // q
        # remove the padding and write back
        EEG_new['data'] = tmpdata[:, nPadAfter:-nPadAfter]
    elif method == 'octave':
        # use port from octave:
        for i in range(EEG['nbchan']):
            tmp, h = resample_raw(EEG['data'][i, :].flatten().astype(np.float64), p, q)
            EEG_new['data'][i, :] = tmp.astype(np.float32)
    else:
        raise ValueError(f"Unsupported method: {method}. Should be 'poly' or 'octave', but got {method}")

    return EEG_new

def upfirdn_raw(x, h, p, q):
    """Upfirdn implementation for resampling.

    Parameters
    ----------
    x : array_like
        Input signal.
    h : array_like
        Filter coefficients.
    p : int
        Upsampling factor.
    q : int
        Downsampling factor.

    Returns
    -------
    y : ndarray
        Filtered and resampled signal.
    """
    # Ensure x is a numpy array and h is 1D.
    x = np.array(x, copy=True)
    h = np.array(h).flatten()

    # If x is a row vector, convert it to a column vector.
    is_row_vector = False
    if x.ndim == 2 and x.shape[0] == 1 and x.shape[1] > 1:
        x = x.T
        is_row_vector = True

    rx, cx = x.shape
    Lh = h.size
    Ly = math.ceil(((rx - 1) * p + Lh) / q)
    y = np.zeros((Ly, cx))

    for c in range(cx):
        for m in range(Ly):
            n = (m * q) // p
            lm = (m * q) % p
            # k goes from max(0, n - rx + 1) to n (inclusive)
            for k in range(max(0, n - rx + 1), n + 1):
                if k * p + lm < Lh:
                    y[m, c] += h[k * p + lm] * x[n - k, c]

    if is_row_vector:
        y = y.T

    return y

def resample_raw(x, p, q, h=None):
    """Change the sample rate of x by a factor of p/q.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    p : int
        The upsampling factor.
    q : int
        The downsampling factor.
    h : array_like, optional
        The filter coefficients. If not provided, a Kaiser-windowed sinc filter is used.

    Returns
    -------
    y : ndarray
        The resampled array. If input is a vector, output will be a vector.
    h : ndarray
        The filter coefficients used.
    """
    # Input validation
    if not isinstance(p, (int, np.integer)) or not isinstance(q, (int, np.integer)):
        raise ValueError("p and q must be positive integers")
    if p <= 0 or q <= 0:
        raise ValueError("p and q must be positive integers")

    # Convert x to numpy array and handle row vectors
    x = np.asarray(x)
    input_shape = x.shape
    is_1d = x.ndim == 1

    # Reshape input to 2D array with shape (samples, channels)
    if is_1d:
        x = x.reshape(-1, 1)
    elif x.ndim == 2 and x.shape[0] == 1:
        x = x.T

    # Simplify decimation and interpolation factors
    great_common_divisor = gcd(p, q)
    if great_common_divisor > 1:
        p = p // great_common_divisor
        q = q // great_common_divisor

    # Filter design if required
    if h is None:
        # Properties of the antialiasing filter
        log10_rejection = -3.0
        stopband_cutoff_f = 1.0 / (2.0 * max(p, q))
        roll_off_width = stopband_cutoff_f / 10.0

        # Determine filter length
        rejection_dB = -20.0 * log10_rejection
        L = ceil((rejection_dB - 8.0) / (28.714 * roll_off_width))

        # Ideal sinc filter
        t = np.arange(-L, L + 1)
        ideal_filter = 2 * p * stopband_cutoff_f * np.sinc(2 * stopband_cutoff_f * t)

        # Determine parameter of Kaiser window
        if 21 <= rejection_dB <= 50:
            beta = 0.5842 * (rejection_dB - 21.0)**0.4 + 0.07886 * (rejection_dB - 21.0)
        elif rejection_dB > 50:
            beta = 0.1102 * (rejection_dB - 8.7)
        else:
            beta = 0.0

        # Apply Kaiser window to ideal filter
        h = ideal_filter * signal.windows.kaiser(2 * L + 1, beta)

    if not np.isrealobj(h):
        raise ValueError("The filter h should be a real vector")

    h = np.asarray(h)
    if h.ndim != 1:
        raise ValueError("The filter h should be a vector")

    Lx = x.shape[0]
    Lh = len(h)
    L = (Lh - 1) / 2.0
    Ly = ceil(Lx * p / q)

    # Pre and postpad filter response
    nz_pre = floor(q - np.mod(L, q))
    h_padded = np.pad(h, (nz_pre, 0), 'constant')

    offset = floor((L + nz_pre) / q)
    nz_post = 0
    while ceil(((Lx - 1) * p + nz_pre + Lh + nz_post) / q) - offset < Ly:
        nz_post += 1
    h_padded = np.pad(h_padded, (0, nz_post), 'constant')

    # Filtering - fixed upfirdn usage
    x_up = np.zeros(p * len(x))
    x_up[::p] = x.flatten()
    # y = signal.upfirdn(h_padded, x_up, up=1, down=q)
    print(x.shape)
    print(h_padded.shape)
    print(p)
    print(q)
    y = upfirdn_raw(x, h_padded, p, q)
    y = y[offset:offset + Ly]

    # Restore original dimensionality
    if is_1d:
        y = y.flatten()
    else:
        y = y.reshape(-1, x.shape[1])

    return y, h

def test_pop_resample_local():
    """Test function for pop_resample."""
    eeglab_file_path = '/Users/arno/Python/eegprep/sample_data/eeglab_data_with_ica_tmp.set'
    EEG = pop_loadset(eeglab_file_path)

    # Test with different engines
    EEG_python = pop_resample(EEG.copy(), 100, engine=None)
    EEG_python = pop_resample(EEG.copy(), 100, engine='poly')
    EEG_python = pop_resample(EEG.copy(), 100, engine='scipy')
    EEG_matlab = pop_resample(EEG.copy(), 100, engine='matlab')
    EEG_octave = pop_resample(EEG.copy(), 100, engine='octave')

    # Print results
    print("Original sampling rate:", EEG['srate'])
    print("Python resampled rate:", EEG_python['srate'])
    print("MATLAB resampled rate:", EEG_matlab['srate'])
    print("Octave resampled rate:", EEG_octave['srate'])

    return EEG_python, EEG_matlab, EEG_octave

if __name__ == '__main__':
    test_pop_resample_local()
