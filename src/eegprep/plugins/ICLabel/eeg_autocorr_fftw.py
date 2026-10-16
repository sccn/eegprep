"""EEG autocorrelation computation using FFTW.

This module provides functions for computing autocorrelation of EEG ICA components using
fast Fourier transform methods.
"""

import numpy as np
from scipy.fft import fft, ifft, next_fast_len
from scipy.signal import resample_poly


def eeg_autocorr_fftw(EEG, pct_data=100):
    """Compute autocorrelation of EEG ICA components using FFT.

    Parameters
    ----------
    EEG : dict
        EEG data structure with 'icaact', 'pnts', 'srate' fields.
    pct_data : float, optional
        Percentage of data to use. Default 100.

    Returns
    -------
    ndarray
        Autocorrelation array.
    """
    # FFT length
    nfft = next_fast_len(2 * EEG['pnts'] - 1)

    ncomp = EEG['icaact'].shape[0]
    ac = np.zeros((ncomp, nfft))

    activations = np.asarray(EEG['icaact'])
    for it in range(activations.shape[0]):
        # MATLAB icaact(it,:,:) still works when trials == 1 drops the last dimension.
        component = activations[it]
        if component.ndim == 1:
            component = component[:, np.newaxis]
        X = fft(component, n=nfft, axis=0)
        ac[it, :] = np.mean(np.abs(X) ** 2, axis=1)

    # Inverse FFT to get autocorrelation
    ac = ifft(ac, axis=1)

    ac = np.real(ac)

    # Adjust the size of autocorrelation array
    if EEG['pnts'] < EEG['srate']:
        # ac = np.hstack(     [ac[:, :EEG['pnts']], np.zeros((ncomp      , EEG['srate'] - EEG['pnts'] + 1))])
        ac = np.concatenate((ac[:, : EEG['pnts']], np.zeros((ac.shape[0], EEG['srate'] - EEG['pnts'] + 1))), axis=1)
    else:
        ac = ac[:, : int(EEG['srate']) + 1]

    ac = ac / ac[:, 0][:, np.newaxis]

    # resample to 1 second at 100 samples/sec
    ac = resample_poly(ac.T, up=100, down=EEG['srate']).T
    ac = ac[:, 1:101]

    return ac
