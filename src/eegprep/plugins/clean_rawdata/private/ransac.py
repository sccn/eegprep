"""RANSAC utilities for EEG data processing."""

from typing import Optional

import numpy as np

from ....functions.adminfunc.eeglabcompat import get_eeglab
from ....functions.miscfunc.parity import rand_sample
from .sphericalSplineInterpolate import sphericalSplineInterpolate


def calc_projector(
    locs: np.ndarray,
    num_samples: int,
    subset_size: int,
    stream: Optional[np.random.RandomState] = None,
    subroutine: str = 'sphericalSplineInterpolate',
) -> np.ndarray:
    """Calculate a bag of reconstruction matrices from random channel subsets.

    Args:
        locs: Nx3 array of channel locations
        num_samples: number of random samples to generate
        subset_size: size of each random subset
        stream: optionally the random number generator to use;
          if not specified, will default to a fixed seed (435656)
        subroutine: which interpolation subroutine to use (for testing)

    Returns
    -------
    P : combined projector matrix
    """
    if stream is None:
        stream = np.random.RandomState(435656)

    # noinspection PyUnresolvedReferences
    rand_samples = np.zeros((locs.shape[0], num_samples, locs.shape[0]))

    if subroutine == 'sphericalSplineInterpolate':

        def op(src, dest):
            return sphericalSplineInterpolate(src.T, dest.T)[0]

    elif subroutine == 'matlab':
        matlab = get_eeglab('MAT')

        def op(src, dest):
            return matlab.sphericalSplineInterpolate(src.T, dest.T)[0]

    elif subroutine == 'octave':
        octave = get_eeglab('OCT')

        def op(src, dest):
            return octave.sphericalSplineInterpolate(src.T, dest.T)[0]

    else:
        raise ValueError(f'Unknown subroutine: {subroutine}')

    # noinspection PyShadowingNames
    for k in range(num_samples - 1, -1, -1):
        sample = rand_sample(locs.shape[0], subset_size, stream)
        tmp = op(locs[sample, :], locs)
        rand_samples[sample, k, :] = np.real(tmp).T
    return np.reshape(rand_samples, (locs.shape[0], -1))
