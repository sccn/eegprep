"""RANSAC utilities for EEG data processing."""

from typing import *

import numpy as np

from .spatial import sphericalSplineInterpolate
from ..eeglabcompat import get_eeglab
from .misc import round_mat

def rand_sample(
        n: int, 
        m: int, 
        stream: np.random.RandomState
) -> np.ndarray:
    """Random sampling without replacement.

    Args:
        n: number of items to sample from
        m: number of items to sample
        stream: random number generator

    Returns:
        random_sample: array of sampled values
    """
    pool = np.arange(n)
    result = np.zeros((m,), dtype=int)

    for k in range(m):
        choice = int(round_mat((pool.shape[0] - 1) * stream.rand()))
        result[k] = pool[choice]
        pool = np.delete(pool, choice)
    return result


def calc_projector(
        locs: np.ndarray, 
        num_samples: int, 
        subset_size: int, 
        stream: Optional[np.random.RandomState] = None,
        subroutine: str = 'sphericalSplineInterpolate'
) -> np.ndarray:
    """Calculate a bag of reconstruction matrices from random channel subsets.

    Args:
        locs: Nx3 array of channel locations
        num_samples: number of random samples to generate
        subset_size: size of each random subset
        stream: optionally the random number generator to use;
          if not specified, will default to a fixed seed (435656)
        subroutine: which interpolation subroutine to use (for testing)
    Returns:
        P: combined projector matrix
    """
    if stream is None:
        stream = np.random.RandomState(435656)

    # noinspection PyUnresolvedReferences
    rand_samples = np.zeros((locs.shape[0], num_samples, locs.shape[0]))

    if subroutine == 'sphericalSplineInterpolate':
        op = lambda src, dest: sphericalSplineInterpolate(src.T, dest.T)[0]
    elif subroutine == 'matlab':
        matlab = get_eeglab('MAT')
        op = lambda src, dest: matlab.sphericalSplineInterpolate(src.T, dest.T)[0]
    elif subroutine == 'octave':
        octave = get_eeglab('OCT')
        op = lambda src, dest: octave.sphericalSplineInterpolate(src.T, dest.T)[0]
    else:
        raise ValueError(f'Unknown subroutine: {subroutine}')

    # noinspection PyShadowingNames
    for k in range(num_samples - 1, -1, -1):
        sample = rand_sample(locs.shape[0], subset_size, stream)
        tmp = op(locs[sample, :], locs)
        rand_samples[sample, k, :] = np.real(tmp).T
    return np.reshape(rand_samples, (locs.shape[0], -1))
