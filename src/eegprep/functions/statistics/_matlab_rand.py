"""MATLAB ``rng('default')`` Mersenne Twister used by EEGLAB resampling."""

from __future__ import annotations

import numpy as np

_N = 624
_M = 397
_UPPER = np.uint32(0x80000000)
_LOWER = np.uint32(0x7FFFFFFF)
_MATRIX_A = np.uint32(0x9908B0DF)
# init_genrand(5489) is the sequence MATLAB documents for rng('default'):
# 0.814723686393179, 0.905791937075619, ...
_DEFAULT_SEED = 5489


class MatlabTwister:
    """Uniform generator matching MATLAB ``rand`` on the default Twister stream."""

    def __init__(self, seed: int = _DEFAULT_SEED) -> None:
        self.mt = np.zeros(_N, dtype=np.uint32)
        self.mti = _N
        self.seed(seed)

    def seed(self, seed: int) -> None:
        self.mt[0] = np.uint32(int(seed) & 0xFFFFFFFF)
        for index in range(1, _N):
            previous = np.uint32(self.mt[index - 1])
            mixed = previous ^ (previous >> np.uint32(30))
            self.mt[index] = np.uint32(np.uint64(1812433253) * np.uint64(mixed) + np.uint64(index))
        self.mti = _N

    def _int32(self) -> np.uint32:
        if self.mti >= _N:
            self._twist()
        value = np.uint32(self.mt[self.mti])
        self.mti += 1
        value ^= value >> np.uint32(11)
        value ^= (value << np.uint32(7)) & np.uint32(0x9D2C5680)
        value ^= (value << np.uint32(15)) & np.uint32(0xEFC60000)
        value ^= value >> np.uint32(18)
        return np.uint32(value)

    def _twist(self) -> None:
        mag01 = (np.uint32(0), _MATRIX_A)
        state = self.mt
        for index in range(_N - _M):
            mixed = (state[index] & _UPPER) | (state[index + 1] & _LOWER)
            state[index] = state[index + _M] ^ (mixed >> np.uint32(1)) ^ mag01[int(mixed & np.uint32(1))]
        for index in range(_N - _M, _N - 1):
            mixed = (state[index] & _UPPER) | (state[index + 1] & _LOWER)
            state[index] = state[index + (_M - _N)] ^ (mixed >> np.uint32(1)) ^ mag01[int(mixed & np.uint32(1))]
        mixed = (state[_N - 1] & _UPPER) | (state[0] & _LOWER)
        state[_N - 1] = state[_M - 1] ^ (mixed >> np.uint32(1)) ^ mag01[int(mixed & np.uint32(1))]
        self.mti = 0

    def rand(self, *shape: int) -> np.ndarray | float:
        """Return MATLAB ``rand(shape)`` values in column-major order."""
        count = int(np.prod(shape)) if shape else 1
        values = np.empty(count, dtype=np.float64)
        scale = 1.0 / 9007199254740992.0
        for index in range(count):
            high = float(self._int32() >> np.uint32(5))
            low = float(self._int32() >> np.uint32(6))
            values[index] = (high * 67108864.0 + low) * scale
        if not shape:
            return float(values[0])
        return values.reshape(shape, order="F")


_STREAM = MatlabTwister(_DEFAULT_SEED)


def reset_matlab_rng(seed: int = _DEFAULT_SEED) -> MatlabTwister:
    """Reset the process-wide MATLAB Twister stream and return it."""
    _STREAM.seed(seed)
    return _STREAM


def matlab_rng() -> MatlabTwister:
    return _STREAM
