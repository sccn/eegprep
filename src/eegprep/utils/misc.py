
"""Miscellaneous utility functions."""

import sys
import math
from typing import Optional

import numpy as np

__all__ = ['is_debug', 'ExceptionUnlessDebug', 'num_jobs_from_reservation', 'humanize_seconds',
           'num_cpus_from_reservation', 'ToolError', 'canonicalize_signs', 'round_mat',
           'aslist', 'get_nested']


def is_debug() -> bool:
    """Check if a debugger is currently attached to the process."""
    return getattr(sys, 'gettrace', None)() is not None


def aslist(arr_or_list: np.ndarray | list) -> list:
    """Return the given array or list in list form."""
    if hasattr(arr_or_list, 'tolist'):
        return arr_or_list.tolist()
    elif isinstance(arr_or_list, list):
        return arr_or_list
    else:
        raise ValueError(f"Input must be a numpy array or a list, but "
                         f"was of type {type(arr_or_list)}.")


# Sentinel value to indicate that KeyError should be raised if key not found
_RAISE_KEYERROR = object()


def get_nested(data: dict, key: str, default=_RAISE_KEYERROR, separator: str = '.'):
    """Deep (recursive) dictionary lookup using dot-notation keys.

    Retrieves a value from a nested dictionary structure using a dot-separated
    key path. For example, 'user.profile.name' would access data['user']['profile']['name'].

    Parameters
    ----------
    data : dict
        The dictionary to search in.
    key : str
        The dot-notation key path (e.g., 'user.profile.name').
    default : object
        The value to return if the key path is not found. If not provided,
        a KeyError will be raised when the key is not found.
    separator : str
        The separator character to use for splitting the key (default: '.').

    Returns
    -------
    object
        The value at the nested location, or the default value if not found.

    Raises
    ------
    KeyError
        If the key path is not found and no default value is provided.

    Examples
    --------
    >>> data = {'user': {'profile': {'name': 'John', 'age': 30}}}
    >>> get_nested(data, 'user.profile.name')
    'John'
    >>> get_nested(data, 'user.profile.age')
    30
    >>> get_nested(data, 'user.email', default='not@found.com')
    'not@found.com'
    >>> get_nested(data, 'user.profile.address.city', default='Unknown')
    'Unknown'
    >>> get_nested(data, 'user.nonexistent')  # Raises KeyError
    Traceback (most recent call last):
        ...
    KeyError: 'user.nonexistent'
    """
    if not isinstance(data, dict):
        if default is _RAISE_KEYERROR:
            raise KeyError(key)
        return default

    keys = key.split(separator) if separator in key else [key]
    current = data

    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            if default is _RAISE_KEYERROR:
                raise KeyError(key)
            return default

    return current


def num_cpus_from_reservation(ReservePerJob: str, *, default: int = 4) -> Optional[int]:
    """Get the number of reserved CPUs per job from the reservation string, if set."""
    ReservePerJob = ReservePerJob.strip().replace(' ', '').upper()
    if ',' in ReservePerJob:
        # scan through multiple reservations, pick the first match
        for part in ReservePerJob.split(','):
            if 'CPU' in part:
                ReservePerJob = part
                break
    # If a margin is specified, take the first part before the margin separator
    if ':' in ReservePerJob:
        ReservePerJob = ReservePerJob.split(':')[0]
    # (legacy syntax for this uses a minus sign)
    if '-' in ReservePerJob:
        ReservePerJob = ReservePerJob.split('-')[0]
    if ReservePerJob.endswith('CPU'):
        # if we got a CPU reservation now
        return int(ReservePerJob[:-3])
    # in all other cases we return a default
    return default


def num_jobs_from_reservation(ReservePerJob: str) -> int:
    """Parse the job reservation string and calculate the number of jobs that can be run.

    This is the resource amount and type to reserve per job, e.g. '4GB' or '2CPU';
    the run will then use as many jobs as possible without exceeding the available resources.
    - Can also contain a total or percentage margin, as in '4GB-10GB', '2CPU-10%'.
    - Can also be specified as a total/maximum, as in '10 total' or '10max'.
    - Can also be a comma-separated list of reservations, e.g. '4GB,2CPU-1CPU,5max'.
    - if not set, will assume a single job.

    Parameters
    ----------
    ReservePerJob : str
        The reservation string.

    Returns
    -------
    int
        The number of jobs that can be run based on the available system resources.
    """
    if not ReservePerJob:
        return 1  # No reservation means we can run one job without restrictions
    if ',' in ReservePerJob:
        # If multiple reservations are specified, take the minimum over all
        # reservations to ensure we don't exceed any of them.
        parts = ReservePerJob.split(',')
        num_jobs = min(num_jobs_from_reservation(part) for part in parts)
        return num_jobs

    ReservePerJob = ReservePerJob.strip().replace(' ', '').upper()
    if ReservePerJob.endswith('TOTAL'):
        return int(ReservePerJob[:-5])
    elif ReservePerJob.endswith('MAX'):
        return int(ReservePerJob[:-3])
    if ':' in ReservePerJob:
        reserve, margin = ReservePerJob.split(':')
    elif '-' in ReservePerJob:
        # legacy syntax used a minus sign
        reserve, margin = ReservePerJob.split('-')
    else:
        reserve, margin = ReservePerJob, '0%'
    if reserve.endswith('B'):
        try:
            import psutil
        except ImportError:
            raise ImportError("psutil is required to determine available system RAM. "
                              "Please install it with 'uv pip install psutil'.")
        avail_amt = psutil.virtual_memory().available
        unit = reserve[-2:].upper()
        multiplier = {'GB': 2 ** 30, 'MB': 2 ** 20, 'KB': 2 ** 10, 'B': 1}[unit]
        reserve_amt = float(reserve[:-2]) * multiplier
    elif reserve.endswith('CPU'):
        import multiprocessing
        avail_amt = multiprocessing.cpu_count()
        reserve_amt = float(reserve[:-3])
    else:
        raise ValueError(f"Invalid reserve amount format: {ReservePerJob}. "
                         "Expected format like '4GB' or '2CPU'.")
    if not margin:
        margin_amt = 0
    elif margin.endswith('%'):
        margin_frac = float(margin[:-1]) / 100
        margin_amt = avail_amt * margin_frac
    elif margin.endswith('B'):
        unit = margin[-2:].upper()
        multiplier = {'GB': 2 ** 30, 'MB': 2 ** 20, 'KB': 2 ** 10, 'B': 1}[unit]
        margin_amt = float(margin[:-2]) * multiplier
    elif margin.endswith('CPU') or margin.endswith('GPU'):
        margin_amt = float(margin[:-3])
    else:
        raise ValueError(f"Invalid margin format: {margin}. "
                         "Expected format like '10%' or '100MB'.")
    avail_amt -= margin_amt
    if reserve_amt > avail_amt:
        raise ValueError(f"Requested reserve amount {reserve_amt} exceeds available "
                         f"system resources after applying margin {margin_amt}. "
                         f"Available: {avail_amt}.")
    num_jobs = int(avail_amt // reserve_amt)
    return num_jobs


def humanize_seconds(sec: float) -> str:
    """Humanize seconds into a readable string."""
    if sec > 3600:
        return f"{sec / 3600:.1f}h"
    elif sec > 180:
        return f"{sec / 60:.1f}m"
    else:
        return f"{sec:.1f}s"


def canonicalize_signs(V):
    """Canonicalize signs of column matrix V so that the largest absolute value is positive."""
    # V: columns are eigenvectors
    idx = np.argmax(np.abs(V), axis=0)
    sgn = np.sign(V[idx, range(V.shape[1])])
    sgn[sgn == 0] = 1
    return V * sgn


def round_mat(x, decimals=0):
    """MATLAB-style rounding function.

    - ties (.5 within fp error) round AWAY from zero
    - supports positive/zero/negative `decimals` like MATLAB round(x, N)
    - NaN/Inf propagate naturally
    - does NOT return integer-typed results

    This can be applied to numpy arrays and acts as a drop-in replacement
    for np.round(), but also works for pure-Python float values; however,
    to get a 1:1 replacement for a use of round(x) you need to write
    int(round_mat(x)) since round() returns integers.

    Parameters
    ----------
    x : array_like
        The value(s) to round.
    decimals : int
        Number of decimals to round to.

    Returns
    -------
    array_like
        The rounded value(s).
    """
    if isinstance(x, (float, int)):
        # Propagate NaN/Inf instead of throwing in math.floor(...)
        if math.isnan(x) or math.isinf(x):
            return x
        xp = math
    else:
        xp = np
        x = np.asarray(x)             # ensure ndarray

    if decimals == 0:
        return xp.copysign(xp.floor(abs(x) + 0.5), x)

    if decimals > 0:
        factor = 10.0 ** decimals
        y = xp.copysign(xp.floor(abs(x) * factor + 0.5), x)
        return y / factor

    # decimals < 0  -> round to tens/hundreds/…
    factor = 10.0 ** (-decimals)
    y = xp.copysign(xp.floor(abs(x) / factor + 0.5), x)
    return y * factor




class SkippableException(Exception):
    """A dummy exception class for use in ExceptionUnlessDebug."""


class ToolError(SkippableException):
    """An exception class to indicate an error in a third-party tool.

    This error cannot be addressed in eegprep and will not stop processing in debug mode.
    """


# a class that defaults to Exception, but uses SkippableException if a debugger is attached
# (this is useful for exceptions that should only be caught in production but not in debug mode)
ExceptionUnlessDebug = SkippableException if is_debug() else Exception
