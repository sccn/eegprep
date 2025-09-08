
import sys
from typing import Optional

__all__ = ['is_debug', 'ExceptionUnlessDebug', 'num_jobs_from_reservation', 'humanize_seconds', 'num_cpus_from_reservation']


def is_debug() -> bool:
    """Check if a debugger is currently attached to the process."""
    return getattr(sys, 'gettrace', None)() is not None


def num_cpus_from_reservation(ReservePerJob: str, *, default: int = 4) -> Optional[int]:
    """Get the number of reserved CPUs per job from the reservation string, if set"""
    ReservePerJob = ReservePerJob.strip().replace(' ', '').upper()
    if ',' in ReservePerJob:
        # scan through multiple reservations, pick the first match
        for part in ReservePerJob.split(','):
            if 'CPU' in part:
                ReservePerJob = part
                break
    if '-' in ReservePerJob:
        # If a margin is specified, take the first part before the dash
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

    Returns:
        the number of jobs that can be run based on the available system resources

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
    if '-' in ReservePerJob:
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
    elif reserve.endswith('GPU'):
        if sys.platform == 'win32':
            raise NotImplementedError("GPU reservation is not supported on Windows. "
                                      "Please use a Linux-based system.")
        try:
            import pynvml
        except ImportError:
            raise ImportError("pynvml is required to determine available GPU resources. "
                              "Please install it with 'uv pip install pynvml'.")
        try:
            pynvml.nvmlInit()
            avail_amt = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError as e:
            avail_amt = 0
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
    if sec > 3600:
        return f"{sec / 3600:.1f}h"
    elif sec > 180:
        return f"{sec / 60:.1f}m"
    else:
        return f"{sec:.1f}s"


class DummyException(Exception):
    """A dummy exception class for use in ExceptionUnlessDebug."""
    pass

# a class that defaullts to Exception, but uses DummyException if a debugger is attached
# (this is useful for exceptions that should only be caught in production but not in debug mode)
ExceptionUnlessDebug = DummyException if is_debug() else Exception
