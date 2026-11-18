"""A :class:`~eegprep_lean.Window` as an ``mne.io.RawArray``.

Named ``mne`` inside the package; the ``import mne`` below is the absolute one, the
third-party package, because Python 3 imports are absolute. The ``mne`` extra supplies it
and it is imported lazily, so a session without it can still import everything else here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .eeg import microvolts_per, require_physical
from .extras import is_missing_extra, missing_extra_error
from .window import Window

if TYPE_CHECKING:
    import mne


def to_mne_raw(window: Window, *, ch_types: str | Sequence[str] | None = None) -> "mne.io.RawArray":
    """Turn a window into an ``mne.io.RawArray`` in volts, named by the window's labels.

    ``ch_types`` is one MNE channel type for every channel or one per channel, and
    defaults to ``"eeg"``. The sampling frequency is the window's ``rate``. Units V, mV
    and uV are converted; any other is refused with ``ValueError``. Needs the ``mne`` extra.
    """
    try:
        import mne
    except ImportError as err:
        if not is_missing_extra(err, "mne"):
            raise
        raise missing_extra_error("to_mne_raw", "mne", err) from err

    labels = require_physical(window)
    volts = window.data * (microvolts_per(window.unit) * 1e-6)
    info = mne.create_info(list(labels), sfreq=float(window.rate), ch_types="eeg" if ch_types is None else ch_types)
    return mne.io.RawArray(volts, info, verbose=False)
