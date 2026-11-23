"""A :class:`~eegprep_lean.Window` as an ``mne.io.RawArray``.

Named ``mne`` inside the package; the ``import mne`` below is the absolute one, the
third-party package, because Python 3 imports are absolute. The ``mne`` extra supplies it
and it is imported lazily, so a session without it can still import everything else here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .channels import MISSING_CHANNEL_TYPE, _channel_type
from .eeg import microvolts_per, require_physical
from .extras import is_missing_extra, missing_extra_error
from .window import Window

if TYPE_CHECKING:
    import mne


def to_mne_raw(window: Window, *, ch_types: str | Sequence[str] | None = None) -> "mne.io.RawArray":
    """Turn a window into an ``mne.io.RawArray`` in volts, named by the window's labels.

    Channel types come from ``window.types`` when ``ch_types`` is left unset. Each value
    is matched, ignoring case, against the types ``mne.create_info`` accepts
    (:func:`mne.io.get_channel_type_constants`), which is MNE's own set rather than a
    second table. ``EOG`` is ``eog``, ``ECG`` is ``ecg``, ``EMG`` is ``emg``, ``MISC`` is
    ``misc`` and ``STIM`` is ``stim``. A declared type outside that set
    (``TRIG``, ``EKG``, ``HEOG``) becomes ``misc``, so a later EEG pick leaves it out.
    MNE's EEGLAB reader instead warns and marks that channel as EEG. A blank type,
    ``n/a`` or ``nan`` is ``eeg``, the same stand-in a missing ``channel_type`` gets, and
    so is a window that carries no types at all.

    ``ch_types`` overrides that: one MNE type for every channel, or one per channel. A
    value outside that set, or the wrong count, raises ``ValueError``.

    The sampling frequency is the window's ``rate``. Units V, mV and uV are converted;
    any other is refused with ``ValueError``. Needs the ``mne`` extra.
    """
    try:
        import mne
    except ImportError as err:
        if not is_missing_extra(err, "mne"):
            raise
        raise missing_extra_error("to_mne_raw", "mne", err) from err

    labels = require_physical(window)
    volts = window.data * (microvolts_per(window.unit) * 1e-6)
    accepted = frozenset(mne.io.get_channel_type_constants(include_defaults=True))
    info = mne.create_info(
        list(labels),
        sfreq=float(window.rate),
        ch_types=_mne_channel_types(window, ch_types, len(labels), accepted),
    )
    return mne.io.RawArray(volts, info, verbose=False)


def _mne_channel_types(
    window: Window,
    ch_types: str | Sequence[str] | None,
    n_channels: int,
    accepted: frozenset[str],
) -> str | list[str]:
    if ch_types is not None:
        return _explicit_mne_types(ch_types, n_channels, accepted)
    if window.types is None:
        return MISSING_CHANNEL_TYPE.lower()
    if len(window.types) != n_channels:
        raise ValueError(
            f"to_mne_raw found {len(window.types)} entries in Window.types for {n_channels} channels. "
            "Set Window.types to one string per channel, or leave types unset."
        )
    return [_mne_channel_type(channel_type, accepted) for channel_type in window.types]


def _explicit_mne_types(ch_types: str | Sequence[str], n_channels: int, accepted: frozenset[str]) -> str | list[str]:
    if isinstance(ch_types, str):
        items = [ch_types]
        single = True
    else:
        items = list(ch_types)
        single = False
        if len(items) != n_channels:
            raise ValueError(
                f"to_mne_raw got {len(items)} ch_types for {n_channels} channels. "
                "Pass one MNE channel type, or one per channel, such as "
                "'eeg', 'eog', 'ecg', 'emg', 'misc', or 'stim'."
            )
    normalized = [_require_mne_type(item, accepted) for item in items]
    return normalized[0] if single else normalized


def _require_mne_type(channel_type: object, accepted: frozenset[str]) -> str:
    if isinstance(channel_type, str):
        candidate = channel_type.strip().lower()
        if candidate in accepted:
            return candidate
    raise ValueError(
        f"to_mne_raw cannot use channel type {channel_type!r}. "
        "Pass an MNE channel type such as 'eeg', 'eog', 'ecg', 'emg', 'misc', or 'stim'."
    )


def _mne_channel_type(channel_type: object, accepted: frozenset[str]) -> str:
    if not isinstance(channel_type, str):
        raise ValueError(
            f"to_mne_raw found channel type {channel_type!r}, which is not text. "
            "Set each Window.types entry to a string such as 'EEG', 'EOG', or 'ECG'."
        )
    candidate = _channel_type(channel_type).lower()
    if candidate in accepted:
        return candidate
    return "misc"
