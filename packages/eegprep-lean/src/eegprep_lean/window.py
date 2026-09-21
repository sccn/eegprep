"""Read a window of samples, in physical units.

:func:`read_window` needs the ``zarr`` extra; :class:`Window` and :func:`to_physical` do
not, and zarr is imported inside the function rather than here so that the ``plot`` extra
is installable and usable on its own.

NEMAR stores level-0 signal as ``int16`` with a per-channel ``scale`` and ``offset``, and
the array itself carries the conversion as ``physical = digital * scale + offset``. The
scaling is per channel, not global, so a window has to be converted against the channels
it actually holds rather than against the first few.

Returning the digital counts is the failure that matters here, because nothing raises:
the numbers are the right shape, the right dtype and the wrong magnitude, and a plot of
them looks like EEG. So :func:`read_window` converts by default and a caller has to ask
for ``physical=False`` in as many words.

**The unit lives on the channel, and this reads it.** The level-0 array carries the
conversion but not the unit it produces, which is why looking only at the array suggests
units are unknowable. They are not: each entry of the channel group's ``channels`` array
declares its own ``unit``, and :mod:`eegprep_lean.channels` reads them. A window reports
a unit only when every channel in it agrees on one, because the store contract says to
read the unit from the channel rather than assume one per modality, and
magnetoencephalography is a Tesla-based unit rather than a voltage.

**Level 0 is not always the rate the recording was acquired at.** The store resamples it
to ``min(native rate, modality cap)``, 250 Hz for electroencephalography, so a 500 Hz
recording arrives here at 250 Hz. :attr:`Window.original_rate` says what it was.

Large offsets are normal and not a sign of a broken conversion: level-0 is raw signal
before referencing or filtering, so a per-channel DC offset of thousands of units is what
the amplifier recorded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .channels import GroupMetadata, read_group_metadata
from .extras import is_missing_extra, missing_extra_error
from .index import ChannelGroup, DatasetIndex, IndexError_, Store
from .transport import Transport

#: What the array declares its conversion to be. Checked rather than assumed: if a future
#: store changes the formula, applying this one would silently produce wrong values.
EXPECTED_FORMULA = "physical = digital * scale + offset"


@dataclass(frozen=True)
class Window:
    """Samples read from one channel group, with what is needed to interpret them."""

    data: np.ndarray
    """Channels by samples.

    In the unit named by :attr:`unit` unless ``physical=False`` was asked for, in which
    case these are the stored integer counts.
    """

    channels: tuple[int, ...]
    """Indices into the group's channels, in the order the rows appear."""

    start_sample: int
    """Index of the first sample, at this group's rate."""

    rate: float
    group_name: str
    physical: bool

    labels: tuple[str, ...] | None = None
    """Channel labels in row order, or None when the group did not supply all of them."""

    unit: str | None = None
    """The unit these rows share, or None when they do not share one.

    None is a real answer, not a gap: a window spanning channels of different units has
    no single unit, and naming one would be wrong for some of its rows.
    """

    original_rate: float = 0.0
    """The rate the recording was acquired at, when it differs from :attr:`rate`.

    Level 0 is resampled to ``min(native rate, modality cap)``, so this is how a caller
    learns that 250 Hz of electroencephalography came from a 500 Hz recording.
    """

    @property
    def was_resampled(self) -> bool:
        return bool(self.original_rate) and self.original_rate != self.rate

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.rate if self.rate else 0.0

    @property
    def times_s(self) -> np.ndarray:
        """Sample times in seconds from the start of the recording, not of the window."""
        start = self.start_sample / self.rate
        return start + np.arange(self.n_samples) / self.rate


def _resolve_channels(
    channels: Sequence[int] | slice | None,
    n_channels: int,
) -> tuple[int, ...]:
    if channels is None:
        return tuple(range(n_channels))
    if isinstance(channels, slice):
        return tuple(range(*channels.indices(n_channels)))
    resolved = tuple(int(c) for c in channels)
    for channel in resolved:
        if not 0 <= channel < n_channels:
            raise IndexError_(f"channel {channel} is outside 0..{n_channels - 1} for this group")
    return resolved


def to_physical(digital: np.ndarray, attributes: dict[str, Any], channels: Sequence[int]) -> np.ndarray:
    """Apply the array's own per-channel conversion to a channels-by-samples block."""
    formula = attributes.get("physical_formula")
    if formula != EXPECTED_FORMULA:
        raise IndexError_(
            f"array declares its conversion as {formula!r}, and this reader implements "
            f"{EXPECTED_FORMULA!r}. Applying the wrong formula returns plausible numbers, "
            "so it refuses instead."
        )
    scale = np.asarray(attributes["scale"], dtype=np.float64)[list(channels)]
    offset = np.asarray(attributes["offset"], dtype=np.float64)[list(channels)]
    # Column vectors so each channel's constants apply along its own row.
    return digital.astype(np.float64) * scale[:, None] + offset[:, None]


async def read_window(
    index: DatasetIndex,
    store: Store,
    *,
    start_sample: int,
    n_samples: int,
    channels: Sequence[int] | slice | None = None,
    group: ChannelGroup | None = None,
    physical: bool = True,
    metadata: GroupMetadata | None = None,
    transport: Transport | None = None,
) -> Window:
    """Read ``n_samples`` from ``start_sample``, converted to physical units.

    Reads only the chunks the window spans: the array is sharded, and Zarr fetches the
    shard index and then the inner chunks the slice touches, rather than the whole store.

    Also fetches the channel group's metadata, for the labels and the unit, unless a
    caller passes one they already hold. That is one small document per call, and the
    alternative is a window that cannot say what it is measured in.
    """
    # First, before validating anything else: a caller without the zarr extra should be
    # told that, not told their arguments are wrong, and not handed a bare
    # ModuleNotFoundError from inside another module at the end of a real read. The
    # import is here rather than at module scope so this module needs only numpy, which
    # is what lets the plot tier stand alone.
    try:
        from .store import open_array
    except ImportError as err:
        if not is_missing_extra(err, "zarr"):
            raise
        raise missing_extra_error("read_window", "zarr", err) from err

    chosen = group or store.group()
    if n_samples <= 0:
        raise IndexError_(f"n_samples must be positive, got {n_samples}")
    if start_sample < 0:
        raise IndexError_(f"start_sample must not be negative, got {start_sample}")
    if start_sample >= chosen.n_samples:
        raise IndexError_(
            f"start_sample {start_sample} is past the end of {chosen.name}, which has {chosen.n_samples} samples"
        )

    # Clamp rather than refuse: asking for the last ten seconds of a recording that has
    # eight is an ordinary thing to do, and the window reports what it actually holds.
    stop = min(start_sample + n_samples, chosen.n_samples)
    wanted = _resolve_channels(channels, chosen.n_channels)

    array = await open_array(index.level0_url(store, chosen), transport=transport)
    # Zarr's asynchronous getitem does basic indexing only, so a list of channels is not
    # a selection it accepts. Read the contiguous span that covers them and take the rows
    # afterwards. That costs no extra bytes: the channel axis is not chunked (the chunk
    # shape spans every channel), so any subset fetches the same chunks regardless.
    first, last = min(wanted), max(wanted)
    span = await array.getitem((slice(first, last + 1), slice(start_sample, stop)))
    digital = np.asarray(span)[[channel - first for channel in wanted], :]

    data = to_physical(digital, dict(array.metadata.attributes), wanted) if physical else np.asarray(digital)

    if metadata is None:
        metadata = await read_group_metadata(index, store, chosen, transport=transport)
    return Window(
        data=data,
        channels=wanted,
        start_sample=start_sample,
        rate=chosen.rate,
        group_name=chosen.name,
        physical=physical,
        labels=metadata.labels(wanted),
        # Only when the values are in it. Stored counts are not in the channel's unit,
        # and carrying the unit alongside them would invite exactly that conflation.
        unit=metadata.unit(wanted) if physical else None,
        original_rate=metadata.original_rate,
    )
