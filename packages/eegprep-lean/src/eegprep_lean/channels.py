"""Read a channel group's own metadata: labels, units, and what was resampled.

Base tier. Only the standard library and this package's transport, so a session can learn
what a recording's channels are called and what they are measured in without installing
zarr or numpy.

**This is where the units are.** The level-0 array carries ``scale``, ``offset`` and
``physical_formula`` but not the unit those produce, which is why reading only the array
leads to the conclusion that units are unknowable. They are not: each entry of the
group's ``channels`` array carries its own ``unit``, and the store contract is explicit
that a client must read it from the channel rather than assume one per modality. MEG is
a Tesla-based unit, not a voltage, so assuming microvolts per modality is wrong by a
factor no one notices.

**Level 0 is not always the rate it was recorded at.** The store resamples level 0 to
``min(native rate, modality cap)``, and the cap for electroencephalography (EEG) and
magnetoencephalography (MEG) is 250 Hz, so a 500 Hz recording arrives at 250 Hz with
``original_rate`` saying so. A reader that reports only ``rate`` lets a caller believe
they are holding the recording as acquired.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .index import ChannelGroup, DatasetIndex, IndexError_, Store
from .transport import Transport, default_transport


@dataclass(frozen=True)
class Channel:
    """One channel, as the group describes it."""

    label: str
    unit: str
    channel_type: str
    row_index: int
    original_rate: float
    target_rate: float
    usable_for_inference: bool

    @classmethod
    def from_entry(cls, entry: dict) -> "Channel":
        return cls(
            label=str(entry.get("label", "")),
            unit=str(entry.get("unit", "")),
            channel_type=str(entry.get("channel_type", "")),
            row_index=int(entry.get("row_index", -1)),
            original_rate=float(entry.get("original_rate") or 0.0),
            target_rate=float(entry.get("target_rate") or 0.0),
            usable_for_inference=bool(entry.get("usable_for_inference", True)),
        )


@dataclass(frozen=True)
class GroupMetadata:
    """A channel group's own attributes, read from the store rather than the index."""

    name: str
    modality: str
    rate: float
    original_rate: float
    n_channels: int
    channels: tuple[Channel, ...]

    @property
    def was_resampled(self) -> bool:
        """True when level 0 is not the rate the recording was acquired at."""
        return bool(self.original_rate) and self.original_rate != self.rate

    def labels(self, rows: tuple[int, ...]) -> tuple[str, ...] | None:
        """Labels for the given rows, or None when this group cannot supply them all.

        None rather than a placeholder for the missing ones: a plot labeled with a mix of
        real names and stand-ins is harder to distrust than one labeled by index.
        """
        by_row = {channel.row_index: channel.label for channel in self.channels}
        labels = tuple(by_row.get(row, "") for row in rows)
        return labels if all(labels) else None

    def unit(self, rows: tuple[int, ...]) -> str | None:
        """The unit these rows share, or None when they do not share one.

        The store contract says to read the unit from the channel, not from the modality.
        A window spanning channels of different units has no single unit to name, and
        naming one would be wrong for some of its rows.
        """
        by_row = {channel.row_index: channel.unit for channel in self.channels}
        units = {by_row.get(row, "") for row in rows}
        return units.pop() if len(units) == 1 and all(units) else None

    @classmethod
    def from_attributes(cls, attributes: dict, *, name: str) -> "GroupMetadata":
        entries = attributes.get("channels")
        if not isinstance(entries, list):
            raise IndexError_(
                f"channel group {name!r} declares no channels array, so its labels and units "
                "cannot be read. Stores converted by biosigIO before 1.2.6 can omit it."
            )
        return cls(
            name=name,
            modality=str(attributes.get("modality", "")),
            rate=float(attributes.get("rate") or 0.0),
            original_rate=float(attributes.get("original_rate") or 0.0),
            # The declared count, not len(channels): they disagree when a document is
            # truncated, and the field is what the store says the group holds.
            n_channels=int(attributes.get("n_channels") or 0),
            channels=tuple(Channel.from_entry(entry) for entry in entries),
        )


async def read_group_metadata(
    index: DatasetIndex,
    store: Store,
    group: ChannelGroup | None = None,
    *,
    transport: Transport | None = None,
) -> GroupMetadata:
    """Fetch one channel group's attributes from the store.

    One small JSON document, next to the arrays. Built on ``contract_base`` like every
    other URL here, never on ``data_base``.
    """
    chosen = group or store.group()
    client = transport or default_transport()
    url = f"{index.group_url(store, chosen)}/zarr.json"
    response = await client.get(url)
    try:
        document = json.loads(response.body)
    except json.JSONDecodeError as err:
        raise IndexError_(f"group metadata at {url} is not JSON: {err}") from err
    return GroupMetadata.from_attributes(document.get("attributes") or {}, name=chosen.name)
