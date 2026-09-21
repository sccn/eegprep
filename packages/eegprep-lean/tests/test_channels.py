"""Tests for the channel-group metadata reader.

Base tier: labels and units are stdlib-only to read, so a session learns what a recording
holds without installing zarr or numpy.

The fixture is a real group document from nm000103, trimmed to four channels. Its
``n_channels`` field still says 129, on purpose: the declared count and the length of the
array it ships disagree, and code that substitutes one for the other should be caught.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from eegprep_lean.channels import Channel, GroupMetadata, read_group_metadata
from eegprep_lean.index import IndexError_

FIXTURE = Path(__file__).parent / "nm000103_group_v3.json"

# Read off the captured document by hand. Comparing against values re-read from the same
# file would pass whatever the parser did with them.
GROUP_NAME = "eeg_250hz"
DECLARED_N_CHANNELS = 129
RATE = 250.0
ORIGINAL_RATE = 500.0
FIRST_LABELS = ("E1", "E2", "E3", "E4")
UNIT = "uV"


@pytest.fixture
def metadata() -> GroupMetadata:
    attributes = json.loads(FIXTURE.read_text())["attributes"]
    return GroupMetadata.from_attributes(attributes, name=GROUP_NAME)


class TestParsing:
    def test_reads_the_channels_the_document_ships(self, metadata: GroupMetadata) -> None:
        assert tuple(c.label for c in metadata.channels) == FIRST_LABELS
        assert all(c.unit == UNIT for c in metadata.channels)

    def test_n_channels_is_the_declared_count_not_the_array_length(self, metadata: GroupMetadata) -> None:
        """They disagree here, and the store's own count is the truthful one."""
        assert metadata.n_channels == DECLARED_N_CHANNELS
        assert len(metadata.channels) == 4

    def test_row_index_is_read_rather_than_assumed_from_position(self, metadata: GroupMetadata) -> None:
        """The contract says row_index is the channel's row in the level-0 array. A
        reader that used list position would agree here and diverge on any group whose
        channels are not listed in row order."""
        assert [c.row_index for c in metadata.channels] == [0, 1, 2, 3]
        shuffled = GroupMetadata(
            name=GROUP_NAME,
            modality="EEG",
            rate=RATE,
            original_rate=ORIGINAL_RATE,
            n_channels=2,
            channels=(
                Channel("B", "uV", "EEG", 1, 500.0, 250.0, True),
                Channel("A", "uV", "EEG", 0, 500.0, 250.0, True),
            ),
        )
        assert shuffled.labels((0, 1)) == ("A", "B")

    def test_a_group_without_a_channels_array_is_refused(self) -> None:
        """Stores converted before biosigIO 1.2.6 can omit it, and inventing labels for
        them would be worse than saying so."""
        with pytest.raises(IndexError_, match="declares no channels array"):
            GroupMetadata.from_attributes({"rate": RATE}, name=GROUP_NAME)


class TestResampling:
    def test_reports_that_level_zero_was_resampled(self, metadata: GroupMetadata) -> None:
        """nm000103 was recorded at 500 Hz and level 0 is 250, because the store caps
        electroencephalography there. A reader that saw only `rate` would believe it had
        the recording as acquired."""
        assert metadata.rate == RATE
        assert metadata.original_rate == ORIGINAL_RATE
        assert metadata.was_resampled is True

    def test_an_unresampled_group_does_not_claim_to_be_one(self) -> None:
        same = GroupMetadata(GROUP_NAME, "EEG", 250.0, 250.0, 1, ())
        assert same.was_resampled is False

    def test_a_group_that_declares_no_original_rate_is_not_resampled(self) -> None:
        """Absent, not zero: older stores omit the field, and reading a missing value as
        0.0 would make every one of them look resampled from nothing."""
        silent = GroupMetadata(GROUP_NAME, "EEG", 250.0, 0.0, 1, ())
        assert silent.was_resampled is False


class TestLabelsAndUnits:
    def test_labels_follow_the_rows_asked_for_in_that_order(self, metadata: GroupMetadata) -> None:
        assert metadata.labels((2, 0)) == ("E3", "E1")

    def test_a_row_the_group_does_not_describe_gives_no_labels_at_all(self, metadata: GroupMetadata) -> None:
        """Rather than a mix of real names and blanks, which is harder to distrust than
        a plot labeled by index."""
        assert metadata.labels((0, 999)) is None

    def test_one_unit_when_the_rows_agree(self, metadata: GroupMetadata) -> None:
        assert metadata.unit((0, 1, 2)) == UNIT

    def test_no_unit_when_the_rows_disagree(self) -> None:
        """The store contract says to read the unit from the channel, not the modality.
        A window spanning channels of different units has none to name, and naming one
        would be wrong for some of its rows."""
        mixed = GroupMetadata(
            GROUP_NAME,
            "EEG",
            RATE,
            ORIGINAL_RATE,
            2,
            (Channel("E1", "uV", "EEG", 0, 500.0, 250.0, True), Channel("T1", "degC", "TEMP", 1, 500.0, 250.0, True)),
        )

        assert mixed.unit((0, 1)) is None
        assert mixed.unit((0,)) == "uV"

    def test_a_blank_unit_is_not_a_unit(self) -> None:
        blank = GroupMetadata(GROUP_NAME, "EEG", RATE, ORIGINAL_RATE, 1, (Channel("E1", "", "EEG", 0, 0.0, 0.0, True),))
        assert blank.unit((0,)) is None


@pytest.mark.network
class TestAgainstTheLiveArchive:
    def test_reads_the_real_groups_labels_and_units(self) -> None:
        from eegprep_lean import read_index

        async def run() -> GroupMetadata:
            index = await read_index("nm000103")
            return await read_group_metadata(index, index.stores[0])

        metadata = asyncio.run(run())

        assert metadata.n_channels == DECLARED_N_CHANNELS
        assert len(metadata.channels) == DECLARED_N_CHANNELS
        assert tuple(c.label for c in metadata.channels[:4]) == FIRST_LABELS
        assert metadata.unit(tuple(range(DECLARED_N_CHANNELS))) == UNIT
        assert metadata.was_resampled is True
