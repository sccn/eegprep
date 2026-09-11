import pytest

from eegprep.functions.adminfunc.pop_delset import pop_delset


def _eeg(name: str = "demo") -> dict:
    return {"setname": name}


def test_pop_delset_rejects_non_positive_indices():
    with pytest.raises(ValueError, match="1-based"):
        pop_delset([_eeg()], -1)
    with pytest.raises(ValueError, match="1-based"):
        pop_delset([_eeg()], 0)


def test_pop_delset_rejects_missing_dataset():
    with pytest.raises(IndexError, match="No dataset at EEGLAB index 2"):
        pop_delset([_eeg()], 2)


def test_pop_delset_empties_slots_in_place_and_drops_trailing_empties():
    # EEGLAB pop_delset blanks the slot, so dataset 2 keeps its number; the emptied
    # last slot is dropped like `eeglab redraw` does.
    alleeg, command = pop_delset([_eeg(name="first"), _eeg(name="second"), _eeg(name="third")], [1, 3, 3])

    assert [eeg.get("setname") for eeg in alleeg] == [None, "second"]
    assert alleeg[0] == {}
    assert command == "ALLEEG = pop_delset( ALLEEG, [1, 3, 3] );"


def test_pop_delset_of_only_dataset_leaves_empty_list():
    alleeg, _command = pop_delset([_eeg()], 1)

    assert alleeg == []
