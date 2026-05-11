import pytest

from eegprep.functions.adminfunc.pop_delset import pop_delset


def _eeg(name: str = "demo") -> dict:
    return {"setname": name}


def test_pop_delset_rejects_non_positive_indices():
    with pytest.raises(ValueError, match="1-based"):
        pop_delset([_eeg()], -1)
    with pytest.raises(ValueError, match="1-based"):
        pop_delset([_eeg()], 0)


def test_pop_delset_deletes_unique_indices_in_reverse_order():
    alleeg, command = pop_delset([_eeg(name="first"), _eeg(name="second"), _eeg(name="third")], [1, 3, 3])

    assert [eeg["setname"] for eeg in alleeg] == ["second"]
    assert command == "ALLEEG = pop_delset( ALLEEG, [1, 3, 3] );"
