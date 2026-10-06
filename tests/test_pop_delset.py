import numpy as np

from eegprep.functions.adminfunc.pop_delset import pop_delset
from tests.eeglab_tests import eeglab_test


def _eeg(name: str = "demo") -> dict:
    return {"setname": name}


@eeglab_test("unittesting_adminfunc/pop_delset/adminfunc_pop_delset_wrapperTest.m", "test_pass_set_in")
def test_reference_delset_original_datasets(eeglab_backend, eeglab_suite_root):
    directory = eeglab_suite_root / "unittesting_adminfunc/pop_delset"
    first = eeglab_backend("pop_loadset", str(directory / "test.set"), "")
    second = eeglab_backend("pop_loadset", str(directory / "test_2.set"), "")
    alleeg = np.array(
        [[tuple(first.values()), tuple(second[field] for field in first)]],
        dtype=[(field, object) for field in first],
    )
    result, _ = eeglab_backend("pop_delset", alleeg, 1.0, nargout=2)
    for field in result.dtype.names:
        value = result[field][0, 0]
        assert np.asarray(value).size == 0 or (isinstance(value, str) and value == "")


def test_pop_delset_empties_slots_in_place_and_drops_trailing_empties():
    # EEGLAB pop_delset blanks the slot, so dataset 2 keeps its number; the emptied
    # last slot is dropped like `eeglab redraw` does.
    alleeg, command = pop_delset([_eeg(name="first"), _eeg(name="second"), _eeg(name="third")], [1, 3, 3])

    assert [eeg.get("setname") for eeg in alleeg] == [None, "second"]
    assert alleeg[0] == {}
    assert command == "ALLEEG = pop_delset( ALLEEG, [1, 3, 3] );"
