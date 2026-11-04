from tests.eeglab_tests import eeglab_test


@eeglab_test("unittesting_popfunc/eeg_emptyset/popfunc_eeg_emptyset_wrapperTest.m", "test_pass_general")
def test_reference_emptyset_returns_structure(eeglab_backend):
    assert isinstance(eeglab_backend("eeg_emptyset"), dict)
