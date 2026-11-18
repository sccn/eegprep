from eegprep import __version__
from eegprep.functions.adminfunc.eeg_getversion import eeg_getversion, numeric_version


def test_eeg_getversion_reports_the_package_version():
    # EEGLAB embeds ``'dev'`` in a source checkout. EEGPrep reports the
    # installed package version and still collapses extra dots the way
    # eeg_getversion.m does (11.3.2.4b -> 11.324).
    result = eeg_getversion()

    assert result.version == __version__
    assert result.version != "dev"
    assert result.matlab_primary == __version__
    assert result.number == numeric_version(__version__)
    assert numeric_version("11.3.2.4b") == 11.324
    assert numeric_version("0.5.0") == 0.5
    assert numeric_version("dev") is None
