"""Read and restore the EEGLAB options file a MATLAB session actually loaded."""

from pathlib import Path

# pop_editoptions.m writes fullfile(homefolder, 'eeg_options.m'). homefolder is
# EEGOPTION_PATH or, when that is empty, '~'. Expand '~' here so the snapshot
# is the same absolute file fopen updates.
_LOCATE_OPTIONS = (
    "eeglab_options; "
    "eegprep_option_dir__ = '~'; "
    "if exist('EEGOPTION_PATH','var') && ~isempty(EEGOPTION_PATH), "
    "eegprep_option_dir__ = EEGOPTION_PATH; "
    "elseif ispc, eegprep_option_dir__ = getenv('USERPROFILE'); "
    "else, eegprep_option_dir__ = char(java.lang.System.getProperty('user.home')); "
    "end; "
    "if isstring(eegprep_option_dir__), eegprep_option_dir__ = char(eegprep_option_dir__); end; "
    "if ~isempty(eegprep_option_dir__) && eegprep_option_dir__(1) == '~', "
    "eegprep_option_dir__ = char(java.lang.System.getProperty('user.home')); "
    "end; "
    "eegprep_option_path__ = char(fullfile(eegprep_option_dir__, 'eeg_options.m'));"
)

# clear functions does not drop a script. eeg_options stays cached with the
# value pop_editoptions wrote until it is cleared by name and run again.
_RELOAD_OPTIONS = "clear functions; clear eeg_options; eeglab_options;"


def eeg_options_snapshot(engine):
    """Return ``(path, original_bytes)`` for the file ``pop_editoptions`` writes."""
    engine.eval(_LOCATE_OPTIONS, nargout=0)
    raw = engine.eval("eegprep_option_path__", nargout=1)
    text = str(raw).strip()
    if not text:
        return None, None
    path = Path(text).expanduser()
    original = path.read_bytes() if path.is_file() else None
    return path, original


def reload_eeg_options(engine) -> None:
    """Drop the cached ``eeg_options`` script and run the file now on disk."""
    engine.eval(_RELOAD_OPTIONS, nargout=0)


def restore_eeg_options(path, original):
    """Write back the options file captured by ``eeg_options_snapshot``.

    When that snapshot found no file, delete ``path`` instead of leaving one
    that ``pop_editoptions`` created.
    """
    if path is None:
        return
    if original is None:
        path.unlink(missing_ok=True)
        return
    path.write_bytes(original)
