"""Read and restore the EEGLAB options file a MATLAB session actually loaded."""

from pathlib import Path


def eeg_options_snapshot(engine):
    """Return ``(path, original_bytes)`` for the file ``eeglab_options`` loaded.

    ``eeglab_options`` is a script. A returned if-statement is illegal in
    MATLAB; the ``option_file`` variable it assigns is a valid expression.
    """
    engine.eval("eeglab_options;", nargout=0)
    raw = engine.eval("char(option_file)", nargout=1)
    text = str(raw).strip()
    if not text:
        return None, None
    path = Path(text).expanduser()
    original = path.read_bytes() if path.is_file() else None
    return path, original


def restore_eeg_options(path, original):
    if path is None:
        return
    if original is None:
        path.unlink(missing_ok=True)
        return
    path.write_bytes(original)
