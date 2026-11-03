"""Import a library that lives behind an extra.

A missing optional library raises ``ImportError`` naming the library, the
feature that needs it, and the exact install command. An ``ImportError`` whose
missing module is something else is re-raised, so a broken dependency is not
reported as a missing extra.
"""

from __future__ import annotations

import importlib
from types import ModuleType

#: Import name to the extra that installs it.
OPTIONAL_LIBRARIES: dict[str, str] = {
    "oct2py": "eeglab",
    "psutil": "sys",
    "pyedflib": "edf",
}

#: Public names that need an extra for one path. The note says which path.
PUBLIC_EXTRAS: dict[str, tuple[str, str]] = {
    "pop_writeeeg": ("edf", "EDF and BDF output; GDF does not need it"),
}

#: What each extra is for. ``full`` restores the packages this release moved out of the base.
EXTRA_SUMMARY: dict[str, str] = {
    "eeglab": "Octave runtime for get_eeglab. Installs oct2py.",
    "sys": "RAM probing for asr_process, asr_process_r, and num_jobs_from_reservation. Installs psutil.",
    "edf": "EDF and BDF output from pop_writeeeg. Installs pyedflib. GDF output does not need it.",
    "full": "The eeglab, sys, and edf extras together. This restores the previous base install.",
}


def install_command(extra: str) -> str:
    """The command a missing-extra error tells the reader to run."""
    return f'pip install "eegprep[{extra}]"'


def import_optional_dependency(name: str, *, feature: str) -> ModuleType:
    """Import ``name``, or raise ``ImportError`` with the extra that provides it.

    Args:
        name: Import name, such as ``pyedflib`` or ``pyedflib.highlevel``.
        feature: Function or feature that needs the library, such as ``pop_writeeeg``.

    Returns:
        The imported module.

    Raises:
        ImportError: If the library is not installed. The message names the
            library, ``feature``, ``pip install "eegprep[<extra>]"``, and
            ``pip install "eegprep[full]"``. An ``ImportError`` for a different
            module is re-raised unchanged.
    """
    root = name.split(".", 1)[0]
    extra = OPTIONAL_LIBRARIES[root]
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        missing = (exc.name or "").split(".", 1)[0]
        if missing != root:
            raise
        raise ImportError(
            f"{feature} requires {root}, which is not installed ({exc}). "
            f"Install it with {install_command(extra)}. "
            f"{install_command('full')} installs this extra with the other packages "
            "that used to be part of the base install."
        ) from exc
