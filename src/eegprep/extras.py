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
    "bids": "pybids",
    "mne": "mne",
    "oct2py": "eeglab",
    "picard": "picard",
}

#: Public names that need an extra for one path. The note says which path.
PUBLIC_EXTRAS: dict[str, tuple[str, str]] = {
    "bids_list_eeg_files": ("pybids", "reading a BIDS dataset"),
    "bids_preproc": ("pybids", "reading a BIDS dataset"),
    "clean_channels": ("eeglab", "the RANSAC octave subroutine"),
    "eeg2mne": ("mne", "converting to an MNE object"),
    "eeg_eeg2mne": ("mne", "converting to an MNE object"),
    "eeg_mne2eeg": ("mne", "converting from an MNE object"),
    "eeg_picard": ("picard", "Picard ICA"),
    "eeglab_clean_artifacts": ("eeglab", "the Octave runtime"),
    "iclabel": ("eeglab", "the Octave runtime"),
    "mne2eeg": ("mne", "converting from an MNE object"),
    "picard": ("picard", "Picard ICA"),
    "pop_biosig": ("mne", "EDF, BDF, and GDF import"),
    "pop_fileio": ("mne", "readers other than EEGLAB, CNT, and BrainVision"),
    "pop_importbids": ("pybids", "reading a BIDS dataset"),
    "pop_load_frombids": ("pybids", "reading a BIDS dataset"),
    "pop_resample": ("eeglab", "engine='octave'"),
}

#: What each extra is for. ``full`` restores the packages this release moved out of the base.
EXTRA_SUMMARY: dict[str, str] = {
    "eeglab": (
        "Octave runtime for pop_resample(engine='octave'), iclabel(engine='octave'), "
        "the clean_channels RANSAC octave subroutine, and eeglab_clean_artifacts. Installs oct2py."
    ),
    "mne": (
        "MNE conversion and the EDF, BDF, and GDF readers in pop_biosig and pop_fileio. "
        "EEGLAB .set, CNT, and BrainVision do not need it. Installs mne."
    ),
    "pybids": "BIDS import and bids_preproc. Installs pybids.",
    "picard": (
        "Picard ICA (eeg_picard, and pop_runica with icatype='picard'). "
        "The default runica path does not need it. Installs python-picard."
    ),
    "full": "The eeglab, mne, pybids, and picard extras together. This restores the previous complete install.",
}


def install_command(extra: str) -> str:
    """The command a missing-extra error tells the reader to run."""
    return f'pip install "eegprep[{extra}]"'


def import_optional_dependency(name: str, *, feature: str) -> ModuleType:
    """Import ``name``, or raise ``ImportError`` with the extra that provides it.

    Args:
        name: Import name, such as ``mne`` or ``mne.export``.
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
