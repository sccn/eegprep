"""Import a library that lives behind an extra.

A missing optional library raises ``ImportError`` naming the library, the
function that needs it, and the install command. When another path does not
need the library, the message says what to do instead. An ``ImportError`` for
a different module is re-raised.
"""

from __future__ import annotations

import importlib
from types import ModuleType

#: Import name to the extra that installs it.
OPTIONAL_LIBRARIES: dict[str, str] = {
    "bids": "pybids",
    "mne": "mne",
    "defusedxml": "mne",
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
    "pop_fileio": ("mne", "EDF, BDF, GDF, EGI .mff, and BrainVision .vhdr with a block range"),
    "pop_iclabel": ("eeglab", "engine='octave'"),
    "pop_importbids": ("pybids", "reading a BIDS dataset"),
    "pop_load_frombids": ("pybids", "reading a BIDS dataset"),
    "pop_resample": ("eeglab", "engine='octave'"),
    "pop_runica": ("picard", "icatype='picard'"),
    "resample": ("eeglab", "engine='octave'"),
}

#: What each extra is for. ``full`` restores the packages this release moved out of the base.
EXTRA_SUMMARY: dict[str, str] = {
    "eeglab": (
        "Octave runtime for pop_resample(engine='octave'), iclabel(engine='octave'), "
        "the clean_channels RANSAC octave subroutine, and eeglab_clean_artifacts. Installs oct2py."
    ),
    "mne": (
        "MNE conversion and the EDF, BDF, GDF, and EGI .mff readers in pop_biosig and pop_fileio. "
        "BrainVision .vhdr files when a block range is set also use it. "
        "EEGLAB .set, CNT, Neuroscan .EEG, and BrainVision .vhdr without a block range do not. "
        "Installs mne and defusedxml for MFF XML parsing."
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


_LIBRARY_LABEL = {
    "bids": "PyBIDS",
    "mne": "MNE",
    "defusedxml": "defusedxml",
    "oct2py": "Octave (oct2py)",
    "picard": "Picard",
}

_INSTEAD = {
    "clean_channels": "The default subroutine, sphericalSplineInterpolate, does not need Octave or MATLAB.",
    "eeg_picard": "pop_runica() uses runica unless icatype is 'picard'.",
    "eeglab_clean_artifacts": "clean_artifacts is the Python cleaner and does not need Octave or MATLAB.",
    "get_eeglab": "Pass runtime='MAT' to use the MATLAB engine instead.",
    "iclabel": 'Leave engine unset for the Python network, which needs pip install "eegprep[iclabel]".',
    "picard": "pop_runica() uses runica unless icatype is 'picard'.",
    "pop_fileio": "EEGLAB .set, CNT, and BrainVision .vhdr without a block range do not need MNE.",
    "pop_iclabel": 'Leave engine unset for the Python network, which needs pip install "eegprep[iclabel]".',
    "pop_resample": "Leave engine unset to resample in Python.",
    "pop_runica": "Leave icatype unset to use runica, which does not need Picard.",
    "resample": "Leave engine unset to resample in Python.",
}


def _instead(feature: str) -> str:
    note = _INSTEAD.get(feature)
    return f" {note}" if note else ""


def missing_extra_message(feature: str, library: str) -> str:
    """Plain-language ``ImportError`` text for a library that lives behind an extra."""
    extra = OPTIONAL_LIBRARIES[library]
    return (
        f"{feature} needs {_LIBRARY_LABEL[library]}, and it is not installed. "
        f"Install it with {install_command(extra)}. "
        f"{install_command('full')} installs MNE, PyBIDS, Octave, and Picard."
        f"{_instead(feature)}"
    )


def missing_onnxruntime_message(feature: str) -> str:
    """Plain-language ``ImportError`` text for the packaged ICLabel network."""
    return (
        f"{feature} needs onnxruntime for the Python ICLabel network, and it is not installed. "
        f"Install it with {install_command('iclabel')}. "
        f"{install_command('gui')} and {install_command('all')} include it. "
        "engine='matlab' or engine='octave' runs ICLabel through EEGLAB instead. "
        f"Octave needs {install_command('eeglab')}."
    )


def missing_matlab_message(feature: str) -> str:
    """Plain-language ``ImportError`` text when the MATLAB engine is absent."""
    return (
        f"{feature} needs the MATLAB Engine for Python, which is not installed. "
        "Install it from your MATLAB installation with: "
        "pip install <matlabroot>/extern/engines/python. "
        f"To use Octave instead, run: {install_command('eeglab')}."
        f"{_instead(feature)}"
    )


def import_optional_dependency(name: str, *, feature: str) -> ModuleType:
    """Import ``name``, or raise ``ImportError`` with the extra that provides it.

    Args:
        name: Import name, such as ``mne`` or ``mne.export``.
        feature: Function that needs the library, such as ``pop_fileio``.

    Returns:
        The imported module.

    Raises:
        ImportError: If the library is not installed. The message names the
            library, ``feature``, ``pip install "eegprep[<extra>]"``, and
            ``pip install "eegprep[full]"``. An ``ImportError`` for a different
            module is re-raised unchanged.
    """
    root = name.split(".", 1)[0]
    if root not in OPTIONAL_LIBRARIES:
        raise KeyError(root)
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        missing = (exc.name or "").split(".", 1)[0]
        if missing != root:
            raise
        raise ImportError(missing_extra_message(feature, root)) from exc
