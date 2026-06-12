"""EEG boundary finding functions."""

from eegprep.functions.miscfunc.event_utils import boundary_event_indices


def eeg_findboundaries(*, EEG):
    """
    EEG_FINDBOUNDARIES - return indices of boundary events.

    Usage:
        boundaries = eeg_findboundaries(EEG)
        boundaries = eeg_findboundaries(EEG=EEG['event'])  # if passing events directly

    Inputs:
        EEG - input EEG dataset structure (dict) or EEG['event'] (list of dicts)

    Outputs:
        boundaries - indices of boundary events (0-based)
    """
    # nargin check
    if EEG is None:
        # In MATLAB: help eeg_findboundaries; return
        return []

    return boundary_event_indices(EEG)
