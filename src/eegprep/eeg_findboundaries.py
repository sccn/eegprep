from eegprep.eeg_options import EEG_OPTIONS

def eeg_findboundaries(*, EEG):
    """
    EEG_FINDBOUNDARIES - return indices of boundary events

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

    boundaries = []
    # isempty(EEG)
    if EEG == {} or EEG == []:
        return boundaries

    # Determine tmpevent
    if isinstance(EEG, dict) and ('event' in EEG and 'setname' in EEG):
        tmpevent = EEG['event']
    else:
        tmpevent = EEG

    # If tmpevent lacks 'type'
    if isinstance(tmpevent, list):
        if len(tmpevent) == 0 or not isinstance(tmpevent[0], dict) or 'type' not in tmpevent[0]:
            return boundaries
    elif isinstance(tmpevent, dict):
        if 'type' not in tmpevent:
            return boundaries
        # Normalize to list for unified handling
        tmpevent = [tmpevent]
    else:
        return boundaries

    first_type = tmpevent[0].get('type')
    if isinstance(first_type, str):
        # boundaries = strmatch('boundary', { tmpevent.type });
        boundaries = [i for i, ev in enumerate(tmpevent)
                      if isinstance(ev.get('type'), str) and ev.get('type', '').startswith('boundary')]
    elif EEG_OPTIONS['option_boundary99']:
        # boundaries = find([ tmpevent.type ] == -99);
        boundaries = [i for i, ev in enumerate(tmpevent)
                      if ev.get('type') == -99]
    else:
        boundaries = []

    return boundaries