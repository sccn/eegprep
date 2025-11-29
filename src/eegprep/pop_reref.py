import logging
from copy import deepcopy
import numpy as np
from eegprep.eeg_checkset import eeg_checkset

logger = logging.getLogger(__name__)

def pop_reref(EEGin, ref, exclude=None):
    EEG = deepcopy(EEGin)

    # check if ref is not empty and not none
    if ref is not None and ref != []:
        raise ValueError('Feature not implemented: The ref parameter must be empty or None')

    # Determine which channels to re-reference
    if exclude is not None:
        # Convert exclude to 0-based indices if needed
        exclude_indices = np.array(exclude)
        # Create a mask of channels to include in re-referencing
        all_channels = np.arange(EEG['nbchan'])
        include_channels = np.setdiff1d(all_channels, exclude_indices)

        # Compute mean only from included channels
        mean_signal = np.mean(EEG['data'][include_channels, :], axis=0)
        # Subtract mean only from included channels
        EEG['data'][include_channels, :] = EEG['data'][include_channels, :] - mean_signal
    else:
        # Subtract mean from all EEG data using broadcasting in NumPy
        EEG['data'] = EEG['data'] - np.mean(EEG['data'], axis=0)

    # Handle ICA re-referencing if ICA was performed
    # icachansind indicates which channels were used for ICA
    # After channel removal, len(icachansind) may be less than nbchan
    if len(EEG['icachansind']) and EEG.get('icawinv') is not None and len(EEG['icawinv']) > 0:
        # Check if all current channels were used for ICA
        ica_all_chans = (len(EEG['icachansind']) == EEG['nbchan'] and
                        set(EEG['icachansind']) == set(range(EEG['nbchan'])))

        if ica_all_chans:
            # All channels used for ICA - can do average reference on icawinv
            # Subtract mean from EEG icawinv using broadcasting in NumPy
            EEG['icawinv'] = EEG['icawinv'] - np.mean(EEG['icawinv'], axis=0)

            # Compute the pseudoinverse of EEG['icawinv']
            EEG['icaweights'] = np.linalg.pinv(EEG['icawinv'])

            EEG['icasphere'] = np.eye(EEG['nbchan'])
        else:
            # Only subset of channels used for ICA
            # Re-reference not implemented for this case - skip ICA update
            logger.warning(f"Skipping ICA re-referencing: ICA was done on {len(EEG['icachansind'])} channels "
                          f"but current data has {EEG['nbchan']} channels")

        # Compute the ICA activations
        # data = EEG['data'].reshape(EEG['data'].shape[0], -1)
        # EEG['icaact'] = np.dot(EEG['icaweights'], data)
        # EEG['icaact'] = EEG['icaact'].reshape(EEG['icaweights'].shape[0], EEG['pnts'], EEG['trials'])

        # Set EEG['ref'] to 'average' (always, regardless of ICA status)
        EEG['ref'] = 'average'

        # Update the reference for each channel in EEG['chanlocs']
        for iChan in range(len(EEG['chanlocs'])):
            EEG['chanlocs'][iChan]['ref'] = 'average'

    # Call eeg_checkset to perform RMS scaling (like MATLAB)
    EEG = eeg_checkset(EEG)

    return EEG
