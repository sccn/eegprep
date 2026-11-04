"""ICLabel feature extraction functions."""

from copy import deepcopy
import numpy as np


def ICL_feature_extractor(EEG, flag_autocorr=False):
    """Extract features for ICLabel classification.

    Parameters
    ----------
    EEG : dict
        EEG data structure with ICA
    flag_autocorr : bool, optional
        Whether to include autocorrelation features (default False)

    Returns
    -------
    features : list
        List of feature arrays
    """
    from eegprep import topoplot
    from eegprep import eeg_rpsd
    from eegprep import eeg_autocorr_welch
    from eegprep import eeg_autocorr
    from eegprep import eeg_autocorr_fftw
    from eegprep import pop_reref

    EEG = deepcopy(EEG)

    if 'icawinv' not in EEG.keys() or EEG['icawinv'].size == 0:
        raise ValueError('You must have an ICA decomposition to use ICLabel')

    ncomp = EEG['icawinv'].shape[1]

    # Assuming chanlocs are correct
    if EEG.get('ref') != 'average' and EEG.get('ref') != 'averef':
        EEG = pop_reref(EEG, [])

    if EEG['icaact'] is None:
        raise ValueError('You must have ICA activations to use ICLabel')

    EEG['icaact'] = EEG['icaact'].astype(float)

    assert np.isreal(EEG['icaact']).all(), 'Your ICA decomposition must be real to use ICLabel'

    topo = np.zeros((32, 32, 1, ncomp))
    for it in range(ncomp):
        tmp_chanlocs = [EEG['chanlocs'][i] for i in EEG['icachansind']]
        _, temp_topo, _, _, _ = topoplot(EEG['icawinv'][:, it], tmp_chanlocs, noplot='on', gridscale=32)
        temp_topo[np.isnan(temp_topo)] = 0
        topo[:, :, 0, it] = temp_topo / np.max(np.abs(temp_topo))

    # Cast
    topo = topo.astype(np.float32)

    psd = eeg_rpsd(EEG, 100)

    # Extrapolate or prune as needed
    nfreq = psd.shape[1]
    if nfreq < 100:
        psd = np.hstack((psd, np.tile(psd[:, -1][:, np.newaxis], (1, 100 - nfreq))))

    # Undo notch filter
    for linenoise_ind in [50, 60]:
        linenoise_around = [linenoise_ind - 1, linenoise_ind + 1]
        difference = psd[:, linenoise_around] - psd[:, linenoise_ind][:, np.newaxis]
        notch_ind = np.all(difference > 5, axis=1)
        if np.any(notch_ind):
            psd[notch_ind, linenoise_ind] = np.mean(psd[notch_ind][:, linenoise_around], axis=1)

    psd = psd / np.max(np.abs(psd), axis=1)[:, np.newaxis]

    psd = np.transpose(np.expand_dims(np.expand_dims(psd, axis=-1), axis=-1), (2, 1, 3, 0)).astype(np.float32)

    if flag_autocorr:
        if EEG['trials'] == 1:
            if EEG['pnts'] / EEG['srate'] > 5:
                autocorr = eeg_autocorr_welch(EEG)
            else:
                autocorr = eeg_autocorr(EEG)
        else:
            autocorr = eeg_autocorr_fftw(EEG)

        autocorr = np.transpose(np.expand_dims(np.expand_dims(autocorr, axis=-1), axis=-1), (2, 1, 3, 0)).astype(
            np.float32
        )

    # Format outputs
    if flag_autocorr:
        features = [0.99 * topo, 0.99 * psd, 0.99 * autocorr]
    else:
        features = [0.99 * topo, 0.99 * psd]

    return features
