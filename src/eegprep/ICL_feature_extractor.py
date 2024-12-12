import numpy as np

def ICL_feature_extractor(EEG, flag_autocorr=False):
    from eegprep import topoplot
    from eegprep import eeg_rpsd
    from eegprep import eeg_autocorr_welch
    from eegprep import eeg_autocorr
    from eegprep import eeg_autocorr_fftw
    from eegprep import pop_reref

    # Check inputs
    ncomp = EEG['icawinv'].shape[1]

    # Check for ICA key and if it is not empty
    if not 'icawinv' in EEG.keys() or EEG['icawinv'].size == 0:
        raise ValueError('You must have an ICA decomposition to use ICLabel')

    # Assuming chanlocs are correct
    if EEG['ref'] != 'average' and EEG['ref'] != 'averef':
        EEG = pop_reref(EEG, []) #, exclude=list(set(range(EEG.nbchan)) - set(EEG.icachansind)))
        # raise ValueError('Data must be rereferenced to average to use ICLabel')
        # EEG = pop_reref(EEG, [], exclude=list(set(range(EEG.nbchan)) - set(EEG.icachansind)))

    # Calculate ICA activations if missing and cast to double
    if EEG['icaact'] is None:
        raise ValueError('You must have ICA activations to use ICLabel')
        # EEG['icaact'] = eeg_getica(EEG)
        
    EEG['icaact'] = EEG['icaact'].astype(float)

    # Check ICA is real
    assert np.isreal(EEG['icaact']).all(), 'Your ICA decomposition must be real to use ICLabel'

    # Calculate topo
    topo = np.zeros((32, 32, 1, ncomp))
    for it in range(ncomp):
        _, temp_topo, _, _, _ = topoplot(EEG['icawinv'][:, it], EEG['chanlocs'][EEG['icachansind']], noplot='on')
        temp_topo[np.isnan(temp_topo)] = 0
        topo[:, :, 0, it] = temp_topo / np.max(np.abs(temp_topo))

    # Cast
    topo = topo.astype(np.float32)

    # Calculate PSD
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

    # Normalize
    psd = psd / np.max(np.abs(psd), axis=1)[:, np.newaxis]

    psd = np.transpose(np.expand_dims(np.expand_dims(psd, axis=-1), axis=-1), (2, 1, 3, 0)).astype(np.float32)

    # Calculate autocorrelation
    if flag_autocorr:
        if EEG['trials'] == 1:
            if EEG['pnts'] / EEG['srate'] > 5:
                autocorr = eeg_autocorr_welch(EEG)
            else:
                autocorr = eeg_autocorr(EEG)
        else:
            autocorr = eeg_autocorr_fftw(EEG)

        # Reshape and cast
        autocorr = np.transpose(np.expand_dims(np.expand_dims(autocorr, axis=-1), axis=-1), (2, 1, 3, 0)).astype(np.float32)

    # Format outputs
    if flag_autocorr:
        features = [0.99 * topo, 0.99 * psd, 0.99 * autocorr]
    else:
        features = [0.99 * topo, 0.99 * psd]
        
    return features

def test_ICL_feature_extractor():
    flag_autocorr = True
    EEG = EEG2
    EEG['ref'] = 'averef'
    EEG['chanlocs'] = np.random.randn(32, 3)