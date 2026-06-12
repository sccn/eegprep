"""MNE epochs to EEGLAB dataset conversion utilities."""

import logging
import math

import numpy as np

from eegprep.functions.miscfunc.misc import finite_matmul, finite_pinv

logger = logging.getLogger(__name__)


def eeg_mne2eeg_epochs(epochs, ica):
    """Convert MNE epochs with ICA to EEGLAB dataset format.

    Parameters
    ----------
    epochs : mne.Epochs
        MNE epochs object.
    ica : mne.preprocessing.ICA
        MNE ICA object.

    Returns
    -------
    dict
        EEGLAB-compatible dataset dictionary.
    """
    mne_data = epochs.get_data(copy=True)
    n_epochs, n_channels, n_times = mne_data.shape
    data = np.transpose(mne_data, (1, 2, 0))

    ica_channels = ica.info['ch_names']
    raw_channels = epochs.info['ch_names']  # Assuming you have the raw object
    ica_channel_indices = [raw_channels.index(ch) for ch in ica_channels]
    ica_channel_indices = np.array(ica_channel_indices)

    ica_weights, ica_sphere, ica_inverse_weights, ica_act = _mne_ica_to_eeglab_fields(
        ica,
        data[ica_channel_indices],
        n_times,
        n_epochs,
    )

    if 'custom_ref_applied' in epochs.info and epochs.info['custom_ref_applied']:
        ref = 'common'  # Custom reference was applied
    else:
        ref = 'average'  # Default to average reference
    logger.info("MNE reference metadata converted to EEGPrep ref=%s.", ref)

    eeglab_dict = {
        'setname': '',
        'filename': '',
        'filepath': '',
        'subject': '',
        'group': '',
        'condition': '',
        'session': np.array([]),
        'comments': '',
        'nbchan': n_channels,
        'trials': n_epochs,
        'pnts': n_times,
        'srate': epochs.info['sfreq'],
        'xmin': epochs.times[0],
        'xmax': epochs.times[-1],
        'times': epochs.times,
        'data': data,
        'icaact': ica_act,
        'icawinv': ica_inverse_weights,
        'icasphere': ica_sphere,
        'icaweights': ica_weights,
        'icachansind': ica_channel_indices,
        'chanlocs': np.array([]),
        'urchanlocs': np.array([]),
        'chaninfo': np.array([]),
        'ref': ref,
        'event': np.array([]),
        'urevent': np.array([]),
        'eventdescription': np.array([]),
        'epoch': np.array([]),
        'epochdescription': np.array([]),
        'reject': np.array([]),
        'stats': np.array([]),
        'specdata': np.array([]),
        'specicaact': np.array([]),
        'splinefile': np.array([]),
        'icasplinefile': np.array([]),
        'dipfit': np.array([]),
        'history': np.array([]),
        'saved': np.array([]),
        'etc': np.array([]),
        'datfile': np.array([]),
        'run': np.array([]),
        'roi': np.array([]),
    }

    # create channel locations
    ch_names = epochs.ch_names

    ch_locs = epochs.info['chs']

    theta_all = []
    radius_all = []
    sph_theta_all = []
    sph_phi_all = []
    sph_radius_all = []
    X_all = []
    Y_all = []
    Z_all = []
    for ch in ch_locs:
        loc = ch.get('loc') if isinstance(ch, dict) else None
        if loc is None or len(loc) < 3:
            x = y = z = 0.0
        else:
            x = float(loc[1]) * 1000
            y = -float(loc[0]) * 1000
            z = float(loc[2]) * 1000
        X_all.append(x)
        Y_all.append(y)
        Z_all.append(z)
        hypotxy = math.hypot(x, y)
        sph_radius_all.append(math.hypot(hypotxy, z))

        az = math.atan2(y, x) / math.pi * 180
        horiz = math.atan2(z, hypotxy) / math.pi * 180

        sph_theta_all.append(az)
        sph_phi_all.append(horiz)

        theta_all.append(-az)  # warning inverse notation compared to MATLAB to match
        radius_all.append(0.5 - horiz / 180)  # warning inverse notation compared to MATLAB to match

    d_list = [
        {
            'labels': ch_name,
            'theta': theta,
            'radius': radius,
            'X': X,
            'Y': Y,
            'Z': Z,
            'sph_theta': sph_theta,
            'sph_phi': sph_phi,
            'sph_radius': sph_radius,
            'type': 'EEG',
            'urchan': 0,
            'ref': '',
        }
        for ch_name, theta, radius, X, Y, Z, sph_theta, sph_phi, sph_radius in zip(
            ch_names, theta_all, radius_all, X_all, Y_all, Z_all, sph_theta_all, sph_phi_all, sph_radius_all
        )
    ]
    # Create the list of dictionaries with a string field
    # d_list = [{
    #     'labels': ch_name,
    #     'theta': math.atan2(ch_loc[0], ch_loc[1]),
    #     'radius': math.hypot(ch_loc[1], ch_loc[0]),
    #     'X': ch_loc[1]*1000,
    #     'Y': ch_loc[0]*1000,
    #     'Z': ch_loc[2]*1000,
    #     'sph_theta': 0,
    #     'sph_phi': 0,
    #     'sph_radius': 0,
    #     'type': 'EEG',
    #     'urchan': 0,
    #     'ref': ''
    # } for ch_name, ch_loc in zip(ch_names, ch_locs_xyz)]

    # convert d_list to a numpy array
    d_list = np.array(d_list)
    eeglab_dict['chanlocs'] = d_list

    return eeglab_dict


def _mne_ica_to_eeglab_fields(ica, data, n_times, n_epochs):
    n_components = int(ica.n_components_)
    n_ica_channels = data.shape[0]
    prewhitener = _prewhitener_matrix(ica, n_ica_channels)
    pca_unmixing = finite_matmul(np.asarray(ica.unmixing_matrix_), np.asarray(ica.pca_components_)[:n_components])
    unmixing = finite_matmul(pca_unmixing, prewhitener)
    sphere = np.eye(n_ica_channels)
    inverse_weights = finite_pinv(unmixing)
    activations_2d = finite_matmul(unmixing, data.reshape(n_ica_channels, -1, order="F"))
    activations = activations_2d.reshape(n_components, n_times, n_epochs, order="F")
    return unmixing, sphere, inverse_weights, activations


def _prewhitener_matrix(ica, n_channels):
    prewhitener = np.asarray(ica.pre_whitener_)
    if ica.noise_cov is not None:
        if prewhitener.shape != (n_channels, n_channels):
            raise ValueError("MNE ICA pre-whitener has incompatible shape")
        return prewhitener
    values = prewhitener.reshape(-1)
    if values.size == 1:
        return np.eye(n_channels) / float(values[0])
    if values.size != n_channels:
        raise ValueError("MNE ICA pre-whitener has incompatible shape")
    return np.diag(1.0 / values)
