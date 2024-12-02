# Example to export MNE epochs to EEGLAB dataset
# Events are not handled correctly in this example but it works

import mne
from mne.datasets import sample
from mne.preprocessing import ICA
import math

import numpy as np
from scipy.io import savemat

# Load example data
def eeg_mne2eeglab_epochs(epochs, ica):
    
    # export to EEGLAB dataset
    data = epochs.get_data()  # Get the data from the epochs
    n_epochs, n_channels, n_times = data.shape
    ica_weights = ica.get_components()  # ICA weights (n_components x n_channels)
    
    # create identity matrix of size n_channels x n_channels
    ica_sphere = np.eye(n_channels)  # ICA sphere (n_channels x n_channels)

    # Compute the mixing matrix (inverse weights)
    ica_inverse_weights = np.linalg.pinv(ica_weights)  # Shape: (n_channels, n_components)
    
    ica_channels = ica.info['ch_names']
    raw_channels = epochs.info['ch_names']  # Assuming you have the raw object
    ica_channel_indices = [raw_channels.index(ch) for ch in ica_channels]
    ica_channel_indices = np.array(ica_channel_indices)
    
    ica_act = ica.get_sources(epochs).get_data(copy=True).transpose(1, 2, 0)  # Get the ICA activations
   
    print('Reference conversion may not be accurate...')
    if 'custom_ref_applied' in epochs.info and epochs.info['custom_ref_applied']:
        ref = 'common'  # Custom reference was applied
    else:
        ref = 'average'  # Default to average reference
    
    eeglab_dict = {
        'setname'         : '',
        'filename'        : '',
        'filepath'        : '',
        'subject'         : '',
        'group'           : '',
        'condition'       : '',
        'session'         : np.array([]),
        'comments'        : '',
        'nbchan'          : n_channels,
        'trials'          : n_epochs,
        'pnts'            : n_times,
        'srate'           : epochs.info['sfreq'],
        'xmin'            : epochs.times[0],
        'xmax'            : epochs.times[-1],
        'times'           : epochs.times,
        'data'            : data,
        'icaact'          : ica_act,
        'icawinv'         : ica_inverse_weights,
        'icasphere'       : ica_weights,
        'icaweights'      : ica_sphere,
        'icachansind'     : ica_channel_indices,
        'chanlocs'        : np.array([]),
        'urchanlocs'      : np.array([]),
        'chaninfo'        : np.array([]),
        'ref'             : ref,
        'event'           : np.array([]),
        'urevent'         : np.array([]),
        'eventdescription': np.array([]),
        'epoch'           : np.array([]),
        'epochdescription': np.array([]),
        'reject'          : np.array([]),
        'stats'           : np.array([]),
        'specdata'        : np.array([]),
        'specicaact'      : np.array([]),
        'splinefile'      : np.array([]),
        'icasplinefile'   : np.array([]),
        'dipfit'          : np.array([]),
        'history'         : np.array([]),
        'saved'           : np.array([]),
        'etc'             : np.array([]),
        'datfile'         : np.array([]),
        'run'             : np.array([]),
        'roi'             : np.array([]),
    }

    # create channel locations
    ch_names = epochs.ch_names
    
    ch_locs = epochs.info['chs']

    theta_all = []
    radius_all = []
    sph_theta_all  = []
    sph_phi_all    = []
    sph_radius_all = []
    X_all = []
    Y_all = []
    Z_all = []
    for ch in ch_locs:
        if 'loc' in ch and ch['loc'] is not None:
            X_all.append(ch['loc'][1]*1000)
            Y_all.append(-ch['loc'][0]*1000)
            Z_all.append(ch['loc'][2]*1000)
            hypotxy = math.hypot(X_all[-1],Y_all[-1])
            sph_radius_all.append(math.hypot(hypotxy,Z_all[-1]))
            
            az = math.atan2(Y_all[-1],X_all[-1])/math.pi*180
            horiz = math.atan2(Z_all[-1],hypotxy)/math.pi*180
            
            sph_theta_all.append(az)
            sph_phi_all.append(horiz)

            theta_all.append(-az) # warning inverse notation compared to MATLAB to match
            radius_all.append(0.5 - horiz/180) # warning inverse notation compared to MATLAB to match
        
    d_list = [{
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
        'ref': ''
    } for ch_name, theta, radius, X, Y, Z, sph_theta, sph_phi, sph_radius in zip(ch_names, theta_all, radius_all, X_all, Y_all, Z_all, sph_theta_all, sph_phi_all, sph_radius_all)]
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
    
    # # Step 4: Save the EEGLAB dataset as a .mat file
    return eeglab_dict

    #print("EEGLAB dataset saved successfully!")
    
def test_eeg_mne2eeeglab_epochs():
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = (
        sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
    )

    raw = mne.io.read_raw_fif(sample_data_raw_file)

    # extract data epochs    
    events = mne.find_events(raw, stim_channel="STI 014")
    event_dict = {
        "auditory/left": 1,
        "auditory/right": 2,
        "visual/left": 3,
        "visual/right": 4,
        "smiley": 5,
        "buttonpress": 32,
    }
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=-0.2,
        tmax=0.5,
        preload=True,
    )    

    ica = ICA(n_components=15, random_state=97, max_iter=800)
    ica.fit(raw)
    
    EEG = eeg_mne2eeglab_epochs(epochs, ica)
    savemat('output_file.mat', EEG) # use pop_saveset

# test_eeg_mne2eeeglab_epochs()