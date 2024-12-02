# Example to export MNE epochs to EEGLAB dataset
# Events are not handled correctly in this example but it works

import mne
from mne.datasets import sample
import numpy as np
from scipy.io import savemat

# Load example data
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = (
    sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
)
raw = mne.io.read_raw_fif(sample_data_raw_file)
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

aud_epochs = epochs["auditory"]

# export to EEGLAB dataset
data = epochs.get_data()  # Get the data from the epochs
n_epochs, n_channels, n_times = data.shape

# Create a dictionary to save as a MATLAB file
    #          setname: 'EEG Data epochs'
    #         filename: 'eeglab_data_epochs_ica.set'
    #         filepath: '/System/Volumes/Data/data/matlab/eeglab/sample_data'
    #          subject: ''
    #            group: ''
    #        condition: ''
    #          session: []
    #         comments: [9×769 char]
    #           nbchan: 32
    #           trials: 80
    #             pnts: 384
    #            srate: 128
    #             xmin: -1
    #             xmax: 1.9922
    #            times: [-1000 -992.1875 -984.3750 -976.5625 -968.7500 -960.9375 -953.1250 -945.3125 -937.5000 -929.6875 -921.8750 -914.0625 -906.2500 -898.4375 -890.6250 -882.8125 -875 -867.1875 -859.3750 -851.5625 -843.7500 -835.9375 … ] (1×384 double)
    #             data: [32×384×80 single]
    #           icaact: [32×384×80 single]
    #          icawinv: [32×32 double]
    #        icasphere: [32×32 double]
    #       icaweights: [32×32 double]
    #      icachansind: [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32]
    #         chanlocs: [1×32 struct]
    #       urchanlocs: [1×32 struct]
    #         chaninfo: [1×1 struct]
    #              ref: 'common'
    #            event: [1×157 struct]
    #          urevent: [1×154 struct]
    # eventdescription: {[2×29 char]  [2×63 char]  [2×36 char]  ''  ''}
    #            epoch: [1×80 struct]
    # epochdescription: {}
    #           reject: [1×1 struct]
    #            stats: [1×1 struct]
    #         specdata: []
    #       specicaact: []
    #       splinefile: []
    #    icasplinefile: ''
    #           dipfit: []
    #          history: 'EEG = eeg_checkset( EEG );↵EEG = eeg_checkset( EEG );↵figure;pop_topoplot(EEG,0, 1, 'EEG Data epochs',[1 1] ,0, 'electrodes', 'on', 'masksurf', 'on');↵EEG = eeg_checkset( EEG );↵EEG.chanlocs=pop_chanedit(EEG.chanlocs,  'plotrad',0.5);↵EEG = pop_loadset( 'filename', 'eeglab_data_epochs_ica.set', 'filepath', '/data/common/matlab/eeglab/sample_data/');↵EEG=pop_chanedit(EEG,  'lookup', '/data/common/matlab/eeglab/plugins/dipfit2.0/standard_BESA/standard-10-5-cap385.elp');↵EEG = pop_saveset( EEG,  'savemode', 'resave');↵EEG=pop_chanedit(EEG,  'eval', 'pop_writelocs( chans, ''/data/common/matlab/eeglab/sample_data/eeglab_chan32_2.locs'',  ''filetype'', ''loc'', ''format'',{ ''channum'', ''theta'', ''radius'', ''labels''}, ''header'', ''off'', ''customheader'','''');', 'load',{ '/data/common/matlab/eeglab/sample_data/eeglab_chan32_2.locs', 'filetype', 'autodetect'}, 'lookup', '/data/common/matlab/eeglab/plugins/dipfit2.0/standard_BESA/standard-10-5-cap385.elp');↵EEG=pop_chanedit(EEG,  'settype',{ '1:32', 'EEG'}, 'changefield',{2, 'type', 'EOG'}, 'changefield',{6, 'type', 'EOG'});↵EEG = pop_loadset( 'filename', 'eeglab_data_epochs_ica.set', 'filepath', '/data/common/matlab/eeglab/sample_data/');↵EEG = pop_loadset('filename','eeglab_data_epochs_ica.set','filepath','/Users/arno/eeglab/sample_data/');↵EEG = eeg_checkset( EEG );↵EEG.etc.eeglabvers = '2024.2'; % this tracks which version of EEGLAB is being used, you may ignore it'
    #            saved: 'yes'
    #              etc: [1×1 struct]
    #          datfile: 'eeglab_data_epochs_ica.fdt'
    #              run: []
    #              roi: []
    
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
    'xmin'            : aud_epochs.tmin,
    'xmax'            : aud_epochs.tmax,
    'times'           : epochs.times,
    'data'            : data,
    'icaact'          : np.array([]),
    'icawinv'         : np.array([]),
    'icasphere'       : np.array([]),
    'icaweights'      : np.array([]),
    'icachansind'     : np.array([]),
    'chanlocs'        : np.array([]),
    'urchanlocs'      : np.array([]),
    'chaninfo'        : np.array([]),
    'ref'             : np.array([]),
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
ch_locs = np.array([[i, 0, 0] for i in range(n_channels)])

# Create the list of dictionaries with a string field
d_list = [{
    'labels': ch_name,
    'theta': 0,
    'radius': 0,
    'X': ch_loc[0],
    'Y': ch_loc[1],
    'Z': ch_loc[2],
    'sph_theta': 0,
    'sph_phi': 0,
    'sph_radius': 0,
    'type': 'EEG',
    'urchan': 0,
    'ref': ''
} for ch_name, ch_loc in zip(ch_names, ch_locs)]

# Define the data type for the structured array, including a string field
# dtype = np.dtype([
#     ('labels', np.str_),
#     ('theta', np.float64),
#     ('radius', np.float64),
#     ('X', np.float64),
#     ('Y', np.float64),
#     ('Z', np.float64),
#     ('sph_theta',  np.float64),
#     ('sph_phi',    np.float64),
#     ('sph_radius', np.float64),
#     ('type', np.str_),
#     ('urchan', np.uint32),
#     ('ref', np.str_),
# ])

dtype = np.dtype([
    ('labels', 'U100'),      # String up to 100 characters
    ('theta', np.float64),
    ('radius', np.float64),
    ('X', np.float64),
    ('Y', np.float64),
    ('Z', np.float64),
    ('sph_theta', np.float64),
    ('sph_phi', np.float64),
    ('sph_radius', np.float64),
    ('type', 'U10'),         # String up to 10 characters
    ('urchan', np.int32),
    ('ref', 'U100')          # String up to 100 characters
])

# Convert the list of dictionaries to a structured NumPy array
eeglab_dict['chanlocs'] = np.array([
    (
        item['labels'],
        item['theta'],
        item['radius'],
        item['X'],
        item['Y'],
        item['Z'],
        item['sph_theta'],
        item['sph_phi'],
        item['sph_radius'],
        item['type'],
        item['urchan'],
        item['ref']
    )
    for item in d_list
], dtype=dtype)

# ch_dict = [{'labels': ch_name, 'X': ch_loc[0], 'Y': ch_loc[1], 'Z': ch_loc[2]} for ch_name, ch_loc in zip(ch_names, ch_locs)]
# eeglab_dict['chanlocs'] = np.array(ch_dict)
# eeglab_dict['urchanlocs'] = ch_dict

#    'event': np.array([[event[0], 0, event[2]] for event in events])  # Create event array

# # Step 4: Save the EEGLAB dataset as a .mat file
savemat('output_file.mat', eeglab_dict)

#print("EEGLAB dataset saved successfully!")
