from eegprep import iclabel, pop_loadset, pop_saveset, pop_eegfiltnew, clean_artifacts, eeg_picard

fname = "sample_data/eeglab_data_with_ica_tmp.set"
EEG = pop_loadset(fname)
EEG = clean_artifacts(EEG, FlatlineCriterion=5,ChannelCriterion=0.87, LineNoiseCriterion=4, \
                      Highpass=[0.25, 0.75], BurstCriterion= 20, WindowCriterion=0.25, \
                      BurstRejection=True, WindowCriterionTolerances=[float('-inf'), 7])
EEG = eeg_EEG)
EEG = iclabel(EEG)
pop_saveset(EEG, fname.replace('.set', '_out.set'))

