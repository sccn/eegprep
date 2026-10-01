function exists = eegprep_source_eeg_global()
% Caller-workspace operations from eeg_global/pass_general.m. Assertions stay
% in Python; do not return or replace the actual global datasets.
clear('EEG');
clear('ALLEEG');
clear('CURRENTSET');
clear('LASTCOM');
clear('ALLCOM');
eeg_global;
exists = [exist('EEG') exist('ALLEEG') exist('CURRENTSET') exist('LASTCOM') exist('ALLCOM')];
end
