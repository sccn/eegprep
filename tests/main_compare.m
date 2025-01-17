clear

% this script compares the MATLAB and Python version of the function
cd('..')
pythonFunc = '.venv/bin/python';
pyenv('Version', pythonFunc);
system([pythonFunc ' main.py']);
cd('tests');

addpath(fullfile(pwd, '..', 'eeglab'));
if ~exist('pop_loadset')
    eeglab;
end

currentFolder = fileparts(mfilename('fullpath'));

% Read config.json.example (adjust the file name accordingly)
configFile = fullfile(currentFolder, '../config.json.example');
if exist(configFile, 'file')
    fid = fopen(configFile);
    raw = fread(fid, inf);
    fclose(fid);
    configData = jsondecode(char(raw'));
else
    error('Config file not found: %s', configFile);
end

% Get the filename from the config
fname = [ '../' configData.set ];

% Create temporary file and output file names
fname_tmp = strrep(fname, '.set', '_tmp2.set');
fname_out = strrep(fname, '.set', '_out2.set');

EEG = pop_loadset(fname)
EEG = pop_eegfiltnew(EEG, 'locutoff', 5,'hicutoff',25,'revfilt', 1,'plotfreqz', 0);
EEG = clean_artifacts(EEG, ...
    'FlatlineCriterion', 5, ...
    'ChannelCriterion', 0.87, ...
    'LineNoiseCriterion', 4, ...
    'Highpass', [0.25, 0.75], ...
    'BurstCriterion', 20, ...
    'WindowCriterion', 0.25, ...
    'BurstRejection', true, ...
    'WindowCriterionTolerances', [-inf, 7]);
EEG = pop_runica(EEG, 'icatype', 'picard');
pop_saveset(EEG, fname_out)

% compare
EEG_py = pop_loadset('../data/eeglab_data_with_ica_tmp_out.set');
EEG_mat = pop_loadset('../data/eeglab_data_with_ica_tmp_out2.set');
eeg_eventtypes(EEG_mat)
eeg_eventtypes(EEG_py)
