% Get the directory of the current file
base_dir = '/Users/arno';
path2eeglab = fullfile(base_dir, 'eeglab');
% Alternatively: path2eeglab = 'eeglab'; % init >10 seconds

addpath(fullfile(path2eeglab, 'functions', 'guifunc'));
addpath(fullfile(path2eeglab, 'functions', 'popfunc'));
addpath(fullfile(path2eeglab, 'functions', 'adminfunc'));
addpath(fullfile(path2eeglab, 'plugins', 'firfilt'));
addpath(fullfile(path2eeglab, 'functions', 'sigprocfunc'));
addpath(fullfile(path2eeglab, 'functions', 'miscfunc'));
addpath(fullfile(path2eeglab, 'plugins', 'dipfit'));
addpath(fullfile(path2eeglab, 'plugins', 'iclabel'));
addpath(fullfile(path2eeglab, 'plugins', 'clean_rawdata'));
addpath(fullfile(path2eeglab, 'plugins', 'clean_rawdata2.10'));

EEG = pop_loadset('tmp.set');
EEG = pop_resample(EEG, 100);
