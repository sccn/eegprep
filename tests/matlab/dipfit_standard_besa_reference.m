function result = dipfit_standard_besa_reference(dataset, eeglab_root)
% Leadfields and components 1-3 for standardBESA on one EEGLAB dataset.
%
% pop_multifit grid-searches every component. Components 4 and later do not
% change the fits of components 1-3, so this copy keeps only those three
% ICA columns and then calls pop_multifit.
if nargin < 2
    eeglab_root = '';
end
if nargin < 1 || isempty(dataset)
    here = fileparts(mfilename('fullpath'));
    dataset = fullfile(here, '..', '..', 'sample_data', 'eeglab_data_epochs_ica.set');
end
dipfit_standard_besa_add_fieldtrip(eeglab_root);
if isempty(which('ft_compute_leadfield'))
    error('FieldTrip is not on the MATLAB path (ft_compute_leadfield).');
end
if isempty(which('dipfitdefs'))
    error('DIPFIT is not on the MATLAB path (dipfitdefs).');
end
addpath(fullfile(fileparts(which('dipfitdefs')), 'standard_BESA'));

EEG = pop_loadset(dataset);
EEG = pop_dipfit_settings(EEG, 'model', 'standardBESA', 'coord_transform', []);
EEG.icaweights = EEG.icaweights(1:3, :);
EEG.icawinv = EEG.icawinv(:, 1:3);
if isfield(EEG, 'icaact')
    EEG = rmfield(EEG, 'icaact');
end
EEG = pop_multifit(EEG, 1:3, 'threshold', 100, 'dipplot', 'off');

chansel = EEG.dipfit.chansel;
elec_raw = [[EEG.chanlocs(chansel).X]; [EEG.chanlocs(chansel).Y]; [EEG.chanlocs(chansel).Z]]';
labels = {EEG.chanlocs(chansel).labels}';
loaded = load('standard_BESA.mat');
vol = ft_datatype_headmodel(loaded.vol);
vol.unit = 'mm';
sens = struct('elecpos', elec_raw, 'chanpos', elec_raw, 'label', {labels}, 'unit', 'mm');
[vol, sens] = ft_prepare_vol_sens(vol, sens);
dippos = [
    0 0 0
    0 0 40
    30 -20 10
    10 -50 20
    68.6342 1.3846 -38.0883
];
leadfield = ft_compute_leadfield(dippos, sens, vol);
fitpos = zeros(3, 3);
fitrv = zeros(3, 1);
for index = 1:3
    fitpos(index, :) = EEG.dipfit.model(index).posxyz(1, :);
    fitrv(index) = EEG.dipfit.model(index).rv;
end
result = struct( ...
    'elec_raw', elec_raw, ...
    'elec_projected', sens.elecpos, ...
    'dippos', dippos, ...
    'leadfield', leadfield, ...
    'fitpos', fitpos, ...
    'fitrv', fitrv);
end
