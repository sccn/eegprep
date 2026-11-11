function result = eegprep_dipfit_bem_reference(dataset)
%EEGPREP_DIPFIT_BEM_REFERENCE FieldTrip leadfield and EEGLAB pop_multifit positions.
if exist('ft_compute_leadfield', 'file') ~= 2 || exist('pop_multifit', 'file') ~= 2
    error('eegprep_dipfit_bem_reference:MissingBackend', ...
        'FieldTrip and DIPFIT must be on the path.');
end

EEG = pop_loadset(dataset);
EEG = pop_dipfit_settings(EEG, 'model', 'standardBEM');
hasCoord = false(1, EEG.nbchan);
for index = 1:EEG.nbchan
    coord = [EEG.chanlocs(index).X EEG.chanlocs(index).Y EEG.chanlocs(index).Z];
    hasCoord(index) = numel(coord) == 3 && all(isfinite(coord));
end
converted = eeglab2fieldtrip(EEG, 'componentanalysis', 'dipfit');
labels = {EEG.chanlocs(hasCoord).labels};
[present, order] = ismember(labels, converted.elec.label);
if any(~present)
    error('eegprep_dipfit_bem_reference:Labels', 'A coordinate channel is missing from eeglab2fieldtrip.');
end
elec = struct();
elec.label = labels(:);
elec.elecpos = converted.elec.elecpos(order, :);
elec.unit = 'mm';

vol = ft_read_headmodel('standard_vol.mat');
if ~isfield(vol, 'unit') || isempty(vol.unit)
    vol.unit = 'mm';
elseif ~strcmp(vol.unit, 'mm')
    vol = ft_convert_units(vol, 'mm');
end
[vol, elec] = ft_prepare_vol_sens(vol, elec);
dipoles = [0 -20 40; 30 -50 30; -25 10 45];
leadfield = ft_compute_leadfield(dipoles, elec, vol);

fitted = pop_multifit(EEG, 1:3, 'threshold', 100, 'dipplot', 'off');
positions = zeros(3, 3);
for component = 1:3
    xyz = fitted.dipfit.model(component).posxyz;
    xyz = xyz(:)';
    positions(component, :) = xyz(1:3);
end

result.labels = strjoin(labels, sprintf('\n'));
result.leadfield = leadfield;
result.positions = positions;
result.dipoles = dipoles;
end
