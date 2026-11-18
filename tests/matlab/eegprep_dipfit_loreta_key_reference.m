function result = eegprep_dipfit_loreta_key_reference(dataset, model, points, eeglab_root)
% Leadfield of one source grid for the head model selected in pop_dipfit_settings.
if nargin < 4
    eeglab_root = '';
end
dipfit_standard_besa_add_fieldtrip(eeglab_root);
if exist('ft_compute_leadfield', 'file') ~= 2 || exist('ft_inside_headmodel', 'file') ~= 2
    error('eegprep_dipfit_loreta_key_reference:MissingFieldTrip', ...
        'FieldTrip must be on the path.');
end
if exist('pop_dipfit_settings', 'file') ~= 2
    error('eegprep_dipfit_loreta_key_reference:MissingDipfit', 'DIPFIT must be on the path.');
end

points = double(points);
if size(points, 2) ~= 3 && size(points, 1) == 3
    points = points.';
end
if size(points, 2) ~= 3
    error('eegprep_dipfit_loreta_key_reference:Points', 'Source points must have shape N x 3.');
end

EEG = pop_loadset(dataset);
EEG = pop_dipfit_settings(EEG, 'model', model);
if contains(lower(char(model)), 'besa')
    addpath(fullfile(fileparts(which('dipfitdefs')), 'standard_BESA'));
    chansel = EEG.dipfit.chansel;
    elecpos = [[EEG.chanlocs(chansel).X]; [EEG.chanlocs(chansel).Y]; [EEG.chanlocs(chansel).Z]]';
    labels = {EEG.chanlocs(chansel).labels};
    loaded = load('standard_BESA.mat');
    vol = ft_datatype_headmodel(loaded.vol);
    vol.unit = 'mm';
    sens = struct('elecpos', elecpos, 'chanpos', elecpos, 'label', {labels(:)}, 'unit', 'mm');
    [vol, sens] = ft_prepare_vol_sens(vol, sens);
else
    hasCoord = false(1, EEG.nbchan);
    for index = 1:EEG.nbchan
        coord = [EEG.chanlocs(index).X EEG.chanlocs(index).Y EEG.chanlocs(index).Z];
        hasCoord(index) = numel(coord) == 3 && all(isfinite(coord));
    end
    converted = eeglab2fieldtrip(EEG, 'componentanalysis', 'dipfit');
    labels = {EEG.chanlocs(hasCoord).labels};
    [present, order] = ismember(labels, converted.elec.label);
    if any(~present)
        error('eegprep_dipfit_loreta_key_reference:Labels', ...
            'A coordinate channel is missing from eeglab2fieldtrip.');
    end
    sens = struct();
    sens.label = labels(:);
    sens.elecpos = converted.elec.elecpos(order, :);
    sens.unit = 'mm';
    vol = ft_read_headmodel('standard_vol.mat');
    if ~isfield(vol, 'unit') || isempty(vol.unit)
        vol.unit = 'mm';
    elseif ~strcmp(vol.unit, 'mm')
        vol = ft_convert_units(vol, 'mm');
    end
    [vol, sens] = ft_prepare_vol_sens(vol, sens);
end

leadfield = ft_compute_leadfield(points, sens, vol);
inside = ft_inside_headmodel(points, vol);
result.labels = strjoin(labels, sprintf('\n'));
result.leadfield = leadfield;
result.inside = double(inside(:));
result.points = points;
end
