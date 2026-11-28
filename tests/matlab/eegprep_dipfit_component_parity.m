function result = eegprep_dipfit_component_parity(dataset, model, kind, eeglab_root)
% Positions, residual variance, or inside-brain counts for one DIPFIT model.
% kind is 'grid', 'single', or 'symmetric'.
if nargin < 4
    eeglab_root = '';
end
dipfit_standard_besa_add_fieldtrip(eeglab_root);
if exist('pop_multifit', 'file') ~= 2 || exist('pop_dipfit_settings', 'file') ~= 2
    error('eegprep_dipfit_component_parity:MissingDipfit', 'DIPFIT must be on the path.');
end
if exist('ft_inside_headmodel', 'file') ~= 2
    error('eegprep_dipfit_component_parity:MissingFieldTrip', 'FieldTrip must be on the path.');
end

EEG = pop_loadset(dataset);
model = char(model);
kind = char(kind);
EEG = pop_dipfit_settings(EEG, 'model', model);

if strcmp(kind, 'grid')
    vol = local_volume(EEG, model);
    [x, y, z] = ndgrid(linspace(-85, 85, 24), linspace(-85, 85, 24), linspace(0, 85, 12));
    inside = ft_inside_headmodel([x(:) y(:) z(:)], vol);
    result.count = sum(inside(:));
    return
end

if strcmp(kind, 'symmetric')
    components = 1:10;
    dipoles = 2;
elseif strcmp(kind, 'single')
    components = 1:size(EEG.icaweights, 1);
    dipoles = 1;
else
    error('eegprep_dipfit_component_parity:Kind', ...
        'kind must be ''grid'', ''single'', or ''symmetric''.');
end

eegprep_require_fminunc('DIPFIT component parity');
fitted = pop_multifit(EEG, components, 'threshold', 100, 'dipoles', dipoles, 'dipplot', 'off');
positions = zeros(numel(components), 3 * dipoles);
residual = zeros(1, numel(components));
for index = 1:numel(components)
    component = components(index);
    xyz = fitted.dipfit.model(component).posxyz;
    if isempty(xyz) || size(xyz, 1) < dipoles || size(xyz, 2) < 3
        error('eegprep_dipfit_component_parity:EmptyFit', ...
            'Component %d did not return %d dipole positions.', component, dipoles);
    end
    xyz = xyz(1:dipoles, 1:3);
    positions(index, :) = reshape(xyz.', 1, []);
    residual(index) = fitted.dipfit.model(component).rv;
end
result.positions = positions;
result.rv = residual;
result.components = components;
end

function vol = local_volume(EEG, model)
if contains(lower(model), 'besa')
    addpath(fullfile(fileparts(which('dipfitdefs')), 'standard_BESA'));
    loaded = load('standard_BESA.mat');
    vol = ft_datatype_headmodel(loaded.vol);
    vol.unit = 'mm';
    chansel = EEG.dipfit.chansel;
    elecpos = [[EEG.chanlocs(chansel).X]; [EEG.chanlocs(chansel).Y]; [EEG.chanlocs(chansel).Z]]';
    labels = {EEG.chanlocs(chansel).labels};
    sens = struct('elecpos', elecpos, 'chanpos', elecpos, 'label', {labels(:)}, 'unit', 'mm');
    [vol, ~] = ft_prepare_vol_sens(vol, sens);
    return
end
hasCoord = false(1, EEG.nbchan);
for index = 1:EEG.nbchan
    coord = [EEG.chanlocs(index).X EEG.chanlocs(index).Y EEG.chanlocs(index).Z];
    hasCoord(index) = numel(coord) == 3 && all(isfinite(coord));
end
converted = eeglab2fieldtrip(EEG, 'componentanalysis', 'dipfit');
labels = {EEG.chanlocs(hasCoord).labels};
[present, order] = ismember(labels, converted.elec.label);
if any(~present)
    error('eegprep_dipfit_component_parity:Labels', ...
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
[vol, ~] = ft_prepare_vol_sens(vol, sens);
end
