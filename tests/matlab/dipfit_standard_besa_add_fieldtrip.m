function dipfit_standard_besa_add_fieldtrip(eeglab_root)
% Add the FieldTrip copy shipped under an EEGLAB plugins directory.
if exist('ft_compute_leadfield', 'file')
    return
end
if nargin < 1 || isempty(eeglab_root)
    return
end
candidates = [ ...
    dir(fullfile(eeglab_root, 'plugins', 'Fieldtrip*', 'ft_defaults.m')); ...
    dir(fullfile(eeglab_root, 'plugins', 'fieldtrip*', 'ft_defaults.m'))];
if isempty(candidates)
    return
end
addpath(candidates(1).folder);
ft_defaults;
end
