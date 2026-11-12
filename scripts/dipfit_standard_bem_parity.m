function dipfit_standard_bem_parity(dataset)
%DIPFIT_STANDARD_BEM_PARITY Compare EEGPrep with FieldTrip and EEGLAB DIPFIT.
%
%   cd('/path/to/eegprep')
%   addpath('scripts')
%   eeglab nogui
%   dipfit_standard_bem_parity
%
% Prints the leadfield max relative error, the 3-D distance in mm, and the
% residual variance of components 1 to 3 on sample_data/eeglab_data_epochs_ica.set.
% Both sides use standardBEM, the cap385 coregistration stored by
% pop_dipfit_settings, and EEGLAB's default grid.

if exist('ft_compute_leadfield', 'file') ~= 2
    error('dipfit_standard_bem_parity:FieldTrip', ...
        'ft_compute_leadfield is not on the path. Start EEGLAB with DIPFIT first.');
end

repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo, 'tests', 'matlab'));
if nargin < 1 || isempty(dataset)
    dataset = fullfile(repo, 'sample_data', 'eeglab_data_epochs_ica.set');
end
jsonFile = [tempname '.json'];
cleaner = onCleanup(@() delete_if_present(jsonFile));
[status, message] = system(python_command(repo, dataset, jsonFile, true));
if status ~= 0
    [status, message] = system(python_command(repo, dataset, jsonFile, false));
end
if status ~= 0
    error('dipfit_standard_bem_parity:Python', 'EEGPrep parity command failed:\n%s', message);
end
payload = jsondecode(fileread(jsonFile));

reference = eegprep_dipfit_bem_reference(dataset);
pythonLabels = cellstr(payload.labels);
[present, order] = ismember(pythonLabels, strsplit(reference.labels, sprintf('\n')));
if any(~present)
    error('dipfit_standard_bem_parity:Labels', 'Python and EEGLAB electrode labels do not match.');
end
pythonLeadfield = payload.leadfield;
matlabLeadfield = reference.leadfield(order, :);
relative = max(abs(pythonLeadfield(:) - matlabLeadfield(:))) / max(abs(matlabLeadfield(:)));
fprintf('leadfield max relative error: %.6e\n', relative);

pythonPositions = payload.positions;
pythonRv = payload.rv(:);
matlabRv = reference.rv(:);
for component = 1:size(reference.positions, 1)
    distance = norm(pythonPositions(component, :) - reference.positions(component, :));
    fprintf('component %d distance_mm: %.4f python_rv: %.8f eeglab_rv: %.8f\n', ...
        component, distance, pythonRv(component), matlabRv(component));
end
end

function command = python_command(repo, dataset, jsonFile, useUv)
script = fullfile(repo, 'scripts', 'dipfit_standard_bem_parity.py');
if useUv
    runner = 'uv run python';
else
    runner = 'python';
end
if ispc
    command = sprintf('cd /d "%s" && %s "%s" --dataset "%s" --output "%s"', repo, runner, script, dataset, jsonFile);
else
    command = sprintf('cd "%s" && %s "%s" --dataset "%s" --output "%s"', repo, runner, script, dataset, jsonFile);
end
end

function delete_if_present(jsonFile)
if isfile(jsonFile)
    delete(jsonFile);
end
end
