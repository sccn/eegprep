function dipfit_standard_besa_parity(eeglab_root)
% Compare EEGPrep standardBESA fits with EEGLAB DIPFIT and FieldTrip.
%
% From the EEGPrep repository root, with MATLAB able to see this EEGLAB checkout:
%
%   matlab -batch "addpath('tools'); dipfit_standard_besa_parity('/path/to/eeglab')"
%
% The script saves a reference MAT file and runs tools/dipfit_standard_besa_parity.py,
% which prints the 3-D distance of components 1-3 and the leadfield max relative error.
if nargin < 1 || isempty(eeglab_root)
    error('Pass the EEGLAB checkout, for example dipfit_standard_besa_parity(''/path/to/eeglab'').');
end
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(eeglab_root);
addpath(fullfile(repo, 'tests', 'matlab'));
eeglab('nogui');
dipfit_standard_besa_add_fieldtrip(eeglab_root);
if isempty(which('ft_compute_leadfield'))
    error('FieldTrip was not found under %s/plugins.', eeglab_root);
end
dataset = fullfile(repo, 'sample_data', 'eeglab_data_epochs_ica.set');
fprintf('Fitting components 1-3 with pop_multifit (standardBESA, threshold 100).\n');
result = dipfit_standard_besa_reference(dataset, eeglab_root);
mat_path = fullfile(tempdir, 'dipfit_standard_besa_reference.mat');
save(mat_path, '-struct', 'result');
fprintf('Wrote %s\n', mat_path);
python = python_executable(repo);
command = sprintf('"%s" "%s" "%s"', python, fullfile(repo, 'tools', 'dipfit_standard_besa_parity.py'), mat_path);
status = system(command);
if status ~= 0
    fprintf('Python comparison did not run. Repeat it with:\n  %s\n', command);
    error('dipfit_standard_besa_parity.py exited with status %d.', status);
end
end

function python = python_executable(repo)
candidates = {
    fullfile(repo, '.venv', 'bin', 'python')
    fullfile(repo, '.venv', 'Scripts', 'python.exe')
    };
for index = 1:numel(candidates)
    if exist(candidates{index}, 'file')
        python = candidates{index};
        return
    end
end
python = 'python3';
end
