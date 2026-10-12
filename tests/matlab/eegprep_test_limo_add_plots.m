function eegprep_test_limo_add_plots(varargin)
% Plot every supplied file, then explicitly decline one optional extra file.
key = 'eegprep_test_limo_file_choice';
assert(~isappdata(groot, key), 'eegprep_test_limo_add_plots:NestedFixture', ...
    'The file-choice fixture must not be nested.');
assert(~isempty(varargin) && iscell(varargin{1}) && ~isempty(varargin{1}), ...
    'eegprep_test_limo_add_plots:MissingFiles', 'Supply the original file list.');
expected = sprintf('Select %g Central tendency file', numel(varargin{1}) + 1);
previous_path = path;
cleanup = onCleanup(@() restore_choice(previous_path, key));
setappdata(groot, key, struct('expected', expected, 'consumed', false));
addpath(fullfile(fileparts(mfilename('fullpath')), 'limo_file_choice'), '-begin');
limo_add_plots(varargin{:});
choice = getappdata(groot, key);
assert(choice.consumed, 'eegprep_test_limo_add_plots:UnusedChoice', ...
    'The native plot did not reach its optional extra-file prompt.');
end

function restore_choice(previous_path, key)
path(previous_path);
rmappdata(groot, key);
end
