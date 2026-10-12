function [file, directory, index] = uigetfile(varargin)
% Narrow filesystem-input fixture; never select or fabricate a data file.
key = 'eegprep_test_limo_file_choice';
choice = getappdata(groot, key);
assert(~choice.consumed && isequal(varargin, {'*mat', choice.expected}), ...
    'eegprep_test_limo_add_plots:UnexpectedChoice', ...
    'Only one optional extra-file prompt may be cancelled.');
choice.consumed = true;
setappdata(groot, key, choice);
file = 0;
directory = 0;
index = 0;
end
