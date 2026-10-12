function eegprep_test_gui_call(function_name, output_count, varargin)
% Preserve native output count without serializing unasserted graphics handles.
if output_count == 0
    feval(function_name, varargin{:});
else
    outputs = cell(1, output_count);
    [outputs{:}] = feval(function_name, varargin{:});
end
end
