function handle = eegprep_test_gui_handle(function_name, varargin)
% Transport a native graphics handle without serializing the graphics object.
handle = double(feval(function_name, varargin{:}));
end
