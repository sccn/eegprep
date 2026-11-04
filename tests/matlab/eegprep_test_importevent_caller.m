function events = eegprep_test_importevent_caller(myEventValues, varargin)
% Bind the original test's named caller variable without replacing its import.
events = importevent('myEventValues', varargin{:});
end
