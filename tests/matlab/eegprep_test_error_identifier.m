function [raised, identifier] = eegprep_test_error_identifier(function_name, varargin)
% Preserve the native identifier for a Python-owned verifyError assertion.
raised = false;
identifier = '';
try
    feval(function_name, varargin{:});
catch exception
    raised = true;
    identifier = exception.identifier;
end
end
