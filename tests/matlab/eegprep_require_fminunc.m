function eegprep_require_fminunc(context)
%EEGPREP_REQUIRE_FMINUNC Require Optimization Toolbox (fminunc) via FieldTrip's check.
% Matches FieldTrip's ft_warp_optim / ft_inverse_dipolefit check so MATLAB does
% not silently fall back to fminsearch during parity runs.
if exist('ft_hastoolbox', 'file') == 2
    has_optim = logical(ft_hastoolbox('optim'));
else
    has_optim = exist('fminunc', 'file') == 2 ...
        && exist('optimset', 'file') == 2 ...
        && license('checkout', 'optimization_toolbox') == 1;
end
if ~has_optim
    if nargin < 1 || isempty(context)
        context = 'this parity check';
    end
    error('eegprep:missingOptimizationToolbox', ...
        'Optimization Toolbox (fminunc) is required for %s.', char(context));
end
end
