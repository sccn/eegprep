function result = eegprep_warp_optim_parity(eeglab_root)
% Traditional ft_warp_optim on the synthetic electrodes coregister fits.
% Options stay inside ft_warp_optim: MaxIter 1500, TolFun 1e-4, LargeScale off.
dipfit_standard_besa_add_fieldtrip(char(eeglab_root));
if exist('ft_warp_optim', 'file') ~= 2 || exist('ft_warp_apply', 'file') ~= 2
    error('eegprep_warp_optim_parity:MissingFieldTrip', ...
        'FieldTrip ft_warp_optim must be on the path.');
end
source = [0 1 0; -1 0 -0.2; 1 0 -0.2; 0 0 1; 0 -1 0];
truth = [5 -3 2 rad2deg([0.05 -0.04 0.03]) 2 1.5 1.2];
target = ft_warp_apply(truth, source, 'traditional');
[points, params] = ft_warp_optim(source, target, 'traditional');
result.params = params(:)';
result.points = points;
end
