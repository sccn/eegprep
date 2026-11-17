function eegprep_reference_plugins(eeglab_root)
% Put optional plugin functions on the path. eeglab('nogui') adds each plugin
% folder but does not run eegplugin_* hooks, which is where JSONio and Biosig
% add their nested folders.
jsonio = fullfile(eeglab_root, 'plugins', 'EEG-BIDS', 'JSONio');
if exist(fullfile(jsonio, 'jsonread.m'), 'file')
    addpath(jsonio);
end
biosig = dir(fullfile(eeglab_root, 'plugins', 'Biosig*', 'biosig', 't200_FileAccess', 'sopen.m'));
for index = 1:numel(biosig)
    addpath(biosig(index).folder);
end
% eeglab('nogui') does not run eegplugin_dipfit, so standard_1005.elc stays
% off the path and bids_chan_lookup drops the imported dataset.
if ~exist('standard_mri.mat', 'file')
    dipfitp = fileparts(which('dipfitdefs.m'));
    if ~isempty(dipfitp)
        addpath(fullfile(dipfitp, 'standard_BEM'));
        addpath(fullfile(dipfitp, 'standard_BEM', 'elec'));
        addpath(fullfile(dipfitp, 'standard_BEM', 'skin'));
        addpath(fullfile(dipfitp, 'standard_BESA'));
    end
end
end
