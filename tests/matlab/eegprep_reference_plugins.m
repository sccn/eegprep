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
end
