function eegprep_batch_engine_startup(eeglab_root)
% Rebuild the path after a wedged matlab-batch process is replaced.
addpath(eeglab_root);
addpath(fullfile(eeglab_root, 'functions'));
set(0, 'DefaultFigureVisible', 'off');
here = pwd;
eeglab('nogui');
cd(here);
eegprep_reference_plugins(eeglab_root);
addpath(fileparts(mfilename('fullpath')));
end
