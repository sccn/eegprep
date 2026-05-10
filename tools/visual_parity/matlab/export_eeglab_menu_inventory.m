function export_eeglab_menu_inventory(output_file, eeglab_root, state)
%EXPORT_EEGLAB_MENU_INVENTORY Export EEGLAB menus as JSON for UI parity checks.
%
% Usage:
%   export_eeglab_menu_inventory('.visual-parity/eeglab_menus.json', 'src/eegprep/eeglab', 'startup')
%
% If arguments are omitted, EEGPREP_VISUAL_OUTPUT and EEGPREP_EEGLAB_ROOT are
% read from the environment.

if nargin < 1 || isempty(output_file)
    output_file = getenv('EEGPREP_VISUAL_OUTPUT');
end
if nargin < 2 || isempty(eeglab_root)
    eeglab_root = getenv('EEGPREP_EEGLAB_ROOT');
end
if nargin < 3 || isempty(state)
    state = getenv('EEGPREP_MENU_STATE');
end
if isempty(output_file)
    error('output_file or EEGPREP_VISUAL_OUTPUT is required');
end
if isempty(eeglab_root)
    eeglab_root = fullfile(pwd, 'src', 'eegprep', 'eeglab');
end
if isempty(state)
    state = 'startup';
end

addpath(eeglab_root);
eeglab;
drawnow;

fig = findobj('tag', 'EEGLAB');
if isempty(fig)
    error('Could not find EEGLAB main window');
end
add_viewprops_menu_if_present(eeglab_root, fig(1));
make_demo_state(state);

fig = findobj('tag', 'EEGLAB');
if isempty(fig)
    error('Could not find EEGLAB main window');
end

menus = collect_menu_children(fig(1));
payload = struct('menus', {menus});

fid = fopen(output_file, 'w');
if fid < 0
    error('Could not open output file: %s', output_file);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(payload));
end

function add_viewprops_menu_if_present(eeglab_root, fig)
if ~isempty(findobj(fig, 'Label', 'View extended channel properties'))
    return;
end
viewprops_root = fullfile(eeglab_root, 'plugins', 'ICLabel', 'viewprops');
if ~exist(fullfile(viewprops_root, 'eegplugin_viewprops.m'), 'file')
    return;
end
addpath(viewprops_root);
try_strings = struct('no_check', '');
catch_strings = struct('add_to_hist', '');
eegplugin_viewprops(fig, try_strings, catch_strings);
drawnow;
end

function make_demo_state(state)
state = lower(char(state));
if strcmp(state, 'startup')
    return;
end

global EEG ALLEEG CURRENTSET;
ALLEEG = [];
EEG = eeg_emptyset;
CURRENTSET = 0;

if strcmp(state, 'continuous')
    EEG = demo_eeg('continuous', 'menu continuous');
    ALLEEG = EEG;
    CURRENTSET = 1;
elseif strcmp(state, 'epoched')
    EEG = demo_eeg('epoched', 'menu epoched');
    ALLEEG = EEG;
    CURRENTSET = 1;
elseif strcmp(state, 'multiple')
    EEG = demo_eeg('continuous', 'menu one');
    ALLEEG = EEG;
    ALLEEG(2) = demo_eeg('continuous', 'menu two');
    CURRENTSET = [1 2];
    EEG = ALLEEG(CURRENTSET);
else
    error('Unknown menu inventory state: %s', state);
end

eeglab redraw;
drawnow;
pause(0.5);
end

function EEG = demo_eeg(kind, setname)
EEG = eeg_emptyset;
EEG.setname = setname;
EEG.filename = [setname '.set'];
EEG.filepath = tempdir;
EEG.nbchan = 4;
EEG.srate = 250;
if strcmp(kind, 'epoched')
    EEG.pnts = 250;
    EEG.trials = 2;
    EEG.xmin = -0.2;
    EEG.xmax = EEG.xmin + (EEG.pnts - 1) / EEG.srate;
    EEG.data = zeros(4, EEG.pnts, EEG.trials);
else
    EEG.pnts = 1000;
    EEG.trials = 1;
    EEG.xmin = 0;
    EEG.xmax = (EEG.pnts - 1) / EEG.srate;
    EEG.data = zeros(4, EEG.pnts);
end
EEG.chanlocs = struct( ...
    'labels', {'Fp1', 'Fp2', 'Cz', 'Oz'}, ...
    'ref', {'common', 'common', 'common', 'common'}, ...
    'theta', {-18, 18, 0, 180}, ...
    'radius', {0.42, 0.42, 0, 0.42}, ...
    'X', {-0.25, 0.25, 0, 0}, ...
    'Y', {0.75, 0.75, 0, -0.8}, ...
    'Z', {0.55, 0.55, 1, 0.55}, ...
    'type', {'EEG', 'EEG', 'EEG', 'EEG'});
EEG.chaninfo = struct();
EEG.event = struct( ...
    'type', {'stim', 'resp'}, ...
    'latency', {100, 350}, ...
    'duration', {0, 0});
EEG.urevent = [];
EEG.epoch = [];
EEG.history = '';
EEG.icaweights = eye(4);
EEG.icasphere = eye(4);
EEG.icawinv = eye(4);
EEG.icachansind = 1:4;
if strcmp(kind, 'epoched')
    EEG.icaact = zeros(4, EEG.pnts, EEG.trials);
else
    EEG.icaact = zeros(4, EEG.pnts);
end
end

function menus = collect_menu_children(parent)
children = allchild(parent);
menu_handles = [];
for idx = 1:numel(children)
    if strcmp(get(children(idx), 'Type'), 'uimenu')
        menu_handles(end + 1) = children(idx); %#ok<AGROW>
    end
end

positions = zeros(1, numel(menu_handles));
for idx = 1:numel(menu_handles)
    positions(idx) = get(menu_handles(idx), 'Position');
end
[~, order] = sort(positions);
menu_handles = menu_handles(order);

menus = struct('label', {}, 'enabled', {}, 'separator', {}, 'checked', {}, 'tag', {}, 'children', {});
for idx = 1:numel(menu_handles)
    handle = menu_handles(idx);
    menus(idx).label = get(handle, 'Label');
    menus(idx).enabled = get(handle, 'Enable');
    menus(idx).separator = get(handle, 'Separator');
    menus(idx).checked = get(handle, 'Checked');
    menus(idx).tag = get(handle, 'Tag');
    menus(idx).children = collect_menu_children(handle);
end
end
