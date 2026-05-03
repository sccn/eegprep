function export_eeglab_menu_inventory(output_file, eeglab_root)
%EXPORT_EEGLAB_MENU_INVENTORY Export EEGLAB menus as JSON for UI parity checks.
%
% Usage:
%   export_eeglab_menu_inventory('.visual-parity/eeglab_menus.json', 'src/eegprep/eeglab')
%
% If arguments are omitted, EEGPREP_VISUAL_OUTPUT and EEGPREP_EEGLAB_ROOT are
% read from the environment.

if nargin < 1 || isempty(output_file)
    output_file = getenv('EEGPREP_VISUAL_OUTPUT');
end
if nargin < 2 || isempty(eeglab_root)
    eeglab_root = getenv('EEGPREP_EEGLAB_ROOT');
end
if isempty(output_file)
    error('output_file or EEGPREP_VISUAL_OUTPUT is required');
end
if isempty(eeglab_root)
    eeglab_root = fullfile(pwd, 'src', 'eegprep', 'eeglab');
end

addpath(eeglab_root);
eeglab;
drawnow;

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

menus = struct('label', {}, 'enabled', {}, 'separator', {}, 'tag', {}, 'children', {});
for idx = 1:numel(menu_handles)
    handle = menu_handles(idx);
    menus(idx).label = get(handle, 'Label');
    menus(idx).enabled = get(handle, 'Enable');
    menus(idx).separator = get(handle, 'Separator');
    menus(idx).tag = get(handle, 'Tag');
    menus(idx).children = collect_menu_children(handle);
end
end
