function result = eegprep_test_getallmenus_fixture()
% Build the getallmenus fixture on gcf. HG2 handles are objects; a double
% copied through a MAT file is not a Parent/uimenu handle.
window = gcf;
uicontrol('Parent', window, ...
    'Units', 'points', ...
    'BackgroundColor', [1 0 0], ...
    'ListboxTop', 0, ...
    'HorizontalAlignment', 'left', ...
    'Position', [100 200 300 400], ...
    'Style', 'frame', ...
    'Tag', 'Frame1');
set(window, 'MenuBar', 'none');
for label = 'abcdefg'
    menu = uimenu(window, 'Label', label);
    if label == 'a'
        children = {'aa', 'ab', 'ac', 'ad'};
    elseif label == 'd'
        children = {'da'};
    else
        children = {};
    end
    for index = 1:numel(children)
        uimenu(menu, 'Label', children{index});
    end
end
result = struct('tree', getallmenus(window), 'count', numel(findobj(0)));
close(window);
end
