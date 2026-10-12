function eegprep_test_prop_parent(action, varargin)
% Keep the modern pop_prop parent as a native graphics object, not double(h).
persistent parent
switch action
    case 'create'
        parent = figure;
    case 'call'
        pop_prop(varargin{1}, varargin{2}, varargin{3}, parent, varargin{4});
    case 'close'
        close(parent);
        parent = [];
end
end
