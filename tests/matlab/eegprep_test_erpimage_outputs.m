function result = eegprep_test_erpimage_outputs(action, varargin)
% Keep graphics native; expose only outputs/properties asserted by the source.
persistent first_axis
switch action
    case 'call'
        outputs = cell(1, 15);
        first_compared_axis = varargin{1};
        [outputs{:}] = erpimage(varargin{2:end});
        handles = outputs{5};
        if iscell(handles)
            first_axis = handles{1};
        else
            first_axis = handles(1);
        end
        % Keep a modern cell tail. The old numeric handle vector is a different
        % class, and flattening it would hide that.
        outputs{5} = handles(first_compared_axis:end);
        result = outputs;
    case 'axis_type'
        % Deliberately retain () rather than correcting the source to {}.
        result = get(first_axis, 'Type');
end
end
