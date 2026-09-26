function result = eegprep_test_erpimage_outputs(action, varargin)
% Keep graphics native; expose only outputs/properties asserted by the source.
persistent first_axis
switch action
    case 'call'
        outputs = cell(1, 15);
        [outputs{:}] = erpimage(varargin{:});
        first_axis = outputs{5}(1);
        % The source never compares the first handle numerically. Preserve
        % the remaining handles' cell/numeric class, including modern cells.
        outputs{5} = outputs{5}(2:end);
        result = outputs;
    case 'axis_type'
        % Deliberately retain () rather than correcting the source to {}.
        result = get(first_axis, 'Type');
end
end
