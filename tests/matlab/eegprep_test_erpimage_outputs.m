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
        % near() compares the tail as a numeric row. Modern axes tails are a
        % cell of handles or placeholders; doubles survive the MAT transport.
        outputs{5} = numeric_handle_row(handles, first_compared_axis);
        result = outputs;
    case 'axis_type'
        % Deliberately retain () rather than correcting the source to {}.
        result = get(first_axis, 'Type');
end
end

function row = numeric_handle_row(handles, first)
if iscell(handles)
    tail = handles(first:end);
    row = nan(size(tail));
    for index = 1:numel(tail)
        row(index) = handle_to_double(tail{index});
    end
    return
end
row = handles(first:end);
if ~isnumeric(row)
    numeric = nan(size(row));
    for index = 1:numel(row)
        numeric(index) = handle_to_double(row(index));
    end
    row = numeric;
end
end

function value = handle_to_double(value)
if isnumeric(value) && isscalar(value)
    return
end
try
    converted = double(value);
catch
    converted = NaN;
end
if ~(isnumeric(converted) && isscalar(converted))
    converted = NaN;
end
value = converted;
end
