function eegprep_test_call(input_file, output_file, function_name, output_count)
% Transport test calls without EEGLAB/EEGPrep production dataset serializers.
loaded = load(input_file, 'arguments');
outputs = cell(1, output_count);
if output_count == 0
    feval(function_name, loaded.arguments{:});
else
    [outputs{:}] = feval(function_name, loaded.arguments{:});
end
outputs = materialize_empty_fields(outputs);
save(output_file, 'outputs', '-v7');
end

function value = materialize_empty_fields(value)
% Unassigned cells/struct fields use zero-length MAT records, which SciPy
% reads as 1x0. Explicitly store their actual 0x0 double value, retaining
% distinct 1x0, 0xN, typed-empty and nonempty values unchanged.
if iscell(value)
    for index = 1:numel(value)
        value{index} = materialize_empty_fields(value{index});
    end
elseif isstruct(value)
    fields = fieldnames(value);
    for index = 1:numel(value)
        for field = 1:numel(fields)
            value(index).(fields{field}) = materialize_empty_fields(value(index).(fields{field}));
        end
    end
elseif isa(value, 'double') && isreal(value) && ~issparse(value) && isequal(size(value), [0 0])
    value = [];
end
end
