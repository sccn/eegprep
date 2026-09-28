function eegprep_test_call(input_file, output_file, function_name, output_count, inline_limit_bytes)
% Transport test calls without EEGLAB/EEGPrep production dataset serializers.
if nargin < 5
    inline_limit_bytes = 2^30;
end
loaded = load(input_file, 'arguments');
arguments = restore_tables(loaded.arguments);
outputs = cell(1, output_count);
if output_count == 0
    feval(function_name, arguments{:});
else
    [outputs{:}] = feval(function_name, arguments{:});
end
outputs = materialize_empty_fields(outputs);
info = whos('outputs');
if info.bytes > inline_limit_bytes
    % MAT v7 limits each variable, including the aggregate output cell, to 2GB.
    % Keep the outer structure and spill large arrays without changing values.
    outputs = spill_arrays(outputs, fileparts(output_file), min(2^20, inline_limit_bytes));
    info = whos('outputs');
    if info.bytes > inline_limit_bytes
        % Many individually small arrays can also exceed the aggregate limit.
        outputs = spill_arrays(outputs, fileparts(output_file), 1);
        info = whos('outputs');
    end
    assert(info.bytes < 2^31, 'eegprep_test_call:OversizedMetadata', ...
        'Output metadata still exceeds the MAT v7 variable limit after array sharding.');
end
save(output_file, 'outputs', '-v7');
end

function value = spill_arrays(value, directory, leaf_limit_bytes)
if iscell(value)
    for index = 1:numel(value)
        value{index} = spill_arrays(value{index}, directory, leaf_limit_bytes);
    end
elseif isstruct(value)
    if isscalar(value) && isequal(fieldnames(value), {'eegprep_test_array_mat_v1'; 'shape'})
        return;
    end
    fields = fieldnames(value);
    for index = 1:numel(value)
        for field = 1:numel(fields)
            value(index).(fields{field}) = spill_arrays(value(index).(fields{field}), directory, leaf_limit_bytes);
        end
    end
elseif (isnumeric(value) || islogical(value)) && ~issparse(value) && ~isempty(value)
    info = whos('value');
    if info.bytes > leaf_limit_bytes
        shape = size(value);
        elements_per_part = max(1, floor(2^28 / (info.bytes / numel(value))));
        paths = cell(1, ceil(numel(value) / elements_per_part));
        for index = 1:numel(paths)
            first = (index - 1) * elements_per_part + 1;
            part = value(first:min(first + elements_per_part - 1, numel(value)));
            filename = [tempname(directory) '.mat'];
            save(filename, 'part', '-v7');
            [~, name, extension] = fileparts(filename);
            paths{index} = [name extension];
        end
        value = struct('eegprep_test_array_mat_v1', {paths}, 'shape', shape);
    end
end
end

function value = materialize_empty_fields(value)
% Unassigned cells/struct fields use zero-length MAT records, which SciPy
% reads as 1x0. Explicitly store their actual 0x0 double value, retaining
% distinct 1x0, 0xN, typed-empty and nonempty values unchanged.
if istable(value)
    % SciPy cannot decode MATLAB's opaque table storage. Carry the exact native
    % MAT bytes instead, including variable classes and all table properties.
    filename = [tempname '.mat'];
    cleanup = onCleanup(@() delete(filename));
    save(filename, 'value', '-v7');
    file = fopen(filename, 'rb');
    bytes = fread(file, Inf, '*uint8');
    fclose(file);
    value = struct('eegprep_test_table_mat_v1', bytes);
elseif iscell(value)
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

function value = restore_tables(value)
if isstruct(value) && isscalar(value) && ...
        isequal(fieldnames(value), {'eegprep_test_table_mat_v1'})
    filename = [tempname '.mat'];
    cleanup = onCleanup(@() delete(filename));
    file = fopen(filename, 'wb');
    fwrite(file, value.eegprep_test_table_mat_v1, 'uint8');
    fclose(file);
    loaded = load(filename, 'value');
    value = loaded.value;
    assert(istable(value), 'eegprep_test_call:InvalidTableEnvelope', ...
        'The table envelope must contain a native MATLAB table.');
elseif iscell(value)
    for index = 1:numel(value)
        value{index} = restore_tables(value{index});
    end
elseif isstruct(value)
    fields = fieldnames(value);
    for index = 1:numel(value)
        for field = 1:numel(fields)
            value(index).(fields{field}) = restore_tables(value(index).(fields{field}));
        end
    end
end
end
