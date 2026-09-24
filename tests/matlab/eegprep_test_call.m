function eegprep_test_call(input_file, output_file, function_name, output_count)
% Transport test calls without EEGLAB/EEGPrep production dataset serializers.
loaded = load(input_file, 'arguments');
outputs = cell(1, output_count);
if output_count == 0
    feval(function_name, loaded.arguments{:});
else
    [outputs{:}] = feval(function_name, loaded.arguments{:});
end
save(output_file, 'outputs', '-v7');
end
