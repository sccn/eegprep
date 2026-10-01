function variables = eegprep_test_run_script(script_name, variable_names)
% Execute a real reference script and expose requested caller-workspace values.
eval(script_name);
variables = struct();
for index = 1:numel(variable_names)
    name = variable_names{index};
    if exist(name, 'var')
        variables.(name) = eval(name);
    end
end
end
