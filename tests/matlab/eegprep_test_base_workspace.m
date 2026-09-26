function eegprep_test_base_workspace(action)
% Preserve base/global state around Python-owned menu workflows, without
% executing any EEGLAB operations or assertions.
persistent base_names base_values base_globals global_names global_values;
if strcmp(action, 'snapshot')
    base_info = evalin('base', 'whos');
    base_names = {base_info.name};
    base_globals = [base_info.global];
    base_values = cell(size(base_names));
    for index = 1:numel(base_names)
        base_values{index} = evalin('base', base_names{index});
    end
    global_names = who('global');
    global_values = cell(size(global_names));
    for index = 1:numel(global_names)
        global_values{index} = read_global(global_names{index});
    end
else
    current_globals = who('global');
    for index = 1:numel(current_globals)
        eval(['clear global ' current_globals{index}]);
    end
    evalin('base', 'clear');
    for index = 1:numel(global_names)
        write_global(global_names{index}, global_values{index});
    end
    for index = 1:numel(base_names)
        if base_globals(index)
            evalin('base', ['global ' base_names{index}]);
        end
        assignin('base', base_names{index}, base_values{index});
    end
    clear base_names base_values base_globals global_names global_values;
end
end

function value = read_global(name)
eval(['global ' name]);
value = eval(name);
end

function write_global(name, value)
eval(['global ' name]);
eval([name ' = value;']);
end
