function state = eegprep_test_tutorial_redraw(state)
% MATLAB's caller-workspace boundary; the tutorial steps remain in Python.
names = fieldnames(state);
for index = 1:numel(names)
    assignin('base', names{index}, state.(names{index}));
end
evalin('base', 'eeglab redraw');
names = {'ALLEEG', 'EEG', 'CURRENTSET', 'ALLCOM', 'STUDY', 'CURRENTSTUDY'};
state = struct;
for index = 1:numel(names)
    state.(names{index}) = evalin('base', names{index});
end
end
