function [rnew, com] = eegprep_test_pop_comments_concat()
% Same inputs as pass_newcomments_concat.m, built in MATLAB so the cell rows
% are not reshaped by the MAT transport.
comments = {'Line nr. 1', 'This is line nr.2', ' ', ...
    'And this is the final line                           '};
newcomments = {'New line 1', 'And this one is also new                             '};
[rnew, com] = pop_comments(comments, '', newcomments, 1);
end
