function values = eegprep_test_mmo_returned(input)
% Preserve the source's helper-returned workspace and its internal assertion.
test = checkmmo_sub7(input);
values = test(:,:,:);
test = checkmmo_sub8(input);
end
