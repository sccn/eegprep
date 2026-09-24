function result = eegprep_test_mmo_assignment(values, subscripts, assigned, transposed)
% Keep the source test's mmo alive until its explicit data extraction.
if transposed
    physical = permute(values, [2:ndims(values) 1]);
else
    physical = values;
end
floatwrite(physical, 'testfile.fdt');
test = mmo('testfile.fdt', size(values), true, transposed, true);
% Subscripts are literal expressions supplied by the Python test cases.
eval(['test(' subscripts ') = assigned;']);
result = test(:,:,:);
end
