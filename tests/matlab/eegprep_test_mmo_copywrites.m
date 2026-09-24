function messages = eegprep_test_mmo_copywrites(values)
% The Python caller first runs the checkmmo setup that creates testfile2.fdt.
floatwrite(values, 'testfile.fdt');
test = mmo('testfile.fdt', [1 10], true, false, true);
testcheck = mmo('testfile2.fdt', [1 10], true, false, true);
a.test3 = test;
messages{1} = evalc('a.test3(5) = 5;');
clear test;
messages{2} = evalc('a.test3(6) = 5;');
clear test;
test2 = a.test3;
messages{3} = evalc('a.test3(7) = 5;');
clear test a test2;
test = mmo('testfile.fdt', [1 10], true, false, true);
messages{4} = evalc('a = checkmmo_sub1(test);');
messages{5} = evalc('a = checkmmo_sub2(test);');
messages{6} = evalc('test(2) = 3.2;');
messages{7} = evalc('a = checkmmo_sub5;');
clear test testcheck test2 a;
end
