function messages = eegprep_test_mmo_copywrites(values)
% checkmmo2 expects the second file left by checkmmo. Supply only that fixture,
% not the unrelated twelve-step copy-count workflow.
floatwrite(values, 'testfile.fdt');
floatwrite(values, 'testfile2.fdt');
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
