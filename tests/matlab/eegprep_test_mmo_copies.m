function counts = eegprep_test_mmo_copies(values)
% Source-shaped checkmmo workspace; exporting objects would change alias counts.
floatwrite(values(:), 'testfile.fdt');
floatwrite(values(:), 'testfile2.fdt');
test = mmo('testfile.fdt', [1 10], true, false, true);
testcheck = mmo('testfile2.fdt', [1 10]);
counts(1) = checkcopies(test);
test2 = test;
counts(2) = checkcopies(test);
clear test2;
test3 = test;
counts(3) = checkcopies(test);
clear test3;
a.mmo = test;
counts(4) = checkcopies(test);
clear a;
test2 = test;
a.mmo2 = {test test2};
counts(5) = checkcopies(test);
clear a test2;
counts(6) = checkcopies(test);
clear test2 a;
counts(7) = checkmmo_sub1(test);
le = checkmmo_sub2(test);
counts(8) = le;
checkmmo_sub3;
counts(9) = le;
checkmmo_sub4;
counts(10) = le;
test2 = test;
checkmmo_sub3;
counts(11) = le;
checkmmo_sub4;
counts(12) = le;
clear test test2 testcheck;
end
