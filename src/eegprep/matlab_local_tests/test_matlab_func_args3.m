function out1 = test_matlab_func_args3(in1)
    fprintf(2, 'in1 is a %s\n', class(in1));
    out1 = in1(1).value;