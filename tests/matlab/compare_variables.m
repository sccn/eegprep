function res = compare_variables(a, b, tol)

if nargin < 3
    tol = 0.005;
end
res = [];
if iscell(a)
    if ~iscell(b)
        disp('Error - data type')
        return
    end

    if length(a) ~= length(b)
        disp('Error - length')
        return
    end
    for iCell = 1:length(a)
        compare_variables(a{iCell}, b{iCell}, tol)
    end
else
    % assumes array
    if ~isnumeric(a) || ~isnumeric(b)
        disp('Warning - cannot compare (not implemented')
        return
    end

    if ndims(a) == ndims(b) && all(size(a) == size(b))
        disp('Pass - size')
    else
        disp('Error - size')
    end

    aa = a(~isnan(a(:)));
    bb = b(~isnan(b(:)));
    if ~isequal(a, aa) || ~isequal(b, bb)
        if numel(aa) ~= numel(bb)
            disp('Error - not the same number or nans')
        end
    end

    res.meandiff = mean(abs((aa(:)-bb(:))));
    res.stddiff  = std(abs((aa(:)-bb(:))));
    res.maxdiff  = max(abs((aa(:)-bb(:))));
    res.meanpercentdiff = mean(abs((aa(:)-bb(:)))/max(abs(aa(:))));
    res.stdpercentdiff  = std(abs((aa(:)-bb(:)))/max(abs(aa(:))));
    res.maxpercentdiff = max(abs((aa(:)-bb(:)))/max(abs(aa(:))));
    fprintf('Mean difference: %1.8f (+- %1.8f)\n', res.meandiff, res.stddiff)
    fprintf('Max difference: %1.8f\n', res.maxdiff)
    if mean(abs((aa(:)-bb(:)))/max(abs(aa(:)))) < tol
        disp('Pass - value')
    else
        disp('Error - value')
    end

end
