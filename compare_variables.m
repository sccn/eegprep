function compare_variable(a, b, tol)

if nargin < 3
    tol = 0.005;
end

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

    fprintf('Mean difference: %1.8f (+- %1.8f)\n', mean(abs((aa(:)-bb(:)))/max(abs(aa(:)))), std(abs((aa(:)-bb(:)))/max(abs(aa(:)))))
    fprintf('Max difference: %1.8f\n', max(abs((aa(:)-bb(:)))/max(abs(aa(:)))))
    if mean(abs((aa(:)-bb(:)))/max(abs(aa(:)))) < tol
        disp('Pass - value')
    else
        disp('Error - value')
    end

end
