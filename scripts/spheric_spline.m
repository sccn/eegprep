

function [xbad, ybad, zbad, allres] = spheric_spline( xelec, yelec, zelec, xbad, ybad, zbad, values, params)

newchans = length(xbad);
numpoints = size(values,2);
params = double(params);

%SPHERERES = 20;
%[x,y,z] = sphere(SPHERERES);
%x(1:(length(x)-1)/2,:) = []; xbad = [ x(:)'];
%y(1:(length(x)-1)/2,:) = []; ybad = [ y(:)'];
%z(1:(length(x)-1)/2,:) = []; zbad = [ z(:)'];

Gelec = computeg(xelec,yelec,zelec,xelec,yelec,zelec,params);
Gsph  = computeg(xbad,ybad,zbad,xelec,yelec,zelec,params);

meanvalues = mean(values);
values = values - repmat(meanvalues, [size(values,1) 1]); % make mean zero

values = [values;zeros(1,numpoints)];
lambda = params(1);
C = pinv([Gelec+eye(size(Gelec))*lambda;ones(1,length(Gelec))]) * values;
%C = pinv([Gelec;ones(1,length(Gelec))]) * values;

clear values;
allres = zeros(newchans, numpoints);

for j = 1:size(Gsph,1)
    allres(j,:) = sum(C .* repmat(Gsph(j,:)', [1 size(C,2)]));
end
allres = allres + repmat(meanvalues, [size(allres,1) 1]);


