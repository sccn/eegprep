% -----------------
% spherical splines
% -----------------
% function [x, y, z, Res] = spheric_spline_old( xelec, yelec, zelec, values)
% 
% SPHERERES = 20;
% [x,y,z] = sphere(SPHERERES);
% x(1:(length(x)-1)/2,:) = []; x = [ x(:)' ];
% y(1:(length(y)-1)/2,:) = []; y = [ y(:)' ];
% z(1:(length(z)-1)/2,:) = []; z = [ z(:)' ];
% 
% Gelec = computeg(xelec,yelec,zelec,xelec,yelec,zelec);
% Gsph  = computeg(x,y,z,xelec,yelec,zelec);
% 
% % equations are 
% % Gelec*C + C0  = Potential (C unknown)
% % Sum(c_i) = 0
% % so 
% %             [c_1]
% %      *      [c_2]
% %             [c_ ]
% %    xelec    [c_n]
% % [x x x x x]         [potential_1]
% % [x x x x x]         [potential_ ]
% % [x x x x x]       = [potential_ ]
% % [x x x x x]         [potential_4]
% % [1 1 1 1 1]         [0]
% 
% % compute solution for parameters C
% % ---------------------------------
% meanvalues = mean(values); 
% values = values - meanvalues; % make mean zero
% C = pinv([Gelec;ones(1,length(Gelec))]) * [values(:);0];
% 
% % apply results
% % -------------
% Res = zeros(1,size(Gsph,1));
% for j = 1:size(Gsph,1)
%     Res(j) = sum(C .* Gsph(j,:)');
% end
% Res = Res + meanvalues;
% Res = reshape(Res, length(x(:)),1);

function [xbad, ybad, zbad, allres] = spheric_spline( xelec, yelec, zelec, xbad, ybad, zbad, values, params)

newchans = length(xbad);
numpoints = size(values,2);

%SPHERERES = 20;
%[x,y,z] = sphere(SPHERERES);
%x(1:(length(x)-1)/2,:) = []; xbad = [ x(:)'];
%y(1:(length(x)-1)/2,:) = []; ybad = [ y(:)'];
%z(1:(length(x)-1)/2,:) = []; zbad = [ z(:)'];

Gelec = computeg(xelec,yelec,zelec,xelec,yelec,zelec,params);
Gsph  = computeg(xbad,ybad,zbad,xelec,yelec,zelec,params);

% compute solution for parameters C
% ---------------------------------
meanvalues = mean(values); 
values = values - repmat(meanvalues, [size(values,1) 1]); % make mean zero

values = [values;zeros(1,numpoints)];
lambda = params(1);
C = pinv([Gelec+eye(size(Gelec))*lambda;ones(1,length(Gelec))]) * values;
%C = pinv([Gelec;ones(1,length(Gelec))]) * values;

clear values;
allres = zeros(newchans, numpoints);

% apply results
% -------------
for j = 1:size(Gsph,1)
    allres(j,:) = sum(C .* repmat(Gsph(j,:)', [1 size(C,2)]));        
end
allres = allres + repmat(meanvalues, [size(allres,1) 1]);
