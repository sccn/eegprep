function [rx, ry, rz] = griddata_octave (x, y, z, xi, yi)

% Meshgrid if x and y are vectors but z is matrix
if (isvector (x) && isvector (y) && all ([numel(y), numel(x)] == size (z)))
    [x, y] = meshgrid (x, y);
end

% Meshgrid xi and yi if one is a row vector and the other is a column
% vector, but not if they are vectors with the same orientation.
if ((isrow (xi) && iscolumn (yi)) || (iscolumn (xi) && isrow (yi)))
    [xi, yi] = meshgrid (xi, yi);
end

x = x(:);
y = y(:);
z = z(:);
zi = NaN (size (xi));

% if (strcmp (method, "v4"))
% Use Biharmonic Spline Interpolation Green's Function method.
% Compatible with Matlab v4 interpolation method, based on
% D. Sandwell 1987 and Deng & Tang 2011.

% The free space Green Function which solves the two dimensional
% Biharmonic PDE
%
% Delta(Delta(G(X))) = delta(X)
%
% for a point source yields
%
% G(X) = |X|^2 * (ln|X|-1) / (8 * pi)
%
% An N-point Biharmonic Interpolation at the point X is given by
%
% z(X) = sum_j_N (alpha_j * G(X-Xj))
%      = sum_j_N (alpha_j * G(rj))
%
% in which the coefficients alpha_j are the unknowns.  rj is the
% Euclidean distance between X and Xj.
% From N datapoints {zi, Xi} an equation system can be formed:
%
% zi(Xi) = sum_j_N (alpha_j * G(Xi-Xj))
%        = sum_j_N (alpha_j * G(rij))
%
% Its inverse yields the unknowns alpha_j.

% Step1: Solve for weight coefficients alpha_j depending on the
% Euclidean distances and the training data set {x,y,z}
r = sqrt ((x - x.').^2 + (y - y.').^2);  % size N^2
D = (r.^2) .* (log (r) - 1);
D(isnan (D)) = 0;  % Fix Green Function for r=0
alpha_j = D \ z;

% Step2 - Use alphas and Green's functions to get interpolated points.
% Use dim3 projection for vectorized calculation to avoid loops.
% Memory usage is proportional to Ni x N.
% FIXME: if this approach is too memory intensive, revert portion to loop
x = permute (x, [3, 2, 1]);
y = permute (y, [3, 2, 1]);
alpha_j = permute (alpha_j, [3, 2, 1]);
r_i = sqrt ((xi - x).^2 + (yi - y).^2);  % size Ni x N
Di = (r_i.^2) .* (log (r_i) - 1);
Di(isnan (Di)) = 0;  % Fix Green's Function for r==0
zi = sum (Di .* alpha_j, 3);


if (nargout > 1)
    rx = xi;
    ry = yi;
    rz = zi;
else
    rx = zi;
end
