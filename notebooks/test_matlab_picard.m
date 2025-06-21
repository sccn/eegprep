rng(0); % for reproducibility

% load the same data
d = load('data.mat','X','A');
X = d.X;
A = d.A;
% run MATLAB Picard
[Y_mat, W_mat] = picard_standard3(X, 10, 200, 2, 1e-6, 0.01, 10, true);
fprintf('MATLAB finished\n');