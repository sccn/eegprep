function tests = test_numdim
% TEST_NUMDIM  Tests for numdim.m (effective number of sources via
% eigenvalue-entropy of the channel second-order matrix A*A'/100).
%
% Expectations are closed-form / release-invariant:
%   - single channel             -> lambda == 1
%   - orthogonal equal-energy    -> lambda == nchan (uniform eigenvalues)
%   - identical channels (rank-1) -> lambda == 1 (one dominant eigenvalue)
%   - full-rank                  -> real, 1 < lambda < nchan
tests = functiontests(localfunctions);
end

function testSingleChannel(testCase)
% One channel -> one (normalized) eigenvalue == 1 -> entropy 0 -> lambda 1.
a = rand(1, 50) + 0.5;            % nonzero energy
verifyEqual(testCase, numdim(a), 1, 'AbsTol', 1e-10);
end

function testTwoChannelOrthogonal(testCase)
% A*A' = 2*I -> equal eigenvalues -> lambda == nchan == 2.
A = [1 1; 1 -1];
verifyEqual(testCase, numdim(A), 2, 'AbsTol', 1e-10);
end

function testOrthogonalEqualEnergy(testCase)
% Hadamard rows are orthogonal with equal norm -> A*A' = 4*I -> lambda == 4.
A = hadamard(4);
verifyEqual(testCase, numdim(A), 4, 'AbsTol', 1e-9);
end

function testRankDeficientApproxOne(testCase)
% Identical channels -> rank-1 -> one dominant eigenvalue -> ~1 effective dim
% (eig yields tiny nonzero eigenvalues, so the result is finite, not NaN).
A = ones(3, 10);
verifyEqual(testCase, numdim(A), 1, 'AbsTol', 1e-6);
end

function testFullRankBounds(testCase)
% Full-rank random data: real scalar strictly between 1 and nchan.
rng(42);
A = randn(5, 200);
v = numdim(A);
verifyTrue(testCase, isreal(v));
verifyGreaterThan(testCase, v, 1);
verifyLessThan(testCase, v, 5);
end

function testNoArgShowsHelp(testCase)
% nargin<1 branch: prints help and returns without error.
out = evalc('numdim()');
verifyTrue(testCase, ~isempty(out));
end
