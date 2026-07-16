function tests = test_spher
% TEST_SPHER  Tests for spher.m (sphering matrix: 2*inv(sqrtm(cov(data')))).
% Properties: symmetric; whitens the channel covariance so that
% S*C*S' = 4*I (since the factor is 2). Single channel has a closed form.
tests = functiontests(localfunctions);
end

function testSingleChannelClosedForm(testCase)
% 1 channel: cov(data') = var(data) (N-1 normalized) = 2.5 -> 2/sqrt(2.5).
data = [1 2 3 4 5];
expected = 2 / sqrt(2.5);
verifyEqual(testCase, spher(data), expected, 'RelTol', 1e-10);
end

function testSymmetric(testCase)
% Sphering matrix of a symmetric PSD covariance is symmetric.
data = [2 1 0 1 3 2; 0 2 1 3 1 0; 1 0 3 1 2 1];
S = spher(data);
verifyLessThan(testCase, norm(S - S.'), 1e-9);
end

function testWhitensCovarianceToFourI(testCase)
% S = 2*inv(sqrtm(C)) -> S*C*S' = 4*I.
data = [2 1 0 1 3 2; 0 2 1 3 1 0; 1 0 3 1 2 1];
S = spher(data);
C = cov(data');
verifyLessThan(testCase, norm(S * C * S.' - 4 * eye(3)), 1e-6);
end

function testNoArgShowsHelp(testCase)
% nargin<1 branch: prints help and returns without error.
out = evalc('spher()');
verifyTrue(testCase, ~isempty(out));
end
