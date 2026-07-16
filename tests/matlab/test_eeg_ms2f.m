function tests = test_eeg_ms2f
% TEST_EEG_MS2F  Tests for eeg_ms2f.m (epoch latency in ms -> nearest 1-based
% epoch frame). outf = 1 + round((pnts-1)*(ms/1000 - xmin)/(xmax - xmin)).
% Closed-form, release-invariant expectations.
tests = functiontests(localfunctions);
end

function eeg = makeEEG(xmin, xmax, pnts)
eeg = struct('xmin', xmin, 'xmax', xmax, 'pnts', pnts);
end

function testFirstFrameAtXmin(testCase)
% ms at xmin -> first frame (1-based).
EEG = makeEEG(0, 1, 1001);
verifyEqual(testCase, eeg_ms2f(EEG, 0), 1);
end

function testLastFrameAtXmax(testCase)
% ms at xmax -> last frame == pnts.
EEG = makeEEG(0, 1, 1001);
verifyEqual(testCase, eeg_ms2f(EEG, 1000), 1001);
end

function testMidpoint(testCase)
% 500 ms -> 0.5 s -> frame 1 + 1000*0.5 = 501.
EEG = makeEEG(0, 1, 1001);
verifyEqual(testCase, eeg_ms2f(EEG, 500), 501);
end

function testRoundsToNearest(testCase)
% 499.4 ms -> 1 + round(499.4) = 500.
EEG = makeEEG(0, 1, 1001);
verifyEqual(testCase, eeg_ms2f(EEG, 499.4), 500);
end

function testEpochCenterNegativeXmin(testCase)
% Epoched data xmin<0: ms=0 -> centre frame.
EEG = makeEEG(-1, 1, 2001);
verifyEqual(testCase, eeg_ms2f(EEG, 0), 1001);
end

function testBelowRangeErrors(testCase)
% ms below xmin -> error('time out of range').
EEG = makeEEG(0, 1, 1001);
threw = false;
try
    eeg_ms2f(EEG, -1);
catch ME
    threw = true;
    verifyTrue(testCase, contains(ME.message, 'out of range'));
end
verifyTrue(testCase, threw);
end

function testAboveRangeErrors(testCase)
% ms above xmax -> error('time out of range').
EEG = makeEEG(0, 1, 1001);
threw = false;
try
    eeg_ms2f(EEG, 2000);
catch ME
    threw = true;
    verifyTrue(testCase, contains(ME.message, 'out of range'));
end
verifyTrue(testCase, threw);
end
