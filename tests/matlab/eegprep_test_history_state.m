function [allcom, lastcom] = eegprep_test_history_state(varargin)
% Set/read the real globals used by eegh, without invoking history operations.
if nargin
    clear global ALLCOM LASTCOM;
end
global ALLCOM LASTCOM;
if nargin
    ALLCOM = varargin{1};
    LASTCOM = varargin{2};
end
allcom = ALLCOM;
lastcom = LASTCOM;
end
