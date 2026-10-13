function eegprep_test_setfont_gcf(varargin)
% setfont needs the live HG2 figure. double(gcf) does not survive MAT transport.
setfont(gcf, varargin{:});
end
