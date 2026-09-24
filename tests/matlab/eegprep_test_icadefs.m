function defaults = eegprep_test_icadefs()
% Expose only the workspace variables asserted by the upstream script test.
icadefs;
defaults = struct();
if exist('ICABINARY', 'var')
    defaults.ICABINARY = ICABINARY;
end
if exist('DEFAULT_SRATE', 'var')
    defaults.DEFAULT_SRATE = DEFAULT_SRATE;
end
end
