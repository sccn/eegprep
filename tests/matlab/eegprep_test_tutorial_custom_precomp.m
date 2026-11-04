function eegprep_test_tutorial_custom_precomp(kind, STUDY, ALLEEG, EEG)
% Native anonymous-function boundary; source workflow order is Python-owned.
if strcmp(kind, 'baseline')
    callback = @(data)bsxfun(@minus, data, mean(data(:,1:410,:),2));
else
    callback = @(data)reshape(eegfilt(data(:,:), EEG(1).srate, 0,10,EEG(1).pnts,60,0,'fir1'), size(data));
end
std_precomp(STUDY, ALLEEG, 'channels', 'customfunc', callback, 'interp', 'on');
end
