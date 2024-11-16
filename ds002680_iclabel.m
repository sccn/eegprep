clear
count = 1;
[STUDY ALLEEG] = pop_loadstudy('filename', 'stern.study', 'filepath', '/System/Volumes/Data/data/data/STUDIES/STERN');

for iEEG = 1:3:length(ALLEEG)
    fileName = fullfile(STUDY.datasetinfo(iEEG).filepath, STUDY.datasetinfo(iEEG).filename);
    res(count) = eeg_iclabelcompare(fileName);
    count = count+1;
end
