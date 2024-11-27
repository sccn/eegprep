clear
count = 1;
[STUDY, ALLEEG] = pop_loadstudy('filename', 'stern_3designs.study', 'filepath', '/Users/arno/Downloads/STUDYstern');

for iEEG = 1:3:length(ALLEEG)
    fileName = fullfile(STUDY.datasetinfo(iEEG).filepath, STUDY.datasetinfo(iEEG).filename);
    results(count) = eeg_iclabelcompare_features(fileName);
    count = count+1;
end

% Compute averages for each field
fields = fieldnames(results); % Get field names
numFields = length(fields);
numElements = length(results);
maxes = struct();
for i = 1:numFields
    fieldName = fields{i};
    values = [results.(fieldName)]; % Concatenate field values across the structure array
    maxes.(fieldName) = max(values); % Compute the mean
end

% Display the results
disp('Max for each field:');
disp(maxes);

%%
figure('Position',[470   731   904   505], 'color', 'w')
subplot(1,3,1);
hist(-log10([results.topoplotdiff])); 
xlabel('% difference (decimals)')
title('A. Scalp topography')

subplot(1,3,2);
hist(-log10([results.psddiff])); 
xlabel('% difference (decimals)')
title('B. Power spectrum')

subplot(1,3,3);
hist(-log10([results.autocorrdiff])); 
xlabel('% difference (decimals)')
title('C. Autocorrelation')

setfont(gcf, 'fontsize', 18)

