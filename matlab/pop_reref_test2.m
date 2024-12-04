eeglab cont
EEG = pop_runica(EEG, 'icatype', 'runica', 'rndreset','yes','interrupt','on');

EEG2 = EEG;
EEG2.icaact = [];
EEG2.icaweights = pinv(EEG.icawinv);
EEG2.icasphere  = eye(EEG.nbchan);
EEG2 = eeg_checkset(EEG2);
% EEG.icaact(1:10,1:10) - EEG2.icaact(1:10,1:10)

% regular average reference
EEG3 = pop_reref(EEG,[]);
EEG3.icaact = [];
EEG3 = eeg_checkset(EEG3);

% manual average reference
EEG4 = EEG;
EEG4.data = bsxfun(@minus, EEG4.data, mean(EEG4.data));
EEG4.icawinv = bsxfun(@minus, EEG4.icawinv , mean(EEG4.icawinv));
EEG4.icaweights = pinv(EEG4.icawinv);
EEG4.icasphere  = eye(EEG4.nbchan);
EEG4.icaact     = EEG4.icaweights*EEG4.data(:,:);

EEG.icaact(1:10,1:10) - EEG4.icaact(1:10,1:10)


% Compute the inverse weight matrix (mixing matrix)
X = rand(10,1000);
[W,S] = runica(X);
W = W*S;
M = pinv(W);
A = W*X;

X2 = X - mean(X, 1);
M2 = M - mean(M, 1);
W2 = pinv(M2);
A2 = W2*X2;

A2(:,1:10)-A(:,1:10)

X2 = X - mean(X, 1);
W2 = W - mean(W, 2);
A2 = W2 * X2;

difference = norm(A2 - A); % Should be close to zero
disp(['Difference between A2 and A: ' num2str(difference)]);
