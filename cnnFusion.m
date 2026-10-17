function acc = cnnFusion(classifiers, X, y)
%CNNFUSION Vote using the numeric names of CNN output categories.
if isempty(classifiers)
    error('Research:EmptyEnsemble', 'At least one CNN is required.');
end
predictions = zeros(size(X, 1), numel(classifiers));
for i = 1:numel(classifiers)
    predictions(:, i) = getCNNPred(classifiers{i}, X);
end
acc = mean(votePredictions(predictions) == numericLabels(y));
end
