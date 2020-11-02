function ensemble = cnnEnsembleSelection(classifiers, X, y)

y = categorical(y);

accuracies = zeros(1, length(classifiers));
for i=1:length(classifiers)
    accuracies(1,i) = mean(getCNNPred(classifiers{1,i}, X) == y);
end

if length(classifiers) == 1
    ensemble = classifiers;
    return
end
threshold = max(accuracies);
selected = [];

for i=1:length(accuracies)
    if accuracies(i) >= threshold
        selected(i) = 1;
    else
        selected(i) = 0;
    end
end
selected = find(selected);
ensemble = {};

for j=1:length(selected)
    ensemble{1,j} = classifiers{1,selected(j)};
end
end





