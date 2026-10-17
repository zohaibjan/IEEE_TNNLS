function acc = fusion(ensemble, centroids, test)
%FUSION Route each sample using feature-only centroids in raw feature space.
if isempty(ensemble) || numel(ensemble) ~= numel(centroids)
    error('Research:CentroidAlignment', 'Supply one centroid for each trained model.');
end
X = test(:, 1:end-1);
truth = numericLabels(test(:, end));
centers = zeros(numel(centroids), size(X, 2));
for j = 1:numel(centroids)
    if numel(centroids{j}) ~= size(X, 2) || any(~isfinite(centroids{j}(:)))
        error('Research:CentroidDimension', 'Centroids must match the feature dimensions.');
    end
    centers(j, :) = centroids{j};
end
labels = zeros(size(X, 1), 1);
for i = 1:size(X, 1)
    offsets = bsxfun(@minus, centers, X(i, :));
    distances = sum(offsets .^ 2, 2);
    nearest = find(distances == min(distances));
    votes = zeros(1, numel(nearest));
    for k = 1:numel(nearest)
        j = nearest(k);
        votes(k) = predictLabels(ensemble{j}, X(i, :));
    end
    labels(i) = votePredictions(votes);
end
acc = mean(labels == truth);
end
