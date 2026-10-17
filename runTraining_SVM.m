function results = runTraining_SVM(p_name , params, j)

params.eachClass = j;
params.p_name = p_name;
mVote = [];
e_mVote = [];
clusterBased = [];
allMVote = [];
data = load(p_name);
data = [data.X , data.y];

%% Create CROSS VALIDATION FOLDS
numOfFolds = params.numOfFolds;
cvFolds = cvpartition(data(:,end), 'KFold', numOfFolds);

%% ITERATE OVER THE NUMBER OF FOLDS
% Zero runs the same parfor body locally for a lightweight CI smoke test.
maxWorkers = Inf;
if isfield(params,'maxWorkers')
    maxWorkers = params.maxWorkers;
end
parfor (f=1:numOfFolds, maxWorkers)
    classifiers= {};
    classifierIndex = 1;
    allClassifiers= {};
    allClassifierIndex = 1;
    
    idx = cvFolds.test(f);
    trainData = data(~idx,:); % trainSVM standardizes predictors using training data.
    testData = data(idx,:);
    idxs = validationHoldout(trainData(:,end),0.01);
    validationData = trainData(idxs,:);
    trainData = trainData(~idxs, :);
    
    trainX = trainData(:, 1:end-1);
    trainy = trainData(:, end);
    
    testX = testData(:, 1:end-1);
    testy = testData(:, end);
    
    valX = validationData(:, 1:end-1);
    valy = validationData(:, end);
    
    trainX(isnan(trainX)) = -1;
    testX(isnan(testX)) = -1;
    valX(isnan(valX)) = -1;
    
    allClusters = generateClusters(trainX, trainy, params);
    [balancedClusters, centroids] = balanceClusters(allClusters, [trainX trainy]);
    
    for allClusterIndex = 1:numel(allClusters)
        cluster = allClusters{allClusterIndex};
        X = cluster.train(:,1:end-1);
        y = cluster.train(:,end);
        allClassifiers{allClassifierIndex} = trainSVM(X, y, valX, valy);
        allClassifierIndex = allClassifierIndex + 1;
    end
    
    
    for balancedIndex = 1:numel(balancedClusters)
        clusterData = balancedClusters{balancedIndex};
        X = clusterData(:,1:end-1);
        y = clusterData(:,end);
        classifiers{classifierIndex} = trainSVM(X, y, valX, valy);
        classifierIndex = classifierIndex + 1;
    end
    
    
    ensemble = ensembleSelection(classifiers, valX, valy);
    mVote(f) = majVote(classifiers, testX, testy);
    e_mVote(f) = majVote(ensemble, testX, testy);
    clusterBased(f) = fusion(classifiers, centroids, [testX, testy]);
    allMVote(f) = majVote(allClassifiers, testX, testy);
    
end
results.majVote = mean(mVote);
results.clusterBased = mean(clusterBased);
results.e_mVote = mean(e_mVote);
results.allMVote = mean(allMVote);
end




