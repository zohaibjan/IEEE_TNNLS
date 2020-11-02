function results = runTraining(p_name , params, j)
warning('off','all');
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
parfor f=1:numOfFolds
    classifiers= {};
    classifierIndex = 1;
    allClassifiers= {};
    allClassifierIndex = 1;
    
    idx = cvFolds.test(f);
    trainData = zscore(data(~idx,:));
    testData = data(idx,:);
    cv = cvpartition(trainData(:,end), 'holdout', 0.01);
    idxs = cv.test;
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
    
    for c=allClusters
        X = c{1,1}.train(:, 1:end-1);
        y = c{1,1}.train(:, end);
        allClassifiers{allClassifierIndex} = trainSVM(X, y, valX, valy);
        allClassifierIndex = allClassifierIndex + 1;
    end
    
    
    for c=balancedClusters
        X = c{1,1}(:, 1:end-1);
        y = c{1,1}(:, end);
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




