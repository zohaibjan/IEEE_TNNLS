function tests = testToolboxes
% Small synthetic runtime checks; no paper dataset or accuracy benchmark.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
testCase.TestData.originalPath = path;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root, fullfile(root, 'tests', 'fixtures'), '-begin');
rng(42, 'twister');
end

function teardownOnce(testCase)
path(testCase.TestData.originalPath);
end

function [X, y] = syntheticData()
X = [randn(16, 6) - 1; randn(16, 6) + 1];
y = [zeros(16, 1); 10 * ones(16, 1)];
end

function testActualSvmAndTreePredictions(testCase)
[X, y] = syntheticData();
a.name = 'SVM';
a.model = fitcecoc(X, y, 'Learners', templateSVM('Standardize', true));
b.name = 'DT'; b.model = fitctree(X, y);
preds = predictionMatrix({a, b}, X);
verifySize(testCase, preds, [32 2]);
verifyTrue(testCase, all(ismember(preds(:), [0 10])));
end

function testActualNeuralNetworkLabelVocabulary(testCase)
[X, y] = syntheticData();
params.trainFunctionANN = {'trainscg'};
net = trainNN(X, y, params, [3 1 3]);
labels = getNNPredict(net, X(1:3, :));
verifySize(testCase, labels, [3 1]);
verifyTrue(testCase, all(ismember(labels, [0 10])));
verifyEqual(testCase, net.userdata.classLabels, [0; 10]);
end

function testActualCnnTrainingAndNumericLabels(testCase)
[X, y] = syntheticData();
net = getCNN(X, y, 2);
labels = getCNNPred(net, X(1:3, :));
verifySize(testCase, labels, [3 1]);
verifyTrue(testCase, all(ismember(labels, [0 10])));
end
