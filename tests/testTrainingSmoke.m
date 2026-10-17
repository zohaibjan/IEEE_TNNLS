function tests = testTrainingSmoke
tests = functiontests(localfunctions);
end
function testBothFoldDrivers(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
rng(22,'twister');
X = [randn(80,3)-1;randn(80,3)+1];
y = [zeros(80,1);10*ones(80,1)];
file = [tempname,'.mat'];
save(file,'X','y');
cleanup = onCleanup(@() delete(file));
params.numOfFolds = 2;
params.maxWorkers = 0;
params.classifiers = {'DT'};
drivers = {@runTraining,@runTraining_SVM};
for k = 1:numel(drivers)
    result = drivers{k}(file,params,1);
    values = [result.majVote,result.clusterBased,result.e_mVote,result.allMVote];
    verifyTrue(testCase,all(isfinite(values) & values>=0 & values<=1));
end
end
function testTinyHoldoutPreservesBothSides(testCase)
y = [0;0;10;10];
selected = validationHoldout(y,0.01);
verifyEqual(testCase,sort(y(selected)),[0;10]);
verifyEqual(testCase,sort(y(~selected)),[0;10]);
large = [zeros(1000,1);10*ones(1000,1)];
verifyEqual(testCase,sum(validationHoldout(large,0.01)),20);
end
function testSingletonClassRejectsSplit(testCase)
verifyError(testCase,@() validationHoldout([0;10;10],0.1),'Research:InsufficientClassSamples');
end
