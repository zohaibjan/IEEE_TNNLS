function program = mainProgram-singlerun()

% Problem = dataSetNames();                 % Get list of dataset names
addpath(genpath('P-Data'));
Problem = {'appendicitis','balance','banknote','breast-cancer-wisconsin',...
    'bupa','fertility','glass','haberman',....
    'hayes-roth','heart','hepatitis','iris',....
    'pima_diabetec','planrelax','segment2','sonar',....
    'spectfheart','statimag','teaching','thyroid',....
    'vehicle','wdbc','wine','zoo'};
% 
% Problem = {'segment'};

%% Model SETTINGS
params.eachClass = 20;                     % how many clusters for each class
params.numOfFolds = 10;                   % Create CROSS VALIDATION FOLDS
params.noOfClusters = 5;                  % For nth root of clustering
params.classifiers = {'KNN'};
params.trainFunctionANN={'trainlm','trainbfg','trainrp','trainscg','traincgb','traincgf','traincgp','trainoss','traingdx'};
params.trainFunctionDiscriminant = {'pseudoLinear','pseudoQuadratic'};
params.kernelFunctionSVM={'gaussian','polynomial','linear'};

%% MAIN LOOP
for i=1:length(Problem)
    for j=1:20
        majVote = [];
        clusterBased = [];
        e_mVote = [];
        for runs = 1:30
            p_name = Problem{i};
            params.eachClass = 20;
            results = runTraining(p_name, params);
            majVote(runs) = results.majVote;
            clusterBased(runs) = results.clusterBased;
            e_mVote(runs) = results.e_mVote;
        end
        results.eachClass = j;
        results.majVote = mean(majVote);
        results.mstd_dev = std(majVote);
        results.clusterBased = mean(clusterBased);
        results.cstd_dev = std(clusterBased);
        results.e_mVote = mean(e_mVote);
        results.e_mVote_dev = std(e_mVote);
        saveResults(results, p_name);
        disp(fprintf('%s:******\nMajvote :  %f \nFusion :  %f\n\n',p_name, results.majVote, results.clusterBased));
    end
end









