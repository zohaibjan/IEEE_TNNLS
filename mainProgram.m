function program = mainProgram()

% Problem = dataSetNames();                 % Get list of dataset names
addpath(genpath('P-Data'));
% Problem = {'breast-cancer-wisconsin','spect','flags','colic',...
%     'glass','haberman','heart','BC',...
%     'hepatitis', 'iris', 'segment2', 'sonar'};

Problem = {'students'};

%% Model SETTINGS
params.eachClass = 20;                     % how many clusters for each class
params.numOfFolds = 5;                   % Create CROSS VALIDATION FOLDS
params.noOfClusters = 5;                  % For nth root of clustering
params.trainFunctionANN={'trainlm','trainbfg','trainrp','trainscg','traincgb','traincgf','traincgp','trainoss','traingdx'};
params.trainFunctionDiscriminant = {'pseudoLinear','pseudoQuadratic'};
params.kernelFunctionSVM={'gaussian','polynomial','linear'};
% params.classifiers = {'KNN', 'SVM', 'NB', 'DT', 'ANN', 'DISCR'};
params.classifiers = {'SVM'};


%% MAIN LOOP
for i=1:length(Problem)
    parfor j=1:10
        majVote = [];
        clusterBased = [];
        e_mVote = [];
        allMVote = [];
        for runs = 1:5
            p_name = Problem{i};
            results = runTraining(p_name, params, j);
            majVote = results.majVote;
            clusterBased = results.clusterBased;
            e_mVote = results.e_mVote;
            allMVote = results.allMVote;  
            results.eachClass = j;
            results.majVote = majVote;
            results.mstd_dev = 0;
            results.clusterBased = clusterBased;
            results.cstd_dev = 0;
            results.e_mVote = e_mVote;
            results.e_mVote_dev = 0;
            results.allMVote = allMVote;
            results.allMVote_dev = 0;
            results.runs = runs;
            saveResults(results, p_name);
            disp(fprintf('%s:******\nMajvote :  %f \nFusion :  %f\n\n',p_name, results.majVote, results.clusterBased));
        end
    end
end









