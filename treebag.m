Problem={'adult','australian','balance','banknote',...
    'breast-cancer-wisconsin','ecoli','haberman','ionosphere','iris'....
    'liver','page-blocks','pima_diabetec','segment','sonar','statimag',...
    'teaching','thyroid','vehicle','vowel','wdbc','wine','DNA',...
    'fertility','heart','letter-recognition','hepatitis','bupa',...
    'transfusion','zoo','hayes-roth'};

% Problem={'letter-recognition'};
parfor i=1:length(Problem)
    accuracy = []
    for runs = 1:10
        warning('off','all');
        p_name = Problem{i};
        %% Create CROSS VALIDATION FOLDS
        numOfFolds = 10;
        data = load([pwd,filesep,'P-Data',filesep, p_name]);
        data = [data.X data.y];
        cvFolds = cvpartition(data(:,end), 'KFold', numOfFolds);
        
        %% RECORD KEEPING VARIABLES
        avgAccuracy = [];
        
        %% ITERATE OVER THE NUMBER OF FOLDS
        for fold=1:numOfFolds
            trainData = data(cvFolds.training(fold),:);
            testData = data(cvFolds.test(fold),:);
            
            trainX = trainData(:,1:end-1);
            trainY = trainData(:,end);
            testX = testData(:,1:end-1);
            testY = testData(:, end);
            
            classifier = TreeBagger(50, trainX, trainY);
            preds = predict(classifier, testX);
            
            for n=1:length(preds)
                predictY(n,1)=str2num(cell2mat(preds(n)));
            end
            eval = mean(testY==predictY);
            avgAccuracy(fold) = eval;
        end
        accuracy(runs) = mean(avgAccuracy)
    end
    if (exist([pwd filesep 'results-Tree.csv'], 'file') == 0)
        fid = fopen([pwd filesep 'results-Tree.csv'], 'w');
        fprintf(fid, '%s,%s,%s\n', ...
            'Data Set', 'Method','Accuracy' ...
            );
    elseif (exist([pwd filesep 'results-Tree.csv'], 'file') == 2)
        fid = fopen([pwd filesep 'results-Tree.csv'], 'a');
    end
    acc
    fprintf(fid, '%s,', p_name);
    fprintf(fid, '%s,', "Tree");
    fprintf(fid, '%f±%f\n',mean(accuracy),std(accuracy));
    fclose(fid);
end


