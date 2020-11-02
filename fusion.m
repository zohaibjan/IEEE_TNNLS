function acc=fusion(ensemble, centroids, test)
decisionMatrix = zeros(size(test,1),1);

for i=1:size(test,1)
    distances = zeros(1, length(centroids));
    for j =1:length(centroids)
        distances(j) = norm(double(test(i,1:end-1) - centroids{1,j}));
    end
    index = find(distances == min(distances));
    if length(index) > 1
        preds = zeros(1, length(index));
        for k=1:length(index)
            if strcmp(ensemble{1,index(k)}.name, 'ANN') == 1
                preds(1,k) = getNNPredict(ensemble{1,index(k)}.model, test(i,1:end-1));
            else
                preds(1,k) = predict(ensemble{1,index(k)}.model, test(i,1:end-1));
            end
            
        end
        decisionMatrix(i,:) = mode(preds,2);
    elseif length(index) == 1
        if strcmp(ensemble{1,index}.name, 'ANN') == 1
            decisionMatrix(i,:) = getNNPredict(ensemble{1,index}.model, test(i,1:end-1));
        else
            decisionMatrix(i,:) = predict(ensemble{1,index}.model, test(i,1:end-1));
        end
    end
end
acc = mean(decisionMatrix == test(:,end));

end