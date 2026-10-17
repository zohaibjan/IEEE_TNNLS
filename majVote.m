function acc = majVote(classifiers, X, Y)
%MAJVOTE Score validated votes without placeholder predictions.
acc = mean(votePredictions(predictionMatrix(classifiers, X)) == numericLabels(Y));
end
