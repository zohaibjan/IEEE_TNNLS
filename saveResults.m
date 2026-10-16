function saveResults(results, p_name)
if (exist([pwd filesep 'results.csv'], 'file') == 0)
    fid = fopen([pwd filesep 'results.csv'], 'w');
    if fid == -1
        error('Research:ResultsOpenFailed', 'Cannot open results.csv for writing.');
    end
    fileCleanup = onCleanup(@() fclose(fid));
    fprintf(fid, '%s, %s,%s,%s, %s, %s, %s, %s, %s, %s, %s \n', ...
        'No of Runs','Data Set','MajVote Accuracy', 'Std dev.', 'Cluster based accuracy', 'Std. dev', ...
        'ES Accuracy', 'ES Std Dev.', 'Each Class', 'Without balance acc', 'Without balance std');
elseif (exist([pwd filesep 'results.csv'], 'file') == 2)
    fid = fopen([pwd filesep 'results.csv'], 'a');
    if fid == -1
        error('Research:ResultsOpenFailed', 'Cannot open results.csv for writing.');
    end
    fileCleanup = onCleanup(@() fclose(fid));
end
fprintf(fid, '%f, ', results.runs);
fprintf(fid, '%s, ', p_name);
fprintf(fid, '%f,%f, %f, %f, %f, %f, %f, %f, %f\n', ...
    results.majVote, results.mstd_dev, results.clusterBased,...
    results.cstd_dev, results.e_mVote, results.e_mVote_dev, results.eachClass,...
    results.allMVote, results.allMVote_dev);

end



