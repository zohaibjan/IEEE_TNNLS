function selected = validationHoldout(y, fraction)
%VALIDATIONHOLDOUT Stratified split with training and validation rows per class.
y = numericLabels(y);
validateattributes(fraction,{'numeric'},{'scalar','real','finite','>',0,'<=',0.5});
selected = false(numel(y),1);
classes = unique(y);
for k = 1:numel(classes)
    rows = find(y==classes(k));
    if numel(rows)<2
        error('Research:InsufficientClassSamples', ...
            'Each training-fold class needs at least two rows for validation separation.');
    end
    count = max(1,min(numel(rows)-1,round(fraction*numel(rows))));
    order = randperm(numel(rows),count);
    selected(rows(order)) = true;
end
end
