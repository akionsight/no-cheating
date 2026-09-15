outcomes = readtable('outcome_labels.csv');

% Build matching keys for both sides
mat_keys = strcat("task", string(task_id), "_", string(difficulty), "_", ...
    string(condition), "_seed", string(seed));
csv_keys = strcat("task", string(outcomes.task_id), "_", string(outcomes.difficulty), "_", ...
    string(outcomes.condition), "_seed", string(outcomes.seed));

[tf, loc] = ismember(mat_keys, csv_keys);
assert(all(tf), "Some rollouts in X have no matching outcome label — check for missing/extra files");

outcome = outcomes.outcome(loc);   % now correctly aligned to X's row order

honest_idx = strcmp(outcome, "honest_correct");
fabricated_idx = strcmp(outcome, "fabricated");

for L = 1:numel(LAYERS)
    feats = squeeze(X(:,L,:));
    grp = [repmat({'honest'}, sum(honest_idx),1); repmat({'fabricated'}, sum(fabricated_idx),1)];
    combined = [feats(honest_idx,:); feats(fabricated_idx,:)];

    cv = cvpartition(grp, 'KFold', 5);
    acc = zeros(cv.NumTestSets,1);
    for k = 1:cv.NumTestSets
        mdl = fitcdiscr(combined(training(cv,k),:), grp(training(cv,k)), 'DiscrimType','linear');
        pred = predict(mdl, combined(test(cv,k),:));
        acc(k) = mean(strcmp(pred, grp(test(cv,k))));
    end
    fprintf("Layer %d: %.2f%% (chance = 50%%)\n", LAYERS(L), 100*mean(acc));
end