clc;
clear vars;
close all;

files = dir('/home/akionsight/Desktop/downloaded_data/results_7b/*.mat');
LAYERS = [7 14 21 27];  % must match the indices you actually saved

X = [];       % [n_rollouts x n_layers x hidden_dim]
condition = {};
difficulty = {};
violated = [];
task_id = [];
seed = [];

for i = 1:numel(files)
    d = load(fullfile(files(i).folder, files(i).name));
    acts = d.acts;                      % [n_tokens x n_layers x hidden_dim]
    pooled = squeeze(mean(acts, 1));    % [n_layers x hidden_dim]
    X(i,:,:) = pooled;
    condition{i} = strtrim(string(d.condition));
    difficulty{i} = strtrim(string(d.difficulty));
    violated(i) = double(d.violated);
    task_id(i) = double(d.task_id);
    seed(i) = double(d.seed);
end

condition = categorical(string(condition));
difficulty = categorical(string(difficulty));


condition = categorical(condition);
difficulty = categorical(difficulty);

for L = 1:numel(LAYERS)
    layer_feats = squeeze(X(:,L,:));         % [n_rollouts x hidden_dim]
    [coeff, score] = pca(layer_feats);
    figure;
    gscatter(score(:,1), score(:,2), condition);
    title(sprintf('Layer %d - PCA colored by condition', LAYERS(L)));
end

results = table();
for L = 1:numel(LAYERS)
    feats = squeeze(X(:,L,:));
    labels = condition;   % or do pairwise: internals vs control only

    % Group k-fold by seed so no seed's rollouts appear in both train/test
    cv = cvpartition(seed, 'KFold', 5);   % partitions by unique seed groups
    acc = zeros(cv.NumTestSets,1);

    for k = 1:cv.NumTestSets
        trainIdx = training(cv,k);
        testIdx  = test(cv,k);
        mdl = fitcdiscr(feats(trainIdx,:), labels(trainIdx), 'DiscrimType','linear');
        pred = predict(mdl, feats(testIdx,:));

        true_labels = labels(testIdx);
        acc(k) = mean( pred(:) == true_labels(:) );
    end

    results = [results; table(LAYERS(L), mean(acc), std(acc), ...
        'VariableNames', {'Layer','MeanAcc','StdAcc'})];
end
disp(results)