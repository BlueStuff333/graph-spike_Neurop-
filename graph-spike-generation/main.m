clear all; close all; clc;

%% Setup

% Multithreading
if isempty(gcp('nocreate')) % if no pool, create new
    parpool("threads");   % or parpool(N) for process-based pool
end

% generated network storage directory
batchstamp = string(datetime('now'), "dd-MMM-yyyy_HH-mm-ss");
batchdir = fullfile('networks', batchstamp);
mkdir(batchdir);
mkdir(fullfile('networks', batchstamp, 'train'));
mkdir(fullfile('networks', batchstamp, 'test'));

%% Randomized parameter sets of the following forms:
% K = 1:5, M = 2, p_ij(r) and l_i same
% K = 4, M = 2, R = 3, 4 different sets of initial unit square model parameters
% K = 5, M = 2, R = 3, l_i [(0.1, 0.9), (0.3, 0.7), (0.5, 0.5), (0.7, 0.3), (0.9, 0.1)], p_ij(r) same
% R = 1:4
%% generate n graph-spiking pairs
n_graphs = 200; % num graphs to make
train_test = 0.8;
n_zeros = ceil(log10(n_graphs));
isDirected = 1; isBinary = 0;
% adj_lists = cell(n_graphs, 1);    % These cause memory issues for large n_graphs
% firings_list = cell(n_graphs, 1); % TODO develop a separate pipeline for visualization from saved .mat files
P0 = [0.8 0.5; 0.5 0.4]; % use a base probability matrix and apply noisy perturbations
noise_level = 0.1;
densities = cell(n_graphs, 1);
tic
parfor i = 1:n_graphs
    % randomize parameters
    n = 500 + randi([-100, 100]); % num neurons to generate
    r = 0.8; % proportion of neurons that are E
    Ne = floor(0.8*n); Ni = n-Ne;
    i_scaling = 3; % how much stronger than excitation we want inhibition to be

    %% multifractal parameters
    R = randi(2); % resolution, prefer lower R for sparser graphs
    % build p_ij matrices
    P = zeros(2,2,R);
    for r_idx = 1:R
        P(:,:,r_idx) = P0 + noise_level*(rand(2,2) - 0.5);
        P(:,:,r_idx) = min(max(P(:,:,r_idx), 0.1), 0.9); % clamp
    end
    % P = repmat(P, 1, 1, R);
    % randomize side length
    l1 = 0.2 + 0.6 * rand(); % [0.2, 0.8]
    L = [l1, 1-l1];
    M = length(L); 
    K = randi([3,4]); % prefer higher K for sparser graphs

    % generate the graph
    PK = calcPK_weighted(M, K, P);
    [LK, LKcum] = calcLK(M, K, L);
    [adj] = generateNetworkWMGM(PK, LK, LKcum, n, isDirected, isBinary);
    densities{i} = nnz(adj) / numel(adj);

    % decide which neurons are e and i
    idx = randperm(n, Ni);
    i_locs = zeros(n, 1);
    i_locs(idx) = 1;
    e_locs = ones(n,1) - i_locs;
    adj(:,logical(i_locs)) = -1 * i_scaling * adj(:,logical(i_locs));

    % generate the spikes
    [firings, stl, y] = generateSpikes(adj, e_locs, i_locs);

    % save the graph and its spikes in a .mat file
    filename = sprintf(['graph-spike_%0', num2str(n_zeros), 'd.mat'], i);
    if( i <= n_graphs * train_test)
        out_path = fullfile(batchdir, 'train', filename);
    else
        out_path = fullfile(batchdir, 'test', filename);
    end
    
    S = struct( ...
        'adj', adj, ... 
        'e_locs', e_locs, ... 
        'i_locs', i_locs, ...
        'firings', firings, ...
        ... %%% WMGM params
        'P', P, ...
        'L', L, ...
        'M', M, ...
        'K', K, ...
        'R', R ...
    );
    save(out_path, '-fromstruct', S);

    % free memory
    adj = []; firings = []; e_locs = []; i_locs = [];

    %clc; 
    disp(num2str(i));
    disp("Density: " + num2str(densities{i}));
    disp("WMGM params:");
    disp(['N: ', num2str(n), ', K: ', num2str(K), ', M: ', num2str(M), ', R: ', num2str(R)]);
    disp(['L: ', num2str(L)]);
end
disp(sprintf('Graph spike gen time: %.2f', toc));
avg_density = mean(cell2mat(densities));
disp(['Average density: ', num2str(avg_density)]);

% TODO enable/disable visualization at will
% for i = 1:n_graphs
%     % create & save jpg of the graph
%     plotname = sprintf('graph_%d', i);
%     plotsdir = fullfile(batchdir, 'plots'); mkdir (plotsdir);
%     node_colors = saveCliqueColors(adj_lists{i}, plotsdir, plotname);
%     % save the rasters in a jpg
%     raster_dir = fullfile(batchdir, 'rasters'); mkdir(raster_dir);
%     saveRasterPlots(firings_list{i}, node_colors, raster_dir, i);
% end

disp('complete');