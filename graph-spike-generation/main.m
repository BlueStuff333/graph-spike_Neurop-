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
n_graphs = 10^6; % num graphs to make
train_test = 0.8;
n_zeros = ceil(log10(n_graphs));
adj_lists = cell(n_graphs, 1);
firings_list = cell(n_graphs, 1);
tic
parfor i = 1:n_graphs
    % randomize parameters
    n = 500 + randi([-100, 100]); % num neurons to generate
    r = 0.8; % proportion of neurons that are E
    Ne = floor(0.8*n); Ni = n-Ne;
    i_scaling = 3; % how much stronger than excitation we want inhibition to be

    %% multifractal parameters
    R = randi(5); % resolution
    % build p_ij matrices
    P = zeros(2,2,R);
    for r_idx = 1:R
        P(:,:,r_idx) = rand(2,2);
    end
    % P = repmat(P, 1, 1, R);
    % randomize side length
    l1 = rand();
    L = [l1, 1-l1];
    M = length(L); K = randi([1,5]);
    isDirected = 1; isBinary = 0;


    % generate the graph
    PK = calcPK_weighted(M, K, P);
    [LK, LKcum] = calcLK(M, K, L);
    adj_lists{i} = generateNetworkWMGM(PK, LK, LKcum, n, isDirected, isBinary);

    % decide which neurons are e and i
    idx = randperm(n, Ni);
    i_locs = zeros(n, 1);
    i_locs(idx) = 1;
    e_locs = ones(n,1) - i_locs;
    adj_lists{i}(:,logical(i_locs)) = -1 * i_scaling * adj_lists{i}(:,logical(i_locs));

    % generate the spikes
    [firings_list{i}, stl, y] = generateSpikes(adj_lists{i}, e_locs, i_locs);

    % save the graph and its spikes in a .mat file
    filename = sprintf(['graph-spike_%0', num2str(n_zeros), 'd.mat'], i);
    if( i <= n_graphs * train_test)
        out_path = fullfile(batchdir, 'train', filename);
    else
        out_path = fullfile(batchdir, 'test', filename);
    end
    
    S = struct('adj', adj_lists{i}, 'e_locs', e_locs, 'i_locs', i_locs, 'firings', firings_list{i});
    save(out_path, '-fromstruct', S);

    %clc; 
    disp(num2str(i));
end
disp(sprintf('Graph spike gen time: %.2f', toc));

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