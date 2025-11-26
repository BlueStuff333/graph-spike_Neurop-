clear all; close all; clc;

batchstamp = string(datetime('now'), "dd-MMM-yyyy_HH-mm-ss");
batchdir = fullfile('networks', batchstamp);
mkdir(batchdir);

%% parameters
n_graphs = 100; % num graphs to make
n = 500 + randi([-100, 100]); % num neurons to generate
r = 0.8; % proportion of neurons that are E
Ne = floor(0.8*n); Ni = n-Ne;
i_scaling = 3; % how much stronger than excitation we want inhibition to be

%% multifractal parameters
P = [
    0.8,0.5;
    0.5,0.4
    ]; % p_ij(r) matrix
% Resolution # TODO it's just the length of P, for now duplicating P
R = 3;
P = repmat(P, 1, 1, R);
% randomize side length
l1 = rand();
L = [l1, 1-l1];
M = length(L); K = randi([1,5]);
isDirected = 1; isBinary = 0;

%% Randomized parameter sets of the following forms:
% K = 1:5, M = 2, p_ij(r) and l_i same
% K = 4, M = 2, R = 3, 4 different sets of initial unit square model parameters
% K = 5, M = 2, R = 3, l_i [(0.1, 0.9), (0.3, 0.7), (0.5, 0.5), (0.7, 0.3), (0.9, 0.1)], p_ij(r) same
% R = 1:4

%% generate n graph-spiking pairs
n_zeros = ceil(log10(n_graphs));
for i = 1:n_graphs
    % generate the graph
    idx = calcIdx(M, K);
    
    PK = calcPK_weighted(M, K, idx, P);
    % PK = calcPK(M, K, idx, P);
    [LK, LKcum] = calcLK(M, K, idx, L);
    % [adj] = generateNetworkMF(PK, LK, LKcum, n, isDirected, isBinary);
    [adj] = generateNetworkWMGM(PK, LK, LKcum, n, isDirected, isBinary);

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
    filepath = fullfile(batchdir, filename);
    save(filepath, 'adj', 'e_locs', 'i_locs', 'firings');
    
    % % create & save jpg of the graph
    % plotname = sprintf('graph_%d', i);
    % plotsdir = fullfile(batchdir, 'plots'); mkdir (plotsdir);
    % node_colors = saveCliqueColors(adj, plotsdir, plotname);
    % % save the rasters in a jpg
    % raster_dir = fullfile(batchdir, 'rasters'); mkdir(raster_dir);
    % saveRasterPlots(firings, node_colors, raster_dir);

    clc; disp(num2str(i));
end