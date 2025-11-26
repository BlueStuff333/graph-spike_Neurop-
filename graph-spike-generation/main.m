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

% multifractal parameters
P = [
    0.8,0.5;
    0.5,0.4
    ]; % p_ij(r) matrix
L = [0.7,0.3]; % side lengths
M = length(L); K = 3;
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
    PK = calcPK(M, K, idx, P);
    [LK, LKcum] = calcLK(M, K, idx, L);
    [adj] = generateNetworkMF(PK, LK, LKcum, n, isDirected, isBinary);
    
    % display the graph
    G = digraph(adj);
    plot(G);
    % % display graph statistics
    % fprintf('Connections per neuron: %.1f\n', nnz(adj)/n);
    % fprintf('Density: %.4f\n', nnz(adj)/n^2);
    % histogram(sum(adj>0, 2)); title('In-degree distribution');

    % decide which neurons are e and i
    idx = randperm(n, Ni);
    i_locs = zeros(n, 1);
    i_locs(idx) = 1;
    e_locs = ones(n,1) - i_locs;
    adj(:,logical(i_locs)) = -1 * i_scaling * adj(:,logical(i_locs));

    % generate the spikes
    [firings, stl, y] = generateSpikes(adj, e_locs, i_locs);

    % save the graph
    filename = sprintf(['graph-spike_%0', num2str(n_zeros), 'd.mat'], i);
    filepath = fullfile(batchdir, filename);
    save(filepath, 'adj', 'e_locs', 'i_locs', 'firings');
    
end
disp("complete")